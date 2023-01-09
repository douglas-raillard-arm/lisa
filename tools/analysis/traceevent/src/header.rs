use core::{borrow::Borrow, convert::TryFrom, fmt::Debug, str::from_utf8};
use std::string::ToString;

use std::{collections::BTreeMap, string::String as StdString};

use smartstring::alias::String;

use nom::{
    branch::alt,
    bytes::complete::{is_a, is_not, tag},
    character::complete::{char, multispace0, u32 as txt_u32, u64 as txt_u64},
    combinator::{all_consuming, map_res, opt},
    error::{context, ContextError, FromExternalError, ParseError},
    multi::{fold_many0, many0, separated_list0},
    number::complete::{be_u32, be_u64, le_u32, le_u64},
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    Finish as _, Parser,
};

use crate::{
    cparser::{CBasicType, CDeclaration, CExpr, CGrammar, CType},
    grammar::PackratGrammar,
    io::BorrowingRead,
    parser::{lexeme, to_str, Input, NomError},
};

pub type Address = usize;
pub type CPU = u32;
pub type PID = u32;
pub type Size = u64;
pub type SymbolName = String;
pub type TaskName = String;
pub type Identifier = String;
pub type Offset = isize;
pub type EventId = u32;

#[derive(Debug, Clone, Copy)]
pub enum Endianness {
    Big,
    Little,
}

#[derive(Debug)]
pub enum DataLocation {
    Ascii(Offset),
    Flyrecord(Vec<(CPU, (Offset, Size))>),
}

#[derive(Clone, Copy, Debug)]
pub enum LongSize {
    Bits32,
    Bits64,
}

impl From<LongSize> for u64 {
    fn from(size: LongSize) -> Self {
        match size {
            LongSize::Bits32 => 4,
            LongSize::Bits64 => 8,
        }
    }
}

impl TryFrom<u64> for LongSize {
    type Error = u64;

    fn try_from(size: u64) -> Result<Self, Self::Error> {
        match size {
            4 => Ok(LongSize::Bits32),
            8 => Ok(LongSize::Bits64),
            x => Err(x),
        }
    }
}

#[derive(Debug)]
pub struct EventDesc {}

#[derive(Debug)]
pub struct Abi {
    pub endianness: Endianness,
    pub long_size: Option<LongSize>,
}

impl Abi {
    #[inline]
    fn u32_parser<'a, E>(&self) -> impl nom::Parser<Input<'a>, u32, E>
    where
        E: ParseError<Input<'a>>,
    {
        match self.endianness {
            Endianness::Little => le_u32,
            Endianness::Big => be_u32,
        }
    }

    #[inline]
    fn u64_parser<'a, E>(&self) -> impl nom::Parser<Input<'a>, u64, E>
    where
        E: ParseError<Input<'a>>,
    {
        match self.endianness {
            Endianness::Little => le_u64,
            Endianness::Big => be_u64,
        }
    }

    #[inline]
    fn parse_u32<'a, E>(&self, input: Input<'a>) -> nom::IResult<Input<'a>, u32, E>
    where
        E: ParseError<Input<'a>>,
    {
        self.u32_parser().parse(input)
    }

    #[inline]
    fn parse_u64<'a, E>(&self, input: Input<'a>) -> nom::IResult<Input<'a>, u64, E>
    where
        E: ParseError<Input<'a>>,
    {
        self.u64_parser().parse(input)
    }
}

#[derive(Debug)]
pub struct Header {
    version: StdString,
    kernel_abi: Abi,
    userspace_long_size: LongSize,
    page_size: Size,
    event_page_size: Size,
    event_descs: Vec<EventDesc>,
    symbols: BTreeMap<Address, SymbolName>,
    const_str: BTreeMap<Address, String>,
    comms: BTreeMap<PID, TaskName>,
    options: Vec<(isize, StdString)>,
    data_loc: DataLocation,
}

#[derive(Debug, Clone, PartialEq)]
struct FieldFmt {
    declaration: CDeclaration,
    offset: Offset,
    size: Size,
}

#[derive(Debug, Clone, PartialEq)]
struct StructFmt {
    fields: Vec<FieldFmt>,
}

impl StructFmt {
    fn field_by_name<Q>(&self, name: &Q) -> Option<&FieldFmt>
    where
        Q: ?Sized,
        Identifier: Borrow<Q> + PartialEq<Q>,
    {
        for field in &self.fields {
            if &field.declaration.identifier == name.borrow() {
                return Some(field);
            }
        }
        None
    }
}

fn fixup_c_type(typ: CType, size: Size, signed: bool, abi: &Abi) -> Result<CType, HeaderError> {
    let (inferred_signed, inferred_size) = match typ {
        CType::Basic(CBasicType::Void) => (None, 0),

        CType::Basic(CBasicType::U8) => (Some(false), 1),
        CType::Basic(CBasicType::I8) => (Some(true), 1),

        CType::Basic(CBasicType::U16) => (Some(false), 2),
        CType::Basic(CBasicType::I16) => (Some(true), 2),

        CType::Basic(CBasicType::U32) => (Some(false), 4),
        CType::Basic(CBasicType::I32) => (Some(true), 4),

        CType::Basic(CBasicType::U64) => (Some(false), 8),
        CType::Basic(CBasicType::I64) => (Some(true), 8),

        CType::Struct(..) => (None, size),
        CType::Union(..) => (None, size),
        CType::Enum(..) => (Some(signed), size),

        CType::Pointer(..) => (None, abi.long_size.map(|x| x.into()).unwrap_or(size)),

        _ => (Some(signed), size),
    };
    let inferred_signed = inferred_signed.unwrap_or(signed);

    if inferred_size != size {
        return Err(HeaderError::InvalidTypeSize {
            typ,
            inferred_size,
            size,
        });
    }

    if inferred_signed != signed {
        return Err(HeaderError::InvalidTypeSign {
            typ,
            inferred_signed,
            signed,
        });
    }
    Ok(typ)
}

fn struct_fmt_parser<'abi, 'a, E>(
    abi: &'abi Abi,
) -> impl nom::Parser<Input<'a>, StructFmt, E> + 'abi
where
    E: ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, HeaderError>
        + Debug
        + 'abi,
    'a: 'abi,
{
    separated_list0(
        char('\n'),
        map_res(
            preceded(
                lexeme(tag(b"field:")),
                separated_pair(
                    is_not(";"),
                    char(';'),
                    terminated(
                        separated_list0(
                            char(';'),
                            separated_pair(
                                preceded(is_a("\t "), is_not("\n:").map(to_str)),
                                char(':'),
                                is_not(";").map(to_str),
                            ),
                        ),
                        char(';'),
                    ),
                ),
            ),
            move |(declaration, props)| {
                let props = BTreeMap::from_iter(props.into_iter());
                macro_rules! get {
                    ($name:expr) => {
                        props
                            .get($name)
                            .expect(concat!("Expected field property", $name))
                            .parse()
                            .expect("Failed to parse field property value")
                    };
                }

                let (_, mut declaration) =
                    all_consuming(CGrammar::wrap_rule(CGrammar::declaration(abi), |_: ()| {
                        HeaderError::InvalidCDeclaration
                    }))
                    .parse(declaration)
                    .finish()
                    .map_err(|err: NomError<_, ()>| err.data)?;

                let signed = {
                    let signed: u8 = get!("signed");
                    signed > 0
                };
                let size = get!("size");
                declaration.typ = fixup_c_type(declaration.typ, size, signed, abi)?;

                Ok(FieldFmt {
                    declaration,
                    offset: get!("offset"),
                    size,
                })
            },
        ),
    )
    .map(|fields| StructFmt { fields })
}

fn header_event_parser<'a, E>() -> impl nom::Parser<Input<'a>, (), E>
where
    E: ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, HeaderError>
        + Debug,
{
    map_res(
        preceded(
            opt(lexeme(preceded(char('#'), many0(is_not("\n"))))),
            fold_many0(
                terminated(
                    alt((
                        separated_pair(
                            lexeme(CGrammar::wrap_rule(CGrammar::identifier(), |_: ()| {
                                HeaderError::InvalidCIdentifier
                            })),
                            char(':'),
                            delimited(
                                opt(pair(lexeme(tag("type")), lexeme(tag("==")))),
                                lexeme(txt_u64),
                                opt(lexeme(tag("bits"))),
                            ),
                        )
                        .map(|(id, n)| match (id.as_ref(), n) {
                            ("type_len", 5) => Ok(()),
                            ("type_len", x) => Err(HeaderError::InvalidEventHeader {
                                field: "type_len".into(),
                                value: x.to_string(),
                            }),

                            ("time_delta", 27) => Ok(()),
                            ("time_delta", x) => Err(HeaderError::InvalidEventHeader {
                                field: "time_delta".into(),
                                value: x.to_string(),
                            }),

                            ("array", 32) => Ok(()),
                            ("array", x) => Err(HeaderError::InvalidEventHeader {
                                field: "array".into(),
                                value: x.to_string(),
                            }),

                            ("padding", 29) => Ok(()),
                            ("padding", x) => Err(HeaderError::InvalidEventHeader {
                                field: "padding".into(),
                                value: x.to_string(),
                            }),

                            ("time_extend", 30) => Ok(()),
                            ("time_extend", x) => Err(HeaderError::InvalidEventHeader {
                                field: "time_extend".into(),
                                value: x.to_string(),
                            }),

                            ("time_stamp", 31) => Ok(()),
                            ("time_stamp", x) => Err(HeaderError::InvalidEventHeader {
                                field: "time_stamp".into(),
                                value: x.to_string(),
                            }),
                            _ => Ok(()),
                        }),
                        preceded(
                            tuple((
                                lexeme(tag("data")),
                                lexeme(tag("max")),
                                lexeme(tag("type_len")),
                                lexeme(char('=')),
                            )),
                            lexeme(txt_u64).map(|bits| match bits {
                                28 => Ok(()),
                                x => Err(HeaderError::InvalidEventHeader {
                                    field: "data max type_len".into(),
                                    value: x.to_string(),
                                }),
                            }),
                        ),
                    )),
                    opt(many0(char('\n'))),
                ),
                || Ok(()),
                |acc, i| match acc {
                    Ok(..) => i,
                    Err(..) => acc,
                },
            ),
        ),
        // Simplify the return type by "promoting" validation errors into a parse
        // error.
        |res| res,
    )
}

#[derive(Debug, PartialEq)]
struct EventFmt {
    name: Identifier,
    id: EventId,
    fmt: StructFmt,
    print_fmt: EventPrintFmt,
}

#[derive(Debug, PartialEq)]
struct EventPrintFmt {
    components: Vec<PrintComponent>,
}

#[derive(Debug, PartialEq)]
enum PrintConvSpec {
    String,
    Pointer,
    SignedInteger,
    UnsignedInteger,
}

#[derive(Debug, PartialEq)]
enum PrintComponent {
    Literal(String),
    Expr(PrintConvSpec, CExpr),
}

fn event_print_fmt_parser<'a, 'abi, E>(
    abi: &'abi Abi,
) -> impl nom::Parser<Input<'a>, EventPrintFmt, E> + 'a
where
    E: 'abi
        + ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, HeaderError>
        + Debug,
    'abi: 'a,
{
    separated_pair(
        context(
            "printk format string",
            lexeme(map_res(
                CGrammar::wrap_rule(CGrammar::string_literal(), |_: ()| {
                    HeaderError::InvalidStringLiteral
                }),
                |expr| match expr {
                    CExpr::StringLiteral(s) => Ok(s),
                    _ => Err(HeaderError::InvalidStringLiteral),
                },
            )),
        ),
        lexeme(char(',')),
        context(
            "printk args",
            lexeme(separated_list0(
                lexeme(char(',')),
                CGrammar::wrap_rule(CGrammar::expr(abi), |_: ()| HeaderError::InvalidPrintkArg),
            )),
        ),
    )
    .map(|(fmt, exprs)| {
        // let fmt = &fmt[..];
        println!("fmt={}", fmt);
        println!("exprs={exprs:?}");
        EventPrintFmt { components: vec![] }
    })
}

fn event_fmt_parser<'a, 'abi, E>(abi: &'abi Abi) -> impl nom::Parser<Input<'a>, EventFmt, E> + 'abi
where
    E: 'a
        + ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, HeaderError>
        + Debug,
    'abi: 'a,
{
    move |input| {
        context(
            "event format",
            map_res(
                tuple((
                    context(
                        "event name",
                        preceded(
                            lexeme(tag("name:")),
                            lexeme(CGrammar::wrap_rule(CGrammar::identifier(), |_: ()| {
                                HeaderError::InvalidCIdentifier
                            })),
                        ),
                    ),
                    context("event ID", preceded(lexeme(tag("ID:")), lexeme(txt_u32))),
                    context(
                        "event format",
                        preceded(
                            pair(lexeme(tag("format:")), multispace0),
                            struct_fmt_parser(&abi),
                        ),
                    ),
                    context(
                        "event print fmt",
                        preceded(
                            pair(multispace0, lexeme(tag("print fmt:"))),
                            event_print_fmt_parser(abi),
                        ),
                    ),
                )),
                |(name, id, fmt, print_fmt)| {
                    Ok(EventFmt {
                        name,
                        id,
                        fmt,
                        print_fmt,
                    })
                },
            ),
        )
        .parse(input)
    }
}

#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum HeaderError {
    #[error("Bad magic found")]
    BadMagic,

    #[error("Could not decode UTF-8 string: {0}")]
    DecodeUtf8(StdString),

    #[error("Expected 0 or 1 for endianness, got: {0}")]
    InvalidEndianness(u8),

    #[error("Could not parse C declaration")]
    InvalidCDeclaration,

    #[error("Could not parse C identifier")]
    InvalidCIdentifier,

    #[error("Could not parse printk argument")]
    InvalidPrintkArg,

    #[error(
        "Size of type \"{typ:?}\" was inferred to be {inferred_size} but kernel reported {size}"
    )]
    InvalidTypeSize {
        typ: CType,
        inferred_size: Size,
        size: Size,
    },

    #[error("Sign of type \"{typ:?}\" was inferred to be {inferred_signed:?} but kernel reported {signed}")]
    InvalidTypeSign {
        typ: CType,
        inferred_signed: bool,
        signed: bool,
    },

    #[error("Invalid string literal")]
    InvalidStringLiteral,

    #[error("Invalid long size: {0:?}")]
    InvalidLongSize(Size),

    #[error("Expected header page start")]
    ExpectedHeaderPage,

    #[error("Could not load header page as it is too large: {0} bytes")]
    HeaderPageSizeTooLarge(u64),

    #[error("Expected header event start")]
    ExpectedHeaderEvent,

    #[error("Could not load header event as it is too large: {0} bytes")]
    HeaderEventSizeTooLarge(u64),

    #[error("Could not load event formats as they are too large: {0} bytes")]
    NrEventFmtTooLarge(u32),

    #[error("Unexpected event header value \"{value}\" for field \"{field}\"")]
    InvalidEventHeader { field: StdString, value: StdString },

    #[error(
        "Could not find the \"commit\" field in header page and therefore the kernel long size"
    )]
    NoCommitField,

    #[error("Error while loading data: {0:?}")]
    IoError(std::io::Error),

    #[error("Unknown error")]
    Unknown,
}

impl Default for HeaderError {
    fn default() -> Self {
        HeaderError::Unknown
    }
}

impl From<std::io::Error> for HeaderError {
    fn from(err: std::io::Error) -> Self {
        HeaderError::IoError(err)
    }
}

impl<E> From<NomError<HeaderError, E>> for HeaderError {
    fn from(err: NomError<HeaderError, E>) -> Self {
        err.data
    }
}

pub fn header<I>(mut input: I) -> Result<Header, HeaderError>
where
    I: BorrowingRead,
{
    input.read_tag(b"\x17\x08\x44tracing", HeaderError::BadMagic)??;

    // TODO: check version and handle v6 and v7 headers
    let _version = from_utf8(&input.read_null_terminated()?)
        .map_err(|err| HeaderError::DecodeUtf8(err.to_string()))?;

    let endianness: u8 = input.read_int(Endianness::Little)?;
    let endianness = match endianness {
        0 => Ok(Endianness::Little),
        1 => Ok(Endianness::Big),
        x => Err(HeaderError::InvalidEndianness(x)),
    }?;

    let abi = Abi {
        long_size: None,
        endianness,
    };

    let _long_size: u8 = input.read_int(endianness)?;
    let _page_size: u32 = input.read_int(endianness)?;

    // Header page
    input.read_tag(b"header_page\0", HeaderError::ExpectedHeaderPage)??;
    let header_page_size: u64 = input.read_int(endianness)?;
    let header_page_size: usize = header_page_size
        .try_into()
        .map_err(|_| HeaderError::HeaderPageSizeTooLarge(header_page_size))?;

    let header_fields = input.parse(header_page_size, |input| {
        struct_fmt_parser(&abi).parse(input)
    })??;

    // Fixup ABI with long_size
    let long_size = match header_fields.field_by_name("commit") {
        Some(commit) => commit
            .size
            .try_into()
            .map_err(|size| HeaderError::InvalidLongSize(size)),
        None => Err(HeaderError::NoCommitField),
    }?;

    let _abi = Abi {
        long_size: Some(long_size),
        ..abi
    };

    // Header event
    input.read_tag(b"header_event\0", HeaderError::ExpectedHeaderEvent)??;
    let header_event_size: u64 = input.read_int(endianness)?;
    let header_event_size: usize = header_event_size
        .try_into()
        .map_err(|_| HeaderError::HeaderEventSizeTooLarge(header_event_size))?;
    input.parse(header_event_size, |input| {
        header_event_parser().parse(input)
    })??;

    let nr_event_fmts: u32 = input.read_int(endianness)?;
    let _nr_events_fmt: usize = nr_event_fmts
        .try_into()
        .map_err(|_| HeaderError::NrEventFmtTooLarge(nr_event_fmts))?;

    panic!("it worked");

    // let (_input, _event_fmts) = many_m_n(nr_events_fmt, nr_events_fmt, |input| {
    //     let (input, size) = abi.parse_u64(input)?;
    //     let (input, fmt) = take(size)(input)?;
    //     eprintln!("XXXXXXXXXX {:?}", input.len());
    //     let fmt = event_fmt_parser(&abi).parse(fmt)?;
    //     Ok((input, fmt))
    // })
    // .parse(input)?;
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::parser::tests::test_parser;

    #[test]
    fn event_fmt_parser_test() {
        fn test<'a>(fmt: Input<'a>, expected: EventFmt) {
            let abi = Abi {
                long_size: Some(LongSize::Bits64),
                endianness: Endianness::Little,
            };
            let parser = event_fmt_parser(&abi);
            test_parser(expected, fmt, parser)
        }

        test(
            // b"name: wakeup\nID: 3\nformat:\n\tfield:unsigned short common_type;\toffset:0;\tsize:2;\tsigned:0;\n\tfield:unsigned char common_flags;\toffset:2;\tsize:1;\tsigned:0;\n\tfield:unsigned char common_preempt_count;\toffset:3;\tsize:1;\tsigned:0;\n\tfield:int common_pid;\toffset:4;\tsize:4;\tsigned:1;\n\n\tfield:unsigned int prev_pid;\toffset:8;\tsize:4;\tsigned:0;\n\tfield:unsigned int next_pid;\toffset:12;\tsize:4;\tsigned:0;\n\tfield:unsigned int next_cpu;\toffset:16;\tsize:4;\tsigned:0;\n\tfield:unsigned char prev_prio;\toffset:20;\tsize:1;\tsigned:0;\n\tfield:unsigned char prev_state;\toffset:21;\tsize:1;\tsigned:0;\n\tfield:unsigned char next_prio;\toffset:22;\tsize:1;\tsigned:0;\n\tfield:unsigned char next_state;\toffset:23;\tsize:1;\tsigned:0;\n\nprint fmt: \"%u:%u:%u  ==+ %u:%u:%u \\\" \\t [%03u]\", REC->prev_pid, REC->prev_prio, REC->prev_state, REC->next_pid, REC->next_prio, REC->next_state, REC->next_cpu, 55\n",
            b"name: wakeup\nID: 3\nformat:\n\tfield:unsigned short common_type;\toffset:0;\tsize:2;\tsigned:0;\n\tfield:unsigned char common_flags;\toffset:2;\tsize:1;\tsigned:0;\n\tfield:unsigned char common_preempt_count;\toffset:3;\tsize:1;\tsigned:0;\n\tfield:int common_pid;\toffset:4;\tsize:4;\tsigned:1;\n\n\tfield:unsigned int prev_pid;\toffset:8;\tsize:4;\tsigned:0;\n\tfield:unsigned int next_pid;\toffset:12;\tsize:4;\tsigned:0;\n\tfield:unsigned int next_cpu;\toffset:16;\tsize:4;\tsigned:0;\n\tfield:unsigned char prev_prio;\toffset:20;\tsize:1;\tsigned:0;\n\tfield:unsigned char prev_state;\toffset:21;\tsize:1;\tsigned:0;\n\tfield:unsigned char next_prio;\toffset:22;\tsize:1;\tsigned:0;\n\tfield:unsigned char next_state;\toffset:23;\tsize:1;\tsigned:0;\n\nprint fmt: \"%u:%u:%u  ==+ %u:%u:%u \\\" \\t [%03u]\", 55\n",
            EventFmt {
                name: "wakeup".into(),
                id: 3,
                print_fmt: EventPrintFmt {
                    components: vec![],
                },
                fmt: StructFmt {
                    fields: vec![
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "common_type".into(),
                                typ: CType::Basic(CBasicType::U16),
                            },
                            offset: 0,
                            size: 2,
                        },
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "common_flags".into(),
                                typ: CType::Basic(CBasicType::U8),
                            },
                            offset: 2,
                            size: 1,
                        },
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "common_preempt_count".into(),
                                typ: CType::Basic(CBasicType::U8),
                            },
                            offset: 3,
                            size: 1,
                        },
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "common_pid".into(),
                                typ: CType::Basic(CBasicType::I32),
                            },
                            offset: 4,
                            size: 4,
                        },
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "prev_pid".into(),
                                typ: CType::Basic(CBasicType::U32),
                            },
                            offset: 8,
                            size: 4,
                        },
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "next_pid".into(),
                                typ: CType::Basic(CBasicType::U32),
                            },
                            offset: 12,
                            size: 4,
                        },
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "next_cpu".into(),
                                typ: CType::Basic(CBasicType::U32),
                            },
                            offset: 16,
                            size: 4,
                        },
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "prev_prio".into(),
                                typ: CType::Basic(CBasicType::U8),
                            },
                            offset: 20,
                            size: 1,
                        },
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "prev_state".into(),
                                typ: CType::Basic(CBasicType::U8),
                            },
                            offset: 21,
                            size: 1,
                        },
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "next_prio".into(),
                                typ: CType::Basic(CBasicType::U8),
                            },
                            offset: 22,
                            size: 1,
                        },
                        FieldFmt {
                            declaration: CDeclaration {
                                identifier: "next_state".into(),
                                typ: CType::Basic(CBasicType::U8),
                            },
                            offset: 23,
                            size: 1,
                        },
                    ]
                }
            }
        );
    }
}

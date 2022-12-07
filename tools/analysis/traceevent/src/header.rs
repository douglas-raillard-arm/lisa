use core::{
    borrow::Borrow,
    convert::TryFrom,
    fmt::Debug,
    str::{from_utf8, FromStr},
};
use std::string::ToString;

use std::collections::BTreeMap;

use smartstring::alias::String;

use nom::{
    branch::alt,
    bytes::complete::{escaped, is_a, is_not, tag, take, take_until},
    character::complete::{
        alpha1, alphanumeric1, anychar, char, digit1, multispace0, one_of, u32 as txt_u32,
        u64 as txt_u64,
    },
    combinator::{fail, map_res, not, opt, recognize, success},
    error::{context, ContextError, FromExternalError, ParseError},
    multi::{fold_many0, many0, many0_count, many1, many_m_n, separated_list0},
    number::complete::u8,
    // TODO: Use complete unless there is a really good reason.
    number::streaming::{be_u32, be_u64, le_u32, le_u64, le_u8},
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    Finish as _,
    Parser,
};

use crate::{IResult, Input, StringError};

pub type Address = usize;
pub type CPU = u32;
pub type PID = u32;
pub type Size = u64;
pub type SymbolName = String;
pub type TaskName = String;
pub type Identifier = String;
pub type Offset = isize;
pub type EventId = u32;

#[derive(Debug, Clone)]
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
    endianness: Endianness,
    long_size: Option<LongSize>,
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
    version: String,
    kernel_abi: Abi,
    userspace_long_size: LongSize,
    page_size: Size,
    event_page_size: Size,
    event_descs: Vec<EventDesc>,
    symbols: BTreeMap<Address, SymbolName>,
    const_str: BTreeMap<Address, String>,
    comms: BTreeMap<PID, TaskName>,
    options: Vec<(isize, String)>,
    data_loc: DataLocation,
}

trait ToIResult<I, O, E> {
    type E2;
    fn into_iresult(self, input: I) -> Result<O, nom::Err<Self::E2>>;
}

impl<I, O, E, S> ToIResult<I, O, E> for Result<O, S>
where
    O: Clone,
    E: ParseError<I>,
    S: Into<std::string::String>,
{
    type E2 = StringError<E>;

    #[inline]
    fn into_iresult(self, input: I) -> Result<O, nom::Err<Self::E2>> {
        match self {
            Ok(x) => Ok(x),
            Err(msg) => Err(nom::Err::Error(StringError {
                msg: msg.into(),
                inner: E::from_error_kind(input, nom::error::ErrorKind::Fail),
            })),
        }
    }
}

pub fn no_backtrack<P, I, O, E>(mut parser: P) -> impl nom::Parser<I, O, E>
where
    P: nom::Parser<I, O, E>,
    E: ParseError<I>,
{
    move |input: I| match parser.parse(input) {
        Err(nom::Err::Error(e)) => Err(nom::Err::Failure(e)),
        x => x,
    }
}

pub fn success_with<F, I, O, E>(mut f: F) -> impl FnMut(I) -> nom::IResult<I, O, E>
where
    F: FnMut() -> O,
    E: ParseError<I>,
{
    move |input: I| Ok((input, f()))
}

pub fn str_parser<'a, E>() -> impl nom::Parser<&'a [u8], &'a str, StringError<E>>
where
    E: ParseError<&'a [u8]> + FromExternalError<&'a [u8], std::string::String>,
{
    map_res(
        terminated(
            take_until(&[0][..]),
            // Consume the null terminator
            tag([0]),
        ),
        |s| from_utf8(s).map_err(|err| err.to_string()),
    )
}

pub fn parenthesized<'a, I, O, E, P>(parser: P) -> impl nom::Parser<I, O, E>
where
    P: nom::Parser<I, O, E>,
    E: ParseError<I>,
    I: nom::Slice<std::ops::RangeFrom<usize>> + nom::InputIter,
    <I as nom::InputIter>::Item: nom::AsChar,
{
    delimited(char('('), parser, char(')'))
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum CBasicType {
    Void,

    U8,
    I8,

    U16,
    I16,

    U32,
    I32,

    U64,
    I64,

    Typedef(Identifier),
    // Complete black box used in cases where we want to completely hide any
    // information about the type.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CType {
    Basic(CBasicType),
    Struct(Identifier),
    Enum(Identifier),
    Union(Identifier),
    Pointer(Box<CType>),
    // The size is expected to be mostly useless as the item type's size should
    // be known and we know the length of the array in bytes already.
    Array(Box<CType>, Option<Size>),
    DynamicArray(Box<CType>),
    DynamicScalar(Box<CType>),
}

struct CDeclarator {
    identifier: Option<Identifier>,
    modify_typ: Box<dyn FnOnce(CType) -> CType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CDeclaration {
    identifier: Identifier,
    typ: CType,
}

fn to_str(s: &[u8]) -> &str {
    from_utf8(s).unwrap()
}

fn interpret_c_numeric_expr<'a>(expr: &'a [u8]) -> Option<u64> {
    match expr {
        // It used to be a macro, but is an enum in recent kernels for compat
        // with eBPF ...
        b"TASK_COMM_LEN" => Some(16),
        // Number literal
        expr => {
            let expr = from_utf8(expr).ok()?;
            str::parse(expr).ok()
        }
    }
}

fn print<'a, O, E, P>(name: &'static str, mut inner: P) -> impl nom::Parser<Input<'a>, O, E>
where
    E: ParseError<Input<'a>>,
    P: nom::Parser<Input<'a>, O, E>,
    // O: Debug,
{
    move |input| {
        let (i, x) = inner.parse(input)?;
        // println!("{name}={x:?} input={:?}", from_utf8(input).unwrap());
        println!("{name} input={:?} new_input={:?}", to_str(input), to_str(i),);
        Ok((i, x))
    }
}

fn lexeme<'a, O, E, P>(inner: P) -> impl nom::Parser<Input<'a>, O, E>
where
    E: ParseError<Input<'a>>,
    P: nom::Parser<Input<'a>, O, E>,
{
    delimited(multispace0, inner, multispace0)
}

fn c_identifier_parser<'a, E>() -> impl nom::Parser<Input<'a>, Identifier, E>
where
    E: ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, std::string::String>,
{
    context(
        "identifier",
        map_res(
            lexeme(recognize(pair(
                alt((alpha1, tag("_"))),
                many0(alt((alphanumeric1, tag("_")))),
            ))),
            |s| {
                from_utf8(s)
                    .map_err(|err| err.to_string())
                    .map(|s| s.into())
            },
        ),
    )
}

fn c_type_qualifier_parser<'a, E>() -> impl nom::Parser<Input<'a>, (), E>
where
    E: ParseError<Input<'a>> + ContextError<Input<'a>>,
{
    context(
        "type qualifier",
        lexeme(alt((
            tag("const"),
            tag("restrict"),
            tag("volatile"),
            tag("_Atomic"),
        )))
        .map(|_| ()),
    )
}

fn c_declarator_parser<'a, E>(
    abstract_declarator: bool,
) -> impl nom::Parser<Input<'a>, CDeclarator, E>
where
    E: Debug
        + ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, std::string::String>,
{
    context(
        "declarator",
        lexeme(pair(
            context(
                "pointer",
                many0_count(pair(
                    lexeme(char('*')),
                    many0_count(c_type_qualifier_parser()),
                )),
            ),
            c_direct_declarator_parser(abstract_declarator),
        ))
        .map(|(indirection_level, declarator)| {
            let mut modify_typ: Box<dyn FnOnce(CType) -> CType> = Box::new(move |typ| typ);
            for _ in 0..indirection_level {
                modify_typ = Box::new(move |typ| CType::Pointer(Box::new((modify_typ)(typ))))
            }

            // Apply the declarator's modification last, since they have the least
            // precedence (arrays)
            modify_typ = Box::new(move |typ| (declarator.modify_typ)(modify_typ(typ)));

            CDeclarator {
                modify_typ,
                ..declarator
            }
        }),
    )
}

fn c_direct_declarator_parser<'a, E>(
    abstract_declarator: bool,
) -> impl nom::Parser<Input<'a>, CDeclarator, E>
where
    E: Debug
        + ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, std::string::String>,
{
    let name = if abstract_declarator {
        "abstract direct declarator"
    } else {
        "direct declarator"
    };
    context(name, move |input| {
        let _id_parser = || {
            lexeme(c_identifier_parser().map(|id| CDeclarator {
                identifier: Some(id),
                modify_typ: Box::new(|typ| typ),
            }))
        };
        let id_parser = || {
            move |input| {
                if abstract_declarator {
                    alt((
                        _id_parser(),
                        success_with(|| CDeclarator {
                            identifier: None,
                            modify_typ: Box::new(|typ| typ),
                        }),
                    ))
                    .parse(input)
                } else {
                    _id_parser().parse(input)
                }
            }
        };

        let parenthesized_parser = || {
            context(
                "parenthesized",
                lexeme(parenthesized(c_declarator_parser(abstract_declarator))),
            )
        };

        let array = context(
            "array",
            pair(
                // We should call c_direct_declarator_parser() instead here to
                // handle nested arrays, but since this would make the grammar left
                // recursive, it leads to infinite recursion.
                alt((parenthesized_parser(), id_parser())),
                context(
                    "array size",
                    many1(lexeme(delimited(
                        char('['),
                        lexeme(opt(is_not("]"))),
                        char(']'),
                    ))),
                ),
            ),
        )
        .map(|(declarator, array_sizes)| {
            let array_sizes = array_sizes.iter().map(|array_size| match array_size {
                Some(array_size) => {
                    if array_size.len() > 0 {
                        interpret_c_numeric_expr(array_size).map(|x| x.into())
                    } else {
                        None
                    }
                }
                _ => None,
            });

            let mut modify_typ: Box<dyn FnOnce(CType) -> CType> = Box::new(move |typ| typ);
            for array_size in array_sizes.rev() {
                modify_typ =
                    Box::new(move |typ| CType::Array(Box::new((modify_typ)(typ)), array_size));
            }
            modify_typ = Box::new(move |typ| (declarator.modify_typ)(modify_typ(typ)));
            CDeclarator {
                modify_typ,
                identifier: declarator.identifier,
            }
        });

        let parser = alt((array, parenthesized_parser(), id_parser()));
        lexeme(parser).parse(input)
    })
}

fn c_declaration_specifier_parser<'a, 'abi, E>(
    abi: &'abi Abi,
) -> impl nom::Parser<Input<'a>, CType, E> + 'abi
where
    'a: 'abi,
    E: 'abi
        + ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, std::string::String>,
{
    context(
        "declaration specifier",
        lexeme(move |mut input| {
            #[derive(Debug, Clone, Copy)]
            enum Signedness {
                Signed,
                Unsigned,
                // This is important to represent unknown signedness so that we can
                // differentiate a lone "signed" from nothing, as a lone "signed" is
                // equivalent to "signed int"
                Unknown,
            }

            #[derive(Debug, Clone)]
            enum State {
                Unknown(Signedness),
                Char(Signedness),
                Short(Signedness),
                Int(Signedness),
                Long(Signedness),
                LongLong(Signedness),
            }

            // Tokens that we simply discard as they don't impact layout or pretty
            // representation
            let discard_parser = || {
                context(
                    "discarded",
                    many0_count(lexeme(alt((
                        tag("extern").map(|_| ()),
                        tag("static").map(|_| ()),
                        tag("auto").map(|_| ()),
                        tag("register").map(|_| ()),
                        tag("_Thread_local").map(|_| ()),
                        c_type_qualifier_parser(),
                    )))),
                )
            };

            // Parse the tokens using a state machine to deal with various
            // combinations of "signed int" "int signed" "signed", "long unsigned
            // long" etc.
            let mut state = State::Unknown(Signedness::Unknown);

            loop {
                // tokens we simply ignore
                (input, _) = discard_parser().parse(input)?;

                macro_rules! fsm {
                ($($tag:expr => $transition:expr,)*) => {
                    ($(
                        tag($tag).map(|_| {$transition})
                    ),*)
                }
            }
                let res = lexeme::<_, (), _>(alt(fsm! {
                    "signed" => match &state {
                        State::Unknown(_) => State::Unknown(Signedness::Signed),
                        State::Char(_) => State::Char(Signedness::Signed),
                        State::Short(_) => State::Short(Signedness::Signed),
                        State::Int(_) => State::Int(Signedness::Signed),
                        State::Long(_) => State::Long(Signedness::Signed),
                        State::LongLong(_) => State::LongLong(Signedness::Signed),
                    },
                    "unsigned" => match &state {
                        State::Unknown(_) => State::Unknown(Signedness::Unsigned),
                        State::Char(_) => State::Char(Signedness::Unsigned),
                        State::Short(_) => State::Short(Signedness::Unsigned),
                        State::Int(_) => State::Int(Signedness::Unsigned),
                        State::Long(_) => State::Long(Signedness::Unsigned),
                        State::LongLong(_) => State::LongLong(Signedness::Unsigned),
                    },

                    "char" => match &state {
                        State::Unknown(signedness) => State::Char(*signedness),
                        x => x.clone(),
                    },
                    "short" => match &state {
                        State::Unknown(signedness) => State::Short(*signedness),
                        State::Int(signedness) => State::Short(*signedness),
                        _ => panic!("Got \"short\" with state={state:?}"),
                    },
                    "int" => match &state {
                        State::Unknown(signedness) => State::Int(*signedness),
                        x => x.clone(),
                    },
                    "long" => match &state {
                        State::Unknown(signedness) => State::Long(*signedness),
                        State::Int(signedness) => State::Long(*signedness),
                        State::Long(signedness) => State::LongLong(*signedness),
                        _ => panic!("Got \"long\" with state={state:?}"),
                    },
                }))
                .parse(input);

                (input, state) = match res {
                    Ok(x) => x,
                    // We stop parsing when we can't recognize anything anymore.
                    // Either we hit something else (e.g. an "(" or "[") or we
                    // simply encountered a user-defined type.
                    Err(_) => break,
                }
            }
            let (input, typ) = match state {
                // If we did not hit any of "int", "signed" etc, then it's just a
                // user-defined type that we can consume now.
                State::Unknown(Signedness::Unknown) => lexeme(alt((
                    context(
                        "struct",
                        preceded(lexeme(tag("struct")), c_identifier_parser())
                            .map(|id| CType::Struct(id)),
                    ),
                    context(
                        "enum",
                        preceded(lexeme(tag("enum")), c_identifier_parser())
                            .map(|id| CType::Enum(id)),
                    ),
                    context(
                        "union",
                        preceded(lexeme(tag("union")), c_identifier_parser())
                            .map(|id| CType::Union(id)),
                    ),
                    context(
                        "scalar",
                        c_identifier_parser().map(|id| match id.as_ref() {
                            "void" => CType::Basic(CBasicType::Void),

                            "u8" | "__u8" | "uint8_t" => CType::Basic(CBasicType::U8),
                            "s8" | "__s8" | "int8_t" => CType::Basic(CBasicType::I8),

                            "s16" | "__s16" | "int16_t" => CType::Basic(CBasicType::I16),
                            "u16" | "__u16" | "uint16_t" => CType::Basic(CBasicType::U16),

                            "s32" | "__s32" | "int32_t" => CType::Basic(CBasicType::I32),
                            "u32" | "__u32" | "uint32_t" => CType::Basic(CBasicType::U32),

                            "s64" | "__s64" | "int64_t" => CType::Basic(CBasicType::I64),
                            "u64" | "__u64" | "uint64_t" => CType::Basic(CBasicType::U64),

                            _ => CType::Basic(CBasicType::Typedef(id)),
                        }),
                    ),
                )))
                .parse(input),

                // "signed" alone is equivalent to "signed int"
                State::Unknown(Signedness::Signed) => Ok((input, CType::Basic(CBasicType::I32))),
                State::Unknown(Signedness::Unsigned) => Ok((input, CType::Basic(CBasicType::U32))),

                State::Char(Signedness::Signed) => Ok((input, CType::Basic(CBasicType::I8))),
                State::Char(Signedness::Unsigned) => Ok((input, CType::Basic(CBasicType::U8))),

                // Use U8 for "char" since both typical use-cases are well handled by that:
                // * byte buffer use case is best served by u8
                // * ASCII use case does not care as all ASCII codes are smaller than 127
                State::Char(Signedness::Unknown) => Ok((input, CType::Basic(CBasicType::U8))),

                State::Short(Signedness::Signed | Signedness::Unknown) => {
                    Ok((input, CType::Basic(CBasicType::I16)))
                }
                State::Short(Signedness::Unsigned) => Ok((input, CType::Basic(CBasicType::U16))),

                State::Int(Signedness::Signed | Signedness::Unknown) => {
                    Ok((input, CType::Basic(CBasicType::I32)))
                }
                State::Int(Signedness::Unsigned) => Ok((input, CType::Basic(CBasicType::U32))),

                State::Long(Signedness::Signed | Signedness::Unknown) => Ok((
                    input,
                    CType::Basic(match &abi.long_size {
                        Some(LongSize::Bits32) => CBasicType::I32,
                        Some(LongSize::Bits64) => CBasicType::I64,
                        None => CBasicType::Unknown,
                    }),
                )),
                State::Long(Signedness::Unsigned) => Ok((
                    input,
                    CType::Basic(match &abi.long_size {
                        Some(LongSize::Bits32) => CBasicType::U32,
                        Some(LongSize::Bits64) => CBasicType::U64,
                        None => CBasicType::Unknown,
                    }),
                )),

                State::LongLong(Signedness::Signed | Signedness::Unknown) => {
                    Ok((input, CType::Basic(CBasicType::I64)))
                }
                State::LongLong(Signedness::Unsigned) => Ok((input, CType::Basic(CBasicType::U64))),
            }?;

            let (input, _) = discard_parser().parse(input)?;
            Ok((input, typ))
        }),
    )
}

fn c_declaration_parser<'a, 'abi, E>(
    abi: &'abi Abi,
) -> impl nom::Parser<Input<'a>, CDeclaration, StringError<E>> + 'abi
where
    E: 'abi
        + ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, std::string::String>
        + Debug,
    'a: 'abi,
{
    // Parser for ISO C
    let iso = context(
        "iso C declaration",
        map_res(
            lexeme(pair(
                c_declaration_specifier_parser(abi),
                // We only have to deal with declarations containing only one
                // declarator, i.e. we only handle "int foo" and not "int foo, bar;"
                c_declarator_parser(false),
            )),
            |(typ, declarator)| {
                let identifier = declarator
                    .identifier
                    .ok_or("no identifier found in that declaration")?;
                Ok(CDeclaration {
                    identifier: identifier,
                    typ: (declarator.modify_typ)(typ),
                })
            },
        ),
    );

    // Invalid C syntax that ftrace outputs for its __data_loc and __rel_loc, e.g.:
    // __data_loc char[] name
    let data_loc = context(
        "__data_loc declaration",
        lexeme(preceded(
            lexeme(alt((tag("__data_loc"), tag("__rel_loc")))),
            // Once we consumed "__data_loc", we don't want to allow
            // backtracking back to the ISO C declaration, as we know it will
            // never yield something sensible.
            no_backtrack(map_res(
                lexeme(tuple((
                    c_declaration_specifier_parser(abi),
                    // This will be an abstract declarator, i.e. a declarator with
                    // no identifier (like parameters in a function prototype), as
                    // the name comes after the last "[]"
                    c_declarator_parser(true),
                    opt(context(
                        "__data_loc identifier",
                        lexeme(c_identifier_parser()),
                    )),
                ))),
                |(typ, abstract_declarator, identifier)| {
                    // Push the array sizes down the stack. The 2nd nested array takes the size of the 1st etc.
                    fn push_array_size(
                        typ: CType,
                        size: Option<Size>,
                    ) -> Result<(bool, CType), &'static str> {
                        match typ {
                            CType::Array(typ, _size) => Ok((
                                true,
                                CType::Array(Box::new(push_array_size(*typ, _size)?.1), size),
                            )),
                            CType::Pointer(typ) => {
                                let (_, typ) = push_array_size(*typ, size)?;
                                // If an array is behind a pointer, it can be ignored.
                                Ok((false, CType::Pointer(Box::new(typ))))
                            }
                            typ @ CType::Basic(_)
                            | typ @ CType::Struct(_)
                            | typ @ CType::Enum(_)
                            | typ @ CType::Union(..) => Ok((false, typ)),
                            CType::DynamicArray(..) | CType::DynamicScalar(..) => {
                                Err("found nested __data_loc")
                            }
                        }
                    }

                    let typ = (abstract_declarator.modify_typ)(typ);

                    // Remove the inner array, which is corresponding to the last "[]"
                    // parsed. It only acts as a separator between the type specifier
                    // and the identifier, and actually corresponds to a top-level
                    // DynamicArray.

                    // The innermost array is a fixed "[]" that is part of the format.
                    // It is actually acting as a top-level array, so we push the array
                    // sizes down the stack and replace the top-level by a DynamicArray.
                    let (is_array, pushed_typ) = push_array_size(typ.clone(), None)?;

                    let typ = if is_array {
                        match pushed_typ {
                            CType::Array(typ, _) => Ok(CType::DynamicArray(typ)),
                            typ => Err(format!(
                                "The outermost type of a dynamic array must be an array: {typ:?}"
                            )),
                        }
                    } else {
                        Ok(CType::DynamicScalar(Box::new(typ)))
                    }?;

                    let identifier = if is_array {
                        match identifier {
                            None => Err("A __data_loc declaration is expected to have an identifier following the last []"),
                            Some(id) => Ok(id),
                        }
                    } else {
                        match (abstract_declarator.identifier, identifier) {
                            (Some(id), None) => Ok(id),
                            (None, Some(id)) => Ok(id),
                            (None, None) => {
                                Err("Found no identifier in the scalar __data_loc declaration")
                            }
                            _ => Err(
                                "Found ambiguous identifiers in the scalar __data_loc declaration",
                            ),
                        }
                    }?;

                    // TODO: handle __rel_loc
                    Ok(CDeclaration {
                        identifier: identifier.clone(),
                        typ: typ.clone(),
                    })
                },
            )),
        )),
    );

    let parser = alt((data_loc, iso));
    context("declaration", lexeme(parser))
}

fn fixup_c_type(
    typ: CType,
    size: Size,
    signed: bool,
    abi: &Abi,
) -> Result<CType, std::string::String> {
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

    if inferred_size != size {
        return Err(format!("Size of type \"{typ:?}\" was inferred to be {inferred_size} but kernel reported {size}"));
    }

    if inferred_signed.unwrap_or(signed) != signed {
        return Err(format!("Sign of type \"{typ:?}\" was inferred to be {inferred_signed:?} but kernel reported {signed}"));
    }
    Ok(typ)
}

fn struct_fmt_parser<'a, 'abi, E>(
    abi: &'abi Abi,
) -> impl nom::Parser<Input<'a>, StructFmt, E> + 'abi
where
    E: 'abi
        + ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, std::string::String>
        + Debug,
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

                let (_, mut declaration) = c_declaration_parser::<'a, 'abi, E>(abi)
                    .parse(declaration)
                    .map_err(|_| "failed to parse C declaration")?;

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
        + FromExternalError<Input<'a>, std::string::String>
        + Debug,
{
    map_res(
        preceded(
            opt(lexeme(preceded(char('#'), many0(is_not("\n"))))),
            fold_many0(
                terminated(
                    alt((
                        separated_pair(
                            lexeme(c_identifier_parser()),
                            char(':'),
                            delimited(
                                opt(pair(lexeme(tag("type")), lexeme(tag("==")))),
                                lexeme(txt_u64),
                                opt(lexeme(tag("bits"))),
                            ),
                        )
                        .map(|(id, n)| match (id.as_ref(), n) {
                            ("type_len", 5) => Ok(()),
                            ("type_len", _) => Err("Unexpected type_len in header_event".into()),

                            ("time_delta", 27) => Ok(()),
                            ("time_delta", _) => {
                                Err("Unexpected time_delta in header_event".into())
                            }

                            ("array", 32) => Ok(()),
                            ("array", _) => Err("Unexpected array in header_event".into()),

                            ("padding", 29) => Ok(()),
                            ("padding", _) => Err("Unexpected padding in header_event".into()),

                            ("time_extend", 30) => Ok(()),
                            ("time_extend", _) => {
                                Err("Unexpected time_extend in header_event".into())
                            }

                            ("time_stamp", 31) => Ok(()),
                            ("time_stamp", _) => {
                                Err("Unexpected time_stamp in header_event".into())
                            }
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
                                _ => Err("Unexpected data max type_len".into()),
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

#[derive(Clone, Debug, PartialEq)]
enum CExpr {
    EventField(Identifier),
    IntLiteral(u64),
    Addr(Box<CExpr>),
    Deref(Box<CExpr>),
    Plus(Box<CExpr>),
    Minus(Box<CExpr>),
    Tilde(Box<CExpr>),
    Cast(CType, Box<CExpr>),
    SizeofType(CType),
    SizeofExpr(Box<CExpr>),
    Preinc(Box<CExpr>),
    Predec(Box<CExpr>),
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

fn c_expr_parser<'a, 'abi, E>(abi: &'abi Abi) -> impl nom::Parser<Input<'a>, CExpr, E> + 'abi
where
    E: 'abi
        + ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, std::string::String>
        + Debug,
    'a: 'abi,
{
    // Allow defining grammar production rules with most of the boilerplate
    // removed and automatic context() added
    macro_rules! grammar {
        ($($vis:vis rule $name:ident $(<$($generics:tt $(: $bound:tt)?),*>)? ($($param:ident: $param_ty:ty),*) -> $ret:ty $body:block)*) => {
            $(
                $vis fn $name<'ret, $($($generics $(: $bound)?,)*)? E>($($param: $param_ty),*) -> impl nom::Parser<Input<'ret>, $ret, E> + 'ret
                where
                    E: 'ret
                    + ParseError<Input<'ret>>
                    + ContextError<Input<'ret>>
                    + FromExternalError<Input<'ret>, std::string::String>
                    + Debug,
                    $($($generics: 'ret),*)?
                {
                    // Wrap the body in a closure to avoid recursive type issues
                    // when a rule is recursive, and add a context for free
                    move |input| context(stringify!($name), $body).parse(input)
                }
            )*
        }
    }

    // Grammar inspired by N1570 ISO C draft (latest C11 draft):
    // https://port70.net/~nsz/c/c11/n1570.html#A.2.1
    grammar! {

        rule postfix_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    primary_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
        rule unary_op() -> Box<dyn Fn(CExpr) -> CExpr> {
            lexeme(
                alt((
                    context("unary &", char('&').map(|_| Box::new(|e| CExpr::Addr(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                    context("unary *", char('*').map(|_| Box::new(|e| CExpr::Deref(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                    context("unary +", char('+').map(|_| Box::new(|e| CExpr::Plus(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                    context("unary -", char('-').map(|_| Box::new(|e| CExpr::Minus(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                    context("unary ~", char('~').map(|_| Box::new(|e| CExpr::Tilde(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
        rule unary_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    postfix_expr(abi),
                    context("preinc expr",
                            preceded(
                                lexeme(tag("++")),
                                unary_expr(abi),
                            ).map(|e| CExpr::Preinc(Box::new(e)))
                    ),
                    context("predec expr",
                            preceded(
                                lexeme(tag("--")),
                                unary_expr(abi),
                            ).map(|e| CExpr::Predec(Box::new(e)))
                    ),
                    context("unary op expr",
                        tuple((
                            unary_op(),
                            cast_expr(abi),
                        )).map(|(modify, e)| modify(e))
                    ),
                    context("sizeof type",
                        preceded(
                            lexeme(tag("sizeof")),
                            parenthesized(
                                type_name(abi),
                            )
                        ).map(|typ| CExpr::SizeofType(typ))
                    ),
                    context("sizeof expr",
                        preceded(
                            lexeme(tag("sizeof")),
                            unary_expr(abi),
                        ).map(|e| CExpr::SizeofExpr(Box::new(e)))
                    ),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.7p1
        rule type_name<'abi>(abi: &'abi Abi) -> CType {
            lexeme(
                tuple((
                    c_declaration_specifier_parser(abi),
                    c_declarator_parser(true),
                )).map(|(typ, abstract_declarator)|
                    (abstract_declarator.modify_typ)(typ)
                )
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.4p1
        rule cast_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    unary_expr(abi),
                    context("cast",
                        tuple((
                            parenthesized(
                                type_name(abi),
                            ),
                            cast_expr(abi),
                        )).map(|(typ, e)| CExpr::Cast(typ, Box::new(e)))
                    )
                ))
            )
        }


        // TODO: parse literals properly, including hex
        rule literal() -> CExpr {
            lexeme(
                alt((
                    context("integer literal", txt_u64.map(CExpr::IntLiteral)),
                ))
            )
        }
        // https://port70.net/~nsz/c/c11/n1570.html#6.5.1p1
        rule primary_expr<'abi>(abi: &'abi Abi) -> CExpr
        {
            lexeme(
                alt((
                    // TODO: this is a postfix expression
                    lexeme(literal()),
                    // TODO: this is a postfix expression
                    parenthesized(
                        c_expr_parser(abi),
                    ),
                    unary_expr(abi),
                )).map(|x| CExpr::EventField("foo".into()))
            )
        }
    }

    // E --> P {B P}
    // P --> v | "(" E ")" | U P
    // B --> "+" | "-" | "*" | "/" | "^"
    // U --> "-"

    context(
        "C expression",
        lexeme(alt((
            preceded(tag("REC->"), c_identifier_parser()).map(|id| CExpr::EventField(id)),
        ))),
    )
}

pub fn c_literal_str_parser<'a, E>() -> impl nom::Parser<Input<'a>, Vec<u8>, E>
where
    E: ParseError<Input<'a>> + ContextError<Input<'a>>,
{
    delimited(
        char('"'),
        context("C string literal", |mut input| {
            let mut fmt = Vec::with_capacity(128);
            loop {
                // Parse escape sequence
                let res = context::<_, E, _, _>("escaped char", preceded(char('\\'), u8))
                    .map(|c| {
                        match c {
                            b'"' => b'\"',
                            b'\'' => b'\'',
                            b'\\' => b'\\',
                            b'n' => b'\n',
                            b't' => b'\t',
                            b'a' => 0x07u8,
                            b'b' => 0x08u8,
                            b'e' => 0x1Bu8,
                            b'f' => 0x0Cu8,
                            b'r' => 0x0Du8,
                            b'v' => 0x0Bu8,
                            b'?' => 0x3Fu8,
                            // TODO: handle numerical escape
                            // sequences
                            _ => c,
                        }
                    })
                    .parse(input);
                match res {
                    Ok((_input, c)) => {
                        fmt.push(c);
                        input = _input;
                        continue;
                    }
                    _ => (),
                }

                // Parse runs of non-escaped chars
                let res = is_not::<_, _, E>(r#"\""#).parse(input);
                match res {
                    Ok((_input, s)) => {
                        fmt.extend(s);
                        input = _input;
                        continue;
                    }
                    _ => return Ok((input, fmt)),
                }
            }
        }),
        char('"'),
    )
}

fn event_print_fmt_parser<'a, 'abi, E>(
    abi: &'abi Abi,
) -> impl nom::Parser<Input<'a>, EventPrintFmt, E> + 'abi
where
    E: 'abi
        + ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, std::string::String>
        + Debug,
    'a: 'abi,
{
    separated_pair(
        context("printk format string", lexeme(c_literal_str_parser())),
        lexeme(char(',')),
        context(
            "printk args",
            lexeme(separated_list0(lexeme(char(',')), c_expr_parser(abi))),
        ),
    )
    .map(|(fmt, exprs)| {
        // let fmt = &fmt[..];
        println!("fmt={}", to_str(&fmt));
        println!("exprs={exprs:?}");
        EventPrintFmt { components: vec![] }
    })
}

fn event_fmt_parser<'a, 'abi, E>(abi: &'abi Abi) -> impl nom::Parser<Input<'a>, EventFmt, E> + 'abi
where
    E: ParseError<Input<'a>>
        + ContextError<Input<'a>>
        + FromExternalError<Input<'a>, std::string::String>
        + Debug,
{
    move |input| {
        context(
            "event format",
            map_res(
                tuple((
                    context(
                        "event name",
                        preceded(lexeme(tag("name:")), lexeme(c_identifier_parser())),
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

pub fn header<'a>(input: Input<'a>) -> IResult<Header> {
    let (input, _) = tuple((tag([0x17, 0x08, 0x44]), tag(b"tracing"))).parse(input)?;
    let (input, version) = str_parser().parse(input)?;
    let _version: String = version.into();

    let (input, endianness) = le_u8(input)?;
    let endianness = match endianness {
        0 => Ok(Endianness::Little),
        1 => Ok(Endianness::Big),
        x => Err(format!("Invalid endianness value in header: {x:?}")),
    }
    .into_iresult(input)?;

    let abi = Abi {
        long_size: None,
        endianness,
    };

    let (input, long_size) = le_u8(input)?;
    let (input, page_size) = abi.parse_u32(input)?;

    // Header page
    let (input, _) = tag(b"header_page\0")(input)?;
    let (input, header_page_size) = abi.parse_u64(input)?;
    let (input, header_page) = take(header_page_size)(input)?;
    let (_, header_fields) = struct_fmt_parser(&abi).parse(header_page)?;

    let header_page = to_str(header_page);
    println!("{header_page}");

    // Fixup ABI with long_size
    let long_size = match header_fields.field_by_name("commit") {
        Some(commit) => commit.size.try_into().map_err(|size| format!("Invalid long size: {size}")),
        None => Err(format!("Could not find the \"commit\" field in header page and therefore the kernel long size: {header_page:?}"))
    }.into_iresult(input)?;
    let abi = Abi {
        long_size: Some(long_size),
        ..abi
    };

    // Header event
    let (input, _) = tag(b"header_event\0")(input)?;
    let (input, header_event_size) = abi.parse_u64(input)?;
    let (input, header_event) = take(header_event_size)(input)?;
    header_event_parser().parse(header_event)?;

    let (input, nr_event_fmts) = abi.parse_u32(input)?;
    let nr_events_fmt = nr_event_fmts
        .try_into()
        .expect("could not convert u32 to usize");

    let (input, event_fmts) = many_m_n(nr_events_fmt, nr_events_fmt, |input| {
        let (input, size) = abi.parse_u64(input)?;
        let (input, fmt) = take(size)(input)?;
        event_fmt_parser(&abi).parse(input)
    })
    .parse(input)?;

    let header_event = to_str(header_event);
    println!("{header_event}");
    panic!("it worked");

    // let (_, x) = take(10usize)(input)?;
    // let x = from_utf8(x);
    // println!("{x:?}");
    // tag(b"tracing")(input)?;
    todo!()
    // Header {}

    // let (input, length) = be_u16(input)?;
    // take(length)(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parser<'a, O, P>(expected: O, input: Input<'a>, mut parser: P)
    where
        O: Debug + PartialEq,
        P: Parser<Input<'a>, O, StringError<nom::error::VerboseError<Input<'a>>>>,
    {
        let parsed = parser.parse(input).finish();
        let input = to_str(input);
        match parsed {
            Ok((_, parsed)) => {
                assert_eq!(parsed, expected, "while parsing: {input:?}");
            }
            Err(err) => {
                // Convert input from &[u8] to &str so convert_error() can
                // display it.
                let mut seen_context = false;
                let inner = nom::error::VerboseError {
                    errors: err
                        .inner
                        .errors
                        .into_iter()
                        // Preserve the leaf-most levels that don't have a
                        // context, but after the first context is
                        // encountered, display only levels with a context.
                        // This makes the path much easier to follow if all
                        // relevant levels are annotated correctly.
                        .filter(|(_, kind)| match kind {
                            nom::error::VerboseErrorKind::Context(..) => {
                                seen_context = true;
                                true
                            }
                            _ => !seen_context,
                        })
                        .map(|(s, err)| (to_str(s), err))
                        .collect(),
                };
                let loc = nom::error::convert_error(input, inner);
                panic!("Could not parse {input:?}:\n{loc}")
            }
        }
    }

    #[test]
    fn c_type_parser_test() {
        fn test<'a>(decl: Input<'a>, id: &'a str, typ: CType) {
            let expected = CDeclaration {
                identifier: id.into(),
                typ: typ,
            };
            let abi = Abi {
                long_size: Some(LongSize::Bits64),
                endianness: Endianness::Little,
            };
            let parser = c_declaration_parser(&abi);
            test_parser(expected, decl, parser)
        }

        // Basic
        test(b"u64 foo", "foo", CType::Basic(CBasicType::U64));
        test(b" u64  \t foo\t", "foo", CType::Basic(CBasicType::U64));
        test(
            b" const volatile u64 foo",
            "foo",
            CType::Basic(CBasicType::U64),
        );
        test(
            b" u64 const volatile foo",
            "foo",
            CType::Basic(CBasicType::U64),
        );
        test(
            b" const\t volatile  _Atomic u64  \t foo\t",
            "foo",
            CType::Basic(CBasicType::U64),
        );

        // Structs + Enum + Union
        test(
            b"struct mystruct foo",
            "foo",
            CType::Struct("mystruct".into()),
        );
        test(b"enum mystruct foo", "foo", CType::Enum("mystruct".into()));
        test(
            b"union mystruct foo",
            "foo",
            CType::Union("mystruct".into()),
        );

        // Signed/Unsigned
        test(
            b"int signed extern const  foo  ",
            "foo",
            CType::Basic(CBasicType::I32),
        );
        test(
            b"signed extern const  foo  ",
            "foo",
            CType::Basic(CBasicType::I32),
        );
        test(
            b"_Atomic\t unsigned    extern const  foo  ",
            "foo",
            CType::Basic(CBasicType::U32),
        );
        test(
            b"int unsigned extern const  foo  ",
            "foo",
            CType::Basic(CBasicType::U32),
        );
        test(
            b" long \t long unsigned extern const  foo  ",
            "foo",
            CType::Basic(CBasicType::U64),
        );

        test(
            b"int long long unsigned extern const  foo  ",
            "foo",
            CType::Basic(CBasicType::U64),
        );

        test(
            b"long extern int long unsigned const  foo  ",
            "foo",
            CType::Basic(CBasicType::U64),
        );

        // Pointers
        test(
            b"u64 *foo",
            "foo",
            CType::Pointer(Box::new(CType::Basic(CBasicType::U64))),
        );
        test(
            b" u64 * \tfoo ",
            "foo",
            CType::Pointer(Box::new(CType::Basic(CBasicType::U64))),
        );
        test(
            b"u64 *foo",
            "foo",
            CType::Pointer(Box::new(CType::Basic(CBasicType::U64))),
        );
        test(
            b" u64 * \tfoo ",
            "foo",
            CType::Pointer(Box::new(CType::Basic(CBasicType::U64))),
        );
        test(
            b"u64 * const foo",
            "foo",
            CType::Pointer(Box::new(CType::Basic(CBasicType::U64))),
        );
        test(
            b" u64 * \tconst\tfoo ",
            "foo",
            CType::Pointer(Box::new(CType::Basic(CBasicType::U64))),
        );
        test(
            b" long unsigned long * \tconst\tfoo ",
            "foo",
            CType::Pointer(Box::new(CType::Basic(CBasicType::U64))),
        );
        test(
            b" const volatile u64 * const foo",
            "foo",
            CType::Pointer(Box::new(CType::Basic(CBasicType::U64))),
        );
        test(
            b" const\tvolatile u64 * \tconst\tfoo ",
            "foo",
            CType::Pointer(Box::new(CType::Basic(CBasicType::U64))),
        );
        test(
            b" const\tvolatile u64 * const * \tconst\tfoo ",
            "foo",
            CType::Pointer(Box::new(CType::Pointer(Box::new(CType::Basic(
                CBasicType::U64,
            ))))),
        );
        test(
            b" const\tvolatile u64 _Atomic * const * \tconst\tfoo ",
            "foo",
            CType::Pointer(Box::new(CType::Pointer(Box::new(CType::Basic(
                CBasicType::U64,
            ))))),
        );

        // Arrays
        test(
            b" u64 foo\t []",
            "foo",
            CType::Array(Box::new(CType::Basic(CBasicType::U64)), None),
        );
        test(
            b" u64 foo\t []\t\t",
            "foo",
            CType::Array(Box::new(CType::Basic(CBasicType::U64)), None),
        );
        test(
            b" u64 foo\t [124]",
            "foo",
            CType::Array(Box::new(CType::Basic(CBasicType::U64)), Some(124)),
        );

        test(
            b" u64 (*foo) [1]",
            "foo",
            CType::Pointer(Box::new(CType::Array(
                Box::new(CType::Basic(CBasicType::U64)),
                Some(1),
            ))),
        );
        test(
            b" u64 ((*foo)) [1]",
            "foo",
            CType::Pointer(Box::new(CType::Array(
                Box::new(CType::Basic(CBasicType::U64)),
                Some(1),
            ))),
        );
        test(
            b" u64 (*foo[]) [1]",
            "foo",
            CType::Array(
                Box::new(CType::Pointer(Box::new(CType::Array(
                    Box::new(CType::Basic(CBasicType::U64)),
                    Some(1),
                )))),
                None,
            ),
        );
        test(
            b" u64(*foo[]\t)[1]",
            "foo",
            CType::Array(
                Box::new(CType::Pointer(Box::new(CType::Array(
                    Box::new(CType::Basic(CBasicType::U64)),
                    Some(1),
                )))),
                None,
            ),
        );
        test(
            b" u64 foo\t [A+B]\t\t",
            "foo",
            CType::Array(Box::new(CType::Basic(CBasicType::U64)), None),
        );

        // Nested arrays
        test(
            b" u64 foo\t [][]",
            "foo",
            CType::Array(
                Box::new(CType::Array(Box::new(CType::Basic(CBasicType::U64)), None)),
                None,
            ),
        );
        test(
            b" u64  foo\t [1][2]",
            "foo",
            CType::Array(
                Box::new(CType::Array(
                    Box::new(CType::Basic(CBasicType::U64)),
                    Some(2),
                )),
                Some(1),
            ),
        );
        test(
            b" u64 (*foo[3]) [2][1] ",
            "foo",
            CType::Array(
                Box::new(CType::Pointer(Box::new(CType::Array(
                    Box::new(CType::Array(
                        Box::new(CType::Basic(CBasicType::U64)),
                        Some(1),
                    )),
                    Some(2),
                )))),
                Some(3),
            ),
        );

        // Scalar __data_loc
        test(
            b"__data_loc u64 foo",
            "foo",
            CType::DynamicScalar(Box::new(CType::Basic(CBasicType::U64))),
        );

        test(
            b"__data_loc u64* foo",
            "foo",
            CType::DynamicScalar(Box::new(CType::Pointer(Box::new(CType::Basic(
                CBasicType::U64,
            ))))),
        );

        test(
            b"__data_loc unsigned volatile * const foo",
            "foo",
            CType::DynamicScalar(Box::new(CType::Pointer(Box::new(CType::Basic(
                CBasicType::U32,
            ))))),
        );

        test(
            b"__data_loc u64 (*)[3] foo",
            "foo",
            CType::DynamicScalar(Box::new(CType::Pointer(Box::new(CType::Array(
                Box::new(CType::Basic(CBasicType::U64)),
                Some(3),
            ))))),
        );

        test(
            b"__data_loc const u64 _Atomic ( * volatile)[3] foo",
            "foo",
            CType::DynamicScalar(Box::new(CType::Pointer(Box::new(CType::Array(
                Box::new(CType::Basic(CBasicType::U64)),
                Some(3),
            ))))),
        );

        // Array __data_loc
        test(
            b"__data_loc u64[] foo",
            "foo",
            CType::DynamicArray(Box::new(CType::Basic(CBasicType::U64))),
        );

        test(
            b"   __data_loc\t u64  []foo",
            "foo",
            CType::DynamicArray(Box::new(CType::Basic(CBasicType::U64))),
        );
        test(
            b"   __data_loc\t u64  [42][]foo",
            "foo",
            CType::DynamicArray(Box::new(CType::Array(
                Box::new(CType::Basic(CBasicType::U64)),
                Some(42),
            ))),
        );

        test(
            b"   __data_loc\t u64  [42][43][]foo",
            "foo",
            CType::DynamicArray(Box::new(CType::Array(
                Box::new(CType::Array(
                    Box::new(CType::Basic(CBasicType::U64)),
                    Some(43),
                )),
                Some(42),
            ))),
        );

        test(
            b"   __data_loc u64 (*[3]) [2][]foo",
            "foo",
            CType::DynamicArray(Box::new(CType::Pointer(Box::new(CType::Array(
                Box::new(CType::Array(
                    Box::new(CType::Basic(CBasicType::U64)),
                    Some(2),
                )),
                Some(3),
            ))))),
        );

        // All together
        test(
            b" const\tvolatile unsigned int volatile*const _Atomic* \tconst\tfoo [sizeof(struct foo)] \t[42] ",
            "foo",
            CType::Array(
                Box::new(CType::Array(
                    Box::new(CType::Pointer(Box::new(CType::Pointer(Box::new(
                        CType::Basic(CBasicType::U32),
                    ))))),
                    Some(42),
                )),
                None,
            ),
        );
        test(
            b" const\tvolatile int volatile signed*const _Atomic* \tconst\tfoo [sizeof(struct foo)] \t[42] ",
            "foo",
            CType::Array(
                Box::new(CType::Array(
                    Box::new(CType::Pointer(Box::new(CType::Pointer(Box::new(
                        CType::Basic(CBasicType::I32),
                    ))))),
                    Some(42),
                )),
                None,
            ),
        );

        test(
            b" __data_loc\tconst\tvolatile int volatile signed*const _Atomic* \tconst \t[] foo",
            "foo",
            CType::DynamicArray(Box::new(CType::Pointer(Box::new(CType::Pointer(
                Box::new(CType::Basic(CBasicType::I32)),
            ))))),
        );

        test(
            b"  __data_loc \tconst\tvolatile int volatile signed*const _Atomic* \tconst\t [sizeof(struct foo)] \t[42] []foo",
            "foo",
            CType::DynamicArray(
            Box::new(CType::Array(
                Box::new(CType::Array(
                    Box::new(CType::Pointer(Box::new(CType::Pointer(Box::new(
                        CType::Basic(CBasicType::I32),
                    ))))),
                    Some(42),
                )),
                None,
            ),
        )));
        // TODO: test __rel_loc
    }

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
            b"name: wakeup\nID: 3\nformat:\n\tfield:unsigned short common_type;\toffset:0;\tsize:2;\tsigned:0;\n\tfield:unsigned char common_flags;\toffset:2;\tsize:1;\tsigned:0;\n\tfield:unsigned char common_preempt_count;\toffset:3;\tsize:1;\tsigned:0;\n\tfield:int common_pid;\toffset:4;\tsize:4;\tsigned:1;\n\n\tfield:unsigned int prev_pid;\toffset:8;\tsize:4;\tsigned:0;\n\tfield:unsigned int next_pid;\toffset:12;\tsize:4;\tsigned:0;\n\tfield:unsigned int next_cpu;\toffset:16;\tsize:4;\tsigned:0;\n\tfield:unsigned char prev_prio;\toffset:20;\tsize:1;\tsigned:0;\n\tfield:unsigned char prev_state;\toffset:21;\tsize:1;\tsigned:0;\n\tfield:unsigned char next_prio;\toffset:22;\tsize:1;\tsigned:0;\n\tfield:unsigned char next_state;\toffset:23;\tsize:1;\tsigned:0;\n\nprint fmt: \"%u:%u:%u  ==+ %u:%u:%u \\\" \\t [%03u]\", REC->prev_pid, REC->prev_prio, REC->prev_state, REC->next_pid, REC->next_prio, REC->next_state, REC->next_cpu\n",
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

use core::{borrow::Borrow, cell::Cell, convert::TryFrom, fmt::Debug, str::from_utf8};
use std::{rc::Rc, string::ToString};

use std::string::String as StdString;

use smartstring::alias::String;
use thiserror::Error;

use nom::{
    branch::alt,
    bytes::complete::{is_a, is_not, tag, take, take_until},
    character::complete::{
        alpha1, alphanumeric1, char, multispace0, u32 as txt_u32, u64 as txt_u64,
    },
    combinator::{fail, map_res, opt, recognize, success},
    error::{context, ErrorKind, FromExternalError, ParseError},
    multi::{fold_many0, many0, many0_count, many1, many_m_n, separated_list0},
    number::complete::u8,
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    IResult, Parser,
};

use crate::{
    grammar::grammar,
    header::{Abi, Identifier, LongSize, Size},
    parser::{lexeme, no_backtrack, parenthesized, print, success_with, to_str, Input},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CBasicType {
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
pub enum CType {
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

#[derive(Clone)]
pub struct CDeclarator {
    identifier: Option<Identifier>,
    // Use Rc<> so that CDeclarator can be Clone, which is necessary to for the
    // packrat cache.
    modify_typ: Rc<dyn Fn(CType) -> CType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CDeclaration {
    pub identifier: Identifier,
    pub typ: CType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CExpr {
    EventField(Identifier),
    IntLiteral(u64),
    StringLiteral(String),
    Addr(Box<CExpr>),
    Deref(Box<CExpr>),
    Plus(Box<CExpr>),
    Minus(Box<CExpr>),
    Tilde(Box<CExpr>),
    Bang(Box<CExpr>),
    Cast(CType, Box<CExpr>),
    SizeofType(CType),
    SizeofExpr(Box<CExpr>),
    Preinc(Box<CExpr>),
    Predec(Box<CExpr>),
}

#[derive(Error, Debug, Default)]
#[non_exhaustive]
pub enum CParseError {
    #[error("Could not decode UTF-8 string: {0}")]
    DecodeUtf8(StdString),
    #[error("No identifier found in a declaration of type {0:?}")]
    DeclarationWithoutIdentifier(CType),
    #[error("Found nested __data_loc")]
    NestedDataLoc,
    #[error("The outermost type of a dynamic array must be an array: {0:?}")]
    DataLocArrayNotArray(CType),
    #[error(
        "A __data_loc array declaration is expected to have an identifier following the last []"
    )]
    DataLocArrayWithoutIdentifier,
    #[error("Found no identifier in the scalar __data_loc declaration")]
    DataLocWithoutIdentifier,
    #[error(
        "Found ambiguous identifiers in the scalar __data_loc declaration: \"{0}\" or \"{1}\""
    )]
    DataLocAmbiguousIdentifier(Identifier, Identifier),

    #[error("Invalid type name (incompatible int/long/short/char usage)")]
    InvalidTypeName,

    #[default]
    #[error("Unknown error")]
    Unknown,
}

// TODO: replace by the proper expr evaluator
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

// TODO: link to the ISO C spec URL for each rule

// Grammar inspired by N1570 ISO C draft (latest C11 draft):
// https://port70.net/~nsz/c/c11/n1570.html#A.2.1
grammar! {
    name: pub CGrammar,
    error: CParseError,
    rules: {

        rule identifier() -> Identifier {
            map_res(
                lexeme(recognize(pair(
                    alt((alpha1, tag("_"))),
                    many0(alt((alphanumeric1, tag("_")))),
                ))),
                // For some reason rustc fails to infer CGrammar properly, even
                // though it looks like it should. Maybe this will be fixed in
                // the future.
                |s: Span<CGrammar>| {
                    from_utf8(s.fragment())
                        .map_err(|err| CParseError::DecodeUtf8(err.to_string()))
                        .map(|s| s.into())
                },
            )
        }

        rule type_qualifier() -> () {
            lexeme(alt((
                tag("const"),
                tag("restrict"),
                tag("volatile"),
                tag("_Atomic"),
            ))).map(|_| ())
        }

        rule declarator(abstract_declarator: bool) -> CDeclarator {
            lexeme(pair(
                context(
                    "pointer",
                    many0_count(pair(
                        lexeme(char('*')),
                        many0_count(Self::type_qualifier()),
                    )),
                ),
                Self::direct_declarator(abstract_declarator),
            ))
            .map(|(indirection_level, declarator)| {
                let mut modify_typ: Box<dyn Fn(CType) -> CType> = Box::new(move |typ| typ);
                for _ in 0..indirection_level {
                    modify_typ = Box::new(move |typ| CType::Pointer(Box::new((modify_typ)(typ))))
                }

                // Apply the declarator's modification last, since they have the least
                // precedence (arrays)
                let modify_typ = Rc::new(move |typ| (declarator.modify_typ)(modify_typ(typ)));

                CDeclarator {
                    modify_typ,
                    ..declarator
                }
            })
        }

        rule direct_declarator(abstract_declarator: bool) -> CDeclarator {
            let name = if abstract_declarator {
                "abstract direct declarator"
            } else {
                "direct declarator"
            };
            context(name, move |input| {
                let _id_parser = || {
                    lexeme(Self::identifier().map(|id| CDeclarator {
                        identifier: Some(id),
                        modify_typ: Rc::new(|typ| typ),
                    }))
                };
                let id_parser = || {
                    move |input| {
                        if abstract_declarator {
                            alt((
                                _id_parser(),
                                success_with(|| CDeclarator {
                                    identifier: None,
                                    modify_typ: Rc::new(|typ| typ),
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
                        lexeme(parenthesized(Self::declarator(abstract_declarator))),
                    )
                };

                let array = context(
                    "array",
                    pair(
                        Self::direct_declarator(abstract_declarator),
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

                    let mut modify_typ: Box<dyn Fn(CType) -> CType> = Box::new(move |typ| typ);
                    for array_size in array_sizes.rev() {
                        modify_typ =
                            Box::new(move |typ| CType::Array(Box::new((modify_typ)(typ)), array_size));
                    }
                    let modify_typ = Rc::new(move |typ| (declarator.modify_typ)(modify_typ(typ)));
                    CDeclarator {
                        modify_typ,
                        identifier: declarator.identifier,
                    }
                });

                let parser = alt((array, parenthesized_parser(), id_parser()));
                lexeme(parser).parse(input)
            })
        }

        rule declaration_specifier<'abi>(abi: &'abi Abi) -> CType {
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
                            Self::type_qualifier(),
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
                            alt((
                                $(
                                    tag($tag).map(|_| {
                                        let res: Result<State, CParseError> = $transition;
                                        res
                                    })
                                ),*
                            ))
                        }
                    }
                    let res = lexeme::<_, _, (), _>(fsm! {
                        "signed" => Ok(match &state {
                            State::Unknown(_) => State::Unknown(Signedness::Signed),
                            State::Char(_) => State::Char(Signedness::Signed),
                            State::Short(_) => State::Short(Signedness::Signed),
                            State::Int(_) => State::Int(Signedness::Signed),
                            State::Long(_) => State::Long(Signedness::Signed),
                            State::LongLong(_) => State::LongLong(Signedness::Signed),
                        }),
                        "unsigned" => Ok(match &state {
                            State::Unknown(_) => State::Unknown(Signedness::Unsigned),
                            State::Char(_) => State::Char(Signedness::Unsigned),
                            State::Short(_) => State::Short(Signedness::Unsigned),
                            State::Int(_) => State::Int(Signedness::Unsigned),
                            State::Long(_) => State::Long(Signedness::Unsigned),
                            State::LongLong(_) => State::LongLong(Signedness::Unsigned),
                        }),

                        "char" => match &state {
                            State::Unknown(signedness) => Ok(State::Char(*signedness)),
                            x@State::Char(_) => Ok(x.clone()),
                            _ => Err(CParseError::InvalidTypeName),
                        },
                        "short" => match &state {
                            State::Unknown(signedness) => Ok(State::Short(*signedness)),
                            State::Int(signedness) => Ok(State::Short(*signedness)),
                            _ => Err(CParseError::InvalidTypeName),
                        },
                        "int" => match &state {
                            State::Unknown(signedness) => Ok(State::Int(*signedness)),
                            State::Char(_) => Err(CParseError::InvalidTypeName),
                            State::Int(_) => Err(CParseError::InvalidTypeName),
                            x => Ok(x.clone()),
                        },
                        "long" => match &state {
                            State::Unknown(signedness) => Ok(State::Long(*signedness)),
                            State::Int(signedness) => Ok(State::Long(*signedness)),
                            State::Long(signedness) => Ok(State::LongLong(*signedness)),
                            _ => Err(CParseError::InvalidTypeName),
                        },
                    })
                    .parse(input.clone());

                    (input, state) = match res {
                        Ok((i, Ok(x))) => (i, x),
                        Ok((i, Err(err))) => return Err(nom::Err::Error(FromExternalError::from_external_error(i, ErrorKind::Fail, err))),
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
                            preceded(lexeme(tag("struct")), Self::identifier())
                                .map(|id| CType::Struct(id)),
                        ),
                        context(
                            "enum",
                            preceded(lexeme(tag("enum")), Self::identifier())
                                .map(|id| CType::Enum(id)),
                        ),
                        context(
                            "union",
                            preceded(lexeme(tag("union")), Self::identifier())
                                .map(|id| CType::Union(id)),
                        ),
                        context(
                            "scalar",
                            Self::identifier().map(|id| match id.as_ref() {
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
            })
        }

        rule declaration<'abi>(abi: &'abi Abi) -> CDeclaration {
            // Parser for ISO C
            let iso = context(
                "iso C declaration",
                map_res(
                    lexeme(pair(
                        Self::declaration_specifier(abi),
                        // We only have to deal with declarations containing only one
                        // declarator, i.e. we only handle "int foo" and not "int foo, bar;"
                        Self::declarator(false),
                    )),
                    |(typ, declarator)| {
                        let typ = (declarator.modify_typ)(typ);
                        match declarator.identifier {
                            Some(identifier) => {
                                Ok(CDeclaration {
                                    identifier,
                                    typ
                                })
                            },
                            None => Err(CParseError::DeclarationWithoutIdentifier(typ))
                        }
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
                            Self::declaration_specifier(abi),
                            // This will be an abstract declarator, i.e. a declarator with
                            // no identifier (like parameters in a function prototype), as
                            // the name comes after the last "[]"
                            Self::declarator(true),
                            opt(context(
                                "__data_loc identifier",
                                lexeme(Self::identifier()),
                            )),
                        ))),
                        |(typ, abstract_declarator, identifier)| {
                            // Push the array sizes down the stack. The 2nd nested array takes the size of the 1st etc.
                            fn push_array_size(
                                typ: CType,
                                size: Option<Size>,
                            ) -> Result<(bool, CType), CParseError> {
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
                                        Err(CParseError::NestedDataLoc)
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
                                    typ => Err(CParseError::DataLocArrayNotArray(typ))
                                }
                            } else {
                                Ok(CType::DynamicScalar(Box::new(typ)))
                            }?;

                            let identifier = if is_array {
                                match identifier {
                                    None => Err(CParseError::DataLocArrayWithoutIdentifier),
                                    Some(id) => Ok(id),
                                }
                            } else {
                                match (abstract_declarator.identifier, identifier) {
                                    (Some(id), None) => Ok(id),
                                    (None, Some(id)) => Ok(id),
                                    (None, None) => {
                                        Err(CParseError::DataLocWithoutIdentifier)
                                    }
                                    (Some(id1), Some(id2)) => Err(
                                        CParseError::DataLocAmbiguousIdentifier(id1, id2)
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

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.2p1
        rule postfix_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    Self::primary_expr(abi),

                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
        rule unary_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    Self::postfix_expr(abi),
                    context("preinc expr",
                            preceded(
                                lexeme(tag("++")),
                                Self::unary_expr(abi),
                            ).map(|e| CExpr::Preinc(Box::new(e)))
                    ),
                    context("predec expr",
                            preceded(
                                lexeme(tag("--")),
                                Self::unary_expr(abi),
                            ).map(|e| CExpr::Predec(Box::new(e)))
                    ),
                    context("unary op expr",
                        tuple((
                            lexeme(
                                // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
                                alt((
                                    context("unary &", char('&').map(|_| Box::new(|e| CExpr::Addr(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                    context("unary *", char('*').map(|_| Box::new(|e| CExpr::Deref(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                    context("unary +", char('+').map(|_| Box::new(|e| CExpr::Plus(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                    context("unary -", char('-').map(|_| Box::new(|e| CExpr::Minus(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                    context("unary ~", char('~').map(|_| Box::new(|e| CExpr::Tilde(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                    context("unary !", char('~').map(|_| Box::new(|e| CExpr::Bang(Box::new(e))) as Box<dyn Fn(_) -> _>)),
                                ))
                            ),
                            Self::cast_expr(abi),
                        )).map(|(modify, e)| modify(e))
                    ),
                    context("sizeof type",
                        preceded(
                            lexeme(tag("sizeof")),
                            parenthesized(
                                Self::type_name(abi),
                            )
                        ).map(|typ| CExpr::SizeofType(typ))
                    ),
                    context("sizeof expr",
                        preceded(
                            lexeme(tag("sizeof")),
                            Self::unary_expr(abi),
                        ).map(|e| CExpr::SizeofExpr(Box::new(e)))
                    ),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.7p1
        rule type_name<'abi>(abi: &'abi Abi) -> CType {
            lexeme(
                tuple((
                    Self::declaration_specifier(abi),
                    Self::declarator(true),
                )).map(|(typ, abstract_declarator)|
                    (abstract_declarator.modify_typ)(typ)
                )
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.4p1
        rule cast_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    Self::unary_expr(abi),
                    context("cast",
                        tuple((
                            parenthesized(
                                Self::type_name(abi),
                            ),
                            Self::cast_expr(abi),
                        )).map(|(typ, e)| CExpr::Cast(typ, Box::new(e)))
                    )
                ))
            )
        }

        rule string_literal() -> CExpr {
            tuple((
                // TODO: do we want to handle the encoding prefix ?
                context("string encoding prefix",
                    lexeme(opt(alt((
                        tag("u8"),
                        tag("u"),
                        tag("U"),
                        tag("L"),
                    ))))
                ),
                delimited(
                    char('"'),
                    context("string char sequence",
                        map_res(
                            // Regrettably, type inference with map_res() breaks
                            // and we are forced to spell out the type of input,
                            // including a reference to a lifetime introduced
                            // inside the grammar!() macro.
                            |mut input: Span<'i, CGrammar>| {
                                let mut seq = Vec::with_capacity(128);
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
                                        .parse(input.clone());
                                    match res {
                                        Ok((_input, c)) => {
                                            seq.push(c);
                                            input = _input;
                                            continue;
                                        }
                                        _ => (),
                                    }

                                    // Parse runs of non-escaped chars
                                    let res = is_not::<_, _, E>(r#"\""#).parse(input.clone());
                                    match res {
                                        Ok((_input, s)) => {
                                            seq.extend(*s.fragment());
                                            input = _input;
                                            continue;
                                        }
                                        _ => return Ok((input, seq)),
                                    }
                                }
                            },
                            move |s| {
                                from_utf8(&s)
                                    .map_err(|err| CParseError::DecodeUtf8(err.to_string()))
                                    .map(|s| s.into())
                            },
                        ),
                    ),
                    char('"'),
                )
            )).map(|(prefix, seq)| CExpr::StringLiteral(seq))
        }

        // TODO: parse literals properly, including hex
        rule literal() -> CExpr {
            lexeme(
                alt((
                    Self::string_literal(),
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
                    lexeme(Self::literal()),
                    // TODO: this is a postfix expression
                    parenthesized(Self::expr(abi)),
                    Self::unary_expr(abi),
                ))
            )
        }

        rule expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(alt((
                Self::primary_expr(abi),
                // preceded(tag("REC->"), c_identifier_parser()).map(|id| CExpr::EventField(id)),
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{grammar::PackratGrammar, header::Endianness, parser::test_parser};

    #[test]
    fn expr_test() {
        fn test<'a>(src: Input<'a>, expected: CExpr) {
            let abi = Abi {
                long_size: Some(LongSize::Bits64),
                endianness: Endianness::Little,
            };
            let parser = CGrammar::expr(&abi);
            let src = CGrammar::make_span(src);
            test_parser(expected, src, parser);
        }

        // Literal
        test(b"1", CExpr::IntLiteral(1));
        test(b"42", CExpr::IntLiteral(42));
        test(b" 1 ", CExpr::IntLiteral(1));
        test(b" 42 ", CExpr::IntLiteral(42));
        test(br#""a""#, CExpr::StringLiteral("a".into()));
        test(
            br#"" hello world ""#,
            CExpr::StringLiteral(" hello world ".into()),
        );
        test(
            br#""1 hello world ""#,
            CExpr::StringLiteral("1 hello world ".into()),
        );
        test(
            br#""hello \n world""#,
            CExpr::StringLiteral("hello \n world".into()),
        );
        test(br#""\n\t\\""#, CExpr::StringLiteral("\n\t\\".into()));

        // Address of
        test(b" &1 ", CExpr::Addr(Box::new(CExpr::IntLiteral(1))));

        // Deref
        test(
            b" *&1 ",
            CExpr::Deref(Box::new(CExpr::Addr(Box::new(CExpr::IntLiteral(1))))),
        );

        // Arithmetic
        test(b"+1", CExpr::Plus(Box::new(CExpr::IntLiteral(1))));
        test(b" +1", CExpr::Plus(Box::new(CExpr::IntLiteral(1))));
        test(b"-1", CExpr::Minus(Box::new(CExpr::IntLiteral(1))));
        test(b" - 1 ", CExpr::Minus(Box::new(CExpr::IntLiteral(1))));
        test(b" ~ 1 ", CExpr::Tilde(Box::new(CExpr::IntLiteral(1))));

        // Cast
        test(
            b"-(int)1 ",
            CExpr::Minus(Box::new(CExpr::Cast(
                CType::Basic(CBasicType::I32),
                Box::new(CExpr::IntLiteral(1)),
            ))),
        );
        test(
            b"-(int)(unsigned long)1 ",
            CExpr::Minus(Box::new(CExpr::Cast(
                CType::Basic(CBasicType::I32),
                Box::new(CExpr::Cast(
                    CType::Basic(CBasicType::U64),
                    Box::new(CExpr::IntLiteral(1)),
                )),
            ))),
        );

        // Sizeof type
        test(
            b"sizeof(unsigned long)",
            CExpr::SizeofType(CType::Basic(CBasicType::U64)),
        );

        test(
            b"sizeof (s32)",
            CExpr::SizeofType(CType::Basic(CBasicType::I32)),
        );

        // Sizeof expr
        test(
            b"sizeof(1)",
            CExpr::SizeofExpr(Box::new(CExpr::IntLiteral(1))),
        );

        test(
            b"sizeof(-(int)1)",
            CExpr::SizeofExpr(Box::new(CExpr::Minus(Box::new(CExpr::Cast(
                CType::Basic(CBasicType::I32),
                Box::new(CExpr::IntLiteral(1)),
            ))))),
        );
        test(
            b"sizeof - (int ) 1 ",
            CExpr::SizeofExpr(Box::new(CExpr::Minus(Box::new(CExpr::Cast(
                CType::Basic(CBasicType::I32),
                Box::new(CExpr::IntLiteral(1)),
            ))))),
        );

        // Pre-increment
        test(b"++ 42 ", CExpr::Preinc(Box::new(CExpr::IntLiteral(42))));
        test(
            b"++ sizeof - (int ) 1 ",
            CExpr::Preinc(Box::new(CExpr::SizeofExpr(Box::new(CExpr::Minus(
                Box::new(CExpr::Cast(
                    CType::Basic(CBasicType::I32),
                    Box::new(CExpr::IntLiteral(1)),
                )),
            ))))),
        );

        // Pre-decrement
        test(
            b"-- -42 ",
            CExpr::Predec(Box::new(CExpr::Minus(Box::new(CExpr::IntLiteral(42))))),
        );
    }

    #[test]
    fn declaration_test() {
        fn test<'a>(decl: Input<'a>, id: &'a str, typ: CType) {
            let expected = CDeclaration {
                identifier: id.into(),
                typ: typ,
            };
            let abi = Abi {
                long_size: Some(LongSize::Bits64),
                endianness: Endianness::Little,
            };
            let parser = CGrammar::declaration(&abi);
            let decl = CGrammar::make_span(decl);
            test_parser(expected, decl, parser);
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
            b" u64  foo\t [1][2][3]",
            "foo",
            CType::Array(
                Box::new(CType::Array(
                    Box::new(CType::Array(
                        Box::new(CType::Basic(CBasicType::U64)),
                        Some(3),
                    )),
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
}

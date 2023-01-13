use core::{fmt::Debug, str::from_utf8};
use std::{rc::Rc, string::ToString};

use std::string::String as StdString;

use smartstring::alias::String;
use thiserror::Error;

use nom::{
    branch::alt,
    bytes::complete::{is_not, tag},
    character::complete::{alpha1, alphanumeric1, char, u64 as txt_u64},
    combinator::{fail, map_res, opt, recognize, success},
    error::{ErrorKind, FromExternalError},
    multi::{fold_many1, many0, many0_count, many1, separated_list0, separated_list1},
    number::complete::u8,
    sequence::{delimited, pair, preceded, separated_pair, terminated, tuple},
    Parser,
};

use crate::{
    grammar::grammar,
    header::{Abi, Identifier, LongSize, Size},
    parser::{lexeme, no_backtrack, parenthesized, success_with},
};

// TODO: merge CBasicType and CType
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

    // Complete black box used in cases where we want to completely hide any
    // information about the type.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CType {
    Basic(CBasicType),
    Typedef(Identifier),
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
    Variable(Identifier),

    Uninit,

    ScalarInitializer(Box<CExpr>),
    ListInitializer(Vec<CExpr>),
    DesignatedInitializer(Box<CExpr>, Box<CExpr>),
    CompoundLiteral(CType, Vec<CExpr>),

    IntConstant(u64),
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
    PreInc(Box<CExpr>),
    PreDec(Box<CExpr>),
    PostInc(Box<CExpr>),
    PostDec(Box<CExpr>),

    MemberAccess(Box<CExpr>, Identifier),
    FuncCall(Box<CExpr>, Vec<CExpr>),
    Subscript(Box<CExpr>, Box<CExpr>),
    Assign(Box<CExpr>, Box<CExpr>),

    Mul(Box<CExpr>, Box<CExpr>),
    Div(Box<CExpr>, Box<CExpr>),
    Mod(Box<CExpr>, Box<CExpr>),
    Add(Box<CExpr>, Box<CExpr>),
    Sub(Box<CExpr>, Box<CExpr>),

    Eq(Box<CExpr>, Box<CExpr>),
    NEq(Box<CExpr>, Box<CExpr>),
    LoEq(Box<CExpr>, Box<CExpr>),
    HiEq(Box<CExpr>, Box<CExpr>),
    Hi(Box<CExpr>, Box<CExpr>),
    Lo(Box<CExpr>, Box<CExpr>),

    And(Box<CExpr>, Box<CExpr>),
    Or(Box<CExpr>, Box<CExpr>),

    LShift(Box<CExpr>, Box<CExpr>),
    RShift(Box<CExpr>, Box<CExpr>),
    BitAnd(Box<CExpr>, Box<CExpr>),
    BitOr(Box<CExpr>, Box<CExpr>),
    BitXor(Box<CExpr>, Box<CExpr>),

    Ternary(Box<CExpr>, Box<CExpr>, Box<CExpr>),
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
fn interpret_c_numeric_expr<'a>(expr: CExpr) -> Option<u64> {
    use CExpr::*;
    match expr {
        IntConstant(x) => Some(x),
        // It used to be a macro, but is an enum in recent kernels for compat
        // with eBPF ...
        Variable(id) if &id == "TASK_COMM_LEN" => Some(16),
        _ => None,
    }
}

// TODO: link to the ISO C spec URL for each rule

// Grammar inspired by N1570 ISO C draft (latest C11 draft):
// https://port70.net/~nsz/c/c11/n1570.html#A.2.1
grammar! {
    name: pub CGrammar,
    error: CParseError,
    rules: {

        // https://port70.net/~nsz/c/c11/n1570.html#6.4.2.1
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

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.3p1
        rule type_qualifier() -> () {
            lexeme(alt((
                tag("const"),
                tag("restrict"),
                tag("volatile"),
                tag("_Atomic"),
            ))).map(|_| ())
        }

        // TODO: deal with array types etc
        // https://port70.net/~nsz/c/c11/n1570.html#6.7.6
        rule declarator<'abi>(abi: &'abi Abi, abstract_declarator: bool) -> CDeclarator {
            lexeme(pair(
                context(
                    "pointer",
                    many0_count(pair(
                        lexeme(char('*')),
                        many0_count(Self::type_qualifier()),
                    )),
                ),
                Self::direct_declarator(abi, abstract_declarator),
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

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.6
        rule direct_declarator<'abi>(abi: &'abi Abi, abstract_declarator: bool) -> CDeclarator {
            let name = if abstract_declarator {
                "abstract direct declarator"
            } else {
                "direct declarator"
            };
            context(name, move |input| {
                let _id = || {
                    lexeme(Self::identifier().map(|id| CDeclarator {
                        identifier: Some(id),
                        modify_typ: Rc::new(|typ| typ),
                    }))
                };
                let id = || {
                    move |input| {
                        if abstract_declarator {
                            alt((
                                _id(),
                                success_with(|| CDeclarator {
                                    identifier: None,
                                    modify_typ: Rc::new(|typ| typ),
                                }),
                            ))
                            .parse(input)
                        } else {
                            _id().parse(input)
                        }
                    }
                };

                let paren = || {
                    context(
                        "parenthesized",
                        lexeme(parenthesized(Self::declarator(abi, abstract_declarator))),
                    )
                };

                let array = context(
                    "array",
                    pair(
                        Self::direct_declarator(abi, abstract_declarator),
                        context(
                            "array size",
                            lexeme(delimited(
                                char('['),
                                preceded(
                                    delimited(
                                        lexeme(opt(tag("static"))),
                                        many0(
                                            Self::type_qualifier()
                                        ),
                                        lexeme(opt(tag("static"))),
                                    ),
                                    lexeme(opt(Self::assignment_expr(abi))),
                                ),
                                char(']'),
                            )),
                        ),
                    ),
                )
                .map(|(declarator, array_size)| {
                    let array_size = match array_size {
                        Some(array_size) => {
                            interpret_c_numeric_expr(array_size).map(|x| x.into())
                        }
                        _ => None,
                    };

                    let modify_typ = Rc::new(move |typ| (declarator.modify_typ)(CType::Array(Box::new(typ), array_size)));
                    CDeclarator {
                        modify_typ,
                        identifier: declarator.identifier,
                    }
                });

                let parser = alt((array, paren(), id()));
                lexeme(parser).parse(input)
            })
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7
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

                                _ => CType::Typedef(id),
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

        // https://port70.net/~nsz/c/c11/n1570.html#6.7
        rule declaration<'abi>(abi: &'abi Abi) -> CDeclaration {
            // Parser for ISO C
            let iso = context(
                "iso C declaration",
                map_res(
                    lexeme(pair(
                        Self::declaration_specifier(abi),
                        // We only have to deal with declarations containing only one
                        // declarator, i.e. we only handle "int foo" and not "int foo, bar;"
                        Self::declarator(abi, false),
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
                            Self::declarator(abi, true),
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
                                    | typ @ CType::Typedef(_)
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

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.9p1
        rule initializer<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    delimited(
                        lexeme(char('{')),
                        Self::initializer_list(abi).map(CExpr::ListInitializer),
                        preceded(
                            lexeme(opt(char(','))),
                            lexeme(char('}')),
                        )
                    ),
                    Self::assignment_expr(abi).map(|expr| CExpr::ScalarInitializer(Box::new(expr))),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.9p1
        rule initializer_list<'abi>(abi: &'abi Abi) -> Vec<CExpr> {

            enum DesignatorKind {
                Subscript(CExpr),
                Member(Identifier),
            }
            let designator = || {
                lexeme(
                    alt((
                        delimited(
                            lexeme(char('[')),
                            Self::constant_expr(abi),
                            lexeme(char(']')),
                        ).map(DesignatorKind::Subscript),
                        preceded(
                            lexeme(char('.')),
                            Self::identifier(),
                        ).map(DesignatorKind::Member),
                    )).map(|kind| {
                        move |parent| match kind {
                            DesignatorKind::Subscript(expr) => CExpr::Subscript(Box::new(parent), Box::new(expr)),
                            DesignatorKind::Member(id) => CExpr::MemberAccess(Box::new(parent), id),
                        }
                    })
                )
            };
            lexeme(
                separated_list1(
                    lexeme(char(',')),
                    alt((
                        separated_pair(
                            fold_many1(
                                designator(),
                                || CExpr::Uninit,
                                |parent, combine| combine(parent)
                            ),
                            lexeme(char('=')),
                            Self::initializer(abi),
                        ).map(|(designation, expr)| CExpr::DesignatedInitializer(Box::new(designation), Box::new(expr))),
                        Self::initializer(abi)
                    )),
                )
            )
        }


        // TODO: finish this rule
        // https://port70.net/~nsz/c/c11/n1570.html#6.5.2p1
        rule postfix_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("postinc expr",
                        terminated(
                            Self::postfix_expr(abi),
                            lexeme(tag("++")),
                        ).map(|expr| CExpr::PostInc(Box::new(expr)))
                    ),
                    context("postdec expr",
                        terminated(
                            Self::postfix_expr(abi),
                            lexeme(tag("--")),
                        ).map(|expr| CExpr::PostDec(Box::new(expr)))
                    ),
                    context("subscript expr",
                        tuple((
                            Self::postfix_expr(abi),
                            delimited(
                                lexeme(char('[')),
                                Self::expr(abi),
                                lexeme(char(']')),
                            ),
                        )).map(|(array, index)| CExpr::Subscript(Box::new(array), Box::new(index)))
                    ),
                    context("func call expr",
                        tuple((
                            Self::postfix_expr(abi),
                            delimited(
                                lexeme(char('(')),
                                separated_list0(
                                    lexeme(char(',')),
                                    Self::assignment_expr(abi)
                                ),
                                lexeme(char(')')),
                            ),
                        )).map(|(func, args)| CExpr::FuncCall(Box::new(func), args))
                    ),
                    context("member access expr",
                        separated_pair(
                            Self::postfix_expr(abi),
                            lexeme(char('.')),
                            Self::identifier(),
                        ).map(|(value, member)| CExpr::MemberAccess(Box::new(value), member))
                    ),
                    context("deref member access expr",
                        separated_pair(
                            Self::postfix_expr(abi),
                            lexeme(tag("->")),
                            Self::identifier(),
                        ).map(|(value, member)| CExpr::MemberAccess(Box::new(CExpr::Deref(Box::new(value))), member))
                    ),

                    context("compound literal",
                        tuple((
                            delimited(
                                lexeme(char('(')),
                                Self::type_name(abi),
                                lexeme(char(')')),
                            ),
                            delimited(
                                lexeme(char('{')),
                                Self::initializer_list(abi),
                                preceded(
                                    lexeme(opt(char(','))),
                                    lexeme(char('}')),
                                )
                            ),
                        )).map(|(typ, init)| CExpr::CompoundLiteral(typ, init))
                    ),
                    Self::primary_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.5
        rule multiplicative_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("* expr",
                        separated_pair(
                            Self::multiplicative_expr(abi),
                            lexeme(char('*')),
                            Self::cast_expr(abi),
                        ).map(|(lop, rop)| CExpr::Mul(Box::new(lop), Box::new(rop)))
                    ),
                    context("/ expr",
                        separated_pair(
                            Self::multiplicative_expr(abi),
                            lexeme(char('/')),
                            Self::cast_expr(abi),
                        ).map(|(lop, rop)| CExpr::Div(Box::new(lop), Box::new(rop)))
                    ),
                    context("% expr",
                        separated_pair(
                            Self::multiplicative_expr(abi),
                            lexeme(char('%')),
                            Self::cast_expr(abi),
                        ).map(|(lop, rop)| CExpr::Mod(Box::new(lop), Box::new(rop)))
                    ),
                    Self::cast_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.6
        rule additive_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("+ expr",
                        separated_pair(
                            Self::additive_expr(abi),
                            lexeme(char('+')),
                            Self::multiplicative_expr(abi),
                        ).map(|(lop, rop)| CExpr::Add(Box::new(lop), Box::new(rop)))
                    ),
                    context("- expr",
                        separated_pair(
                            Self::additive_expr(abi),
                            lexeme(char('-')),
                            Self::multiplicative_expr(abi),
                        ).map(|(lop, rop)| CExpr::Sub(Box::new(lop), Box::new(rop)))
                    ),
                    Self::multiplicative_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.7
        rule shift_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("<< expr",
                        separated_pair(
                            Self::shift_expr(abi),
                            lexeme(tag("<<")),
                            Self::additive_expr(abi),
                        ).map(|(lop, rop)| CExpr::LShift(Box::new(lop), Box::new(rop)))
                    ),
                    context(">> expr",
                        separated_pair(
                            Self::shift_expr(abi),
                            lexeme(tag(">>")),
                            Self::additive_expr(abi),
                        ).map(|(lop, rop)| CExpr::RShift(Box::new(lop), Box::new(rop)))
                    ),
                    Self::additive_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.8
        rule relational_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("<= expr",
                        separated_pair(
                            Self::relational_expr(abi),
                            lexeme(tag("<=")),
                            Self::shift_expr(abi),
                        ).map(|(lop, rop)| CExpr::LoEq(Box::new(lop), Box::new(rop)))
                    ),
                    context(">= expr",
                        separated_pair(
                            Self::relational_expr(abi),
                            lexeme(tag(">=")),
                            Self::shift_expr(abi),
                        ).map(|(lop, rop)| CExpr::HiEq(Box::new(lop), Box::new(rop)))
                    ),
                    context("< expr",
                        separated_pair(
                            Self::relational_expr(abi),
                            lexeme(char('<')),
                            Self::shift_expr(abi),
                        ).map(|(lop, rop)| CExpr::Lo(Box::new(lop), Box::new(rop)))
                    ),
                    context("> expr",
                        separated_pair(
                            Self::relational_expr(abi),
                            lexeme(char('>')),
                            Self::shift_expr(abi),
                        ).map(|(lop, rop)| CExpr::Hi(Box::new(lop), Box::new(rop)))
                    ),
                    Self::shift_expr(abi),
                ))
            )
        }


        // https://port70.net/~nsz/c/c11/n1570.html#6.5.9
        rule equality_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("== expr",
                        separated_pair(
                            Self::equality_expr(abi),
                            lexeme(tag("==")),
                            Self::relational_expr(abi),
                        ).map(|(lop, rop)| CExpr::Eq(Box::new(lop), Box::new(rop)))
                    ),
                    context("!=",
                        separated_pair(
                            Self::equality_expr(abi),
                            lexeme(tag("!=")),
                            Self::relational_expr(abi),
                        ).map(|(lop, rop)| CExpr::NEq(Box::new(lop), Box::new(rop)))
                    ),
                    Self::relational_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.10
        rule and_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("& expr",
                        separated_pair(
                            Self::and_expr(abi),
                            lexeme(char('&')),
                            Self::equality_expr(abi),
                        ).map(|(lop, rop)| CExpr::BitAnd(Box::new(lop), Box::new(rop)))
                    ),
                    Self::equality_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.11
        rule exclusive_or_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("^ expr",
                        separated_pair(
                            Self::exclusive_or_expr(abi),
                            lexeme(char('^')),
                            Self::and_expr(abi),
                        ).map(|(lop, rop)| CExpr::BitXor(Box::new(lop), Box::new(rop)))
                    ),
                    Self::and_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.12
        rule inclusive_or_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("| expr",
                        separated_pair(
                            Self::inclusive_or_expr(abi),
                            lexeme(char('|')),
                            Self::exclusive_or_expr(abi),
                        ).map(|(lop, rop)| CExpr::BitOr(Box::new(lop), Box::new(rop)))
                    ),
                    Self::exclusive_or_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.13
        rule logical_and_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("&& expr",
                        separated_pair(
                            Self::logical_and_expr(abi),
                            lexeme(tag("&&")),
                            Self::inclusive_or_expr(abi),
                        ).map(|(lop, rop)| CExpr::And(Box::new(lop), Box::new(rop)))
                    ),
                    Self::inclusive_or_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.14
        rule logical_or_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("|| expr",
                        separated_pair(
                            Self::logical_or_expr(abi),
                            lexeme(tag("||")),
                            Self::logical_and_expr(abi),
                        ).map(|(lop, rop)| CExpr::Or(Box::new(lop), Box::new(rop)))
                    ),
                    Self::logical_and_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
        rule conditional_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("ternary expr",
                        separated_pair(
                            context("ternary cond expr",
                                Self::logical_or_expr(abi)
                            ),
                            lexeme(char('?')),
                            separated_pair(
                                context("ternary true expr",
                                    Self::expr(abi)
                                ),
                                lexeme(char(':')),
                                context("ternary false expr",
                                    Self::conditional_expr(abi)
                                ),
                            ),
                        ).map(|(cond, (true_, false_))| CExpr::Ternary(Box::new(cond), Box::new(true_), Box::new(false_)))
                    ),
                    Self::logical_or_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.6
        rule constant_expr<'abi>(abi: &'abi Abi) -> CExpr {
            Self::conditional_expr(abi)
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
        rule assignment_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("assignment",
                        tuple((
                            Self::unary_expr(abi),
                            lexeme(alt((
                                tag("="),
                                tag("*="),
                                tag("/="),
                                tag("%="),
                                tag("+="),
                                tag("-="),
                                tag("<<="),
                                tag(">>="),
                                tag("&="),
                                tag("^="),
                                tag("|="),
                            ))),
                            Self::assignment_expr(abi),
                        )).map(|(lexpr, op, rexpr)| {
                            use CExpr::*;
                            match &op.fragment()[..] {
                                b"=" => Assign(Box::new(lexpr), Box::new(rexpr)),
                                b"*=" => Assign(Box::new(lexpr.clone()), Box::new(Mul(Box::new(lexpr), Box::new(rexpr)))),
                                b"/=" => Assign(Box::new(lexpr.clone()), Box::new(Div(Box::new(lexpr), Box::new(rexpr)))),
                                b"%=" => Assign(Box::new(lexpr.clone()), Box::new(Mod(Box::new(lexpr), Box::new(rexpr)))),
                                b"+=" => Assign(Box::new(lexpr.clone()), Box::new(Add(Box::new(lexpr), Box::new(rexpr)))),
                                b"-=" => Assign(Box::new(lexpr.clone()), Box::new(Sub(Box::new(lexpr), Box::new(rexpr)))),
                                b"<<=" => Assign(Box::new(lexpr.clone()), Box::new(LShift(Box::new(lexpr), Box::new(rexpr)))),
                                b">>=" => Assign(Box::new(lexpr.clone()), Box::new(RShift(Box::new(lexpr), Box::new(rexpr)))),
                                b"&=" => Assign(Box::new(lexpr.clone()), Box::new(BitAnd(Box::new(lexpr), Box::new(rexpr)))),
                                b"^=" => Assign(Box::new(lexpr.clone()), Box::new(BitXor(Box::new(lexpr), Box::new(rexpr)))),
                                b"|=" => Assign(Box::new(lexpr.clone()), Box::new(BitOr(Box::new(lexpr), Box::new(rexpr)))),
                                _ => panic!("unhandled assignment operator")
                            }
                        })
                    ),
                    Self::conditional_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.3p1
        rule unary_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("preinc expr",
                            preceded(
                                lexeme(tag("++")),
                                Self::unary_expr(abi),
                            ).map(|e| CExpr::PreInc(Box::new(e)))
                    ),
                    context("predec expr",
                            preceded(
                                lexeme(tag("--")),
                                Self::unary_expr(abi),
                            ).map(|e| CExpr::PreDec(Box::new(e)))
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
                    Self::postfix_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.7.7p1
        rule type_name<'abi>(abi: &'abi Abi) -> CType {
            lexeme(
                tuple((
                    Self::declaration_specifier(abi),
                    Self::declarator(abi, true),
                )).map(|(typ, abstract_declarator)|
                    (abstract_declarator.modify_typ)(typ)
                )
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.4p1
        rule cast_expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(
                alt((
                    context("cast expr",
                        tuple((
                            parenthesized(
                                Self::type_name(abi),
                            ),
                            Self::cast_expr(abi),
                        )).map(|(typ, e)| CExpr::Cast(typ, Box::new(e)))
                    ),
                    Self::unary_expr(abi),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.4.5
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
            )).map(|(_prefix, seq)| CExpr::StringLiteral(seq))
        }

        // TODO: parse integer constant literals properly, including hex
        rule constant() -> CExpr {
            lexeme(
                alt((
                    context("integer constant", txt_u64.map(CExpr::IntConstant)),
                    // TODO: this will require desambiguation with the idenfitier in primary_expr()
                    // context("enum constant", Self::identifier()),
                ))
            )
        }

        // https://port70.net/~nsz/c/c11/n1570.html#6.5.1p1
        rule primary_expr<'abi>(abi: &'abi Abi) -> CExpr
        {
            lexeme(
                alt((
                    parenthesized(
                        Self::expr(abi)
                    ),
                    // TODO: we should only accept "REC" as a variable
                    Self::identifier().map(|id| CExpr::Variable(id)),
                    Self::constant(),
                    Self::string_literal(),
                ))
            )
        }

        rule expr<'abi>(abi: &'abi Abi) -> CExpr {
            lexeme(alt((
                Self::assignment_expr(abi),
                // preceded(tag("REC->"), c_identifier_parser()).map(|id| CExpr::EventField(id)),
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grammar::PackratGrammar,
        header::Endianness,
        parser::{tests::test_parser, Input},
    };

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
        test(b"1", CExpr::IntConstant(1));
        test(b"42", CExpr::IntConstant(42));
        test(b" 1 ", CExpr::IntConstant(1));
        test(b" 42 ", CExpr::IntConstant(42));
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
        test(b" &1 ", CExpr::Addr(Box::new(CExpr::IntConstant(1))));

        // Deref
        test(
            b" *&1 ",
            CExpr::Deref(Box::new(CExpr::Addr(Box::new(CExpr::IntConstant(1))))),
        );

        // Unary
        test(b"+1", CExpr::Plus(Box::new(CExpr::IntConstant(1))));
        test(b" +1", CExpr::Plus(Box::new(CExpr::IntConstant(1))));
        test(b"-1", CExpr::Minus(Box::new(CExpr::IntConstant(1))));
        test(b" - 1 ", CExpr::Minus(Box::new(CExpr::IntConstant(1))));
        test(b" ~ 1 ", CExpr::Tilde(Box::new(CExpr::IntConstant(1))));

        // Cast
        test(
            b"(int)1 ",
            CExpr::Cast(
                CType::Basic(CBasicType::I32),
                Box::new(CExpr::IntConstant(1)),
            ),
        );
        test(
            b"(type)1 ",
            CExpr::Cast(
                CType::Typedef("type".into()),
                Box::new(CExpr::IntConstant(1)),
            ),
        );
        test(
            b"(type)(1) ",
            CExpr::Cast(
                CType::Typedef("type".into()),
                Box::new(CExpr::IntConstant(1)),
            ),
        );
        test(
            b"-(int)1 ",
            CExpr::Minus(Box::new(CExpr::Cast(
                CType::Basic(CBasicType::I32),
                Box::new(CExpr::IntConstant(1)),
            ))),
        );
        test(
            b"-(int)(unsigned long)1 ",
            CExpr::Minus(Box::new(CExpr::Cast(
                CType::Basic(CBasicType::I32),
                Box::new(CExpr::Cast(
                    CType::Basic(CBasicType::U64),
                    Box::new(CExpr::IntConstant(1)),
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
            CExpr::SizeofExpr(Box::new(CExpr::IntConstant(1))),
        );

        test(
            b"sizeof(-(int)1)",
            CExpr::SizeofExpr(Box::new(CExpr::Minus(Box::new(CExpr::Cast(
                CType::Basic(CBasicType::I32),
                Box::new(CExpr::IntConstant(1)),
            ))))),
        );
        test(
            b"sizeof - (int ) 1 ",
            CExpr::SizeofExpr(Box::new(CExpr::Minus(Box::new(CExpr::Cast(
                CType::Basic(CBasicType::I32),
                Box::new(CExpr::IntConstant(1)),
            ))))),
        );

        // Pre-increment
        test(b"++ 42 ", CExpr::PreInc(Box::new(CExpr::IntConstant(42))));
        test(
            b"++ sizeof - (int ) 1 ",
            CExpr::PreInc(Box::new(CExpr::SizeofExpr(Box::new(CExpr::Minus(
                Box::new(CExpr::Cast(
                    CType::Basic(CBasicType::I32),
                    Box::new(CExpr::IntConstant(1)),
                )),
            ))))),
        );

        // Pre-decrement
        test(
            b"-- -42 ",
            CExpr::PreDec(Box::new(CExpr::Minus(Box::new(CExpr::IntConstant(42))))),
        );

        // Addition
        test(
            b"1+2",
            CExpr::Add(
                Box::new(CExpr::IntConstant(1)),
                Box::new(CExpr::IntConstant(2)),
            ),
        );
        test(
            b" 1 + 2 ",
            CExpr::Add(
                Box::new(CExpr::IntConstant(1)),
                Box::new(CExpr::IntConstant(2)),
            ),
        );
        test(
            b" (1) + (2) ",
            CExpr::Add(
                Box::new(CExpr::IntConstant(1)),
                Box::new(CExpr::IntConstant(2)),
            ),
        );

        // Operator precedence
        test(
            b" 1 + 2 * 3",
            CExpr::Add(
                Box::new(CExpr::IntConstant(1)),
                Box::new(CExpr::Mul(
                    Box::new(CExpr::IntConstant(2)),
                    Box::new(CExpr::IntConstant(3)),
                )),
            ),
        );

        test(
            b" 1 * 2 + 3",
            CExpr::Add(
                Box::new(CExpr::Mul(
                    Box::new(CExpr::IntConstant(1)),
                    Box::new(CExpr::IntConstant(2)),
                )),
                Box::new(CExpr::IntConstant(3)),
            ),
        );

        // Function call
        test(
            b"f(1)",
            CExpr::FuncCall(
                Box::new(CExpr::Variable("f".into())),
                vec![CExpr::IntConstant(1)],
            ),
        );
        test(
            b" f(1, 2, 3)",
            CExpr::FuncCall(
                Box::new(CExpr::Variable("f".into())),
                vec![
                    CExpr::IntConstant(1),
                    CExpr::IntConstant(2),
                    CExpr::IntConstant(3),
                ],
            ),
        );
        test(
            // This is actually not ambiguous with a cast, since the argument
            // list is not valid expression on its own.
            b" (f)(1, 2, 3)",
            CExpr::FuncCall(
                Box::new(CExpr::Variable("f".into())),
                vec![
                    CExpr::IntConstant(1),
                    CExpr::IntConstant(2),
                    CExpr::IntConstant(3),
                ],
            ),
        );

        // Subscript
        test(
            b"arr[1]",
            CExpr::Subscript(
                Box::new(CExpr::Variable("arr".into())),
                Box::new(CExpr::IntConstant(1)),
            ),
        );
        test(
            b"arr[1][2]",
            CExpr::Subscript(
                Box::new(CExpr::Subscript(
                    Box::new(CExpr::Variable("arr".into())),
                    Box::new(CExpr::IntConstant(1)),
                )),
                Box::new(CExpr::IntConstant(2)),
            ),
        );

        // Member access
        test(
            b"x.y",
            CExpr::MemberAccess(Box::new(CExpr::Variable("x".into())), "y".into()),
        );
        test(
            b"x.y.z",
            CExpr::MemberAccess(
                Box::new(CExpr::MemberAccess(
                    Box::new(CExpr::Variable("x".into())),
                    "y".into(),
                )),
                "z".into(),
            ),
        );

        // Compound literal
        test(
            b"(type){0}",
            CExpr::CompoundLiteral(
                CType::Typedef("type".into()),
                vec![CExpr::ScalarInitializer(Box::new(CExpr::IntConstant(0)))],
            ),
        );
        test(
            b"(type){0, 1}",
            CExpr::CompoundLiteral(
                CType::Typedef("type".into()),
                vec![
                    CExpr::ScalarInitializer(Box::new(CExpr::IntConstant(0))),
                    CExpr::ScalarInitializer(Box::new(CExpr::IntConstant(1))),
                ],
            ),
        );
        test(
            b"(type){.x = 0}",
            CExpr::CompoundLiteral(
                CType::Typedef("type".into()),
                vec![CExpr::DesignatedInitializer(
                    Box::new(CExpr::MemberAccess(Box::new(CExpr::Uninit), "x".into())),
                    Box::new(CExpr::ScalarInitializer(Box::new(CExpr::IntConstant(0)))),
                )],
            ),
        );

        // Ambiguous cases

        // Amibiguity of is lifted by 6.4p4 stating that the tokenizer is
        // greedy, i.e. the following is tokenized as "1 ++ + 2":
        // https://port70.net/~nsz/c/c11/n1570.html#6.4p4
        test(
            b" 1 +++ 2 ",
            CExpr::Add(
                Box::new(CExpr::PostInc(Box::new(CExpr::IntConstant(1)))),
                Box::new(CExpr::IntConstant(2)),
            ),
        );
        test(
            b" 1 +++++ 2 ",
            CExpr::Add(
                Box::new(CExpr::PostInc(Box::new(CExpr::PostInc(Box::new(
                    CExpr::IntConstant(1),
                ))))),
                Box::new(CExpr::IntConstant(2)),
            ),
        );

        test(
            b" 1 --- 2 ",
            CExpr::Sub(
                Box::new(CExpr::PostDec(Box::new(CExpr::IntConstant(1)))),
                Box::new(CExpr::IntConstant(2)),
            ),
        );
        test(
            b" 1 ----- 2 ",
            CExpr::Sub(
                Box::new(CExpr::PostDec(Box::new(CExpr::PostDec(Box::new(
                    CExpr::IntConstant(1),
                ))))),
                Box::new(CExpr::IntConstant(2)),
            ),
        );

        // This is genuinely ambiguous: it can be either a cast to type "type"
        // of "+2" or the addition of a "type" variable and 2.
        // We parse it as a cast as the expressions we are interested in only
        // contain one variable (REC).
        test(
            b" (type) + (2) ",
            CExpr::Cast(
                CType::Typedef("type".into()),
                Box::new(CExpr::Plus(Box::new(CExpr::IntConstant(2)))),
            ),
        );

        // Another ambiguous case: could be a function call or a cast. We decide
        // to treat that as a cast, since you can make a call without the extra
        // paren.
        test(
            b" (type)(2) ",
            CExpr::Cast(
                CType::Typedef("type".into()),
                Box::new(CExpr::IntConstant(2)),
            ),
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

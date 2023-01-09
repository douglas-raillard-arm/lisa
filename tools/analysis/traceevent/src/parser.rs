use core::{
    default::Default,
    fmt::{Debug, Display},
};
use std::string::String as StdString;

use nom::{
    bytes::complete::{tag, take_until},
    character::complete::{char, multispace0},
    error::{ContextError, ErrorKind, FromExternalError, ParseError},
    sequence::{delimited, terminated},
    Parser,
};

pub type Input<'a> = &'a [u8];

/// Tie together a nom error and some user-defined data.
#[derive(Debug)]
pub struct NomError<T, E> {
    /// User-defined data.
    pub data: T,
    /// nom error, such as [nom::error::Error]
    pub inner: E,
}

impl<T, E> NomError<T, E>
where
    T: Default,
{
    fn from_inner(e: E) -> Self {
        NomError {
            data: Default::default(),
            inner: e,
        }
    }
}

impl<I, T, E> ParseError<I> for NomError<T, E>
where
    E: ParseError<I>,
    T: Default,
{
    fn from_error_kind(input: I, kind: ErrorKind) -> Self {
        NomError::from_inner(E::from_error_kind(input, kind))
    }
    fn append(input: I, kind: ErrorKind, other: Self) -> Self {
        NomError {
            data: other.data,
            inner: E::append(input, kind, other.inner),
        }
    }

    fn from_char(input: I, c: char) -> Self {
        NomError::from_inner(E::from_char(input, c))
    }
    fn or(self, other: Self) -> Self {
        NomError {
            data: other.data,
            inner: self.inner.or(other.inner),
        }
    }
}

impl<I, T, E> FromExternalError<I, T> for NomError<T, E>
where
    E: ParseError<I>,
{
    fn from_external_error(input: I, kind: ErrorKind, e: T) -> Self {
        NomError {
            data: e,
            inner: E::from_error_kind(input, kind),
        }
    }
}

impl<I, T, E> ContextError<I> for NomError<T, E>
where
    E: ContextError<I>,
{
    fn add_context(input: I, ctx: &'static str, other: Self) -> Self {
        NomError {
            data: other.data,
            inner: E::add_context(input, ctx, other.inner),
        }
    }
}

//////////////
// Conversions
//////////////

pub fn to_str(s: &[u8]) -> StdString {
    StdString::from_utf8_lossy(s).to_string()
}

// pub trait ToIResult<I, O, E> {
//     type E2;
//     fn into_iresult(self, input: I) -> Result<O, nom::Err<Self::E2>>;
// }

// // TODO: generalize to NomError or get rid of it
// impl<I, O, E, T> ToIResult<I, O, E> for Result<O, T>
// where
//     O: Clone,
//     E: ParseError<I>,
// {
//     type E2 = NomError<T, E>;

//     #[inline]
//     fn into_iresult(self, input: I) -> Result<O, nom::Err<Self::E2>> {
//         match self {
//             Ok(x) => Ok(x),
//             Err(err) => Err(nom::Err::Error(NomError::from_external_error(
//                 input,
//                 ErrorKind::Fail,
//                 err,
//             ))),
//         }
//     }
// }

//////////////////////
// Generic combinators
//////////////////////

pub fn print<'a, I, O, E, P>(name: &'static str, mut inner: P) -> impl nom::Parser<I, O, E>
where
    E: ParseError<I>,
    P: nom::Parser<I, O, E>,
    I: core::ops::Deref<Target = Input<'a>>,
    I: Clone,
    O: Debug,
{
    move |input: I| {
        let (i, x) = inner.parse(input.clone())?;
        println!(
            "{name} input={:?} out={x:?} new_input={:?}",
            to_str(&input),
            to_str(&i)
        );
        Ok((i, x))
    }
}

pub fn lexeme<I, O, E, P>(inner: P) -> impl nom::Parser<I, O, E>
where
    E: ParseError<I>,
    P: nom::Parser<I, O, E>,
    I: Clone + nom::InputLength + nom::InputIter + nom::InputTake + nom::InputTakeAtPosition,
    <I as nom::InputIter>::Item: Clone + nom::AsChar,
    <I as nom::InputTakeAtPosition>::Item: Clone + nom::AsChar,
{
    delimited(multispace0, inner, multispace0)
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

// Not available in nom 7 but maybe will be there in nom 8:
// https://github.com/rust-bakery/nom/issues/1422
pub fn map_err<P, F, I, O, E, E2, MappedE>(mut parser: P, f: F) -> impl nom::Parser<I, O, E2>
where
    P: nom::Parser<I, O, E>,
    E: ParseError<I>,
    E2: ParseError<I> + FromExternalError<I, MappedE>,
    F: Fn(E) -> MappedE,
    I: Clone,
{
    move |input: I| match parser.parse(input.clone()) {
        Err(nom::Err::Error(e)) => Err(nom::Err::Error(E2::from_external_error(
            input,
            ErrorKind::Fail,
            f(e),
        ))),
        Err(nom::Err::Failure(e)) => Err(nom::Err::Failure(E2::from_external_error(
            input,
            ErrorKind::Fail,
            f(e),
        ))),
        Err(nom::Err::Incomplete(x)) => Err(nom::Err::Incomplete(x)),
        Ok(x) => Ok(x),
    }
}

pub fn success_with<F, I, O, E>(mut f: F) -> impl FnMut(I) -> nom::IResult<I, O, E>
where
    F: FnMut() -> O,
    E: ParseError<I>,
{
    move |input: I| Ok((input, f()))
}

pub fn null_terminated_str_parser<'a, E>() -> impl nom::Parser<&'a [u8], &'a [u8], E>
where
    E: ParseError<&'a [u8]>,
{
    terminated(
        take_until(&[0][..]),
        // Consume the null terminator
        tag([0]),
    )
}

pub trait DisplayErr {
    fn display_err(&self) -> StdString;
}
pub trait DisplayErrViaDisplay {}

impl DisplayErrViaDisplay for crate::cparser::CParseError {}
impl DisplayErrViaDisplay for crate::header::HeaderError {}

impl<T> DisplayErr for T
where
    T: DisplayErrViaDisplay + Display,
{
    fn display_err(&self) -> StdString {
        format!("{}", self)
    }
}

impl DisplayErr for () {
    fn display_err(&self) -> StdString {
        "".into()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use nom::{error::VerboseError, Finish as _};
    pub fn test_parser<I, O, T, P>(expected: O, input: I, mut parser: P)
    where
        O: Debug + PartialEq,
        T: DisplayErr,
        P: Parser<I, O, NomError<T, VerboseError<I>>>,
        I: nom::AsBytes + Clone,
    {
        let parsed = parser.parse(input.clone()).finish();
        let input = to_str(input.as_bytes());
        match parsed {
            Ok((_, parsed)) => {
                assert_eq!(parsed, expected, "while parsing: {input:?}");
            }
            Err(err) => {
                // Convert input from &[u8] to &str so convert_error() can
                // display it.
                let mut seen_context = false;
                let inner = VerboseError {
                    errors: err
                        .inner
                        .errors
                        .iter()
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
                        .map(|(s, err)| (to_str(s.as_bytes()), err.clone()))
                        .collect(),
                };
                let loc = nom::error::convert_error(input.clone(), inner);
                let err_data = err.data.display_err();
                panic!("Could not parse {input:?}: {err_data} :\n{loc}")
            }
        }
    }
}

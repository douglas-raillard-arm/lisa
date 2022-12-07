use nom::error;

mod header;

pub type Input<'a> = &'a [u8];
pub type IResult<'a, O, E = nom::error::Error<Input<'a>>> =
    nom::IResult<Input<'a>, O, StringError<E>>;

#[derive(Debug)]
pub struct StringError<E> {
    msg: std::string::String,
    inner: E,
}

impl<E> StringError<E> {
    fn from_inner(e: E) -> Self {
        StringError {
            msg: "".into(),
            inner: e,
        }
    }
}

impl<I, E> error::ParseError<I> for StringError<E>
where
    E: error::ParseError<I>,
{
    fn from_error_kind(input: I, kind: error::ErrorKind) -> Self {
        StringError::from_inner(E::from_error_kind(input, kind))
    }
    fn append(input: I, kind: error::ErrorKind, other: Self) -> Self {
        StringError {
            msg: other.msg,
            inner: E::append(input, kind, other.inner),
        }
    }

    fn from_char(input: I, c: char) -> Self {
        StringError::from_inner(E::from_char(input, c))
    }
    fn or(self, other: Self) -> Self {
        StringError {
            msg: other.msg,
            inner: self.inner.or(other.inner),
        }
    }
}

impl<I, E, Eext> error::FromExternalError<I, Eext> for StringError<E>
where
    E: error::FromExternalError<I, Eext>,
{
    fn from_external_error(input: I, kind: error::ErrorKind, e: Eext) -> Self {
        StringError::from_inner(E::from_external_error(input, kind, e))
    }
}

impl<I, E> error::ContextError<I> for StringError<E>
where
    E: error::ContextError<I>,
{
    fn add_context(input: I, ctx: &'static str, other: Self) -> Self {
        StringError {
            msg: other.msg,
            inner: E::add_context(input, ctx, other.inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_test() {
        let data = include_bytes!("../../../../doc/traces/trace.dat");
        // println!("bytes:\n{}", &data[0..100].to_hex(8));
        let res = header::header(data);
        match res {
            Ok((_, _x)) => {
                // println!("{x:?}")
            }
            Err(err) => {
                println!("{err:?}");
                panic!("failed to parse header")
            }
        }
    }
}

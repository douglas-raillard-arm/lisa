use core::{cell::Cell, fmt::Debug, str::from_utf8};
use std::rc::Rc;

use nom::{
    character::complete::{char, multispace0},
    error::ParseError,
    sequence::delimited,
    Finish as _, Parser,
};
use nom_locate::LocatedSpan;

use crate::parser::{lexeme, parenthesized, print, to_str, Input};

type Location = usize;
pub type Span<'i, G> = LocatedSpan<Input<'i>, Rc<Vec<<G as PackratGrammar>::State<'i>>>>;

pub trait PackratGrammar {
    type State<'i>;

    fn make_span<'i>(input: Input<'i>) -> Span<'i, Self>
    where
        Self::State<'i>: Default,
    {
        // Use len + 1 so that we can store a state for empty strings as well
        // (and for the last rule of the parse when we consumed the full input)
        let len = input.len() + 1;
        let mut vector = Vec::with_capacity(len);
        for _ in 0..len {
            vector.push(Default::default());
        }
        let ctx = Rc::new(vector);
        LocatedSpan::new_extra(input, ctx)
    }

    // Use Box<dyn ...> return type instead of impl as impl is not allowed in
    // traits for now (see return_position_impl_trait_in_trait unstable
    // feature)
    fn wrap_rule<'i, 'p, O, E, P>(mut rule: P) -> Box<dyn nom::Parser<Input<'i>, O, E> + 'p>
    where
        P: 'p
            + nom::Parser<
                Span<'i, Self>,
                O,
                // Instantiate the parserr with a VerboseError as we are
                // dealing with source code. Parse failures must be reported
                // cleanly, and absolute performance is less relevant as the
                // input is typically tiny.
                nom::error::VerboseError<Span<'i, Self>>,
            >,
        E: ParseError<Input<'i>>,
        <Self as PackratGrammar>::State<'i>: Default,
    {
        Box::new(move |input: Input<'i>| {
            let span = Self::make_span(input);
            match rule.parse(span) {
                // Convert back from Span to Input so that the
                // parser can be integrated with the rest of the
                // infrastructure.
                Ok((i, o)) => Ok((*i.fragment(), o)),

                // TODO: find a better mapping that preserves the error content
                Err(_err) => Err(nom::Err::Error(E::from_error_kind(
                    input,
                    nom::error::ErrorKind::Fail,
                ))),
            }
        })
    }
}

pub enum PackratAction<T> {
    Seed,
    Succeed(T),
    Fail,
}

impl<T> Default for PackratAction<T> {
    fn default() -> Self {
        PackratAction::Seed
    }
}

// Allow defining grammar production rules with most of the boilerplate
// removed and automatic context() added
macro_rules! grammar {
    (
        name: $vis:vis $grammar_name:ident,
        error: $grammar_error:ty,
        rules: {
            $(rule $name:ident $(<$($generics:tt $(: $bound:tt)?),*>)? ($($param:ident: $param_ty:ty),*) -> $ret:ty $body:block)*
        }) => {
        $vis struct $grammar_name ();

        // Create the state struct in a fresh scope so it will not
        // conflict with any other state structs. It also allows using "use"
        // without polluting the surrounding scope.
        const _: () = {
            use $crate::grammar::{PackratGrammar, PackratAction, Span};
            use ::nom::error::context;

            #[automatically_derived]
            #[allow(non_camel_case_types)]
            #[derive(Default)]
            pub struct PackratState<'i> {
                $(
                    $name: ::core::cell::Cell<PackratAction<(
                        Span<'i, $grammar_name>,
                        $ret,
                    )>>
                ),*
            }

            impl PackratGrammar for $grammar_name {
                // Using Rc<> allows cloning the LocatedSpan while sharing
                // the packrat state.
                type State<'i> = PackratState<'i>;
            }

            impl $grammar_name {
                $(
                    $vis fn $name<'i, 'ret, $($($generics $(: $bound)?,)*)? E>($($param: $param_ty),*) -> impl ::nom::Parser<Span<'i, $grammar_name>, $ret, E> + 'ret
                    where
                        E: 'ret
                        + ::nom::error::ParseError<Span<'i, $grammar_name>>
                        + ::nom::error::ContextError<Span<'i, $grammar_name>>
                        + ::nom::error::FromExternalError<Span<'i, $grammar_name>, $grammar_error>,
                        $($($generics: 'ret),*)?
                    {
                        // Wrap the body in a closure to avoid recursive type issues
                        // when a rule is recursive, and add a context for free.
                        //
                        // Also, this allows to implement packrat parsing
                        // modified to support left recursive grammar.
                        move |input: Span<'i, $grammar_name>| {
                            let orig_pos = input.location_offset();
                            let state = &input.extra.as_ref()[orig_pos].$name;
                            let parser = move |input| $body.parse(input);

                            // We cannot borrow from a Cell, so instead we take the
                            // state and we will put something back later on. In the
                            // mean time, Default::default() value is used (None).
                            match state.take() {
                                PackratAction::Seed => {
                                    // Will make any recursive invocation of the rule fail
                                    state.replace(PackratAction::Fail);

                                    // Parse once, with no recursive call allowed to
                                    // succeed. This provides the seed result that
                                    // will be reinjected at the next attempt.
                                    let mut res = context(concat!(stringify!($name), " (seed)"), parser).parse(input.clone())?;

                                    loop {
                                        let (i, _x) = &res;
                                        let pos = i.location_offset();

                                        // Set the seed, which will make any
                                        // recursive call to that rule succeed with
                                        // that result.
                                        state.replace(PackratAction::Succeed(res.clone()));

                                        // Parse again with the seed in place, so
                                        // that recursive call succeed and we can
                                        // try to match what comes after
                                        let res2 = context(concat!(stringify!($name), " (reparse)"), parser).parse(input.clone())?;

                                        let (i2, _x2) = &res2;
                                        let pos2 = i2.location_offset();

                                        // If we consumed the whole input, we have
                                        // the best match possible.
                                        if i2.fragment().len() == 0 {
                                            return Ok(res2)
                                        } else if pos >= pos2 {
                                            return Ok(res)
                                        // If this resulted in a longer match, take
                                        // it and loop again. Otherwise, we found
                                        // the best match.
                                        } else {
                                            res = res2;
                                        }
                                    }
                                }
                                PackratAction::Succeed((i, x)) => {
                                    state.set(PackratAction::Succeed((i.clone(), x.clone())));
                                    context(concat!(stringify!($name), " (pre-parsed)"), success(x))(i)
                                }
                                s@PackratAction::Fail => {
                                    state.set(s);
                                    context(concat!(stringify!($name), " (seed recursion block)"), fail)(input)
                                }
                            }
                        }
                    }

                )*
            }
        };
    }
}
pub(crate) use grammar;

// Allow importing the macro like any other item inside this crate

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::parser::test_parser;

    #[test]
    fn packrat_test() {
        fn test<'a>(input: Input<'a>, expected: AST) {
            let parser = TestGrammar::starting_symbol();
            let input = TestGrammar::make_span(input);
            test_parser(expected, input, parser);
        }

        #[derive(Debug, Clone, PartialEq)]
        enum AST {
            Var(std::string::String),
            Add(Box<AST>, Box<AST>),
            Sub(Box<AST>, Box<AST>),
            Mul(Box<AST>, Box<AST>),
            Div(Box<AST>, Box<AST>),
        }

        use nom::{
            branch::alt,
            bytes::complete::tag,
            character::complete::alpha1,
            combinator::{fail, recognize, success},
            error::{context, ContextError, FromExternalError, ParseError},
            multi::many1,
            sequence::separated_pair,
        };
        use std::string::ToString;

        grammar! {
            name: TestGrammar,
            error: (),
            rules: {
                rule literal() -> AST {
                    lexeme(recognize(many1(alpha1))).map(|id: LocatedSpan<Input<'_>, _>| AST::Var(to_str(id.fragment()).into()))
                }

                rule expr() -> AST {
                    lexeme(alt((
                        context("add",
                            separated_pair(
                                Self::expr(),
                                tag("+"),
                                Self::term(),
                            ).map(|(op1, op2)| AST::Add(Box::new(op1), Box::new(op2)))
                        ),
                        context("sub",
                            separated_pair(
                                Self::expr(),
                                tag("-"),
                                Self::term(),
                            ).map(|(op1, op2)| AST::Sub(Box::new(op1), Box::new(op2)))
                        ),
                        Self::term(),
                    )))
                }

                rule term() -> AST {
                    lexeme(alt((
                        context("mul",
                            separated_pair(
                                Self::term(),
                                tag("*"),
                                Self::factor(),
                            ).map(|(op1, op2)| AST::Mul(Box::new(op1), Box::new(op2)))
                        ),
                        context("div",
                            separated_pair(
                                Self::term(),
                                tag("/"),
                                Self::factor(),
                            ).map(|(op1, op2)| AST::Div(Box::new(op1), Box::new(op2)))
                        ),
                        Self::factor(),
                    )))
                }


                rule factor() -> AST {
                    lexeme(alt((
                        Self::literal(),
                        context("paren",
                            parenthesized(Self::expr())
                        ),
                    )))
                }

                rule starting_symbol() -> AST {
                    Self::expr()
                }
            }
        }

        test(b" a ", AST::Var("a".to_string()));
        test(b"((( a)) ) ", AST::Var("a".to_string()));
        test(
            b"a + b ",
            AST::Add(
                Box::new(AST::Var("a".to_string())),
                Box::new(AST::Var("b".to_string())),
            ),
        );
        test(
            b" a + b ",
            AST::Add(
                Box::new(AST::Var("a".to_string())),
                Box::new(AST::Var("b".to_string())),
            ),
        );
        test(
            b" a + b",
            AST::Add(
                Box::new(AST::Var("a".to_string())),
                Box::new(AST::Var("b".to_string())),
            ),
        );

        test(
            b" a + b+c ",
            AST::Add(
                Box::new(AST::Add(
                    Box::new(AST::Var("a".to_string())),
                    Box::new(AST::Var("b".to_string())),
                )),
                Box::new(AST::Var("c".to_string())),
            ),
        );

        test(
            b"a+(b+c)",
            AST::Add(
                Box::new(AST::Var("a".to_string())),
                Box::new(AST::Add(
                    Box::new(AST::Var("b".to_string())),
                    Box::new(AST::Var("c".to_string())),
                )),
            ),
        );

        test(
            b"(a+b)+c",
            AST::Add(
                Box::new(AST::Add(
                    Box::new(AST::Var("a".to_string())),
                    Box::new(AST::Var("b".to_string())),
                )),
                Box::new(AST::Var("c".to_string())),
            ),
        );
        test(
            b"(a+b+c)",
            AST::Add(
                Box::new(AST::Add(
                    Box::new(AST::Var("a".to_string())),
                    Box::new(AST::Var("b".to_string())),
                )),
                Box::new(AST::Var("c".to_string())),
            ),
        );
        test(
            b"(a+b*c)",
            AST::Add(
                Box::new(AST::Var("a".to_string())),
                Box::new(AST::Mul(
                    Box::new(AST::Var("b".to_string())),
                    Box::new(AST::Var("c".to_string())),
                )),
            ),
        );
        test(
            b"a*b+c*d",
            AST::Add(
                Box::new(AST::Mul(
                    Box::new(AST::Var("a".to_string())),
                    Box::new(AST::Var("b".to_string())),
                )),
                Box::new(AST::Mul(
                    Box::new(AST::Var("c".to_string())),
                    Box::new(AST::Var("d".to_string())),
                )),
            ),
        );

        test(
            b"(a*b)/(c*d)",
            AST::Div(
                Box::new(AST::Mul(
                    Box::new(AST::Var("a".to_string())),
                    Box::new(AST::Var("b".to_string())),
                )),
                Box::new(AST::Mul(
                    Box::new(AST::Var("c".to_string())),
                    Box::new(AST::Var("d".to_string())),
                )),
            ),
        );
        test(
            b"(a*(b/(c*d)))",
            AST::Mul(
                Box::new(AST::Var("a".to_string())),
                Box::new(AST::Div(
                    Box::new(AST::Var("b".to_string())),
                    Box::new(AST::Mul(
                        Box::new(AST::Var("c".to_string())),
                        Box::new(AST::Var("d".to_string())),
                    )),
                )),
            ),
        );

        test(
            b"a*b/c*d",
            AST::Mul(
                Box::new(AST::Div(
                    Box::new(AST::Mul(
                        Box::new(AST::Var("a".to_string())),
                        Box::new(AST::Var("b".to_string())),
                    )),
                    Box::new(AST::Var("c".to_string())),
                )),
                Box::new(AST::Var("d".to_string())),
            ),
        );

        test(
            b"a-b/c*d",
            AST::Sub(
                Box::new(AST::Var("a".to_string())),
                Box::new(AST::Mul(
                    Box::new(AST::Div(
                        Box::new(AST::Var("b".to_string())),
                        Box::new(AST::Var("c".to_string())),
                    )),
                    Box::new(AST::Var("d".to_string())),
                )),
            ),
        );
    }
}

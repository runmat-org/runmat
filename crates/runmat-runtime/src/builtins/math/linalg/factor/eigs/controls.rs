//! Borrowed text and typed public-control identities shared by both eigs routes.
//! Only the legacy Unbounded parser materializes a canonical name String.
use super::Sigma;
use crate::builtins::math::linalg::ops::linear_operator::{CooperativeOperatorError, OperatorWork};
use runmat_value::Value;
#[derive(Clone, Copy)]
pub(super) enum Text<'a> {
    Utf8(&'a str),
    Characters(&'a [runmat_value::Utf16CodeUnit]),
}
impl<'a> Text<'a> {
    pub fn of(value: &'a Value) -> Option<Self> {
        match value {
            Value::String(runmat_value::StringElement::Text(value)) => {
                Some(Self::Utf8(value.as_str()))
            }
            Value::StringArray(value)
                if value.data().len() == 1 && !value.data()[0].is_missing() =>
            {
                Some(Self::Utf8(
                    value.data()[0]
                        .text_ref()
                        .expect("checked nonmissing scalar string")
                        .as_str(),
                ))
            }
            Value::CharArray(value)
                if value.rows == 1 && value.shape().iter().skip(2).all(|&n| n == 1) =>
            {
                Some(Self::Characters(value.data()))
            }
            _ => None,
        }
    }
    pub fn eq(self, expected: &str) -> bool {
        match self {
            Self::Utf8(value) => value.eq_ignore_ascii_case(expected),
            Self::Characters(value) => {
                value.len() == expected.encode_utf16().count()
                    && value
                        .iter()
                        .map(|unit| unit.raw())
                        .zip(expected.encode_utf16())
                        .all(|(a, b)| {
                            let lower = |unit: u16| {
                                if (65..=90).contains(&unit) {
                                    unit + 32
                                } else {
                                    unit
                                }
                            };
                            lower(a) == lower(b)
                        })
            }
        }
    }
    pub async fn trim(
        self,
        work: &mut Option<OperatorWork<'_>>,
    ) -> Result<Self, CooperativeOperatorError> {
        let result = match self {
            Self::Utf8(mut value) => {
                while let Some(c) = value.chars().next().filter(|c| c.is_whitespace()) {
                    value = &value[c.len_utf8()..];
                    if let Some(work) = work.as_mut() {
                        work.tick().await?;
                    }
                }
                while let Some(c) = value.chars().next_back().filter(|c| c.is_whitespace()) {
                    value = &value[..value.len() - c.len_utf8()];
                    if let Some(work) = work.as_mut() {
                        work.tick().await?;
                    }
                }
                Self::Utf8(value)
            }
            Self::Characters(mut value) => {
                while value.first().is_some_and(|c| {
                    char::from_u32(u32::from(c.raw())).is_some_and(char::is_whitespace)
                }) {
                    value = &value[1..];
                    if let Some(work) = work.as_mut() {
                        work.tick().await?;
                    }
                }
                while value.last().is_some_and(|c| {
                    char::from_u32(u32::from(c.raw())).is_some_and(char::is_whitespace)
                }) {
                    value = &value[..value.len() - 1];
                    if let Some(work) = work.as_mut() {
                        work.tick().await?;
                    }
                }
                Self::Characters(value)
            }
        };
        if let Some(work) = work.as_mut() {
            work.flush().await?;
        }
        Ok(result)
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ImaginaryAlias {
    Explicit,
    Legacy,
}
impl ImaginaryAlias {
    pub fn resolve(self, sigma: Sigma, complex: bool) -> Sigma {
        if matches!(self, Self::Legacy) && !complex {
            Sigma::BothEndsImag
        } else {
            sigma
        }
    }
}
pub(super) fn selector(text: Text<'_>) -> Option<(Sigma, ImaginaryAlias)> {
    let mut result = None;
    for (names, sigma) in [
        (&["largestabs", "lm"][..], Sigma::LargestAbs),
        (&["smallestabs", "sm"][..], Sigma::SmallestAbs),
        (&["largestreal", "lr", "la"][..], Sigma::LargestReal),
        (&["smallestreal", "sr", "sa"][..], Sigma::SmallestReal),
        (&["bothendsreal", "be"][..], Sigma::BothEndsReal),
        (&["largestimag", "li"][..], Sigma::LargestImag),
        (&["smallestimag", "si"][..], Sigma::SmallestImag),
        (&["bothendsimag"][..], Sigma::BothEndsImag),
    ] {
        if names.iter().any(|name| text.eq(name)) {
            result = Some((
                sigma,
                if text.eq("li") {
                    ImaginaryAlias::Legacy
                } else {
                    ImaginaryAlias::Explicit
                },
            ));
            break;
        }
    }
    result
}
pub(super) use runmat_builtins::EigsOption as Key;
pub(super) fn option(text: Text<'_>) -> Option<Key> {
    Key::ALL
        .into_iter()
        .find(|key| key.aliases().iter().any(|name| text.eq(name)))
}

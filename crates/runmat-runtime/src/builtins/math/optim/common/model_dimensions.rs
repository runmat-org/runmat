//! One public numeric/named dimension and axis-label grammar for model arrays.
//! Variable and expression constructors share this parser and its canonical-indexed
//! geometry; public entry owners supply the existing error/cooperation policy.
pub(in crate::builtins::math::optim) use super::model_policy::Policy;
use super::ScalarCooperation;
use crate::BuiltinResult;
use runmat_value::Value;
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::builtins::math::optim) enum ConstructorSyntax {
    Variable,
    Expression,
}
mod axes;
pub(in crate::builtins::math::optim) use axes::{constructor, property};

//! Dispatches a numeric-limit request to its typed implementation family.

use runmat_value::Value;

use crate::BuiltinResult;

use super::operation::LimitOperation;

pub(super) fn run(arguments: Vec<Value>, operation: LimitOperation) -> BuiltinResult<Value> {
    match operation {
        LimitOperation::Integer { name, kind } => super::integer::execute(arguments, kind, name),
        LimitOperation::Floating { name, kind } => super::floating::execute(arguments, kind, name),
    }
}

mod operations;
mod validation;

use runmat_value::Value;

use super::super::outcome::TransferOutcome;
use crate::BuiltinResult;

fn evaluate(args: &[Value]) -> BuiltinResult<TransferOutcome> {
    futures::executor::block_on(super::execute::evaluate(args))
}

fn text(path: &std::path::Path) -> Value {
    Value::from(path.to_string_lossy().to_string())
}

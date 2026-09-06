use runmat_builtins::MATLABROOT_ERROR_TOO_MANY_INPUTS;
use runmat_value::{CharArray, Value};

const IDENTITY: &str = "matlabroot";

pub(super) fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(super::super::error::from_descriptor(
            IDENTITY,
            &MATLABROOT_ERROR_TOO_MANY_INPUTS,
        ));
    }
    let root = super::super::root();
    Ok(Value::CharArray(CharArray::new_row(
        &crate::builtins::common::fs::path_to_string(&root),
    )))
}

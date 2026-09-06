use runmat_builtins::{TEMPDIR_ERROR_TOO_MANY_INPUTS, TEMPDIR_ERROR_UNAVAILABLE};
use runmat_value::Value;

use super::super::{error, output};

const IDENTITY: &str = "tempdir";

pub(super) fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    if !args.is_empty() {
        return Err(error::builtin(IDENTITY, &TEMPDIR_ERROR_TOO_MANY_INPUTS));
    }
    let path = crate::builtins::common::env::temp_dir();
    output::character_directory(&path)
        .ok_or_else(|| error::builtin(IDENTITY, &TEMPDIR_ERROR_UNAVAILABLE))
}

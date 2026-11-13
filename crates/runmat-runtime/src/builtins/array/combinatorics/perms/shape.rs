use runmat_builtins::PERMS_ERROR_INVALID_INPUT;

use crate::BuiltinResult;

use super::error;

pub(super) fn vector_len(shape: &[usize]) -> BuiltinResult<usize> {
    match shape {
        [] => Err(error::from_descriptor(&PERMS_ERROR_INVALID_INPUT)),
        [n] => Ok(*n),
        [0, 0] => Ok(0),
        [rows, columns] if *rows == 1 || *columns == 1 => Ok(rows.saturating_mul(*columns)),
        _ => Err(error::from_descriptor(&PERMS_ERROR_INVALID_INPUT)),
    }
}

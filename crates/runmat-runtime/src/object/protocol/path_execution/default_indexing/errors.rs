use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub(super) fn invalid_selector() -> RuntimeError {
    semantic_error(
        "InvalidObjectSubscriptPath",
        "subscript kind and selector disagree",
    )
}

pub(super) fn invalid_path() -> RuntimeError {
    semantic_error(
        "InvalidObjectSubscriptPath",
        "object assignment path is empty",
    )
}

pub(super) fn invalid_base() -> RuntimeError {
    semantic_error(
        "InvalidObjectSubscriptBase",
        "value does not support this subscript",
    )
}

pub(super) fn single_value_error() -> RuntimeError {
    semantic_error(
        "CommaSeparatedListRequiresSingleValue",
        "chained indexing requires one value",
    )
}

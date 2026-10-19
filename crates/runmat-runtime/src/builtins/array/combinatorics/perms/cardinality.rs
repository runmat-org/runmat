use runmat_builtins::PERMS_ERROR_TOO_LARGE;

use crate::BuiltinResult;

use super::super::enumeration::{self, EnumerationError, MaterializationLimit};
use super::error;

const MAX_INPUT_LEN: usize = 10;
const MAX_OUTPUT_ELEMENTS: usize = 50_000_000;

pub(super) fn output_rows(elements: usize) -> BuiltinResult<usize> {
    enumeration::permutation_rows(
        elements,
        MaterializationLimit {
            max_input_elements: Some(MAX_INPUT_LEN),
            max_output_elements: MAX_OUTPUT_ELEMENTS,
        },
    )
    .map_err(|failure| match failure {
        EnumerationError::CardinalityOverflow => error::with_message(
            &PERMS_ERROR_TOO_LARGE,
            "perms: output element count overflows",
        ),
        EnumerationError::ElementLimitExceeded => error::with_message(
            &PERMS_ERROR_TOO_LARGE,
            format!("perms: input or output exceeds the supported materialization limit ({MAX_INPUT_LEN} input elements, {MAX_OUTPUT_ELEMENTS} output elements)"),
        ),
        EnumerationError::SequenceInvariant => unreachable!("cardinality calculation cannot enumerate"),
    })
}

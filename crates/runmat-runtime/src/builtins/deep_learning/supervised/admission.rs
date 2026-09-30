//! Public arity and cancellable supervised-loop admission.

use crate::BuiltinResult;

use super::deep_learning_error;

pub(super) fn admit_training_output_count(function: &'static str) -> BuiltinResult<()> {
    if let Some(requested) = crate::output_count::current_output_count() {
        if requested > 2 {
            return Err(deep_learning_error(
                function,
                format!("{function}: requested {requested} outputs, but at most 2 are supported"),
            ));
        }
    }
    Ok(())
}

pub(super) fn admit_training_observations(
    observations: Option<usize>,
    function: &'static str,
) -> BuiltinResult<()> {
    if observations == Some(0) {
        return Err(deep_learning_error(
            function,
            format!("{function}: X must contain at least one observation"),
        ));
    }
    Ok(())
}

pub(super) fn admit_training_cancellation(function: &'static str) -> BuiltinResult<()> {
    if crate::interrupt::is_cancelled() {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:Cancelled",
            format!("{function}: training was cancelled"),
        ));
    }
    Ok(())
}

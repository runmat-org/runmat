//! Public arity and cancellable supervised-loop admission.

use crate::BuiltinResult;

use super::deep_learning_error;

pub(super) fn admit_training_output_count(
    function: &'static str,
    requested_outputs: Option<usize>,
) -> BuiltinResult<()> {
    if let Some(requested) = requested_outputs {
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

pub(super) fn admit_training_cancellation(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &'static str,
) -> BuiltinResult<()> {
    if runtime.map_or_else(crate::interrupt::is_cancelled, |runtime| {
        runtime.is_cancelled()
    }) {
        return Err(crate::runtime_error::semantic_error(
            "RunMat:Cancelled",
            format!("{function}: training was cancelled"),
        ));
    }
    Ok(())
}

//! Leaf callback execution and output-count validation.
use super::*;

pub(super) async fn dlupdate_leaf(
    function: &Value,
    values: &[Value],
    requested_outputs: usize,
) -> BuiltinResult<Vec<Value>> {
    let result = crate::call_feval_async_with_outputs(function.clone(), values, requested_outputs)
        .await
        .map_err(|err| {
            crate::build_runtime_error(format!(
                "dlupdate: function-handle evaluation failed: {err}"
            ))
            .with_builtin("dlupdate")
            .with_identifier("RunMat:deepLearning:InvalidInput")
            .with_source(err)
            .build()
        })?;
    normalize_dlupdate_outputs(result, requested_outputs)
}

fn normalize_dlupdate_outputs(value: Value, requested_outputs: usize) -> BuiltinResult<Vec<Value>> {
    if requested_outputs == 0 {
        return Ok(Vec::new());
    }
    match value {
        Value::OutputList(values) if values.len() == requested_outputs => Ok(values),
        Value::OutputList(values) => Err(deep_learning_error(
            "dlupdate",
            format!(
                "dlupdate: function returned {} outputs but {} were requested",
                values.len(),
                requested_outputs
            ),
        )),
        other if requested_outputs == 1 => Ok(vec![other]),
        _ => Err(deep_learning_error(
            "dlupdate",
            "dlupdate: function did not return the requested number of outputs",
        )),
    }
}

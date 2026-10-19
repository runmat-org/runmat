//! Leaf callback execution and output-count validation.
use super::*;

pub(super) async fn dlupdate_leaf(
    runtime: Option<&crate::context::RuntimeContext>,
    function: &Value,
    values: &[Value],
    requested_outputs: usize,
) -> BuiltinResult<RootedValues> {
    let call = match runtime {
        Some(runtime) => {
            crate::call_feval_async_with_outputs_in_context(
                runtime,
                function.clone(),
                values,
                requested_outputs,
            )
            .await
        }
        None => {
            crate::call_feval_async_with_outputs(function.clone(), values, requested_outputs).await
        }
    };
    let result = call.map_err(|err| {
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

fn normalize_dlupdate_outputs(
    value: RootedValue,
    requested_outputs: usize,
) -> BuiltinResult<RootedValues> {
    value.with_owned(|value| {
        let values = normalize_raw_outputs(value, requested_outputs)?;
        RootedValues::from_output_storage(values).map_err(|error| error.into_runtime_error())
    })
}

fn normalize_raw_outputs(
    value: Value,
    requested_outputs: usize,
) -> BuiltinResult<runmat_value::OutputListStorage> {
    if requested_outputs == 0 {
        return Ok(runmat_value::OutputListStorage::empty());
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
        other if requested_outputs == 1 => {
            Ok(runmat_value::OutputListStorage::unbounded(vec![other]))
        }
        _ => Err(deep_learning_error(
            "dlupdate",
            "dlupdate: function did not return the requested number of outputs",
        )),
    }
}

//! Generic model prediction dispatcher.

use runmat_builtins::{PREDICT_INTEGER_CONTROLS_EXTENSION, PREDICT_INTEGER_PREDICTORS_EXTENSION};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{gather_if_needed_async, BuiltinResult};

use super::classification_linear::predict_classification_linear_object;
use super::classification_tree::predict_classification_tree_object;
use super::linear_model::{predict_invalid, predict_linear_model_dispatch};

#[runtime_builtin(
    name = "predict",
    binding_variant = "default",
    context = true,
    requested_outputs = true,
    builtin_path = "crate::builtins::stats::ml::predict"
)]
async fn predict_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    model: Value,
    xnew: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    predict_in_runtime(runtime.as_ref(), requested_outputs, model, xnew, rest).await
}

#[cfg(test)]
pub(crate) async fn predict_builtin(
    model: Value,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    predict_in_runtime(
        None,
        crate::output_count::current_output_count(),
        model,
        xnew,
        crate::execution::RootedValues::new(rest)
            .map_err(|error| error.into_runtime_error())?
            .into(),
    )
    .await
}

async fn predict_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    model: Value,
    xnew: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    let output = match model {
        Value::Object(object)
            if crate::builtins::deep_learning::model::is_deep_learning_network_object(&object) =>
        {
            crate::builtins::deep_learning::model::predict_deep_learning_object_in_runtime(
                runtime,
                requested_outputs,
                object,
                xnew,
                rest,
            )
            .await?
        }
        model => predict_statistical(runtime, model, xnew, rest).await?,
    };
    match requested_outputs {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(1) => Ok(Value::output_list(vec![output[0].clone()])),
        Some(out_count) if out_count > output.len() => {
            Err(predict_invalid("predict: too many output arguments"))
        }
        Some(out_count) => Ok(crate::output_count::output_list_with_padding(
            out_count, output,
        )),
        None => Ok(output
            .into_iter()
            .next()
            .expect("predict dispatch always returns at least one output")),
    }
}

async fn predict_statistical(
    runtime: Option<&crate::context::RuntimeContext>,
    model: Value,
    xnew: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Vec<Value>> {
    let statistical_model = matches!(
        &model,
        Value::Object(object)
            if object
                .class_name()
                .is(runmat_types::standard::LINEAR_MODEL)
                || object.class_name().is(runmat_types::standard::CLASSIFICATION_TREE)
                || object.class_name().is(runmat_types::standard::CLASSIFICATION_LINEAR)
    );
    if statistical_model {
        crate::builtins::common::validation::reject_typed_complex_integer(&xnew, "predict")?;
        for control in &rest {
            crate::builtins::common::validation::reject_typed_complex_integer(control, "predict")?;
        }
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            &xnew,
            &PREDICT_INTEGER_PREDICTORS_EXTENSION,
            "predict",
            "predictor",
        )
        .await?;
        for control in &rest {
            crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
                control,
                &PREDICT_INTEGER_CONTROLS_EXTENSION,
                "predict",
                "control",
            )
            .await?;
        }
    }
    let model = gather(model)
        .await
        .map_err(|err| predict_invalid(err.message))?;
    let xnew = gather(xnew)
        .await
        .map_err(|err| predict_invalid(err.message))?;
    let rest = gather_all(rest, runtime)
        .await
        .map_err(|err| predict_invalid(err.message))?;
    match model {
        Value::Object(object)
            if object
                .class_name()
                .is(runmat_types::standard::CLASSIFICATION_TREE) =>
        {
            predict_classification_tree_object(object, xnew, rest.view())
        }
        Value::Object(object)
            if object
                .class_name()
                .is(runmat_types::standard::CLASSIFICATION_LINEAR) =>
        {
            predict_classification_linear_object(object, xnew, rest.view())
        }
        other => predict_linear_model_dispatch(other, xnew, rest.view()),
    }
}

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await
}

async fn gather_all(
    values: crate::call::BuiltinArguments,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<crate::call::BuiltinArguments> {
    values
        .try_map_preserving_roots_async(values.len(), runtime, |index| {
            gather(values[index].clone())
        })
        .await
}

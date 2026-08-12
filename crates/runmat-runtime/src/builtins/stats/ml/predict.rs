//! Generic model prediction dispatcher.

use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::{gather_if_needed_async, BuiltinResult};

use super::classification_linear::{
    predict_classification_linear_object, CLASSIFICATION_LINEAR_CLASS,
};
use super::classification_tree::{predict_classification_tree_object, CLASSIFICATION_TREE_CLASS};
use super::linear_model::{predict_invalid, predict_linear_model_dispatch, predict_type};

#[runtime_builtin(
    name = "predict",
    category = "stats/ml",
    summary = "Predict responses from a fitted model.",
    keywords = "predict,fitlm,linear model,classification,deep learning,prediction",
    type_resolver(predict_type),
    descriptor(crate::builtins::stats::ml::linear_model::PREDICT_DESCRIPTOR),
    builtin_path = "crate::builtins::stats::ml::predict"
)]
pub(crate) async fn predict_builtin(
    model: Value,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let model = gather(model)
        .await
        .map_err(|err| predict_invalid(err.message))?;
    let xnew = gather(xnew)
        .await
        .map_err(|err| predict_invalid(err.message))?;
    let rest = gather_all(rest)
        .await
        .map_err(|err| predict_invalid(err.message))?;
    let output = match model {
        Value::Object(object)
            if crate::builtins::deep_learning::model::is_deep_learning_network_object(&object) =>
        {
            crate::builtins::deep_learning::model::predict_deep_learning_object(object, xnew, rest)?
        }
        Value::Object(object) if object.class_name == CLASSIFICATION_TREE_CLASS => {
            predict_classification_tree_object(object, xnew, rest)?
        }
        Value::Object(object) if object.class_name == CLASSIFICATION_LINEAR_CLASS => {
            predict_classification_linear_object(object, xnew, rest)?
        }
        other => predict_linear_model_dispatch(other, xnew, rest)?,
    };
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![output[0].clone()])),
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

async fn gather(value: Value) -> BuiltinResult<Value> {
    gather_if_needed_async(&value).await
}

async fn gather_all(values: Vec<Value>) -> BuiltinResult<Vec<Value>> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        out.push(gather(value).await?);
    }
    Ok(out)
}

//! Prediction-specific input presentation and public option admission.
//!
//! Numerical execution, parameter admission, placement, and state handling
//! belong to the shared network executor. Output wrappers, State publication,
//! and tape recording belong to the common network output presenter.

use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use runmat_builtins::{DeepLearningAxisLabel, DeepLearningAxisLayout};
use runmat_value::{ObjectInstance, Value};

use super::execution::{
    execute_network, NetworkCaller, NetworkExecutionRequest, NetworkOutputLayout,
};
use super::{
    deep_learning_error, object, output, scalar_text, BuiltinResult, DLARRAY_CLASS, DLNETWORK_CLASS,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObservationsIn {
    Rows,
    Columns,
}

#[cfg(test)]
pub(crate) async fn predict_deep_learning_object(
    network: ObjectInstance,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    let rest = crate::call::BuiltinArguments::from_borrowed(&rest, None)?;
    predict_deep_learning_object_in_runtime(
        None,
        crate::output_count::current_output_count(),
        network,
        xnew,
        rest,
    )
    .await
}

pub(in crate::builtins) async fn predict_deep_learning_object_in_runtime(
    runtime: Option<&crate::context::RuntimeContext>,
    requested_outputs: Option<usize>,
    network: ObjectInstance,
    xnew: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Vec<Value>> {
    let requested_state = requested_outputs == Some(2) && network.class_name().is(DLNETWORK_CLASS);
    if requested_outputs.is_some_and(|count| {
        count
            > if network.class_name().is(DLNETWORK_CLASS) {
                2
            } else {
                1
            }
    }) {
        return Err(deep_learning_error(
            "predict",
            "predict: too many output arguments",
        ));
    }
    let observations = parse_observations_in(rest.view())?;
    let formatted = matches!(&xnew, Value::Object(value)
        if value.class_name().is(DLARRAY_CLASS)
            && matches!(value.properties().get("Format"), Some(Value::String(runmat_value::StringElement::Text(format))) if !format.is_empty()));
    let unformatted_dlarray = matches!(&xnew, Value::Object(value)
        if value.class_name().is(DLARRAY_CLASS) && !formatted);
    let input = prepare_input(xnew, observations)?;
    let outcome = execute_network(NetworkExecutionRequest {
        runtime,
        network: &network,
        input: &input,
        caller: NetworkCaller::Predict,
        requested_outputs,
        output_layout: if formatted {
            NetworkOutputLayout::Preserve
        } else if unformatted_dlarray {
            NetworkOutputLayout::MatchInputRank
        } else {
            NetworkOutputLayout::ObservationMajor
        },
    })
    .await?;
    output::present(
        &network,
        &input,
        outcome,
        if formatted {
            output::OutputPresentation::PredictFormatted
        } else if unformatted_dlarray {
            output::OutputPresentation::PredictUnformattedArray
        } else {
            output::OutputPresentation::PredictRaw
        },
        requested_state,
    )
}

fn prepare_input(input: Value, observations: ObservationsIn) -> BuiltinResult<Value> {
    let column_layout = DeepLearningAxisLayout::from_axes(&[
        DeepLearningAxisLabel::Channel,
        DeepLearningAxisLabel::Batch,
    ])
    .map_err(|error| deep_learning_error("predict", format!("predict: {error}")))?;
    let column_format = column_layout.labels();
    match input {
        Value::Object(mut value) if value.class_name().is(DLARRAY_CLASS) => {
            let data = value.properties().get("Data").ok_or_else(|| {
                deep_learning_error("predict", "predict: dlarray is missing Data")
            })?;
            let format = match value.properties().get("Format") {
                Some(Value::String(runmat_value::StringElement::Text(text))) => text.as_str(),
                None => "",
                Some(_) => {
                    return Err(deep_learning_error(
                        "predict",
                        "predict: dlarray Format must be text",
                    ))
                }
            };
            let labels = match value.properties().get("Labels") {
                Some(Value::String(runmat_value::StringElement::Text(text))) => text.as_str(),
                None => format,
                Some(_) => {
                    return Err(deep_learning_error(
                        "predict",
                        "predict: dlarray Labels must be text",
                    ))
                }
            };
            if labels != format {
                return Err(deep_learning_error(
                    "predict",
                    "predict: dlarray Format and Labels disagree",
                ));
            }
            let shape = match data {
                Value::Tensor(tensor) => Some(tensor.shape()),
                Value::ComplexTensor(tensor) => Some(tensor.shape()),
                Value::GpuTensor(handle) => Some(handle.shape()),
                _ => None,
            };
            if !format.is_empty() {
                if let Some(shape) = shape {
                    let layout =
                        DeepLearningAxisLayout::parse(format, shape.len()).map_err(|error| {
                            deep_learning_error(
                                "predict",
                                format!("predict: invalid dlarray axes: {error}"),
                            )
                        })?;
                    if observations == ObservationsIn::Columns && layout != column_layout {
                        return Err(deep_learning_error(
                            "predict",
                            "predict: ObservationsIn='columns' conflicts with formatted dlarray axes",
                        ));
                    }
                }
            } else if observations == ObservationsIn::Columns {
                value.replace_dl_property("Format", Value::string(column_format.clone()))?;
                value.replace_dl_property("Labels", Value::string(column_format))?;
            }
            Ok(Value::Object(value))
        }
        data => {
            let format = if observations == ObservationsIn::Columns {
                column_format
            } else {
                String::new()
            };
            Ok(object(
                "dlarray",
                vec![
                    ("Data", data),
                    ("Format", Value::string(format.clone())),
                    ("Labels", Value::string(format)),
                ],
            )?)
        }
    }
}

fn parse_observations_in(
    values: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<ObservationsIn> {
    if !values.len().is_multiple_of(2) {
        return Err(deep_learning_error(
            "predict",
            "predict: name-value options must be paired",
        ));
    }
    let mut observations = ObservationsIn::Rows;
    for pair in values.chunks_exact(2) {
        let name = scalar_text(&pair[0], "predict")?.to_ascii_lowercase();
        if name != "observationsin" {
            return Err(deep_learning_error(
                "predict",
                format!("predict: unknown option '{name}'"),
            ));
        }
        let value = scalar_text(&pair[1], "predict")?;
        observations = match value.to_ascii_lowercase().as_str() {
            "rows" => ObservationsIn::Rows,
            "columns" => ObservationsIn::Columns,
            other => {
                return Err(deep_learning_error(
                    "predict",
                    format!("predict: unsupported ObservationsIn '{other}'"),
                ))
            }
        };
    }
    Ok(observations)
}

#[cfg(test)]
mod tests;

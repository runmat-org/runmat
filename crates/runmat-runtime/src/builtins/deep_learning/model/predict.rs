//! Prediction-specific input presentation and output adaptation.
//!
//! Numerical execution, parameter admission, placement, and state handling
//! belong to the shared network executor. This module only interprets the
//! public `predict` observations option and returns its raw score array.

use runmat_value::{ObjectInstance, Value};

use super::execution::{
    execute_network, NetworkCaller, NetworkExecutionRequest, NetworkOutputLayout,
};
use super::{deep_learning_error, object, scalar_text, BuiltinResult, DLARRAY_CLASS};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObservationsIn {
    Rows,
    Columns,
}

impl ObservationsIn {
    fn format(self) -> &'static str {
        match self {
            Self::Rows => "BC",
            Self::Columns => "CB",
        }
    }
}

pub(crate) async fn predict_deep_learning_object(
    network: ObjectInstance,
    xnew: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Vec<Value>> {
    let observations = parse_observations_in(&rest)?;
    let data = match xnew {
        Value::Object(mut value) if value.class_name.is(DLARRAY_CLASS) => value
            .properties
            .remove("Data")
            .ok_or_else(|| deep_learning_error("predict", "predict: dlarray is missing Data"))?,
        other => other,
    };
    let format = observations.format();
    let input = object(
        "dlarray",
        vec![
            ("Data", data),
            ("Format", Value::String(format.into())),
            ("Labels", Value::String(format.into())),
        ],
    );
    let outcome = execute_network(NetworkExecutionRequest {
        network: &network,
        input: &input,
        caller: NetworkCaller::Predict,
        output_layout: NetworkOutputLayout::ObservationMajor,
    })
    .await?;
    Ok(vec![outcome.data])
}

fn parse_observations_in(values: &[Value]) -> BuiltinResult<ObservationsIn> {
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

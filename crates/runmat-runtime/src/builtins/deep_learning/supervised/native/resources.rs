//! Training-only, effect-free class admission for canonical Learnables/State.
//!
//! Forward execution may apply its documented integer projection at an
//! operation boundary. An optimizer update cannot preserve an integer
//! Learnable's class, so training rejects that payload before initialization,
//! callbacks, or provider transfer rather than discovering it after backward.

use runmat_value::{NumericDType, Value};

use crate::builtins::deep_learning::{deep_learning_error, model};
use crate::BuiltinResult;

fn require_floating_payload(
    value: &Value,
    label: &str,
    function: &'static str,
) -> BuiltinResult<()> {
    let shape = model::parameters::numeric_payload_shape(value, label, function)?;
    if shape.contains(&0) {
        return Ok(());
    }
    if matches!(
        model::parameters::floating_precision(value),
        Some(NumericDType::F32 | NumericDType::F64)
    ) {
        return Ok(());
    }
    Err(deep_learning_error(
        function,
        format!("{function}: filled {label} must be single or double for training"),
    ))
}

pub(super) fn admit(source: &Value, layers: &[Value], function: &'static str) -> BuiltinResult<()> {
    for row in model::parameters::view_layers(layers, function)? {
        if let Some(value) = row.value {
            require_floating_payload(value, row.key.role.name(), function)?;
        }
    }
    let supplied = match source {
        Value::Object(network) if model::is_deep_learning_network_object(network) => {
            model::state::view(network, function)?
        }
        _ => model::state::supplied_initial_values(layers, function)?,
    };
    for row in supplied {
        require_floating_payload(row.value, row.key.role.name(), function)?;
    }
    Ok(())
}

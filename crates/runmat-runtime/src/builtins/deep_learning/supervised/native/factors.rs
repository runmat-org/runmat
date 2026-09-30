//! Typed, effect-free per-learnable training-factor admission.

use std::collections::HashMap;

use runmat_builtins::DeepLearningLayerOption as O;
use runmat_value::{ObjectInstance, Value};

use crate::builtins::deep_learning::{
    deep_learning_error,
    model::parameters::{self, ParameterKey, ParameterRole},
    numeric_values,
};
use crate::BuiltinResult;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

#[derive(Clone, Copy)]
pub(super) struct Factors {
    pub learning: f64,
    pub l2: f64,
}

pub(super) struct FactorPlan {
    values: HashMap<ParameterKey, Factors>,
}

fn options(role: ParameterRole) -> (O, O) {
    use ParameterRole as R;
    match role {
        R::Weights => (O::WeightLearnRateFactor, O::WeightL2Factor),
        R::InputWeights => (O::InputWeightsLearnRateFactor, O::InputWeightsL2Factor),
        R::RecurrentWeights => (
            O::RecurrentWeightsLearnRateFactor,
            O::RecurrentWeightsL2Factor,
        ),
        R::Bias => (O::BiasLearnRateFactor, O::BiasL2Factor),
        R::Scale => (O::ScaleLearnRateFactor, O::ScaleL2Factor),
        R::Offset => (O::OffsetLearnRateFactor, O::OffsetL2Factor),
    }
}

fn scalar(layer: &ObjectInstance, option: O, function: &'static str) -> BuiltinResult<f64> {
    let mut found = None;
    for (name, value) in &layer.properties {
        if O::from_name(name) != Some(option) {
            continue;
        }
        if found.is_some() {
            return Err(fail(
                function,
                format!("{} is specified more than once", option.canonical_name()),
            ));
        }
        let decoded = numeric_values(value, function, option.canonical_name())?;
        if decoded.len() != 1 || !decoded[0].is_finite() || decoded[0] < 0.0 {
            return Err(fail(
                function,
                format!(
                    "{} must be a finite nonnegative scalar",
                    option.canonical_name()
                ),
            ));
        }
        found = Some(decoded[0]);
    }
    Ok(found.unwrap_or(1.0))
}

impl FactorPlan {
    pub(super) fn admit(layers: &[Value], function: &'static str) -> BuiltinResult<Self> {
        let mut values = HashMap::new();
        for row in parameters::view_layers(layers, function)? {
            let Value::Object(layer) = &layers[row.key.layer_index] else {
                unreachable!("canonical parameter view admitted layer")
            };
            let (learning, l2) = options(row.key.role);
            let factors = Factors {
                learning: scalar(layer, learning, function)?,
                l2: scalar(layer, l2, function)?,
            };
            values.insert(row.key, factors);
        }
        Ok(Self { values })
    }

    pub(super) fn get(&self, key: &ParameterKey, function: &'static str) -> BuiltinResult<Factors> {
        self.values
            .get(key)
            .copied()
            .ok_or_else(|| fail(function, "optimizer key has no admitted factor plan"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::deep_learning::layers::{
        feature_input_layer_builtin, fully_connected_layer_builtin,
    };
    use futures::executor::block_on;

    #[test]
    fn factors_bind_to_typed_roles_and_reject_invalid_mutations() {
        let input = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
        let dense = block_on(fully_connected_layer_builtin(
            Value::Num(3.0),
            vec![
                Value::String("WeightLearnRateFactor".into()),
                Value::Num(0.25),
            ],
        ))
        .unwrap();
        let mut layers = vec![input, dense];
        let plan = FactorPlan::admit(&layers, "test").unwrap();
        let rows = parameters::view_layers(&layers, "test").unwrap();
        assert_eq!(plan.get(&rows[0].key, "test").unwrap().learning, 0.25);
        assert_eq!(plan.get(&rows[1].key, "test").unwrap().learning, 1.0);
        let Value::Object(layer) = &mut layers[1] else {
            panic!("dense layer expected")
        };
        layer
            .properties
            .insert("BiasL2Factor".into(), Value::Num(f64::NAN));
        assert!(FactorPlan::admit(&layers, "test").is_err());
    }
}

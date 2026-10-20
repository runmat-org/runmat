//! Typed, effect-free per-learnable training-factor admission.

#[cfg(test)]
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use std::collections::HashMap;

use runmat_builtins::{DeepLearningLayerKind, DeepLearningLayerOption as O};
use runmat_value::{ObjectInstance, Value};

use crate::builtins::deep_learning::{
    deep_learning_error,
    model::{
        execution::host::{numeric::Real, recurrent::RecurrentGateRows},
        parameters::{self, ParameterKey, ParameterRole, ParameterShape},
    },
    numeric_values,
};
use crate::BuiltinResult;

fn fail(function: &'static str, detail: impl std::fmt::Display) -> crate::RuntimeError {
    deep_learning_error(function, format!("{function}: {detail}"))
}

#[derive(Clone, Copy, Debug)]
enum FactorValues {
    Uniform(f64),
    RecurrentGates([f64; 8]),
}

impl FactorValues {
    fn at_slot(self, slot: usize) -> f64 {
        match self {
            Self::Uniform(value) => value,
            Self::RecurrentGates(values) => values[slot],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Factors {
    learning: FactorValues,
    l2: FactorValues,
    gate_rows: Option<RecurrentGateRows>,
}

impl Factors {
    #[cfg(test)]
    pub(super) fn uniform(learning: f64, l2: f64) -> Self {
        Self {
            learning: FactorValues::Uniform(learning),
            l2: FactorValues::Uniform(l2),
            gate_rows: None,
        }
    }

    pub(super) fn native<T: Real>(
        self,
        learn_rate: f64,
        l2_regularization: f64,
        function: &'static str,
    ) -> BuiltinResult<NativeFactors<T>> {
        let mut rates = [T::zero(); 8];
        let mut regularization = [T::zero(); 8];
        let count = self.gate_rows.map_or(1, RecurrentGateRows::factor_count);
        for slot in 0..count {
            rates[slot] = T::from_f64(learn_rate * self.learning.at_slot(slot));
            regularization[slot] = T::from_f64(l2_regularization * self.l2.at_slot(slot));
            if !rates[slot].is_finite()
                || rates[slot] < T::zero()
                || !regularization[slot].is_finite()
                || regularization[slot] < T::zero()
            {
                return Err(fail(
                    function,
                    "effective training factors exceed parameter precision",
                ));
            }
        }
        Ok(NativeFactors {
            rates,
            regularization,
            gate_rows: self.gate_rows,
        })
    }
}

pub(super) struct NativeFactors<T> {
    rates: [T; 8],
    regularization: [T; 8],
    gate_rows: Option<RecurrentGateRows>,
}

impl<T: Copy> NativeFactors<T> {
    pub(super) fn at(&self, index: usize) -> (T, T) {
        let slot = self.gate_rows.map_or(0, |rows| rows.factor_index(index));
        (self.rates[slot], self.regularization[slot])
    }
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

fn decode(
    layer: &ObjectInstance,
    option: O,
    default: f64,
    gate_rows: Option<RecurrentGateRows>,
    gate_count: usize,
    function: &'static str,
) -> BuiltinResult<FactorValues> {
    let mut found = None;
    for (name, value) in layer.properties() {
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
        if decoded.is_empty()
            || decoded
                .iter()
                .any(|factor| !factor.is_finite() || *factor < 0.0)
        {
            return Err(fail(
                function,
                format!(
                    "{} must contain finite nonnegative factors",
                    option.canonical_name()
                ),
            ));
        }
        let factor = if decoded.len() == 1 {
            FactorValues::Uniform(decoded[0])
        } else {
            let valid_shape = matches!(value, Value::Tensor(tensor) if tensor.shape() == [1, gate_count].as_slice());
            if gate_rows.is_none() || decoded.len() != gate_count || !valid_shape {
                return Err(fail(
                    function,
                    format!(
                        "{} must be a finite nonnegative scalar or a 1-by-{gate_count} recurrent gate vector",
                        option.canonical_name()
                    ),
                ));
            }
            let mut values = [0.0; 8];
            values[..gate_count].copy_from_slice(&decoded);
            FactorValues::RecurrentGates(values)
        };
        found = Some(factor);
    }
    Ok(found.unwrap_or(FactorValues::Uniform(default)))
}

fn recurrent_geometry(
    kind: DeepLearningLayerKind,
    role: ParameterRole,
    shape: &ParameterShape,
    function: &'static str,
) -> BuiltinResult<(Option<RecurrentGateRows>, usize)> {
    let Some(gate_count) = kind.recurrent_gate_count() else {
        return Ok((None, 0));
    };
    if !matches!(
        role,
        ParameterRole::InputWeights | ParameterRole::RecurrentWeights | ParameterRole::Bias
    ) {
        return Err(fail(
            function,
            "recurrent factor role differs from parameter schema",
        ));
    }
    let rows = match shape {
        ParameterShape::Matrix { rows, .. } => *rows,
        ParameterShape::Dimensions(dimensions) => match dimensions.as_slice() {
            [Some(rows), Some(1)] => *rows,
            _ => {
                return Err(fail(
                    function,
                    "recurrent bias geometry differs from parameter schema",
                ))
            }
        },
        ParameterShape::Elements(_) => {
            return Err(fail(function, "recurrent factor requires row geometry"));
        }
    };
    let gate_rows = RecurrentGateRows::for_layer(rows, kind)
        .ok_or_else(|| fail(function, "recurrent gate geometry is invalid"))?;
    Ok((Some(gate_rows), gate_count))
}

impl FactorPlan {
    pub(super) fn admit(layers: &[Value], function: &'static str) -> BuiltinResult<Self> {
        let mut values = HashMap::new();
        for row in parameters::view_layers(layers, function)? {
            let Value::Object(layer) = &layers[row.key.layer_index] else {
                unreachable!("canonical parameter view admitted layer")
            };
            let kind = DeepLearningLayerKind::from_identity(layer.class_name())
                .ok_or_else(|| fail(function, "unknown layer in parameter factor plan"))?;
            let (gate_rows, gate_count) =
                recurrent_geometry(kind, row.key.role, &row.expected_shape, function)?;
            let (learning, l2) = options(row.key.role);
            let default_l2 = if row.key.role == ParameterRole::Bias {
                0.0
            } else {
                1.0
            };
            let factors = Factors {
                learning: decode(layer, learning, 1.0, gate_rows, gate_count, function)?,
                l2: decode(layer, l2, default_l2, gate_rows, gate_count, function)?,
                gate_rows,
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
        bilstm_layer_builtin, feature_input_layer_builtin, lstm_layer_builtin,
        sequence_input_layer_builtin,
    };
    use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
    use futures::executor::block_on;
    use runmat_value::Tensor;

    fn option(name: &str, values: Vec<f64>, shape: Vec<usize>) -> (String, Value) {
        (
            name.into(),
            Value::Tensor(Tensor::new(values, shape).unwrap()),
        )
    }

    fn recurrent_layers(bidirectional: bool) -> Vec<Value> {
        let input = block_on(sequence_input_layer_builtin(
            Value::Num(3.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let recurrent = if bidirectional {
            block_on(bilstm_layer_builtin(
                Value::Num(2.0),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .unwrap()
        } else {
            block_on(lstm_layer_builtin(
                Value::Num(2.0),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .unwrap()
        };
        vec![input, recurrent]
    }

    #[test]
    fn factors_bind_to_typed_roles_and_reject_invalid_mutations() {
        let input = block_on(feature_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let dense = block_on(fully_connected_layer_builtin(
            None,
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
        assert_eq!(
            plan.get(&rows[0].key, "test").unwrap().learning.at_slot(0),
            0.25
        );
        assert_eq!(
            plan.get(&rows[1].key, "test").unwrap().learning.at_slot(0),
            1.0
        );
        assert_eq!(plan.get(&rows[0].key, "test").unwrap().l2.at_slot(0), 1.0);
        assert_eq!(plan.get(&rows[1].key, "test").unwrap().l2.at_slot(0), 0.0);
        let Value::Object(layer) = &mut layers[1] else {
            panic!("dense layer expected")
        };
        layer
            .replace_dl_property("BiasL2Factor", Value::Num(f64::NAN))
            .expect("fixture property");
        assert!(FactorPlan::admit(&layers, "test").is_err());
    }

    #[test]
    fn lstm_gate_vectors_follow_canonical_row_order_for_all_parameter_roles() {
        let mut layers = recurrent_layers(false);
        let Value::Object(layer) = &mut layers[1] else {
            panic!("LSTM layer expected")
        };
        let (name, value) = option(
            "InputWeightsLearnRateFactor",
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 4],
        );
        layer
            .replace_dl_property(&name, value)
            .expect("fixture property");
        let (name, value) = option(
            "RecurrentWeightsL2Factor",
            vec![4.0, 3.0, 2.0, 1.0],
            vec![1, 4],
        );
        layer
            .replace_dl_property(&name, value)
            .expect("fixture property");
        let (name, value) = option("BiasLearnRateFactor", vec![0.0, 1.0, 2.0, 3.0], vec![1, 4]);
        layer
            .replace_dl_property(&name, value)
            .expect("fixture property");
        let rows = parameters::view_layers(&layers, "test").unwrap();
        let plan = FactorPlan::admit(&layers, "test").unwrap();
        for row in rows {
            let native = plan
                .get(&row.key, "test")
                .unwrap()
                .native::<f32>(0.1, 0.5, "test")
                .unwrap();
            let expected_rate = match row.key.role {
                ParameterRole::InputWeights => [0.1, 0.2, 0.3, 0.4],
                ParameterRole::RecurrentWeights => [0.1; 4],
                ParameterRole::Bias => [0.0, 0.1, 0.2, 0.3],
                _ => panic!("unexpected recurrent role"),
            };
            let expected_l2 = match row.key.role {
                ParameterRole::RecurrentWeights => [2.0, 1.5, 1.0, 0.5],
                ParameterRole::Bias => [0.0; 4],
                _ => [0.5; 4],
            };
            for gate in 0..4 {
                for unit in 0..2 {
                    let (rate, l2) = native.at(gate * 2 + unit);
                    assert!((rate - expected_rate[gate]).abs() < 1.0e-6);
                    assert!((l2 - expected_l2[gate]).abs() < 1.0e-6);
                }
            }
            let (second_column_rate, _) = native.at(8 + 4);
            assert!((second_column_rate - expected_rate[2]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn bilstm_gate_vectors_include_backward_direction_and_reject_bad_geometry() {
        let mut layers = recurrent_layers(true);
        let Value::Object(layer) = &mut layers[1] else {
            panic!("BiLSTM layer expected")
        };
        let (name, value) = option("BiasL2Factor", (1..=8).map(f64::from).collect(), vec![1, 8]);
        layer
            .replace_dl_property(&name, value)
            .expect("fixture property");
        let rows = parameters::view_layers(&layers, "test").unwrap();
        let bias = rows
            .iter()
            .find(|row| row.key.role == ParameterRole::Bias)
            .unwrap();
        let plan = FactorPlan::admit(&layers, "test").unwrap();
        let native = plan
            .get(&bias.key, "test")
            .unwrap()
            .native::<f64>(0.1, 0.25, "test")
            .unwrap();
        for gate in 0..8 {
            for unit in 0..2 {
                let (rate, l2) = native.at(gate * 2 + unit);
                assert_eq!(rate, 0.1);
                assert_eq!(l2, 0.25 * (gate + 1) as f64);
            }
        }
        let Value::Object(layer) = &mut layers[1] else {
            unreachable!()
        };
        let (name, column) = option("BiasL2Factor", (1..=8).map(f64::from).collect(), vec![8, 1]);
        layer
            .replace_dl_property(&name, column)
            .expect("fixture property");
        assert!(FactorPlan::admit(&layers, "test").is_err());
        let Value::Object(layer) = &mut layers[1] else {
            unreachable!()
        };
        let (name, short) = option("BiasL2Factor", vec![1.0; 4], vec![1, 4]);
        layer
            .replace_dl_property(&name, short)
            .expect("fixture property");
        assert!(FactorPlan::admit(&layers, "test").is_err());
    }

    #[test]
    fn recurrent_constructor_accepts_only_documented_row_vectors() {
        let correct = option("InputWeightsLearnRateFactor", vec![1.0; 4], vec![1, 4]);
        assert!(block_on(lstm_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(correct.0.into()),
                correct.1
            ]),
        ))
        .is_ok());
        let wrong_orientation = option("InputWeightsLearnRateFactor", vec![1.0; 4], vec![4, 1]);
        assert!(block_on(lstm_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(wrong_orientation.0.into()),
                wrong_orientation.1
            ]),
        ))
        .is_err());
        let wrong_gate_count = option("BiasL2Factor", vec![1.0; 4], vec![1, 4]);
        assert!(block_on(bilstm_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String(wrong_gate_count.0.into()),
                wrong_gate_count.1
            ]),
        ))
        .is_err());
    }
}

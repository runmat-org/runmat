//! Recurrent state row identity and admitted channel geometry.
use super::*;
use runmat_builtins::DeepLearningLayerKind;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(in crate::builtins::deep_learning) enum StateRole {
    HiddenState,
    CellState,
}

impl StateRole {
    pub(in crate::builtins::deep_learning) fn name(self) -> &'static str {
        match self {
            Self::HiddenState => "HiddenState",
            Self::CellState => "CellState",
        }
    }

    pub(in crate::builtins::deep_learning) fn parse(name: &str) -> Option<Self> {
        match name {
            "HiddenState" => Some(Self::HiddenState),
            "CellState" => Some(Self::CellState),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(in crate::builtins::deep_learning) struct StateKey {
    pub(in crate::builtins::deep_learning) layer_index: usize,
    pub(in crate::builtins::deep_learning) layer_name: String,
    pub(in crate::builtins::deep_learning) role: StateRole,
    pub(in crate::builtins::deep_learning) channels: usize,
}

pub(super) fn keys(layers: &[Value], function: &'static str) -> BuiltinResult<Vec<StateKey>> {
    let names = layer_names(layers, function)?;
    if names.iter().collect::<HashSet<_>>().len() != names.len() {
        return Err(deep_learning_error(
            function,
            format!("{function}: layer names must be unique"),
        ));
    }
    let mut keys = Vec::new();
    for (layer_index, (layer, name)) in layers.iter().zip(names).enumerate() {
        let Value::Object(layer) = layer else {
            return Err(deep_learning_error(
                function,
                format!("{function}: network Layers must contain layer objects"),
            ));
        };
        let multiplier = match DeepLearningLayerKind::from_identity(layer.class_name()) {
            Some(DeepLearningLayerKind::Lstm) => 1,
            Some(DeepLearningLayerKind::Bilstm) => 2,
            Some(
                DeepLearningLayerKind::FeatureInput
                | DeepLearningLayerKind::SequenceInput
                | DeepLearningLayerKind::FullyConnected
                | DeepLearningLayerKind::Convolution1D
                | DeepLearningLayerKind::LayerNormalization
                | DeepLearningLayerKind::GlobalAveragePooling1D
                | DeepLearningLayerKind::Relu
                | DeepLearningLayerKind::Elu
                | DeepLearningLayerKind::Softmax
                | DeepLearningLayerKind::ClassificationOutput
                | DeepLearningLayerKind::RegressionOutput,
            ) => continue,
            None => {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: unsupported layer class '{}'",
                        layer.class_name()
                    ),
                ));
            }
        };
        let has_state_inputs = layer
            .properties()
            .get("HasStateInputs")
            .map(|value| logical_scalar(value, function, "HasStateInputs"))
            .transpose()?
            .unwrap_or(false);
        if has_state_inputs {
            for role in [StateRole::HiddenState, StateRole::CellState] {
                if let Some(value) = layer.properties().get(role.name()) {
                    let shape = super::super::parameters::numeric_payload_shape(
                        value,
                        role.name(),
                        function,
                    )?;
                    if !shape.contains(&0) {
                        return Err(deep_learning_error(
                            function,
                            format!(
                                "{function}: {} must be empty when HasStateInputs is true",
                                role.name()
                            ),
                        ));
                    }
                }
            }
            continue;
        }
        let units = positive_property_usize(layer, "NumHiddenUnits", function)?;
        let channels = units.checked_mul(multiplier).ok_or_else(|| {
            deep_learning_error(
                function,
                format!("{function}: recurrent state shape overflows"),
            )
        })?;
        for role in [StateRole::HiddenState, StateRole::CellState] {
            keys.push(StateKey {
                layer_index,
                layer_name: name.clone(),
                role,
                channels,
            });
        }
    }
    Ok(keys)
}

pub(super) fn validate_value(
    value: &Value,
    key: &StateKey,
    function: &'static str,
) -> BuiltinResult<()> {
    let shape = super::super::parameters::numeric_payload_shape(value, key.role.name(), function)?;
    if shape.contains(&0) {
        return Ok(());
    }
    let valid = match shape {
        [channels, batch] => *channels == key.channels && *batch > 0,
        [channels] => *channels == key.channels,
        _ => false,
    };
    if !valid {
        return Err(deep_learning_error(
            function,
            format!(
                "{function}: {} of layer '{}' has shape {shape:?}, expected {}-by-batch",
                key.role.name(),
                key.layer_name,
                key.channels
            ),
        ));
    }
    Ok(())
}

pub(super) fn batch_size(value: &Value, function: &'static str) -> BuiltinResult<Option<usize>> {
    let shape = super::super::parameters::numeric_payload_shape(value, "State", function)?;
    if shape.contains(&0) {
        return Ok(None);
    }
    Ok(Some(if shape.len() == 1 { 1 } else { shape[1] }))
}

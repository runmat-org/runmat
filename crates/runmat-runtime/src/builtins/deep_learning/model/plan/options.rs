//! Borrowed operational requirements for executable network routes.
//!
//! Layer construction may retain a recognized control even when a particular
//! execution route does not consume it. A route must explicitly admit each
//! nondefault control before it prepares data, invokes a callback, or reaches
//! a provider. Inspection here neither clones payloads nor runs callbacks.

use std::collections::BTreeSet;

use runmat_builtins::{
    DeepLearningLayerKind, DeepLearningLayerOption, DeepLearningLayerStructuralProperty,
    LayerOptionDefault, LayerOptionExecution,
};
use runmat_value::Value;

use super::super::super::{deep_learning_error, logical_scalar, scalar_text};
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::deep_learning) enum OptionRequirementStatus {
    Metadata,
    DeclaredDefault,
    NeedsConsumer,
}

/// Only the admission decision is retained. Execution decodes consumed
/// values from the canonical layer object after this pre-effect check.
pub(in crate::builtins::deep_learning) struct LayerOptionRequirement {
    pub layer_index: usize,
    pub kind: DeepLearningLayerKind,
    pub option: DeepLearningLayerOption,
    pub status: OptionRequirementStatus,
}

pub(in crate::builtins::deep_learning) struct LayerOperationalPlan {
    requirements: Vec<LayerOptionRequirement>,
}

impl LayerOperationalPlan {
    pub fn inspect(layers: &[Value], function: &'static str) -> BuiltinResult<Self> {
        let mut requirements = Vec::new();
        for (layer_index, value) in layers.iter().enumerate() {
            let Value::Object(layer) = value else {
                return Err(deep_learning_error(
                    function,
                    format!("{function}: network layers must be layer objects"),
                ));
            };
            let kind =
                DeepLearningLayerKind::from_identity(layer.class_name()).ok_or_else(|| {
                    deep_learning_error(
                        function,
                        format!(
                            "{function}: unsupported layer type '{}'",
                            layer.class_name()
                        ),
                    )
                })?;
            let mut seen = BTreeSet::new();
            for (name, value) in layer.properties() {
                if DeepLearningLayerStructuralProperty::from_name(name)
                    .is_some_and(|property| property.belongs_to(kind))
                {
                    continue;
                }
                let option = kind.option(name).ok_or_else(|| {
                    deep_learning_error(
                        function,
                        format!(
                            "{function}: layer {} has unsupported property '{name}'",
                            layer_index + 1
                        ),
                    )
                })?;
                if !seen.insert(option) {
                    return Err(deep_learning_error(
                        function,
                        format!(
                            "{function}: layer {} specifies {} more than once",
                            layer_index + 1,
                            option.canonical_name()
                        ),
                    ));
                }
                let status = if matches!(option.execution_for(kind), LayerOptionExecution::Metadata)
                {
                    OptionRequirementStatus::Metadata
                } else if option
                    .default_for(kind)
                    .is_some_and(|default| equals_declared_default(value, default))
                {
                    OptionRequirementStatus::DeclaredDefault
                } else {
                    OptionRequirementStatus::NeedsConsumer
                };
                requirements.push(LayerOptionRequirement {
                    layer_index,
                    kind,
                    option,
                    status,
                });
            }
        }
        Ok(Self { requirements })
    }

    #[cfg(test)]
    pub fn requirements(&self) -> &[LayerOptionRequirement] {
        &self.requirements
    }

    /// A caller supplies its *actual* typed consumers, including route-specific
    /// handling of initializers and callbacks. Catalog-wide "executable"
    /// status alone is insufficient for GPU, host, training, or export parity.
    pub fn require_consumed(
        &self,
        function: &'static str,
        mut consumed: impl FnMut(&LayerOptionRequirement) -> bool,
    ) -> BuiltinResult<()> {
        for requirement in &self.requirements {
            if requirement.status == OptionRequirementStatus::NeedsConsumer
                && !consumed(requirement)
            {
                return Err(deep_learning_error(
                    function,
                    format!(
                        "{function}: layer {} option {} is not supported by this execution route",
                        requirement.layer_index + 1,
                        requirement.option.canonical_name()
                    ),
                ));
            }
        }
        Ok(())
    }
}

fn equals_declared_default(value: &Value, default: LayerOptionDefault) -> bool {
    match default {
        LayerOptionDefault::Text(expected) => {
            scalar_text(value, "dlnetwork").is_ok_and(|text| text.eq_ignore_ascii_case(expected))
        }
        LayerOptionDefault::Number(expected) => {
            if let Some(integer) = crate::builtins::common::tensor::scalar_integer_value(value) {
                integer
                    .try_to_i64()
                    .is_some_and(|number| (number as f64) == expected)
            } else {
                match value {
                    Value::Num(number) => *number == expected,
                    Value::Tensor(tensor)
                        if crate::builtins::common::tensor::is_scalar_tensor(tensor) =>
                    {
                        crate::builtins::common::tensor::tensor_value_f64(tensor, 0) == expected
                    }
                    _ => false,
                }
            }
        }
        LayerOptionDefault::Bool(expected) => logical_scalar(value, "dlnetwork", "layer option")
            .is_ok_and(|actual| actual == expected),
    }
}

#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet};

use super::*;
use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_value::{CellArray, StructValue};

pub(in crate::builtins::fea) const FROZEN_GENERATORS_PROPERTY: &str = "frozen_generators";
pub(in crate::builtins::fea) const FROZEN_STUDY_GENERATORS_PROPERTY: &str =
    "frozen_study_generators";

#[derive(Debug, Clone, Default, PartialEq)]
pub(in crate::builtins::fea) struct FrozenGeneratorInventory {
    bindings: Vec<FrozenSignalGenerator>,
}

impl FrozenGeneratorInventory {
    pub(in crate::builtins::fea) fn new(
        builtin: &'static str,
        signals: &[PhysicalSignal],
        bindings: Vec<FrozenSignalGenerator>,
    ) -> BuiltinResult<Self> {
        let inventory = Self { bindings };
        inventory.validate_against(builtin, signals)?;
        Ok(inventory)
    }

    pub(in crate::builtins::fea) fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    pub(in crate::builtins::fea) fn bindings(&self) -> &[FrozenSignalGenerator] {
        &self.bindings
    }

    pub(in crate::builtins::fea) fn attach_to(
        &self,
        object: &mut ObjectInstance,
    ) -> BuiltinResult<()> {
        if self.is_empty() {
            object.properties.remove(FROZEN_GENERATORS_PROPERTY);
            return Ok(());
        }
        object.properties.insert(
            FROZEN_GENERATORS_PROPERTY.into(),
            encode_bindings(&self.bindings)?,
        );
        Ok(())
    }

    pub(in crate::builtins::fea) fn from_property_value(
        builtin: &'static str,
        value: Option<&Value>,
        signals: &[PhysicalSignal],
    ) -> BuiltinResult<Self> {
        let bindings = value
            .map(|value| decode_bindings(builtin, value))
            .transpose()?
            .unwrap_or_default();
        Self::new(builtin, signals, bindings)
    }

    fn validate_against(
        &self,
        builtin: &'static str,
        signals: &[PhysicalSignal],
    ) -> BuiltinResult<()> {
        let mut by_id = BTreeMap::new();
        for signal in signals {
            if by_id.insert(signal.signal_id(), signal).is_some() {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!(
                        "signal `{}` is declared more than once",
                        signal.signal_id().0
                    ),
                ));
            }
        }
        let mut bound = BTreeSet::new();
        for binding in &self.bindings {
            if !bound.insert(&binding.signal_id) {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!(
                        "signal `{}` has more than one frozen callable",
                        binding.signal_id.0
                    ),
                ));
            }
            let signal = by_id.get(&binding.signal_id).ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!(
                        "frozen callable references missing signal `{}`",
                        binding.signal_id.0
                    ),
                )
            })?;
            let generator = signal.generator().ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!(
                        "frozen callable references non-generated signal `{}`",
                        binding.signal_id.0
                    ),
                )
            })?;
            binding
                .callable
                .validate_identity()
                .map_err(|error| builtin_error(builtin, &ERROR_INPUT, error.to_string()))?;
            if !binding.callable.owner_identity.starts_with("bound:")
                || binding.callable.qualified_name != generator.callable.display_name()
            {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!(
                        "frozen callable does not match generated signal `{}`",
                        binding.signal_id.0
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(in crate::builtins::fea) struct FrozenStudyGeneratorInventory {
    studies: Vec<(String, FrozenGeneratorInventory)>,
}

impl FrozenStudyGeneratorInventory {
    pub(in crate::builtins::fea) fn new(
        builtin: &'static str,
        studies: &[(crate::analysis::AnalysisStudySpec, FrozenGeneratorInventory)],
    ) -> BuiltinResult<Self> {
        let mut seen = BTreeSet::new();
        let mut retained = Vec::new();
        for (study, inventory) in studies {
            if !seen.insert(study.study_id.as_str()) {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("study `{}` is declared more than once", study.study_id),
                ));
            }
            let signals = study
                .model
                .as_ref()
                .map(|model| model.signals.as_slice())
                .unwrap_or_default();
            inventory.validate_against(builtin, signals)?;
            if !inventory.is_empty() {
                retained.push((study.study_id.clone(), inventory.clone()));
            }
        }
        Ok(Self { studies: retained })
    }

    pub(in crate::builtins::fea) fn attach_to(
        &self,
        object: &mut ObjectInstance,
    ) -> BuiltinResult<()> {
        if self.studies.is_empty() {
            object.properties.remove(FROZEN_STUDY_GENERATORS_PROPERTY);
            return Ok(());
        }
        let mut rows = Vec::with_capacity(self.studies.len());
        for (study_id, inventory) in &self.studies {
            let mut row = StructValue::new();
            row.insert("study_id", Value::String(study_id.clone()));
            row.insert("bindings", encode_bindings(inventory.bindings())?);
            rows.push(Value::Struct(row));
        }
        object.properties.insert(
            FROZEN_STUDY_GENERATORS_PROPERTY.into(),
            Value::Cell(
                CellArray::new(rows, 1, self.studies.len())
                    .map_err(|error| builtin_error(SWEEP_NAME, &ERROR_INTERNAL, error))?,
            ),
        );
        Ok(())
    }

    fn remove_from(
        builtin: &'static str,
        object: &mut ObjectInstance,
        studies: &[crate::analysis::AnalysisStudySpec],
    ) -> BuiltinResult<Self> {
        let Some(value) = object.properties.remove(FROZEN_STUDY_GENERATORS_PROPERTY) else {
            return Self::new(
                builtin,
                &studies
                    .iter()
                    .cloned()
                    .map(|study| (study, FrozenGeneratorInventory::default()))
                    .collect::<Vec<_>>(),
            );
        };
        let Value::Cell(cell) = value else {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                "frozen study generators must be a cell array",
            ));
        };
        let mut decoded = BTreeMap::new();
        for value in &cell.data {
            let Value::Struct(row) = value else {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "frozen study generator entry must be a structure",
                ));
            };
            if row.fields.len() != 2 {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "frozen study generator entry has unexpected fields",
                ));
            }
            let Some(Value::String(study_id)) = row.fields.get("study_id") else {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "frozen study generator entry requires a study id",
                ));
            };
            let bindings = row.fields.get("bindings").ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "frozen study generator entry requires bindings",
                )
            })?;
            if decoded
                .insert(study_id.clone(), decode_bindings(builtin, bindings)?)
                .is_some()
            {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    format!("study `{study_id}` has duplicate frozen generator data"),
                ));
            }
        }
        let composed = studies
            .iter()
            .cloned()
            .map(|study| {
                let bindings = decoded.remove(&study.study_id).unwrap_or_default();
                let signals = study
                    .model
                    .as_ref()
                    .map(|model| model.signals.as_slice())
                    .unwrap_or_default();
                FrozenGeneratorInventory::new(builtin, signals, bindings)
                    .map(|inventory| (study, inventory))
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        if let Some((study_id, _)) = decoded.into_iter().next() {
            return Err(builtin_error(
                builtin,
                &ERROR_INPUT,
                format!("frozen generator data references missing study `{study_id}`"),
            ));
        }
        Self::new(builtin, &composed)
    }

    pub(in crate::builtins::fea) fn from_property_value(
        builtin: &'static str,
        value: Option<&Value>,
        studies: &[crate::analysis::AnalysisStudySpec],
    ) -> BuiltinResult<Self> {
        let mut carrier = ObjectInstance::new("runmat.internal.fea.sweep-bindings".into());
        if let Some(value) = value {
            carrier
                .properties
                .insert(FROZEN_STUDY_GENERATORS_PROPERTY.into(), value.clone());
        }
        Self::remove_from(builtin, &mut carrier, studies)
    }

    pub(in crate::builtins::fea) fn for_study(&self, study_id: &str) -> FrozenGeneratorInventory {
        self.studies
            .iter()
            .find(|(id, _)| id == study_id)
            .map(|(_, inventory)| inventory.clone())
            .unwrap_or_default()
    }
}

fn encode_bindings(bindings: &[FrozenSignalGenerator]) -> BuiltinResult<Value> {
    let mut values = Vec::with_capacity(bindings.len());
    for binding in bindings {
        let mut row = StructValue::new();
        row.insert("signal_id", Value::String(binding.signal_id.0.clone()));
        row.insert(
            "callable",
            super::callable_to_value("FEA signal", binding.callable.clone())?,
        );
        values.push(Value::Struct(row));
    }
    Ok(Value::Cell(
        CellArray::new(values, 1, bindings.len())
            .map_err(|error| builtin_error("FEA signal", &ERROR_INTERNAL, error))?,
    ))
}

fn decode_bindings(
    builtin: &'static str,
    value: &Value,
) -> BuiltinResult<Vec<FrozenSignalGenerator>> {
    let Value::Cell(cell) = value else {
        return Err(builtin_error(
            builtin,
            &ERROR_INPUT,
            "frozen generators must be a cell array",
        ));
    };
    cell.data
        .iter()
        .map(|value| {
            let Value::Struct(row) = value else {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "frozen generator entry must be a structure",
                ));
            };
            if row.fields.len() != 2 {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "frozen generator entry has unexpected fields",
                ));
            }
            let Some(Value::String(signal_id)) = row.fields.get("signal_id") else {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "frozen generator entry requires a signal id",
                ));
            };
            let callable = row.fields.get("callable").ok_or_else(|| {
                builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "frozen generator entry requires a callable",
                )
            })?;
            let encoded = crate::execution::value_codec::encode_inline_value(callable)
                .map_err(|error| builtin_error(builtin, &ERROR_INPUT, error.to_string()))?;
            let ValuePayload::Inline(encoded) = encoded else {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "frozen generator callable must be inline",
                ));
            };
            let InlineValue::Callable(callable) = *encoded else {
                return Err(builtin_error(
                    builtin,
                    &ERROR_INPUT,
                    "frozen generator entry does not contain a callable",
                ));
            };
            Ok(FrozenSignalGenerator {
                signal_id: SignalId(signal_id.clone()),
                callable,
            })
        })
        .collect()
}

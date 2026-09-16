//! YAML-tree mechanics shared by the canonical document operations.

use runmat_analysis_core::{
    AnalysisBoundary, AnalysisConnector, AnalysisConstraint, AnalysisInterface, ExcitationAction,
    ExcitationCase, InitialCondition, PhysicalSignal, ResponseSpectrum,
};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value as JsonValue;
use serde_yaml::{Mapping, Value as YamlValue};

use super::{FeaStudyNamedEntry, IdInput};

pub(super) fn decode<T: DeserializeOwned>(value: JsonValue, label: &str) -> Result<T, String> {
    serde_json::from_value(value).map_err(|error| format!("invalid {label} input: {error}"))
}

pub(super) fn decode_id(value: JsonValue, label: &str) -> Result<String, String> {
    let input: IdInput = decode(value, label)?;
    validate_id(&input.id, "id")?;
    Ok(input.id)
}

pub(super) fn to_yaml<T: Serialize>(value: &T) -> Result<YamlValue, String> {
    // `serde_yaml` uses native `!variant` tags for externally tagged Rust enums. Those tags cannot
    // be nested reliably inside the document's internally tagged root enum. Passing through JSON
    // gives every enum the canonical singleton-map representation accepted by `.fea` parsing and
    // keeps generated documents portable to YAML tooling that does not understand Rust tags.
    let canonical = serde_json::to_value(value)
        .map_err(|error| format!("failed to encode FEA value: {error}"))?;
    serde_yaml::to_value(canonical).map_err(|error| format!("failed to encode FEA value: {error}"))
}

pub(super) fn parse_document(source: &str) -> Result<YamlValue, String> {
    if source.trim().is_empty() {
        return Ok(YamlValue::Mapping(Mapping::new()));
    }
    let value: YamlValue = serde_yaml::from_str(source)
        .map_err(|error| format!("failed to parse finite element study: {error}"))?;
    if !value.is_mapping() {
        return Err("finite element study document must be a YAML mapping".into());
    }
    Ok(value)
}

pub(super) fn serialize_document(value: &YamlValue) -> Result<String, String> {
    serde_yaml::to_string(value)
        .map_err(|error| format!("failed to serialize finite element study: {error}"))
}

fn root(value: &mut YamlValue) -> &mut Mapping {
    if !value.is_mapping() {
        *value = YamlValue::Mapping(Mapping::new());
    }
    value.as_mapping_mut().expect("mapping")
}

pub(super) fn inline_model_data(value: &YamlValue) -> Option<&YamlValue> {
    inline_source_data(yaml_at(value, &["model", "source"])?)
}

pub(super) fn inline_model_data_mut(value: &mut YamlValue) -> Result<&mut YamlValue, String> {
    let model = root(value)
        .get_mut(YamlValue::String("model".into()))
        .and_then(YamlValue::as_mapping_mut)
        .ok_or_else(|| "study model must be a mapping".to_string())?;
    let source = model
        .get_mut(YamlValue::String("source".into()))
        .and_then(YamlValue::as_mapping_mut)
        .ok_or_else(|| "study model source must be a typed mapping".to_string())?;
    inline_source_data_mut(source).ok_or_else(|| {
        "model-data mutations require an inline model source or an explicit inline overlay in a composed source"
            .to_string()
    })
}

fn inline_source_data(source: &YamlValue) -> Option<&YamlValue> {
    let source = source.as_mapping()?;
    if let Some(inline) = source.get(YamlValue::String("inline".into())) {
        return Some(inline);
    }
    source
        .get(YamlValue::String("compose".into()))?
        .as_sequence()?
        .iter()
        .rev()
        .find_map(inline_source_data)
}

fn inline_source_data_mut(source: &mut Mapping) -> Option<&mut YamlValue> {
    if source.contains_key(YamlValue::String("inline".into())) {
        return source.get_mut(YamlValue::String("inline".into()));
    }
    source
        .get_mut(YamlValue::String("compose".into()))?
        .as_sequence_mut()?
        .iter_mut()
        .rev()
        .find_map(|source| inline_source_data_mut(source.as_mapping_mut()?))
}

pub(super) fn set_value(value: &mut YamlValue, key: &str, entry: YamlValue) {
    root(value).insert(YamlValue::String(key.into()), entry);
}

pub(super) fn remove_value(value: &mut YamlValue, key: &str) {
    root(value).remove(YamlValue::String(key.into()));
}

pub(super) fn mutate_map(
    value: &mut YamlValue,
    section: &str,
    id: String,
    entry: YamlValue,
    add: bool,
) -> Result<(), String> {
    let entries = root(value)
        .entry(YamlValue::String(section.into()))
        .or_insert_with(|| YamlValue::Mapping(Mapping::new()))
        .as_mapping_mut()
        .ok_or_else(|| format!("{section} must be a mapping"))?;
    let key = YamlValue::String(id.clone());
    if add && entries.contains_key(&key) {
        return Err(format!("{section} entry `{id}` already exists"));
    }
    if !add && !entries.contains_key(&key) {
        return Err(format!("{section} entry `{id}` does not exist"));
    }
    entries.insert(key, entry);
    Ok(())
}

pub(super) fn remove_map(value: &mut YamlValue, section: &str, id: &str) -> Result<(), String> {
    let entries = root(value)
        .get_mut(YamlValue::String(section.into()))
        .and_then(YamlValue::as_mapping_mut)
        .ok_or_else(|| format!("{section} entry `{id}` does not exist"))?;
    if entries.remove(YamlValue::String(id.into())).is_none() {
        return Err(format!("{section} entry `{id}` does not exist"));
    }
    Ok(())
}

fn sequence_mut<'a>(
    value: &'a mut YamlValue,
    section: &str,
) -> Result<&'a mut Vec<YamlValue>, String> {
    root(value)
        .entry(YamlValue::String(section.into()))
        .or_insert_with(|| YamlValue::Sequence(Vec::new()))
        .as_sequence_mut()
        .ok_or_else(|| format!("{section} must be a sequence"))
}

pub(super) fn mutate_sequence<F: Fn(&YamlValue) -> Option<String>>(
    value: &mut YamlValue,
    section: &str,
    id: &str,
    entry: YamlValue,
    add: bool,
    identify: F,
) -> Result<(), String> {
    validate_id(id, "id")?;
    let entries = sequence_mut(value, section)?;
    let at = entries
        .iter()
        .position(|value| identify(value).as_deref() == Some(id));
    match (add, at) {
        (true, Some(_)) => Err(format!("{section} entry `{id}` already exists")),
        (true, None) => {
            entries.push(entry);
            Ok(())
        }
        (false, Some(index)) => {
            entries[index] = entry;
            Ok(())
        }
        (false, None) => Err(format!("{section} entry `{id}` does not exist")),
    }
}

pub(super) fn remove_named<F: Fn(&YamlValue) -> Option<String>>(
    value: &mut YamlValue,
    section: &'static str,
    input: JsonValue,
    identify: F,
) -> Result<&'static str, String> {
    let id = decode_id(input, section)?;
    let entries = sequence_mut(value, section)?;
    let index = entries
        .iter()
        .position(|value| identify(value).as_deref() == Some(&id))
        .ok_or_else(|| format!("{section} entry `{id}` does not exist"))?;
    entries.remove(index);
    Ok(section)
}

pub(super) fn append_pair(
    value: &mut YamlValue,
    target: &YamlValue,
    material: &str,
    entry: YamlValue,
) -> Result<(), String> {
    let entries = sequence_mut(value, "material_assignments")?;
    if entries.iter().any(|value| {
        mapping_entry(value, "target") == Some(target)
            && scalar(value, &["material"]).as_deref() == Some(material)
    }) {
        return Err(format!(
            "material_assignments entry for `{material}` and the same target already exists"
        ));
    }
    entries.push(entry);
    Ok(())
}

pub(super) fn remove_pair(
    value: &mut YamlValue,
    target: &YamlValue,
    material: &str,
) -> Result<(), String> {
    let entries = sequence_mut(value, "material_assignments")?;
    let index = entries
        .iter()
        .position(|value| {
            mapping_entry(value, "target") == Some(target)
                && scalar(value, &["material"]).as_deref() == Some(material)
        })
        .ok_or_else(|| {
            format!("material_assignments entry for `{material}` and the requested target does not exist")
        })?;
    entries.remove(index);
    Ok(())
}

fn mapping_entry<'a>(value: &'a YamlValue, key: &str) -> Option<&'a YamlValue> {
    value.as_mapping()?.get(YamlValue::String(key.to_string()))
}

fn typed_id<T: DeserializeOwned>(
    value: &YamlValue,
    identify: impl FnOnce(&T) -> String,
) -> Option<String> {
    from_yaml::<T>(value).map(|value| identify(&value))
}

pub(super) fn constraint_id(value: &YamlValue) -> Option<String> {
    typed_id::<AnalysisConstraint>(value, |value| value.constraint_id().into())
}

pub(super) fn boundary_id(value: &YamlValue) -> Option<String> {
    typed_id::<AnalysisBoundary>(value, |value| value.boundary_id.clone())
}

pub(super) fn connector_id(value: &YamlValue) -> Option<String> {
    typed_id::<AnalysisConnector>(value, |value| value.connector_id.clone())
}

pub(super) fn interface_id(value: &YamlValue) -> Option<String> {
    typed_id::<AnalysisInterface>(value, |value| value.interface_id.clone())
}

pub(super) fn signal_id(value: &YamlValue) -> Option<String> {
    typed_id::<PhysicalSignal>(value, |value| value.signal_id().0.clone())
}

pub(super) fn excitation_case_id(value: &YamlValue) -> Option<String> {
    typed_id::<ExcitationCase>(value, |value| value.case_id.0.clone())
}

pub(super) fn excitation_id(value: &YamlValue) -> Option<String> {
    typed_id::<ExcitationAction>(value, |value| value.excitation().excitation_id.0.clone())
}

pub(super) fn initial_condition_id(value: &YamlValue) -> Option<String> {
    typed_id::<InitialCondition>(value, |value| value.initial_condition_id().into())
}

pub(super) fn response_spectrum_id(value: &YamlValue) -> Option<String> {
    typed_id::<ResponseSpectrum>(value, |value| value.spectrum_id.0.clone())
}

pub(super) fn random_excitation_id(value: &YamlValue) -> Option<String> {
    typed_id::<runmat_analysis_core::RandomExcitationSpectrum>(value, |value| {
        value.excitation_id.0.clone()
    })
}

pub(super) fn named_entries<T: DeserializeOwned>(
    value: &YamlValue,
    section: &str,
    identify: impl Fn(&T) -> String,
) -> Vec<FeaStudyNamedEntry> {
    sequence_values(value, section)
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            from_yaml::<T>(value).map(|value| FeaStudyNamedEntry {
                index,
                id: identify(&value),
            })
        })
        .collect()
}

pub(super) fn map_entries<T>(
    value: &YamlValue,
    section: &str,
    map: impl Fn(String, &YamlValue) -> T,
) -> Vec<T> {
    yaml_at(value, &[section])
        .and_then(YamlValue::as_mapping)
        .into_iter()
        .flat_map(|entries| entries.iter())
        .filter_map(|(key, value)| key.as_str().map(|key| map(key.into(), value)))
        .collect()
}

pub(super) fn sequence_values<'a>(value: &'a YamlValue, section: &str) -> &'a [YamlValue] {
    yaml_at(value, &[section])
        .and_then(YamlValue::as_sequence)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

pub(super) fn yaml_at<'a>(value: &'a YamlValue, path: &[&str]) -> Option<&'a YamlValue> {
    path.iter().try_fold(value, |value, key| {
        value.as_mapping()?.get(YamlValue::String((*key).into()))
    })
}

pub(super) fn typed_at<T: DeserializeOwned>(value: &YamlValue, path: &[&str]) -> Option<T> {
    yaml_at(value, path).and_then(from_yaml)
}

fn from_yaml<T: DeserializeOwned>(value: &YamlValue) -> Option<T> {
    let canonical = serde_json::to_value(value).ok()?;
    serde_json::from_value(canonical).ok()
}

pub(super) fn scalar(value: &YamlValue, path: &[&str]) -> Option<String> {
    match yaml_at(value, path)? {
        YamlValue::String(value) => Some(value.clone()),
        YamlValue::Number(value) => Some(value.to_string()),
        YamlValue::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

pub(super) fn validate_id(value: &str, field: &str) -> Result<(), String> {
    if !value.trim().is_empty()
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.')
        })
    {
        Ok(())
    } else {
        Err(format!(
            "{field} must be a non-empty identifier containing only ASCII letters, numbers, '.', '-', or '_'"
        ))
    }
}

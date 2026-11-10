//! Configurations: named variants of one part, held in the same file.
//!
//! ```text
//! configuration long = { width = 120 mm }
//! configuration analysis = { suppress = [feature.round, feature.h] }
//! configuration small = { width = 60 mm, features = { h = { diameter = 3 mm } } }
//! ```
//!
//! A configuration's record holds three kinds of entry:
//!
//! - `<input> = <value>`: the input's value in this configuration (an input override).
//! - `suppress = [feature.<id>, …]`: features left out of this configuration only.
//! - `features = { <id> = { <field> = <value>, … } }`: a feature's own parameters in this
//!   configuration, written as the fields of its definition (`extent = { distance = 12 mm }`
//!   for an extrude's depth). A nested record merges into the field it names; any other value
//!   replaces it.
//! - `part_number = "<text>"`, `description = "<text>"` and `properties = { <name> = <value> }`:
//!   the part's own identity in this configuration. A left-hand and a right-hand version of one
//!   part are different part numbers; the BOM, drawing title blocks and exports read them
//!   ([`crate::model::PartDocument::in_configuration`]).
//!
//! [`Configuration::apply_features`] writes a configuration's feature overrides into a copy of
//! the history, so regeneration, dependency analysis and every other reader sees one form.

use crate::ids::{ConfigurationName, FeatureId, InputName};
use crate::model::{Feature, PropertyValue};
use crate::value::ValueExpr;
use serde::de::Error as _;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// The record keys a configuration reserves; every other key is an input override.
pub const RESERVED: &[&str] = &[
    "name",
    "suppress",
    "features",
    "part_number",
    "description",
    "properties",
];

/// A feature's parameter overrides: fields of its definition (as the definition serializes).
pub type FeaturePatch = Map<String, Value>;

/// Every feature's parameter overrides in one configuration.
pub type FeaturePatches = BTreeMap<FeatureId, FeaturePatch>;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Configuration {
    pub name: ConfigurationName,
    pub overrides: BTreeMap<InputName, ValueExpr>,
    /// Features suppressed in this configuration only.
    pub suppress: BTreeSet<FeatureId>,
    /// Feature parameters set in this configuration only.
    pub features: FeaturePatches,
    /// The part number of the part in this configuration (a dash number, the other hand).
    pub part_number: Option<String>,
    /// Properties of the part in this configuration, `description` among them.
    pub properties: BTreeMap<String, PropertyValue>,
}

/// `feature.<id>` as a value reference, the form `suppress` lists are written in.
fn feature_reference(id: &FeatureId) -> ValueExpr {
    ValueExpr::Reference {
        path: vec!["feature".into(), id.0.clone()],
    }
}

impl Serialize for Configuration {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("name", &self.name)?;
        for (input, value) in &self.overrides {
            map.serialize_entry(input.as_str(), value)?;
        }
        if !self.suppress.is_empty() {
            let list: Vec<ValueExpr> = self.suppress.iter().map(feature_reference).collect();
            map.serialize_entry("suppress", &list)?;
        }
        if !self.features.is_empty() {
            map.serialize_entry("features", &self.features)?;
        }
        if let Some(pn) = &self.part_number {
            map.serialize_entry("part_number", pn)?;
        }
        let mut rest = Map::new();
        for (name, value) in &self.properties {
            let json = property_json(value);
            if name == "description" && json.is_string() {
                map.serialize_entry("description", &json)?;
            } else {
                rest.insert(name.clone(), json);
            }
        }
        if !rest.is_empty() {
            map.serialize_entry("properties", &rest)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Configuration {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Value::Object(entries) = Value::deserialize(deserializer)? else {
            return Err(D::Error::custom("a configuration is a record"));
        };
        let mut out = Configuration::default();
        for (key, value) in entries {
            match key.as_str() {
                "name" => {
                    out.name = ConfigurationName::deserialize(value).map_err(D::Error::custom)?
                }
                "suppress" => {
                    let Value::Array(items) = value else {
                        return Err(D::Error::custom(
                            "`suppress` lists features: `suppress = [feature.round]`",
                        ));
                    };
                    for item in items {
                        out.suppress
                            .insert(suppressed_feature(item).map_err(D::Error::custom)?);
                    }
                }
                "features" => {
                    let Value::Object(features) = value else {
                        return Err(D::Error::custom(
                            "`features` holds each feature's parameters: `features = { h = { diameter = 3 mm } }`",
                        ));
                    };
                    for (id, patch) in features {
                        let Value::Object(patch) = patch else {
                            return Err(D::Error::custom(format!(
                                "`features.{id}` is a record of the feature's parameters"
                            )));
                        };
                        out.features.insert(FeatureId::new(id), patch);
                    }
                }
                "part_number" => {
                    let Value::String(pn) = value else {
                        return Err(D::Error::custom(
                            "`part_number` is text: `part_number = \"BR-100-L\"`",
                        ));
                    };
                    out.part_number = Some(pn);
                }
                "description" => {
                    let Value::String(text) = value else {
                        return Err(D::Error::custom("`description` is text"));
                    };
                    out.properties
                        .insert("description".into(), PropertyValue::Text(text));
                }
                "properties" => {
                    let Value::Object(entries) = value else {
                        return Err(D::Error::custom(
                            "`properties` is a record: `properties = { finish = \"anodized\" }`",
                        ));
                    };
                    for (name, value) in entries {
                        let value = property_value(value)
                            .map_err(|e| D::Error::custom(format!("property `{name}`: {e}")))?;
                        out.properties.insert(name, value);
                    }
                }
                input => {
                    let value = ValueExpr::deserialize(value).map_err(|e| {
                        D::Error::custom(format!("the value of input `{input}`: {e}"))
                    })?;
                    out.overrides.insert(InputName::new(input), value);
                }
            }
        }
        Ok(out)
    }
}

/// A property as it is written: text, a flag, or a value.
fn property_json(value: &PropertyValue) -> Value {
    match value {
        PropertyValue::Text(t) => Value::String(t.clone()),
        PropertyValue::Boolean(b) => Value::Bool(*b),
        PropertyValue::Quantity(q) => serde_json::to_value(q).unwrap_or(Value::Null),
    }
}

fn property_value(value: Value) -> Result<PropertyValue, String> {
    match value {
        Value::String(t) => Ok(PropertyValue::Text(t)),
        Value::Bool(b) => Ok(PropertyValue::Boolean(b)),
        other => ValueExpr::deserialize(other)
            .map(PropertyValue::Quantity)
            .map_err(|e| e.to_string()),
    }
}

/// A `suppress` list item: `feature.<id>`.
fn suppressed_feature(item: Value) -> Result<FeatureId, String> {
    match ValueExpr::deserialize(item) {
        Ok(ValueExpr::Reference { path }) if path.len() == 2 && path[0] == "feature" => {
            Ok(FeatureId::new(path[1].clone()))
        }
        _ => Err("a `suppress` list names features as `feature.<name>`".into()),
    }
}

/// Merges `patch` into `target`: a record into a record field by field, any other value in
/// place of the field. A value expression (a record with a `form` tag) replaces whole.
fn merge(target: &mut Map<String, Value>, patch: &Map<String, Value>) {
    for (key, value) in patch {
        match (target.get_mut(key), value) {
            (Some(Value::Object(inner)), Value::Object(part))
                if !part.contains_key("form") && !inner.contains_key("form") =>
            {
                merge(inner, part)
            }
            _ => {
                target.insert(key.clone(), value.clone());
            }
        }
    }
}

/// A feature override that could not be applied.
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureOverrideProblem {
    pub feature: FeatureId,
    pub message: String,
}

impl Configuration {
    /// Whether this configuration changes anything about `feature` (suppresses or sets it).
    pub fn touches(&self, feature: &FeatureId) -> bool {
        self.suppress.contains(feature) || self.features.contains_key(feature)
    }

    /// The history with this configuration's feature overrides written in. A feature the part
    /// lacks, or a patch its definition cannot take, is a problem; that feature keeps its own
    /// definition.
    pub fn apply_features(
        &self,
        features: &[Feature],
    ) -> (Vec<Feature>, Vec<FeatureOverrideProblem>) {
        let mut out = features.to_vec();
        let mut problems = Vec::new();
        for (id, patch) in &self.features {
            let Some(feature) = out.iter_mut().find(|f| &f.id == id) else {
                problems.push(FeatureOverrideProblem {
                    feature: id.clone(),
                    message: format!(
                        "configuration `{}` sets `{id}`, which the part does not have",
                        self.name
                    ),
                });
                continue;
            };
            // `suppressed` is the feature's own flag, not a field of its definition: a
            // configuration that sets it brings back (or leaves out) a feature the part
            // suppresses (or keeps) by default.
            let mut patch = patch.clone();
            if let Some(flag) = patch.remove("suppressed") {
                match patch_bool(&flag) {
                    Some(b) => feature.suppressed = b,
                    None => problems.push(FeatureOverrideProblem {
                        feature: id.clone(),
                        message: format!(
                            "configuration `{}` sets `{id}.suppressed`, which is `true` or `false`",
                            self.name
                        ),
                    }),
                }
                if patch.is_empty() {
                    continue;
                }
            }
            match patched_kind(feature, &patch) {
                Ok(kind) => feature.kind = kind,
                Err(e) => problems.push(FeatureOverrideProblem {
                    feature: id.clone(),
                    message: format!(
                        "configuration `{}` sets `{id}` to a definition it cannot take: {e}",
                        self.name
                    ),
                }),
            }
        }
        (out, problems)
    }
}

/// A patch value read as a flag: `true`/`false` as written in the source (a bool, or the word).
fn patch_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::String(s) => s.parse().ok(),
        Value::Object(m) => m.values().find_map(|x| match x {
            Value::Bool(b) => Some(*b),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }),
        _ => None,
    }
}

fn patched_kind(
    feature: &Feature,
    patch: &FeaturePatch,
) -> Result<crate::model::FeatureKind, serde_json::Error> {
    let mut kind = match serde_json::to_value(&feature.kind)? {
        Value::Object(map) => map,
        _ => return Err(serde_json::Error::custom("the definition is not a record")),
    };
    merge(&mut kind, patch);
    serde_json::from_value(Value::Object(kind))
}

/// The value at `path` (dotted field names) of a feature's definition, after `patch`.
pub fn patch_value<'a>(patch: &'a FeaturePatch, path: &[&str]) -> Option<&'a Value> {
    let (first, rest) = path.split_first()?;
    let mut at = patch.get(*first)?;
    for key in rest {
        at = at.as_object()?.get(*key)?;
    }
    Some(at)
}

/// Sets the value at `path` in a patch, making the records on the way.
pub fn set_patch_value(patch: &mut FeaturePatch, path: &[&str], value: Value) {
    let Some((last, parents)) = path.split_last() else {
        return;
    };
    let mut at = patch;
    for key in parents {
        let entry = at
            .entry(key.to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if !entry.is_object() {
            *entry = Value::Object(Map::new());
        }
        at = entry.as_object_mut().expect("a record");
    }
    at.insert(last.to_string(), value);
}

/// Removes the value at `path` from a patch, and the records it leaves empty. Returns whether
/// the patch is empty afterwards.
pub fn remove_patch_value(patch: &mut FeaturePatch, path: &[&str]) -> bool {
    fn remove(at: &mut Map<String, Value>, path: &[&str]) {
        match path {
            [] => {}
            [last] => {
                at.remove(*last);
            }
            [first, rest @ ..] => {
                if let Some(Value::Object(inner)) = at.get_mut(*first) {
                    remove(inner, rest);
                    if inner.is_empty() {
                        at.remove(*first);
                    }
                }
            }
        }
    }
    remove(patch, path);
    patch.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_record_holds_inputs_suppressions_and_feature_parameters() {
        let c: Configuration = serde_json::from_value(json!({
            "name": "small",
            "width": { "form": "quantity", "value": 60.0, "unit": "mm" },
            "suppress": [{ "form": "reference", "path": ["feature", "round"] }],
            "features": { "h": { "diameter": { "form": "quantity", "value": 3.0, "unit": "mm" } } }
        }))
        .unwrap();
        assert_eq!(c.name.as_str(), "small");
        assert_eq!(c.overrides[&InputName::new("width")], ValueExpr::mm(60.0));
        assert!(c.suppress.contains(&FeatureId::new("round")));
        assert!(c.features.contains_key(&FeatureId::new("h")));
        assert!(c.part_number.is_none());
        let back: Configuration =
            serde_json::from_value(serde_json::to_value(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn a_configuration_brings_back_a_feature_the_part_suppresses() {
        let parsed = crate::syntax::parse_part(
            "p.part",
            "runmat 0.7\nstd 1\n\npart p {\n    sketch s on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 5 mm)\n    }\n    feature a = extrude(profile = sketch.s, distance = 2 mm)\n    feature b = extrude(profile = sketch.s, distance = 4 mm, suppressed = true)\n    output p = { body = feature.a.body }\n    configuration upper = { suppress = [feature.a], features = { b = { suppressed = false } } }\n}\n",
        );
        let doc = parsed.document.unwrap_or_else(|| panic!("the part parses ({} diagnostics)", parsed.diagnostics.len()));
        let c = doc.configuration(&ConfigurationName::new("upper")).unwrap();
        let (features, problems) = c.apply_features(&doc.features);
        assert!(problems.is_empty(), "{problems:?}");
        let b = features.iter().find(|f| f.id == FeatureId::new("b")).unwrap();
        assert!(!b.suppressed, "the configuration unsuppresses b");
        assert_eq!(b.kind, doc.features.iter().find(|f| f.id == FeatureId::new("b")).unwrap().kind);
    }

    #[test]
    fn a_configuration_carries_its_own_part_number_and_properties() {
        let c: Configuration = serde_json::from_value(json!({
            "name": "port",
            "side": -1.0,
            "part_number": "FUS-WS-034",
            "description": "Windshield pane, left",
            "properties": { "finish": "clear coat" }
        }))
        .unwrap();
        assert_eq!(c.part_number.as_deref(), Some("FUS-WS-034"));
        assert_eq!(
            c.properties["description"],
            PropertyValue::Text("Windshield pane, left".into())
        );
        let written = serde_json::to_value(&c).unwrap();
        assert_eq!(written["description"], "Windshield pane, left");
        assert_eq!(written["properties"]["finish"], "clear coat");
        let back: Configuration = serde_json::from_value(written).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn a_suppress_list_names_features_by_reference() {
        let bare =
            serde_json::from_value::<Configuration>(json!({ "name": "a", "suppress": ["round"] }));
        assert!(bare.is_err());
    }

    #[test]
    fn nested_records_merge_and_values_replace() {
        let mut target = json!({ "extent": { "extent": "blind", "distance": { "form": "quantity", "value": 10.0, "unit": "mm" } }, "reverse": false });
        let patch = json!({ "extent": { "distance": { "form": "input", "name": "depth" } } });
        merge(target.as_object_mut().unwrap(), patch.as_object().unwrap());
        assert_eq!(target["extent"]["extent"], "blind");
        assert_eq!(
            target["extent"]["distance"],
            json!({ "form": "input", "name": "depth" })
        );
    }

    #[test]
    fn patch_paths_set_and_clear() {
        let mut patch = FeaturePatch::new();
        set_patch_value(&mut patch, &["extent", "distance"], json!(1));
        assert_eq!(
            patch_value(&patch, &["extent", "distance"]),
            Some(&json!(1))
        );
        assert!(remove_patch_value(&mut patch, &["extent", "distance"]));
    }
}

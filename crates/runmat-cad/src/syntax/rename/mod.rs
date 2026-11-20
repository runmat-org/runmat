//! Rename: a part's feature (sketches, planes and mate connectors are features too), input or
//! region takes a new name in its declaration and in every reference to it, and the source
//! changes by the smallest text edit ([`super::edit::diff_source`]).
//!
//! References name a feature in several written forms, and each reads into the model as the
//! feature's id in one of a few places, so the rename works on the model:
//!
//! | Written | Model |
//! |---|---|
//! | `feature body = …` | the feature's `id` |
//! | `profile = sketch.s1`, `sketch s2 on plane.mid` | `sketch`, `feature` (a datum) |
//! | `feature.body.face.side(s1.l1)` | a generated name's `feature`, its curve tag's `sketch` |
//! | `suppress = [feature.round]`, `let.x = sketch.s1.l1.end` | a value reference's `path` |
//! | `seeds = [hole]`, `features = [hole]` | lists of ids |
//! | `rollback_after = hole` | `rollback_after` |
//! | a configuration's `features = { hole = { … } }` | the patch map's keys |
//!
//! Inputs are read as `input.<name>` and set by name in configurations; regions are named by
//! publishes (`region = mount`) and links.
//!
//! [`follow`] carries a rename into the project files that name the part's items: drawings
//! (features), assemblies (inputs) and `.fea` studies (regions).

mod follow;
pub use follow::{
    assemblies_in, rename_in_drawing, rename_input_in_assembly, rename_region_in_study,
};

use super::edit::{diff_source, EditFailure, SourceChange};
use super::{parse_part, PART};
use crate::ids::FeatureId;
use crate::model::PartDocument;
use crate::refs::name::{recipe_of, remember, EntityName, NameToken};
use crate::regen::deps::DependencyGraph;
use serde_json::{Map, Value};

/// What a part rename names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenameKind {
    /// A feature, sketch, plane, axis, point or mate connector.
    Feature,
    Input,
    Region,
}

impl RenameKind {
    fn noun(self) -> &'static str {
        match self {
            RenameKind::Feature => "feature",
            RenameKind::Input => "input",
            RenameKind::Region => "region",
        }
    }
}

/// Words a name can't take: the reference heads and the literals.
const RESERVED: &[&str] = &[
    "feature", "sketch", "plane", "axis", "point", "origin", "input", "let", "region", "publish",
    "part", "material", "true", "false", "none", "std", "runmat",
];

/// Path heads that name a feature in a value reference (`sketch.s1.l1.end`).
const FEATURE_HEADS: &[&str] = &["feature", "sketch", "plane", "axis", "point"];

/// Model keys whose string value (or list of strings) is a feature id.
const FEATURE_KEYS: &[&str] = &[
    "feature",
    "sketch",
    "maker",
    "rollback_after",
    "seeds",
    "features",
    "suppress",
];

pub(super) fn failure(message: impl Into<String>) -> EditFailure {
    EditFailure::Model(message.into())
}

/// Why `name` can't be a name, if it can't.
pub fn invalid_name(name: &str) -> Option<String> {
    let mut chars = name.chars();
    let starts = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    if name.is_empty() {
        return Some("type a name".into());
    }
    if !starts || !chars.all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Some(format!(
            "`{name}` isn't a name: use letters, digits and underscores, starting with a letter"
        ));
    }
    if RESERVED.contains(&name) {
        return Some(format!("`{name}` is a reserved word; pick another name"));
    }
    None
}

fn rename_string(value: &mut Value, from: &str, to: &str) {
    match value {
        Value::String(s) if s == from => *s = to.to_owned(),
        Value::Array(items) => items.iter_mut().for_each(|v| rename_string(v, from, to)),
        _ => {}
    }
}

/// A value reference's `path`: `[head, name, …]`.
fn rename_path(items: &mut [Value], heads: &[&str], from: &str, to: &str) {
    if let [Value::String(head), Value::String(name), ..] = items {
        if heads.contains(&head.as_str()) && name == from {
            *name = to.to_owned();
        }
    }
}

/// Renames feature `from` to `to` wherever the JSON form of a model names it. `tag` skips the
/// entry's own kind tag (`"feature": "extrude"`), which is a kind word.
pub(super) fn rename_feature_json(value: &mut Value, from: &str, to: &str, tag: bool) {
    match value {
        Value::Object(map) => {
            for (key, v) in map.iter_mut() {
                if key == "path" {
                    if let Value::Array(items) = v {
                        rename_path(items, FEATURE_HEADS, from, to);
                    }
                }
                // A sheet-metal bend line: `[sketch, curve]`.
                if key == "line" {
                    if let Value::Array(items) = v {
                        if let Some(first) = items.first_mut() {
                            rename_string(first, from, to);
                        }
                    }
                }
                if FEATURE_KEYS.contains(&key.as_str()) && !(tag && key == "feature") {
                    rename_string(v, from, to);
                }
                rename_feature_json(v, from, to, false);
            }
        }
        Value::Array(items) => {
            // Token sets are kept sorted (a blend's two faces): they stay sorted.
            let tokens = |items: &[Value]| -> Option<Vec<String>> {
                items
                    .iter()
                    .map(|v| v.as_str().filter(|t| is_token(t)).map(str::to_owned))
                    .collect()
            };
            let sorted = tokens(items).is_some_and(|t| t.windows(2).all(|w| w[0] <= w[1]));
            items
                .iter_mut()
                .for_each(|v| rename_feature_json(v, from, to, false));
            if sorted {
                items.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
            }
        }
        // A nested name's token (a blend's faces, a copy's base): the token of the same name
        // with the feature renamed.
        Value::String(text) => {
            if let Some(token) = renamed_token(text, from, to) {
                *text = token;
            }
        }
        _ => {}
    }
}

fn is_token(text: &str) -> bool {
    text.len() == 32 && text.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The token `text` names, renamed, when it is a token of a name that names `from`.
fn renamed_token(text: &str, from: &str, to: &str) -> Option<String> {
    if !is_token(text) {
        return None;
    }
    let token: NameToken = serde_json::from_value(Value::String(text.to_owned())).ok()?;
    let name = recipe_of(&token)?;
    let mut json = serde_json::to_value(&name).ok()?;
    rename_feature_json(&mut json, from, to, false);
    let renamed: EntityName = serde_json::from_value(json).ok()?;
    (renamed != name).then(|| remember(&renamed).to_string())
}

fn rename_key(map: &mut Map<String, Value>, from: &str, to: &str) {
    if let Some(v) = map.remove(from) {
        map.insert(to.to_owned(), v);
    }
}

fn rename_input_json(value: &mut Value, from: &str, to: &str) {
    match value {
        Value::Object(map) => {
            // `input.<name>` in a value: `{ form: input, name }`.
            if map.get("form").and_then(Value::as_str) == Some("input") {
                if let Some(name) = map.get_mut("name") {
                    rename_string(name, from, to);
                }
            }
            for (key, v) in map.iter_mut() {
                if key == "path" {
                    if let Value::Array(items) = v {
                        rename_path(items, &["input"], from, to);
                    }
                }
                rename_input_json(v, from, to);
            }
        }
        Value::Array(items) => items
            .iter_mut()
            .for_each(|v| rename_input_json(v, from, to)),
        _ => {}
    }
}

/// The model with `from` renamed to `to`, through its JSON form.
fn renamed(
    document: &PartDocument,
    kind: RenameKind,
    from: &str,
    to: &str,
) -> Result<PartDocument, EditFailure> {
    let mut json = serde_json::to_value(document).map_err(|e| failure(e.to_string()))?;
    let root = json
        .as_object_mut()
        .ok_or_else(|| failure("the part isn't a record"))?;
    match kind {
        RenameKind::Feature => {
            for (key, v) in root.iter_mut() {
                match key.as_str() {
                    // Templates name their own features.
                    "feature_defs" => {}
                    "features" => {
                        for entry in v.as_array_mut().into_iter().flatten() {
                            let Some(entry) = entry.as_object_mut() else {
                                continue;
                            };
                            for (field, value) in entry.iter_mut() {
                                if field == "id" {
                                    rename_string(value, from, to);
                                } else {
                                    rename_feature_json(value, from, to, field == "kind");
                                }
                            }
                        }
                    }
                    "configurations" => {
                        for configuration in v.as_array_mut().into_iter().flatten() {
                            if let Some(Value::Object(patches)) = configuration.get_mut("features")
                            {
                                rename_key(patches, from, to);
                            }
                            rename_feature_json(configuration, from, to, false);
                        }
                    }
                    "rollback_after" => rename_string(v, from, to),
                    _ => rename_feature_json(v, from, to, false),
                }
            }
        }
        RenameKind::Input => {
            for (key, v) in root.iter_mut() {
                match key.as_str() {
                    "inputs" => {
                        for entry in v.as_array_mut().into_iter().flatten() {
                            if let Some(name) = entry.get_mut("name") {
                                rename_string(name, from, to);
                            }
                        }
                    }
                    "configurations" => {
                        // A configuration sets an input under the input's name.
                        for configuration in v.as_array_mut().into_iter().flatten() {
                            if let Some(map) = configuration.as_object_mut() {
                                rename_key(map, from, to);
                            }
                            rename_input_json(configuration, from, to);
                        }
                    }
                    "feature_defs" => {}
                    _ => rename_input_json(v, from, to),
                }
            }
        }
        RenameKind::Region => {
            for (key, v) in root.iter_mut() {
                match key.as_str() {
                    "regions" => {
                        for entry in v.as_array_mut().into_iter().flatten() {
                            if let Some(name) = entry.get_mut("name") {
                                rename_string(name, from, to);
                            }
                        }
                    }
                    "publishes" | "links" => rename_region_json(v, from, to),
                    _ => {}
                }
            }
        }
    }
    serde_json::from_value(json).map_err(|e| failure(format!("the renamed part: {e}")))
}

fn rename_region_json(value: &mut Value, from: &str, to: &str) {
    match value {
        Value::Object(map) => {
            for (key, v) in map.iter_mut() {
                if key == "region" {
                    rename_string(v, from, to);
                }
                rename_region_json(v, from, to);
            }
        }
        Value::Array(items) => items
            .iter_mut()
            .for_each(|v| rename_region_json(v, from, to)),
        _ => {}
    }
}

/// Feature dependencies with `from` read as `to`: a rename keeps every edge.
fn graph_renamed(graph: &DependencyGraph, from: &str, to: &str) -> DependencyGraph {
    use std::collections::{BTreeMap, BTreeSet};
    let id = |f: &FeatureId| {
        if f.0 == from {
            FeatureId::new(to)
        } else {
            f.clone()
        }
    };
    let set = |s: &BTreeSet<FeatureId>| s.iter().map(id).collect::<BTreeSet<_>>();
    fn keyed<K: Ord + Clone, V>(
        m: &BTreeMap<K, V>,
        k: impl Fn(&K) -> K,
        v: impl Fn(&V) -> V,
    ) -> BTreeMap<K, V> {
        m.iter().map(|(a, b)| (k(a), v(b))).collect()
    }
    DependencyGraph {
        order: graph.order.iter().map(id).collect(),
        feature_inputs: keyed(&graph.feature_inputs, id, Clone::clone),
        feature_deps: keyed(&graph.feature_deps, id, set),
        part_deps: keyed(&graph.part_deps, Clone::clone, set),
        region_deps: keyed(&graph.region_deps, Clone::clone, set),
        publish_deps: keyed(&graph.publish_deps, Clone::clone, set),
        feature_published: keyed(&graph.feature_published, id, Clone::clone),
        input_published: graph.input_published.clone(),
    }
}

fn exists(document: &PartDocument, kind: RenameKind, name: &str) -> bool {
    match kind {
        RenameKind::Feature => document.features.iter().any(|f| f.id.0 == name),
        RenameKind::Input => document.inputs.iter().any(|i| i.name.as_str() == name),
        RenameKind::Region => document.regions.iter().any(|r| r.name.as_str() == name),
    }
}

/// Renames the part's `kind` named `from` to `to`, with every reference to it in the file.
/// Refuses a name that is taken or isn't a name, with the reason.
pub fn rename_part_item(
    source: &str,
    path: &str,
    kind: RenameKind,
    from: &str,
    to: &str,
) -> Result<SourceChange, EditFailure> {
    let noun = kind.noun();
    if let Some(why) = invalid_name(to) {
        return Err(failure(why));
    }
    let parsed = parse_part(path, source);
    let (Some(document), true) = (parsed.document.clone(), parsed.diagnostics.is_empty()) else {
        return Err(EditFailure::Invalid(parsed.diagnostics));
    };
    if !exists(&document, kind, from) {
        return Err(failure(format!("the part has no {noun} `{from}`")));
    }
    if from == to {
        return Err(failure(format!("the {noun} is already called `{to}`")));
    }
    if exists(&document, kind, to) {
        return Err(failure(format!(
            "the part already has a {noun} `{to}`; pick another name"
        )));
    }
    let new = renamed(&document, kind, from, to)?;
    if kind == RenameKind::Feature {
        // Every dependency on the feature follows it: none is left on the old name.
        let before = graph_renamed(&DependencyGraph::analyse(&document), from, to);
        if DependencyGraph::analyse(&new) != before {
            return Err(failure(format!(
                "`{from}` is named in a form rename can't follow yet; rename it in the source"
            )));
        }
    }
    let site = |name: &str| format!("{}.{name}", kind.noun());
    let renames = vec![(site(from), site(to))];
    diff_source(&PART, path, source, &new, &parsed.uses, &renames)
}

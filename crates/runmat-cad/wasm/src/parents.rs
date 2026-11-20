//! `part.parents-children`: what the selected feature or sketch is built on (its parents: the
//! features, sketches and inputs it reads) and what is built on it (its children: the features,
//! regions, publishes and part outputs that read it), from the compiler's dependency analysis
//! ([`runmat_cad::regen::deps::DependencyGraph`]). A query: nothing in the file changes.
//!
//! `part.show-references` runs the same query, and on an input or a region lists what uses it
//! (and what a region selects from).
//!
//! Rows are `Parent` or `Child`, the item's name and its kind, in history order; each names its
//! Outline item, so selecting a row selects it. `data` holds the same lists for agents.

use crate::analyze::{Analysis, Row};
use runmat_cad::ids::FeatureId;
use runmat_cad::model::PartDocument;
use runmat_cad::regen::deps::DependencyGraph;
use runmat_cad::syntax::parse_part;
use serde_json::{json, Value};

/// The kind word of feature `id` (`Extrude`, `Sketch`).
fn kind_of(document: &PartDocument, id: &FeatureId) -> String {
    let word = document
        .feature(id)
        .and_then(|f| serde_json::to_value(&f.kind).ok())
        .and_then(|k| k.get("feature").and_then(Value::as_str).map(str::to_owned))
        .unwrap_or_else(|| "feature".into());
    let spaced = word.replace('_', " ");
    let mut chars = spaced.chars();
    chars
        .next()
        .map(|c| c.to_ascii_uppercase().to_string() + chars.as_str())
        .unwrap_or_default()
}

fn row(relation: &str, name: &str, kind: &str, item: String) -> Row {
    Row {
        cells: vec![relation.to_owned(), name.to_owned(), kind.to_owned()],
        items: vec![item],
        ..Row::default()
    }
}

fn parsed(path: &str, text: &str) -> Result<PartDocument, String> {
    let parsed = parse_part(path, text);
    parsed.document.ok_or_else(|| {
        format!(
            "{path} has errors: {}",
            parsed
                .diagnostics
                .first()
                .map_or("", |d| d.message.as_str())
        )
    })
}

/// What an input or a region uses and what uses it (Show references on those items).
fn item_references(path: &str, text: &str, item: &str) -> Result<Analysis, String> {
    let document = parsed(path, text)?;
    let graph = DependencyGraph::analyse(&document);
    let mut rows = Vec::new();
    let mut parents: Vec<String> = Vec::new();
    let mut children: Vec<String> = Vec::new();
    let name = if let Some(input) = item.strip_prefix("input:") {
        if !document.inputs.iter().any(|i| i.name.as_str() == input) {
            return Err(format!("the part has no input `{input}`"));
        }
        for f in &document.features {
            let reads = graph
                .feature_inputs
                .get(&f.id)
                .is_some_and(|s| s.iter().any(|i| i.as_str() == input));
            if reads {
                rows.push(row(
                    "Child",
                    f.id.as_str(),
                    &kind_of(&document, &f.id),
                    format!("feature:{}", f.id),
                ));
                children.push(f.id.to_string());
            }
        }
        input
    } else if let Some(region) = item.strip_prefix("region:") {
        if !document.regions.iter().any(|r| r.name.as_str() == region) {
            return Err(format!("the part has no region `{region}`"));
        }
        let deps = graph
            .region_deps
            .iter()
            .find(|(n, _)| n.as_str() == region)
            .map(|(_, d)| d.clone())
            .unwrap_or_default();
        for f in document.features.iter().filter(|f| deps.contains(&f.id)) {
            rows.push(row(
                "Parent",
                f.id.as_str(),
                &kind_of(&document, &f.id),
                format!("feature:{}", f.id),
            ));
            parents.push(f.id.to_string());
        }
        for p in &document.publishes {
            let target = serde_json::to_value(&p.target).unwrap_or(Value::Null);
            if target.get("region").and_then(Value::as_str) == Some(region) {
                rows.push(row(
                    "Child",
                    p.name.as_str(),
                    "Publish",
                    format!("publish:{}", p.name),
                ));
                children.push(p.name.to_string());
            }
        }
        region
    } else {
        return Err("select one feature, sketch, input or region".into());
    };
    let count =
        |n: usize, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
    Ok(Analysis {
        title: format!("References of {name}"),
        summary: format!(
            "{name} uses {} and is used by {}",
            count(parents.len(), "item", "items"),
            count(children.len(), "item", "items")
        ),
        columns: vec!["Relation".into(), "Item".into(), "Kind".into()],
        rows,
        data: json!({ "item": name, "parents": parents, "children": children }),
        ..Analysis::default()
    })
}

/// Parents and children of the one feature or sketch in `selection`; for Show references, an
/// input's or a region's too.
pub fn parents_children(path: &str, text: &str, selection: &[String]) -> Result<Analysis, String> {
    if let [one] = selection {
        if one.starts_with("input:") || one.starts_with("region:") {
            return item_references(path, text, one);
        }
    }
    let selected: Vec<&str> = selection
        .iter()
        .filter_map(|s| {
            s.strip_prefix("feature:")
                .or_else(|| s.strip_prefix("sketch:"))
        })
        .collect();
    let [id] = selected.as_slice() else {
        return Err("select one feature or sketch to list its parents and children".into());
    };
    let document = parsed(path, text)?;
    let id = FeatureId::new(*id);
    if document.feature(&id).is_none() {
        return Err(format!("the part has no feature `{id}`"));
    }
    let graph = DependencyGraph::analyse(&document);
    let in_order = |set: &dyn Fn(&FeatureId) -> bool| -> Vec<FeatureId> {
        document
            .features
            .iter()
            .map(|f| &f.id)
            .filter(|f| set(f))
            .cloned()
            .collect()
    };
    let parents = in_order(&|f| graph.feature_deps.get(&id).is_some_and(|d| d.contains(f)));
    let inputs: Vec<String> = graph
        .feature_inputs
        .get(&id)
        .map(|s| s.iter().map(|i| i.as_str().to_owned()).collect())
        .unwrap_or_default();
    let children = in_order(&|f| graph.feature_deps.get(f).is_some_and(|d| d.contains(&id)));
    fn named<K: std::fmt::Display>(
        deps: &std::collections::BTreeMap<K, std::collections::BTreeSet<FeatureId>>,
        id: &FeatureId,
    ) -> Vec<String> {
        deps.iter()
            .filter(|(_, d)| d.contains(id))
            .map(|(name, _)| name.to_string())
            .collect()
    }
    let regions = named(&graph.region_deps, &id);
    let publishes = named(&graph.publish_deps, &id);
    let outputs = named(&graph.part_deps, &id);

    let mut rows = Vec::new();
    for p in &parents {
        rows.push(row(
            "Parent",
            p.as_str(),
            &kind_of(&document, p),
            format!("feature:{p}"),
        ));
    }
    for i in &inputs {
        rows.push(row("Parent", i, "Input", format!("input:{i}")));
    }
    for c in &children {
        rows.push(row(
            "Child",
            c.as_str(),
            &kind_of(&document, c),
            format!("feature:{c}"),
        ));
    }
    for r in &regions {
        rows.push(row("Child", r, "Region", format!("region:{r}")));
    }
    for p in &publishes {
        rows.push(row("Child", p, "Publish", format!("publish:{p}")));
    }
    for o in &outputs {
        rows.push(row("Child", o, "Part", format!("part:{o}")));
    }
    let count =
        |n: usize, one: &str, many: &str| format!("{n} {}", if n == 1 { one } else { many });
    let parent_count = parents.len() + inputs.len();
    let child_count = children.len() + regions.len() + publishes.len() + outputs.len();
    Ok(Analysis {
        title: format!("Parents and children of {id}"),
        summary: format!(
            "{id} is built on {} and has {}",
            count(parent_count, "parent", "parents"),
            count(child_count, "child", "children")
        ),
        columns: vec!["Relation".into(), "Item".into(), "Kind".into()],
        rows,
        problems: Vec::new(),
        data: json!({
            "item": id.as_str(),
            "parents": { "features": parents, "inputs": inputs },
            "children": { "features": children, "regions": regions, "publishes": publishes, "parts": outputs },
        }),
        ..Analysis::default()
    })
}

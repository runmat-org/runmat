//! The top assembly's configurations and display states for the Outline, the document bar's
//! Configuration menu and the Configurations form: each one's input values and the components it
//! suppresses (a configuration) or hides (a display state), which of them is active, and what
//! in it no longer exists.

use crate::assembly::model::{AssemblyDocument, RelPath};
use crate::assembly::resolve::ResolvedAssembly;
use crate::syntax::value::print_value;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

/// One configuration or display state of the top assembly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfigurationRow {
    pub name: String,
    /// `configuration` or `display_state`.
    pub kind: String,
    /// A configuration's input values as written, `width = 40 mm, depth = 20 mm`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub values: String,
    /// The components it suppresses (a configuration) or hides (a display state).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub components: Vec<String>,
    /// Their Outline IDs (`occurrence:<uuid>`), for those that exist.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<String>,
    /// The configuration the assembly is compiled in.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub active: bool,
    /// What it names that the assembly doesn't have (an input, a component).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem: Option<String>,
    /// Its values as the Configurations form takes them (`kind`, `name`, `values`, and
    /// `suppress` or `hidden` as component picks): editing it starts from these.
    pub form: Map<String, Value>,
}

/// The configuration rows, then the display-state rows, in document order.
pub fn configuration_rows(
    tree: &ResolvedAssembly,
    doc: Option<&AssemblyDocument>,
) -> Vec<ConfigurationRow> {
    let Some(doc) = doc else { return Vec::new() };
    let children: BTreeSet<&str> = doc.occurrences.iter().map(|o| o.name.as_str()).collect();
    let item = |name: &str| {
        tree.find(&RelPath::new(&[name]))
            .map(|n| format!("occurrence:{}", tree.node(n).uuid))
    };
    let missing = |names: &BTreeSet<String>| {
        let gone: Vec<&str> = names
            .iter()
            .map(String::as_str)
            .filter(|n| !children.contains(n))
            .collect();
        (!gone.is_empty()).then(|| format!("the assembly has no component {}", gone.join(", ")))
    };
    let mut rows = Vec::new();
    for c in &doc.configurations {
        let values = c
            .overrides
            .iter()
            .map(|(k, v)| format!("{k} = {}", print_value(v).unwrap_or_default()))
            .collect::<Vec<_>>()
            .join(", ");
        let unknown: Vec<&str> = c
            .overrides
            .keys()
            .map(|k| k.as_str())
            .filter(|k| !doc.inputs.iter().any(|i| i.name.as_str() == *k))
            .collect();
        let problem = if unknown.is_empty() {
            missing(&c.suppress)
        } else {
            Some(format!("the assembly has no input {}", unknown.join(", ")))
        };
        let items: Vec<String> = c.suppress.iter().filter_map(|n| item(n)).collect();
        rows.push(ConfigurationRow {
            name: c.name.as_str().to_owned(),
            kind: "configuration".into(),
            form: form("configuration", c.name.as_str(), &values, "suppress", &items),
            values,
            components: c.suppress.iter().cloned().collect(),
            items,
            active: tree.configuration.as_ref() == Some(&c.name),
            problem,
        });
    }
    for d in &doc.display_states {
        let items: Vec<String> = d.hidden.iter().filter_map(|n| item(n)).collect();
        rows.push(ConfigurationRow {
            name: d.name.clone(),
            kind: "display_state".into(),
            form: form("display_state", &d.name, "", "hidden", &items),
            values: String::new(),
            components: d.hidden.iter().cloned().collect(),
            items,
            active: false,
            problem: missing(&d.hidden),
        });
    }
    rows
}

fn form(kind: &str, name: &str, values: &str, field: &str, items: &[String]) -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("kind".into(), json!(kind));
    m.insert("name".into(), json!(name));
    if kind == "configuration" {
        m.insert("values".into(), json!(values));
    }
    m.insert(field.into(), json!(items));
    m
}

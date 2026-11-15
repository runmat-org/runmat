//! The top assembly's interfaces for the Outline and the canvas: its mate connectors (where each
//! sits after the solve, what it stands on, the mates that use it, and its values as the Mate
//! connector form takes them) and what it publishes.

use crate::assembly::model::{
    AnchorRef, AssemblyDocument, AssemblyPublishTarget, MateConnector, RotationExpr,
};
use crate::assembly::resolve::ResolvedAssembly;
use crate::ids::ItemUuid;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// One mate connector of the top assembly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConnectorRow {
    pub name: String,
    /// The occurrence it is fixed in; `None` for the assembly's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occurrence: Option<ItemUuid>,
    /// What it stands on, for people: `published top`, `origin`, `sketch s1.p1`, `frame`.
    pub on: String,
    /// The mates that join it.
    pub mates: Vec<String>,
    /// Its frame in assembly coordinates (m): origin, X and Z, when it resolves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame: Option<[[f64; 3]; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem: Option<String>,
    pub values: Map<String, Value>,
}

fn on(c: &MateConnector) -> String {
    match &c.anchor {
        AnchorRef::Origin => "origin".into(),
        AnchorRef::Published { name } => format!("published {name}"),
        AnchorRef::Explicit { .. } => "frame".into(),
        AnchorRef::Sketch { sketch, entity, .. } => format!("sketch {sketch}.{entity}"),
    }
}

/// The Mate connector form's values for `c`: `name`, `flip`, the offset `x`, `y`, `z` (value
/// edits) and `angle` about its Z (a value edit; a literal when the rotation is a plain turn).
/// `on` stays empty: editing keeps what the connector stands on unless a new pick replaces it.
pub fn connector_form_values(c: &MateConnector) -> Map<String, Value> {
    let edit = |e: &crate::value::ValueExpr| -> Value {
        let text = crate::syntax::value::print_value(e).unwrap_or_default();
        json!({ "kind": "value", "value": e, "text": text, "scope": { "scope": "all" } })
    };
    let mut out = Map::new();
    out.insert("name".into(), Value::String(c.name.clone()));
    out.insert("flip".into(), Value::Bool(c.flip));
    if let Some(offset) = &c.offset {
        for (k, v) in ["x", "y", "z"].iter().zip(&offset.translation) {
            out.insert((*k).into(), edit(v));
        }
        if let RotationExpr::AxisAngle { axis, angle } = &offset.rotation {
            if axis[0].abs() < 1e-12 && axis[1].abs() < 1e-12 {
                out.insert("angle".into(), edit(angle));
            }
        }
    }
    out
}

pub fn connector_rows(tree: &ResolvedAssembly, doc: Option<&AssemblyDocument>) -> Vec<ConnectorRow> {
    let Some(doc) = doc else { return Vec::new() };
    doc.connectors
        .iter()
        .map(|c| {
            let placed = tree.connectors.iter().find(|r| r.name == c.name);
            let frame = placed.and_then(|r| r.world).map(|w| {
                let col = |k: usize| [w.rotation[0][k], w.rotation[1][k], w.rotation[2][k]];
                [w.translation, col(0), col(2)]
            });
            ConnectorRow {
                name: c.name.clone(),
                occurrence: placed
                    .filter(|r| r.host != 0 && r.world.is_some())
                    .map(|r| tree.node(r.host).uuid),
                on: on(c),
                mates: doc
                    .mates
                    .iter()
                    .filter(|m| m.connectors().contains(&c.name.as_str()))
                    .map(|m| m.name.clone())
                    .collect(),
                frame,
                problem: placed.and_then(|r| r.problem.clone()),
                values: connector_form_values(c),
            }
        })
        .collect()
}

/// One `publish` of the top assembly.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PublishRow {
    pub name: String,
    /// `connector`, `occurrence` or `envelope`.
    pub kind: String,
    /// What it names: the connector's or envelope's name, or the occurrence's path.
    pub target: String,
    /// The Outline ID of what it names (`connector:<name>`, `occurrence:<uuid>`), when it exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub problem: Option<String>,
}

pub fn publish_rows(tree: &ResolvedAssembly, doc: Option<&AssemblyDocument>) -> Vec<PublishRow> {
    let Some(doc) = doc else { return Vec::new() };
    doc.publishes
        .iter()
        .map(|p| {
            let (kind, target, item, problem) = match &p.target {
                AssemblyPublishTarget::Connector { connector } => {
                    let found = doc.connector(connector).is_some();
                    (
                        "connector",
                        connector.clone(),
                        found.then(|| format!("connector:{connector}")),
                        (!found).then(|| format!("the connector `{connector}` is gone")),
                    )
                }
                AssemblyPublishTarget::Envelope { envelope } => {
                    let found = doc.envelopes.iter().any(|e| &e.name == envelope);
                    (
                        "envelope",
                        envelope.clone(),
                        None,
                        (!found).then(|| format!("the envelope `{envelope}` is gone")),
                    )
                }
                AssemblyPublishTarget::Occurrence { path } => {
                    let node = tree.find(path);
                    (
                        "occurrence",
                        path.to_string(),
                        node.map(|n| format!("occurrence:{}", tree.node(n).uuid)),
                        node.is_none()
                            .then(|| format!("`{path}` isn't in this configuration")),
                    )
                }
            };
            PublishRow {
                name: p.name.to_string(),
                kind: kind.into(),
                target,
                item,
                problem,
            }
        })
        .collect()
}

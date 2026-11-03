//! The JSON projection: a deterministic view of the part model plus resolved references, for the
//! server index and agents.
//!
//! Every list is in a stable order (document order for authored items, sorted for derived ones)
//! and every map is a `BTreeMap`, so identical inputs give identical bytes. Nothing per-run
//! (body positions, entity indices) appears; entities are given by their structured names.

use crate::diagnostics::Diagnostic;
use crate::ids::{
    FeatureId, FilePath, InputName, ItemUuid, MaterialRef, PartName, PublishName, RegionName,
};
use crate::model::items::{Link, Property};
use crate::model::{PartDocument, VersionHeader};
use crate::refs::lock::{ItemPath, LockSection};
use crate::refs::{
    EntityName, Reference, ResolutionClass, ResolutionStats, ResolveOutcome, TopoKind,
};
use crate::regen::engine::{FeatureStatus, ItemOutcome, ReferenceRecord, RegenResult};
use crate::regen::hash::CacheKey;
use crate::units::Dimension;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Bumped on any change to the projection's shape.
pub const INDEX_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartIndex {
    pub format_version: u32,
    pub file: FilePath,
    pub header: VersionHeader,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<String>,
    pub final_key: CacheKey,
    pub inputs: Vec<InputRow>,
    pub features: Vec<FeatureRow>,
    pub bodies: Vec<BodyRow>,
    pub parts: Vec<PartRow>,
    pub regions: Vec<RegionRow>,
    pub publishes: Vec<PublishRow>,
    pub links: Vec<Link>,
    pub diagnostics: Vec<Diagnostic>,
    pub stats: ResolutionStats,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputRow {
    pub name: InputName,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub si: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension: Option<Dimension>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceRow {
    pub param: String,
    /// For people and agents; never parsed back.
    pub display: String,
    pub reference: Reference,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<ResolutionClass>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<Vec<EntityName>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureRow {
    pub id: FeatureId,
    pub kind: String,
    pub status: FeatureStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<ItemUuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<CacheKey>,
    pub references: Vec<ReferenceRow>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyRow {
    pub names: Vec<EntityName>,
    pub faces: Vec<Vec<EntityName>>,
    pub edge_count: usize,
    pub vertex_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartRow {
    pub name: PartName,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<ItemUuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part_number: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub material: Option<MaterialRef>,
    pub properties: Vec<Property>,
    pub body: Vec<Vec<EntityName>>,
    pub ok: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegionRow {
    pub name: RegionName,
    pub kind: TopoKind,
    pub entities: Vec<Vec<EntityName>>,
    pub ok: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PublishRow {
    pub name: PublishName,
    pub auto: bool,
    pub ok: bool,
    pub entities: Vec<Vec<EntityName>>,
}

fn reference_rows(records: &[ReferenceRecord]) -> Vec<ReferenceRow> {
    records
        .iter()
        .map(|r| {
            let (status, class, entities) = match &r.outcome {
                ResolveOutcome::Resolved(res) => {
                    let entities = match &res.target {
                        crate::refs::Resolved::Foreign { target } => {
                            target.entities.iter().map(|e| e.names.clone()).collect()
                        }
                        crate::refs::Resolved::Local { .. } => Vec::new(),
                    };
                    ("resolved".to_owned(), Some(res.class), entities)
                }
                ResolveOutcome::Waiting(_) => ("waiting".to_owned(), None, Vec::new()),
                ResolveOutcome::Failed(p) => {
                    let v = serde_json::to_value(p).expect("problem serializes");
                    (
                        format!(
                            "failed:{}",
                            v.get("kind").and_then(|x| x.as_str()).unwrap_or("unknown")
                        ),
                        None,
                        Vec::new(),
                    )
                }
            };
            ReferenceRow {
                param: r.param.clone(),
                display: r.reference.to_string(),
                reference: r.reference.clone(),
                status,
                class,
                entities,
            }
        })
        .collect()
}

fn item_entities(o: Option<&ItemOutcome>) -> (Vec<Vec<EntityName>>, bool) {
    match o {
        Some(o) => (o.entities.clone(), o.ok),
        None => (Vec::new(), false),
    }
}

pub fn build_index(
    doc: &PartDocument,
    result: &RegenResult,
    lock: Option<&LockSection>,
) -> PartIndex {
    let uuid = |p: ItemPath| lock.and_then(|l| l.lookup(&p)).map(|l| l.uuid);
    let inputs = doc
        .inputs
        .iter()
        .map(|i| {
            let q = result.values.inputs.get(&i.name);
            InputRow {
                name: i.name.clone(),
                kind: i.kind.0.clone(),
                si: q.map(|q| q.si),
                dimension: q.map(|q| q.dimension),
                error: result
                    .values
                    .failed
                    .get(&format!("input.{}", i.name))
                    .map(|e| e.to_string()),
            }
        })
        .collect();
    let features = result
        .features
        .iter()
        .map(|f| FeatureRow {
            id: f.id.clone(),
            kind: f.kind.clone(),
            status: f.status.clone(),
            uuid: uuid(ItemPath::feature(&doc.path, &f.id)),
            cache_key: f.cache_key.clone(),
            references: reference_rows(&f.references),
        })
        .collect();
    let bodies = result
        .state
        .bodies
        .iter()
        .map(|b| BodyRow {
            names: b.names.body.iter().cloned().collect(),
            faces: b
                .names
                .faces
                .iter()
                .map(|s| s.iter().cloned().collect())
                .collect(),
            edge_count: b.topology.edges.len(),
            vertex_count: b.topology.vertices.len(),
        })
        .collect();
    let parts = doc
        .parts
        .iter()
        .map(|p| {
            let (body, ok) = item_entities(result.parts.get(&p.name));
            PartRow {
                name: p.name.clone(),
                uuid: uuid(ItemPath::part(&doc.path, &p.name)),
                part_number: p.part_number.clone(),
                material: p.material.clone(),
                properties: p.properties.clone(),
                body,
                ok,
            }
        })
        .collect();
    let regions = doc
        .regions
        .iter()
        .map(|r| {
            let (entities, ok) = item_entities(result.regions.get(&r.name));
            RegionRow {
                name: r.name.clone(),
                kind: r.kind,
                entities,
                ok,
            }
        })
        .collect();
    let by_name: BTreeMap<&PublishName, _> = result.publishes.iter().collect();
    let publishes = doc
        .publishes
        .iter()
        .map(|p| {
            let r = by_name.get(&p.name);
            let (entities, ok) = match r {
                Some(Ok(t)) => (t.entities.iter().map(|e| e.names.clone()).collect(), true),
                _ => (Vec::new(), false),
            };
            PublishRow {
                name: p.name.clone(),
                auto: p.auto,
                ok,
                entities,
            }
        })
        .collect();
    PartIndex {
        format_version: INDEX_FORMAT_VERSION,
        file: doc.path.clone(),
        header: doc.header.clone(),
        configuration: result.configuration.as_ref().map(|c| c.0.clone()),
        final_key: result.final_key.clone(),
        inputs,
        features,
        bodies,
        parts,
        regions,
        publishes,
        links: doc.links.clone(),
        diagnostics: result.diagnostics.diagnostics.clone(),
        stats: result.stats.clone(),
    }
}

pub fn to_json(index: &PartIndex) -> String {
    serde_json::to_string_pretty(index).expect("index serializes")
}

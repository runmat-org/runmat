//! What each feature's references resolve to in the regenerated part: the faces, edges and
//! vertices with their kind and size, for the editor's reference hovers.
//!
//! References are resolved again against the final state (after the whole history), so the
//! facts describe the model the viewport shows. A face a later feature consumed has no facts.

use crate::model::items::Cardinality;
use crate::model::PartDocument;
use crate::refs::history::{EdgeCurve, FaceGeom, SurfaceKind};
use crate::refs::name::TopoKind;
use crate::refs::resolve::{ResolveOutcome, Resolver};
use crate::regen::engine::RegenResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Most entities described per reference.
const MOST: usize = 6;

/// One resolved entity, in SI units (m, m², rad).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "entity")]
pub enum EntityFacts {
    Face {
        /// `planar`, `cylindrical` or `other`.
        surface: String,
        area: f64,
        centroid: [f64; 3],
        #[serde(default, skip_serializing_if = "Option::is_none")]
        normal: Option<[f64; 3]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        axis: Option<[f64; 3]>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        radius: Option<f64>,
    },
    Edge {
        /// `line`, `circle` or `other`.
        curve: String,
        length: f64,
        start: [f64; 3],
        end: [f64; 3],
        #[serde(default, skip_serializing_if = "Option::is_none")]
        radius: Option<f64>,
    },
    Vertex {
        point: [f64; 3],
    },
    Body {
        faces: usize,
        edges: usize,
    },
}

/// The facts for every feature reference, keyed `<feature>/<param>`, plus how many entities
/// each resolved to (the facts list stops at a few).
pub fn reference_facts(
    document: &PartDocument,
    result: &RegenResult,
) -> BTreeMap<String, (usize, Vec<EntityFacts>)> {
    let state = &result.state;
    let resolver = Resolver {
        file: &document.path,
        bodies: state.views(),
        features: &state.table,
        consumer: None,
        burials: &state.burials,
        publishes: &document.publishes,
        lock: None,
        catalog: None,
        policy: Default::default(),
        names: Some(&state.names),
    };
    let mut out = BTreeMap::new();
    for feature in &result.features {
        for record in &feature.references {
            let ResolveOutcome::Resolved(resolution) =
                resolver.resolve(&record.reference, None, Cardinality::Any)
            else {
                continue;
            };
            let entities = resolution.entities();
            let facts = entities
                .iter()
                .take(MOST)
                .filter_map(|h| {
                    let body = state.bodies.get(h.body as usize)?;
                    let topology = &body.topology;
                    Some(match h.kind {
                        TopoKind::Face => face(topology.faces.get(h.index as usize)?.geom),
                        TopoKind::Edge => {
                            let g = topology.edges.get(h.index as usize)?.geom;
                            let (curve, radius) = match g.curve {
                                EdgeCurve::Line => ("line", None),
                                EdgeCurve::Circle { radius, .. } => ("circle", Some(radius)),
                                EdgeCurve::Other => ("other", None),
                            };
                            EntityFacts::Edge {
                                curve: curve.into(),
                                length: g.length,
                                start: g.start,
                                end: g.end,
                                radius,
                            }
                        }
                        TopoKind::Vertex => EntityFacts::Vertex {
                            point: topology.vertices.get(h.index as usize)?.point,
                        },
                        TopoKind::Body => EntityFacts::Body {
                            faces: topology.faces.len(),
                            edges: topology.edges.len(),
                        },
                    })
                })
                .collect();
            out.insert(
                format!("{}/{}", feature.id.0, record.param),
                (entities.len(), facts),
            );
        }
    }
    out
}

fn face(g: FaceGeom) -> EntityFacts {
    let (surface, normal, axis, radius) = match g.surface {
        SurfaceKind::Plane { normal } => ("planar", Some(normal), None, None),
        SurfaceKind::Cylinder {
            axis_direction,
            radius,
            ..
        } => ("cylindrical", None, Some(axis_direction), Some(radius)),
        SurfaceKind::Other => ("other", None, None, None),
    };
    EntityFacts::Face {
        surface: surface.into(),
        area: g.area,
        centroid: g.centroid,
        normal,
        axis,
        radius,
    }
}

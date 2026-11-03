//! Layer 2: carry names forward through an operation's history.
//!
//! Given the operands' names and one [`OpHistory`], name every output face and body:
//! - a face continuing input faces answers to all of their names (a merge keeps every name);
//! - several output faces continuing one input face are its pieces and share its names;
//! - a generated face is named by feature, role and the tag translated into this feature's terms;
//! - a body continuing one input body keeps its names; when several bodies continue it, the
//!   names go to the piece whose face names overlap the original's most (ties keep all, and a
//!   reference to the body then reports the ambiguity rather than picking one).
//!
//! Names of input faces and bodies with no image are buried with the feature that removed them,
//! so a later reference reports "consumed by X" instead of "not found".

use crate::ids::FeatureId;
use crate::refs::history::{
    EntityIndex, HistorySource, KernelTag, OpHistory, OperandEntity, Topology,
};
use crate::refs::name::{EntityName, GenerationName, Role, SourceTag, TopoKind};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// A body's names: the body's own, and each face's by index.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyNames {
    pub body: BTreeSet<EntityName>,
    pub faces: Vec<BTreeSet<EntityName>>,
}

impl BodyNames {
    /// The display name of a face: the least of its names.
    pub fn primary(&self, face: EntityIndex) -> Option<&EntityName> {
        self.faces.get(face as usize).and_then(|s| s.iter().next())
    }

    pub fn all_face_names(&self) -> BTreeSet<&EntityName> {
        self.faces.iter().flatten().collect()
    }
}

/// The relational name of an edge: the primary names of its two faces, sorted.
pub fn edge_name(
    names: &BodyNames,
    topology: &Topology,
    edge: EntityIndex,
) -> Option<[EntityName; 2]> {
    let e = topology.edges.get(edge as usize)?;
    let a = names.primary(e.faces[0])?.clone();
    let b = names.primary(e.faces[1])?.clone();
    Some(if a <= b { [a, b] } else { [b, a] })
}

/// What the naming step needs to know about the feature that ran.
#[derive(Clone, Debug)]
pub struct NamingContext {
    pub feature: FeatureId,
    /// Sketch feature per profile index, per seed (`seed: None` uses index 0).
    pub profiles: Vec<Vec<FeatureId>>,
    /// For patterns and mirrors of features: the seed feature per seed index.
    pub seeds: Vec<FeatureId>,
}

impl NamingContext {
    pub fn simple(feature: &FeatureId, profiles: Vec<FeatureId>) -> Self {
        Self {
            feature: feature.clone(),
            profiles: vec![profiles],
            seeds: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "error")]
pub enum NamingError {
    #[error("history has no record for face {face} of output body {body}")]
    MissingFaceRecord { body: u32, face: u32 },
    #[error("history has no record for output body {body}")]
    MissingBodyRecord { body: u32 },
    #[error("history names operand entity {operand}:{entity}, which does not exist")]
    BadOperand { operand: u32, entity: u32 },
    #[error("history names profile {profile}, which the operation did not have")]
    BadProfile { profile: u32 },
    #[error("history names seed {seed}, which the operation did not have")]
    BadSeed { seed: u32 },
    #[error("an output face has no source at all")]
    Unsourced,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NamingOutput {
    pub bodies: Vec<BodyNames>,
    /// Names that no output entity carries any more, by kind.
    pub buried: Vec<(TopoKind, EntityName)>,
}

pub struct OperandView<'a> {
    pub names: &'a BodyNames,
    pub topology: &'a Topology,
}

pub fn name_outputs(
    ctx: &NamingContext,
    operands: &[OperandView<'_>],
    outputs: &[&Topology],
    history: &OpHistory,
) -> Result<NamingOutput, NamingError> {
    let face_records: BTreeMap<(u32, u32), &crate::refs::history::FaceRecord> = history
        .faces
        .iter()
        .map(|r| ((r.body, r.face), r))
        .collect();
    let body_records: BTreeMap<u32, &crate::refs::history::BodyRecord> =
        history.bodies.iter().map(|r| (r.body, r)).collect();

    let operand_face = |oe: &OperandEntity| -> Result<&BTreeSet<EntityName>, NamingError> {
        operands
            .get(oe.operand as usize)
            .and_then(|o| o.names.faces.get(oe.entity as usize))
            .ok_or(NamingError::BadOperand {
                operand: oe.operand,
                entity: oe.entity,
            })
    };

    let mut bodies = Vec::with_capacity(outputs.len());
    for (b, topo) in outputs.iter().enumerate() {
        let mut faces = Vec::with_capacity(topo.faces.len());
        for f in 0..topo.faces.len() {
            let rec =
                face_records
                    .get(&(b as u32, f as u32))
                    .ok_or(NamingError::MissingFaceRecord {
                        body: b as u32,
                        face: f as u32,
                    })?;
            if rec.sources.is_empty() {
                return Err(NamingError::Unsourced);
            }
            let mut names = BTreeSet::new();
            for src in &rec.sources {
                match src {
                    HistorySource::Input { from } => {
                        names.extend(operand_face(from)?.iter().cloned())
                    }
                    HistorySource::Generated {
                        role,
                        tag,
                        ordinal,
                        seed,
                        instance,
                    } => {
                        let (maker, profiles) = match seed {
                            Some(s) => (
                                ctx.seeds
                                    .get(*s as usize)
                                    .ok_or(NamingError::BadSeed { seed: *s })?
                                    .clone(),
                                ctx.profiles.get(*s as usize).cloned().unwrap_or_default(),
                            ),
                            None => (
                                ctx.feature.clone(),
                                ctx.profiles.first().cloned().unwrap_or_default(),
                            ),
                        };
                        let tag = translate_tag(tag, &profiles, operands)?;
                        let base = EntityName::Generated(GenerationName {
                            feature: maker,
                            role: *role,
                            tag,
                            ordinal: *ordinal,
                        });
                        let name = match instance {
                            Some(k) => EntityName::Copy {
                                base: Box::new(base),
                                pattern: ctx.feature.clone(),
                                instance: k.clone(),
                            },
                            None => base,
                        };
                        names.insert(name);
                    }
                    HistorySource::Copy { from, instance } => {
                        for n in operand_face(from)? {
                            names.insert(EntityName::Copy {
                                base: Box::new(n.clone()),
                                pattern: ctx.feature.clone(),
                                instance: instance.clone(),
                            });
                        }
                    }
                }
            }
            faces.push(names);
        }
        bodies.push(BodyNames {
            body: BTreeSet::new(),
            faces,
        });
    }

    // Bodies: generated and copied names first, then inheritance by lineage.
    let mut claims: BTreeMap<u32, Vec<u32>> = BTreeMap::new(); // operand body -> output bodies
    for b in 0..outputs.len() as u32 {
        let rec = body_records
            .get(&b)
            .ok_or(NamingError::MissingBodyRecord { body: b })?;
        for src in &rec.sources {
            match src {
                HistorySource::Input { from } => {
                    if operands.get(from.operand as usize).is_none() {
                        return Err(NamingError::BadOperand {
                            operand: from.operand,
                            entity: 0,
                        });
                    }
                    claims.entry(from.operand).or_default().push(b);
                }
                HistorySource::Generated {
                    role,
                    tag,
                    ordinal,
                    seed,
                    instance,
                } => {
                    let maker = match seed {
                        Some(s) => ctx
                            .seeds
                            .get(*s as usize)
                            .ok_or(NamingError::BadSeed { seed: *s })?
                            .clone(),
                        None => ctx.feature.clone(),
                    };
                    let profiles = ctx.profiles.first().cloned().unwrap_or_default();
                    let tag = translate_tag(tag, &profiles, operands)?;
                    let base = EntityName::Generated(GenerationName {
                        feature: maker,
                        role: *role,
                        tag,
                        ordinal: *ordinal,
                    });
                    let name = match instance {
                        Some(k) => EntityName::Copy {
                            base: Box::new(base),
                            pattern: ctx.feature.clone(),
                            instance: k.clone(),
                        },
                        None => base,
                    };
                    bodies[b as usize].body.insert(name);
                }
                HistorySource::Copy { from, instance } => {
                    let o = operands
                        .get(from.operand as usize)
                        .ok_or(NamingError::BadOperand {
                            operand: from.operand,
                            entity: 0,
                        })?;
                    for n in &o.names.body {
                        bodies[b as usize].body.insert(EntityName::Copy {
                            base: Box::new(n.clone()),
                            pattern: ctx.feature.clone(),
                            instance: instance.clone(),
                        });
                    }
                }
            }
        }
    }
    for (operand, claimants) in &claims {
        let original = &operands[*operand as usize].names;
        let winners: Vec<u32> = if claimants.len() == 1 {
            claimants.clone()
        } else {
            let orig: BTreeSet<&EntityName> = original.all_face_names();
            let score = |b: u32| -> (usize, usize) {
                let piece = bodies[b as usize].all_face_names();
                let inter = piece.intersection(&orig).count();
                let union = piece.union(&orig).count().max(1);
                // Compare inter/union as a fraction without floating point.
                (inter, union)
            };
            let mut best: Vec<u32> = Vec::new();
            let mut best_score = (0usize, 1usize);
            for &b in claimants {
                let s = score(b);
                let ord = (s.0 * best_score.1).cmp(&(best_score.0 * s.1));
                if best.is_empty() || ord == std::cmp::Ordering::Greater {
                    best = vec![b];
                    best_score = s;
                } else if ord == std::cmp::Ordering::Equal {
                    best.push(b);
                }
            }
            best
        };
        for b in winners {
            let inherited: Vec<EntityName> = original.body.iter().cloned().collect();
            bodies[b as usize].body.extend(inherited);
        }
    }

    // Burials: names of deleted inputs that no output carries.
    let mut live: BTreeSet<&EntityName> = BTreeSet::new();
    for b in &bodies {
        live.extend(b.faces.iter().flatten());
    }
    let mut buried = Vec::new();
    for d in &history.deleted_faces {
        for n in operand_face(d)? {
            if !live.contains(n) {
                buried.push((TopoKind::Face, n.clone()));
            }
        }
    }
    let live_bodies: BTreeSet<&EntityName> = bodies.iter().flat_map(|b| b.body.iter()).collect();
    for d in &history.deleted_bodies {
        let o = operands.get(*d as usize).ok_or(NamingError::BadOperand {
            operand: *d,
            entity: 0,
        })?;
        for n in &o.names.body {
            if !live_bodies.contains(n) {
                buried.push((TopoKind::Body, n.clone()));
            }
        }
        for n in o.names.faces.iter().flatten() {
            if !live.contains(n) {
                buried.push((TopoKind::Face, n.clone()));
            }
        }
    }
    buried.sort();
    buried.dedup();
    Ok(NamingOutput { bodies, buried })
}

fn translate_tag(
    tag: &KernelTag,
    profiles: &[FeatureId],
    operands: &[OperandView<'_>],
) -> Result<SourceTag, NamingError> {
    Ok(match tag {
        KernelTag::None => SourceTag::None,
        KernelTag::Curve { profile, curve } => SourceTag::Curve {
            sketch: profiles
                .get(*profile as usize)
                .ok_or(NamingError::BadProfile { profile: *profile })?
                .clone(),
            entity: curve.clone(),
        },
        KernelTag::Key { key } => SourceTag::Key { key: key.clone() },
        KernelTag::Side { side } => SourceTag::Side { side: *side },
        KernelTag::InputEdge { edge } => {
            let o = operands
                .get(edge.operand as usize)
                .ok_or(NamingError::BadOperand {
                    operand: edge.operand,
                    entity: edge.entity,
                })?;
            let between =
                edge_name(o.names, o.topology, edge.entity).ok_or(NamingError::BadOperand {
                    operand: edge.operand,
                    entity: edge.entity,
                })?;
            SourceTag::Edge {
                between: Box::new(between),
            }
        }
        KernelTag::InputFace { face } => {
            let o = operands
                .get(face.operand as usize)
                .ok_or(NamingError::BadOperand {
                    operand: face.operand,
                    entity: face.entity,
                })?;
            let n = o
                .names
                .primary(face.entity)
                .ok_or(NamingError::BadOperand {
                    operand: face.operand,
                    entity: face.entity,
                })?;
            SourceTag::Face {
                face: Box::new(n.clone()),
            }
        }
        KernelTag::CadName { name } => SourceTag::Key { key: name.clone() },
    })
}

/// Names for an imported body: CAD-file names become [`EntityName::CadFile`]. Used instead of
/// generation names when the kernel tags faces with [`KernelTag::CadName`] under an import.
pub fn cad_names_for_import(import: &FeatureId, output: &mut NamingOutput) {
    for b in &mut output.bodies {
        for set in b.faces.iter_mut().chain(std::iter::once(&mut b.body)) {
            let converted: BTreeSet<EntityName> = set
                .iter()
                .map(|n| match n {
                    EntityName::Generated(g) if &g.feature == import => match &g.tag {
                        SourceTag::Key { key } => EntityName::CadFile {
                            import: import.clone(),
                            name: key.clone(),
                        },
                        _ => n.clone(),
                    },
                    _ => n.clone(),
                })
                .collect();
            *set = converted;
        }
    }
}

/// Whether a role is one whose faces replace an edge (so a lost edge can be traced to them).
pub fn role_replaces_edge(role: Role) -> bool {
    matches!(role, Role::Blend | Role::Chamfer)
}

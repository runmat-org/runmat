//! Layer 2: carry names forward through an operation's history.
//!
//! Given the operands' names and one [`OpHistory`], name every output face and body:
//! - a face continuing input faces answers to all of their names (a merge keeps every name);
//! - several output faces continuing one input face are its pieces and share its names;
//! - a generated face is named by feature, role and the tag translated into this feature's terms;
//! - a body continuing one input body keeps its names; when several bodies continue it, the
//!   names go to the piece most similar to the original by lineage. Similarity is the weighted
//!   Jaccard index over face-name tokens (each face contributes its names, which include the
//!   names of the faces it was split from, weighted by how much of that named surface it
//!   carries). A piece takes the slot only when it dominates every other piece by
//!   [`SLOT_MARGIN`], so a sliver never takes the slot and a near-even split never flips it
//!   silently: pieces within the margin tie, keep the names, and are reported ([`SlotTie`]); a
//!   reference to the body then reports the ambiguity, with repairs, rather than picking one.
//!   Each body also carries a MinHash [`BodySignature`] of its names, a fixed-size digest a
//!   pinned slot can store and compare when the original names are no longer at hand.
//! - faces that share their primary name with other faces (pieces of a split) get a bounded
//!   [`PieceQualifier`]: up to three neighbour names that tell the piece apart. It is derived
//!   per state, never carried forward, and is what a picker writes to select one piece.
//!
//! Every name is one level deep ([`crate::refs::name`]): a tag made from other names holds
//! their tokens.
//!
//! Names of input faces and bodies with no image are buried with the feature that removed them,
//! so a later reference reports "consumed by X" instead of "not found".

use crate::ids::FeatureId;
use crate::refs::history::{
    EntityIndex, HistorySource, KernelTag, OpHistory, OperandEntity, Topology,
};
use crate::refs::name::{
    EntityName, GenerationName, NameTable, NameToken, Role, SourceTag, TokenCollision, TopoKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// A body's names: the body's own, each face's by index, the lineage signature, and the
/// qualifiers that tell pieces of one split face apart.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyNames {
    pub body: BTreeSet<EntityName>,
    pub faces: Vec<BTreeSet<EntityName>>,
    /// Names of a surface body's boundary (laminar) edges, by edge index.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub boundaries: BTreeMap<EntityIndex, BTreeSet<EntityName>>,
    #[serde(default)]
    pub signature: BodySignature,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub qualifiers: BTreeMap<EntityIndex, PieceQualifier>,
}

/// Neighbour names that select one piece of a split face: the piece borders every face named
/// in `bounded_by` and none named in `not_bounded_by`. At most [`MAX_QUALIFIERS`] of each.
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PieceQualifier {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bounded_by: Vec<EntityName>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub not_bounded_by: Vec<EntityName>,
}

pub const MAX_QUALIFIERS: usize = 3;

// ------------------------------------------------------------------------- body signatures

/// Positions in a [`BodySignature`].
pub const SIGNATURE_LEN: usize = 128;

/// A MinHash signature over a body's face-name tokens: position `i` holds the least value of
/// hash function `i` over the set. The share of equal positions between two signatures
/// estimates the Jaccard similarity of the two sets. Fixed-size, platform-independent, and
/// storable beside a body slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BodySignature(pub Vec<u64>);

impl Default for BodySignature {
    fn default() -> Self {
        BodySignature(vec![u64::MAX; SIGNATURE_LEN])
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

impl BodySignature {
    pub fn of_tokens<'a>(tokens: impl IntoIterator<Item = &'a NameToken>) -> Self {
        let mut sig = vec![u64::MAX; SIGNATURE_LEN];
        for t in tokens {
            let lo = u64::from_be_bytes(t.0[..8].try_into().expect("8 bytes"));
            let hi = u64::from_be_bytes(t.0[8..].try_into().expect("8 bytes"));
            for (i, slot) in sig.iter_mut().enumerate() {
                let seed = splitmix64(0x5EED_0000_0000_0000 ^ i as u64);
                let h = splitmix64(lo ^ splitmix64(hi ^ seed));
                if h < *slot {
                    *slot = h;
                }
            }
        }
        BodySignature(sig)
    }

    pub fn of_names<'a>(names: impl IntoIterator<Item = &'a EntityName>) -> Self {
        let tokens: BTreeSet<NameToken> = names.into_iter().map(|n| n.token()).collect();
        Self::of_tokens(tokens.iter())
    }

    /// Positions where both signatures hold the same value (an empty set agrees with nothing).
    pub fn agreement(&self, other: &BodySignature) -> u32 {
        self.0
            .iter()
            .zip(&other.0)
            .filter(|(a, b)| a == b && **a != u64::MAX)
            .count() as u32
    }
}

/// Several pieces of one body were equally similar to it, so each kept the body's names.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlotTie {
    /// The operand body that was split.
    pub operand: u32,
    pub names: Vec<EntityName>,
    /// Output bodies that tied.
    pub bodies: Vec<u32>,
    /// The best piece's lineage similarity to the original (weighted Jaccard, 0..1).
    pub similarity: f64,
}

/// A piece takes a split body's slot only when its similarity exceeds every other piece's by
/// this factor; otherwise the pieces within it tie.
pub const SLOT_MARGIN: f64 = 2.0;

/// Each face-name token of a body, weighted by the area of the faces carrying it.
fn name_weights(names: &BodyNames, topo: &Topology) -> BTreeMap<NameToken, f64> {
    let mut w: BTreeMap<NameToken, f64> = BTreeMap::new();
    for (i, set) in names.faces.iter().enumerate() {
        let area = topo
            .faces
            .get(i)
            .map(|f| f.geom.area.max(0.0))
            .unwrap_or(0.0);
        // A face of no measured area still counts, so lineage alone can decide.
        let a = if area > 0.0 { area } else { 1e-12 };
        for n in set {
            *w.entry(n.token()).or_default() += a;
        }
    }
    w
}

/// Σ min / Σ max over the union of tokens: how much of the original's named surface the piece
/// carries, against how much it adds.
fn weighted_jaccard(a: &BTreeMap<NameToken, f64>, b: &BTreeMap<NameToken, f64>) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    let keys: BTreeSet<&NameToken> = a.keys().chain(b.keys()).collect();
    for k in keys {
        let x = a.get(k).copied().unwrap_or(0.0);
        let y = b.get(k).copied().unwrap_or(0.0);
        num += x.min(y);
        den += x.max(y);
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

impl BodyNames {
    /// Recompute the lineage signature from the face names.
    pub fn sign(&mut self) {
        self.signature = BodySignature::of_names(self.faces.iter().flatten());
    }

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
    if e.laminar {
        // A boundary edge: its face and its own boundary name.
        let a = names.primary(e.faces[0])?.clone();
        let b = names.boundaries.get(&edge)?.iter().next()?.clone();
        return Some(if a <= b { [a, b] } else { [b, a] });
    }
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
    /// Body slots that went to several equally similar pieces.
    pub slot_ties: Vec<SlotTie>,
}

impl NamingOutput {
    /// Record every name (and the names of buried entities) in `table`.
    pub fn intern_into(&self, table: &mut NameTable) -> Result<(), TokenCollision> {
        for b in &self.bodies {
            for n in b.body.iter().chain(b.faces.iter().flatten()) {
                table.intern(n)?;
            }
        }
        for (_, n) in &self.buried {
            table.intern(n)?;
        }
        Ok(())
    }
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
                            Some(k) => EntityName::copy(&base, &ctx.feature, k.clone()),
                            None => base,
                        };
                        names.insert(name);
                    }
                    HistorySource::Copy { from, instance } => {
                        for n in operand_face(from)? {
                            names.insert(EntityName::copy(n, &ctx.feature, instance.clone()));
                        }
                    }
                }
            }
            faces.push(names);
        }
        let mut boundaries: BTreeMap<EntityIndex, BTreeSet<EntityName>> = BTreeMap::new();
        for rec in history.boundaries.iter().filter(|r| r.body == b as u32) {
            let mut names = BTreeSet::new();
            for src in &rec.sources {
                match src {
                    HistorySource::Input { from } => {
                        let o = operands.get(from.operand as usize).ok_or(
                            NamingError::BadOperand {
                                operand: from.operand,
                                entity: from.entity,
                            },
                        )?;
                        names.extend(
                            o.names
                                .boundaries
                                .get(&from.entity)
                                .into_iter()
                                .flatten()
                                .cloned(),
                        );
                    }
                    HistorySource::Generated {
                        role, tag, ordinal, ..
                    } => {
                        let profiles = ctx.profiles.first().cloned().unwrap_or_default();
                        let tag = translate_tag(tag, &profiles, operands)?;
                        names.insert(EntityName::Generated(GenerationName {
                            feature: ctx.feature.clone(),
                            role: *role,
                            tag,
                            ordinal: *ordinal,
                        }));
                    }
                    HistorySource::Copy { from, instance } => {
                        if let Some(o) = operands.get(from.operand as usize) {
                            for n in o.names.boundaries.get(&from.entity).into_iter().flatten() {
                                names.insert(EntityName::copy(n, &ctx.feature, instance.clone()));
                            }
                        }
                    }
                }
            }
            if names.is_empty() {
                return Err(NamingError::Unsourced);
            }
            boundaries.insert(rec.edge, names);
        }
        for (ei, e) in topo.edges.iter().enumerate() {
            if e.laminar && !boundaries.contains_key(&(ei as u32)) {
                return Err(NamingError::Unsourced);
            }
        }
        let mut b = BodyNames {
            body: BTreeSet::new(),
            faces,
            boundaries,
            ..Default::default()
        };
        b.sign();
        bodies.push(b);
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
                        Some(k) => EntityName::copy(&base, &ctx.feature, k.clone()),
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
                        bodies[b as usize].body.insert(EntityName::copy(
                            n,
                            &ctx.feature,
                            instance.clone(),
                        ));
                    }
                }
            }
        }
    }
    let mut slot_ties = Vec::new();
    for (operand, claimants) in &claims {
        let original = &operands[*operand as usize].names;
        let winners: Vec<u32> = if claimants.len() == 1 {
            claimants.clone()
        } else {
            let orig = name_weights(original, operands[*operand as usize].topology);
            let mut scored: Vec<(f64, u32)> = claimants
                .iter()
                .map(|&b| {
                    let piece = name_weights(&bodies[b as usize], outputs[b as usize]);
                    (weighted_jaccard(&orig, &piece), b)
                })
                .collect();
            scored.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
            let best = scored[0].0;
            // A slot goes to one piece only when it clearly dominates; every piece within the
            // margin of the best is a tie.
            let tied: Vec<u32> = scored
                .iter()
                .filter(|(s, _)| *s * SLOT_MARGIN > best)
                .map(|(_, b)| *b)
                .collect();
            if tied.len() > 1 && !original.body.is_empty() {
                let mut bs = tied.clone();
                bs.sort();
                slot_ties.push(SlotTie {
                    operand: *operand,
                    names: original.body.iter().cloned().collect(),
                    bodies: bs,
                    similarity: best,
                });
            }
            tied
        };
        for b in winners {
            let inherited: Vec<EntityName> = original.body.iter().cloned().collect();
            bodies[b as usize].body.extend(inherited);
        }
    }

    qualify_pieces(&mut bodies, outputs);

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
    Ok(NamingOutput {
        bodies,
        buried,
        slot_ties,
    })
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
            let [a, b] =
                edge_name(o.names, o.topology, edge.entity).ok_or(NamingError::BadOperand {
                    operand: edge.operand,
                    entity: edge.entity,
                })?;
            SourceTag::edge_between(&a, &b)
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
            SourceTag::face_of(n)
        }
        KernelTag::CadName { name } => SourceTag::Key { key: name.clone() },
    })
}

/// Names for an imported body: CAD-file names become [`EntityName::CadFile`]. Used instead of
/// generation names when the kernel tags faces with [`KernelTag::CadName`] under an import.
pub fn cad_names_for_import(import: &FeatureId, output: &mut NamingOutput) {
    let convert = |n: &EntityName| match n {
        EntityName::Generated(g) if &g.feature == import => match &g.tag {
            SourceTag::Key { key } => EntityName::CadFile {
                import: import.clone(),
                name: key.clone(),
            },
            _ => n.clone(),
        },
        _ => n.clone(),
    };
    for b in &mut output.bodies {
        for set in b.faces.iter_mut().chain(std::iter::once(&mut b.body)) {
            *set = set.iter().map(convert).collect();
        }
        for q in b.qualifiers.values_mut() {
            q.bounded_by = q.bounded_by.iter().map(convert).collect();
            q.not_bounded_by = q.not_bounded_by.iter().map(convert).collect();
        }
        b.sign();
    }
}

/// Whether a role is one whose faces replace an edge (so a lost edge can be traced to them).
pub fn role_replaces_edge(role: Role) -> bool {
    matches!(role, Role::Blend | Role::Chamfer)
}

/// Give every face that shares its primary (whole) name with other faces a bounded qualifier
/// from its neighbours' names.
pub fn qualify_pieces(bodies: &mut [BodyNames], outputs: &[&Topology]) {
    let mut by_primary: BTreeMap<EntityName, Vec<(usize, EntityIndex)>> = BTreeMap::new();
    for (bi, b) in bodies.iter().enumerate() {
        for f in 0..b.faces.len() {
            if let Some(p) = b.primary(f as EntityIndex) {
                by_primary
                    .entry(p.whole())
                    .or_default()
                    .push((bi, f as EntityIndex));
            }
        }
    }
    for b in bodies.iter_mut() {
        b.qualifiers.clear();
    }
    for pieces in by_primary.values() {
        if pieces.len() < 2 {
            continue;
        }
        let neigh: Vec<BTreeSet<EntityName>> = pieces
            .iter()
            .map(|(bi, f)| {
                outputs[*bi]
                    .face_neighbours(*f)
                    .into_iter()
                    .filter_map(|n| bodies[*bi].primary(n).cloned())
                    .collect()
            })
            .collect();
        for (i, (bi, f)) in pieces.iter().enumerate() {
            let mine = &neigh[i];
            let others: Vec<&BTreeSet<EntityName>> = neigh
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, n)| n)
                .collect();
            // Neighbours no other piece has.
            let mut q = PieceQualifier::default();
            for n in mine {
                if q.bounded_by.len() >= MAX_QUALIFIERS {
                    break;
                }
                if others.iter().all(|o| !o.contains(n)) {
                    q.bounded_by.push(n.clone());
                }
            }
            if q.bounded_by.is_empty() {
                // Neighbours every other piece has and this one lacks.
                let union: BTreeSet<&EntityName> = others.iter().flat_map(|o| o.iter()).collect();
                for n in union {
                    if q.not_bounded_by.len() >= MAX_QUALIFIERS {
                        break;
                    }
                    if !mine.contains(n) && others.iter().all(|o| o.contains(n)) {
                        q.not_bounded_by.push(n.clone());
                    }
                }
            }
            if !(q.bounded_by.is_empty() && q.not_bounded_by.is_empty()) {
                bodies[*bi].qualifiers.insert(*f, q);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::SketchEntityId;

    fn n(i: u32) -> EntityName {
        EntityName::generated(
            "e",
            Role::Side,
            SourceTag::Curve {
                sketch: FeatureId::new("s"),
                entity: SketchEntityId::new(format!("l{}", i)),
            },
        )
    }

    #[test]
    fn minhash_agreement_tracks_jaccard() {
        let a: Vec<EntityName> = (0..40).map(n).collect();
        let b: Vec<EntityName> = (10..50).map(n).collect(); // Jaccard 30/50 = 0.6
        let c: Vec<EntityName> = (35..75).map(n).collect(); // Jaccard 5/75
        let sa = BodySignature::of_names(&a);
        let sb = BodySignature::of_names(&b);
        let sc = BodySignature::of_names(&c);
        let ab = sa.agreement(&sb) as f64 / SIGNATURE_LEN as f64;
        let ac = sa.agreement(&sc) as f64 / SIGNATURE_LEN as f64;
        assert!((ab - 0.6).abs() < 0.15, "estimate {}", ab);
        assert!(ac < 0.2, "estimate {}", ac);
        assert_eq!(sa.agreement(&sa), SIGNATURE_LEN as u32);
        assert_eq!(
            BodySignature::default().agreement(&BodySignature::default()),
            0
        );
    }

    #[test]
    fn weighted_similarity_prefers_the_main_piece() {
        let mut orig = BTreeMap::new();
        let mut main = BTreeMap::new();
        let mut sliver = BTreeMap::new();
        for i in 0..4 {
            orig.insert(n(i).token(), 100.0);
            main.insert(n(i).token(), 95.0);
            sliver.insert(n(i).token(), 5.0);
        }
        let s_main = weighted_jaccard(&orig, &main);
        let s_sliver = weighted_jaccard(&orig, &sliver);
        assert!(s_main > SLOT_MARGIN * s_sliver);
    }
}

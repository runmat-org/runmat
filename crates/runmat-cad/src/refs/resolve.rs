//! The resolver: a reference plus the regenerated state gives a set of entities, or a specific
//! reason it does not.
//!
//! Order of checks for a name:
//! 1. **Makers.** Each feature the name mentions must exist (else a lock alias is tried, then
//!    `maker_missing`), sit before the consumer (`forward_reference`), and have run. A
//!    suppressed or rolled-back maker means the reference is **waiting**, not lost; a failed
//!    maker means **blocked** (a cascade of that failure).
//! 2. **Exact.** Every live entity carrying the name. A split gives several (all pieces); a
//!    merged face answers to each of its names.
//! 3. **Why not.** Buried by a later feature (`consumed`), a pattern instance that no longer
//!    exists (`instance_removed`), or the maker ran and made no such entity (`not_generated`,
//!    with same-feature candidates offered as repairs, never chosen).
//! 4. **Fallback.** Only for `Reference::WithFallback`, only when the policy allows it, only
//!    among entities the stored maker made, and only on a unique match within tolerance. It is
//!    reported every time it is used. There is no nearest-entity search anywhere.

use crate::ids::{FeatureId, FilePath};
use crate::model::datum::{dot, sub};
use crate::model::items::{Cardinality, PublishEntry, PublishTarget};
use crate::refs::history::{EntityIndex, SurfaceKind, Topology};
use crate::refs::lock::LockSection;
use crate::refs::name::{EntityName, InstanceKey, Role, SourceTag, TopoKind};
use crate::refs::naming::{edge_name, BodyNames};
use crate::refs::publish::{ForeignTarget, PublishedCatalog, XrefProblem};
use crate::refs::reference::{GeometricFallback, Reference, XrefContext};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// An entity in the current regenerated state: body position plus entity index. Valid for one
/// regeneration only; persistent identity is the name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityHandle {
    pub body: u32,
    pub kind: TopoKind,
    pub index: EntityIndex,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum MakerStatus {
    Ran,
    Suppressed,
    RolledBack,
    /// Suppressed in the active configuration.
    SuppressedInConfiguration,
    Failed,
    /// Did not run because one of its own references was waiting on `root`.
    Waiting {
        root: FeatureId,
    },
    /// Did not run because something it needs failed.
    Blocked {
        root: FeatureId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureEntry {
    pub index: usize,
    pub status: Option<MakerStatus>,
}

/// Position and status of every feature, as regeneration has got so far.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureTable {
    pub entries: BTreeMap<FeatureId, FeatureEntry>,
    /// Pattern instance keys each pattern or mirror produced in this regeneration.
    pub instances: BTreeMap<FeatureId, BTreeSet<InstanceKey>>,
}

/// Fallback settings. Both are off by default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionPolicy {
    /// Allow `Reference::WithFallback` to use its geometric signature.
    pub geometric_fallback: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionClass {
    Exact,
    /// One entity answering to several names (a merge).
    Merged,
    /// Several pieces of one split entity.
    Pieces,
    /// Through a rename alias in the lock section.
    ViaAlias,
    /// Through a published name of another file.
    Published,
    /// Geometric fallback: the class that can be silently wrong elsewhere, counted and reported.
    Geometric,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "note")]
pub enum ResolveNote {
    ViaAlias {
        from: FeatureId,
        to: FeatureId,
        rewritten: Reference,
    },
    Merged {
        name: EntityName,
        also: Vec<EntityName>,
    },
    Pieces {
        name: EntityName,
        count: usize,
    },
    FallbackUsed {
        maker: FeatureId,
    },
    /// Pinned to `pinned`; `latest` is newer.
    PinBehind {
        file: FilePath,
        pinned: crate::ids::RevisionId,
        latest: crate::ids::RevisionId,
    },
    /// Some members of a list were waiting and were left out.
    Dormant {
        waiting: Vec<WaitReason>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "scope")]
pub enum Resolved {
    Local { entities: BTreeSet<EntityHandle> },
    Foreign { target: Box<ForeignTarget> },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Resolution {
    pub target: Resolved,
    pub class: ResolutionClass,
    #[serde(default)]
    pub notes: Vec<ResolveNote>,
}

impl Resolution {
    pub fn entities(&self) -> BTreeSet<EntityHandle> {
        match &self.target {
            Resolved::Local { entities } => entities.clone(),
            Resolved::Foreign { .. } => BTreeSet::new(),
        }
    }
    pub fn len(&self) -> usize {
        match &self.target {
            Resolved::Local { entities } => entities.len(),
            Resolved::Foreign { target } => {
                target.entities.len().max(target.datum.is_some() as usize)
            }
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitCause {
    Suppressed,
    RolledBack,
    SuppressedInConfiguration,
    /// The maker is itself waiting on `root`.
    MakerWaiting {
        root: FeatureId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WaitReason {
    pub maker: FeatureId,
    pub cause: WaitCause,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum RefProblem {
    MakerMissing {
        maker: FeatureId,
    },
    ForwardReference {
        maker: FeatureId,
    },
    /// The maker failed (or is blocked by `root`); this is a cascade.
    Blocked {
        maker: FeatureId,
        root: FeatureId,
    },
    NotGenerated {
        maker: FeatureId,
        name: EntityName,
        candidates: Vec<Reference>,
    },
    Consumed {
        name: EntityName,
        by: FeatureId,
    },
    InstanceRemoved {
        pattern: FeatureId,
        instance: InstanceKey,
        available: Vec<InstanceKey>,
    },
    /// Both face sets resolved but no edge runs between them.
    NotAdjacent {
        a: Reference,
        b: Reference,
    },
    /// The edge between the faces was replaced by a blend or chamfer face.
    EdgeReplaced {
        by: FeatureId,
        blend: EntityName,
    },
    /// A qualifier removed every piece.
    NoPieceMatches {
        base: Reference,
        pieces: usize,
    },
    NoVertex,
    KindMismatch {
        expected: TopoKind,
        found: TopoKind,
    },
    Ambiguous {
        count: usize,
        candidates: Vec<Reference>,
    },
    Xref {
        problem: XrefProblem,
    },
    UnknownPublish {
        name: crate::ids::PublishName,
    },
    /// The geometric fallback was tried and did not find exactly one match.
    FallbackFailed {
        inner: Box<RefProblem>,
        matches: usize,
    },
    /// The reference carries a fallback, but the policy does not allow it.
    FallbackDisabled {
        inner: Box<RefProblem>,
    },
    /// A required reference has no entities (an empty selection that was not allowed).
    Empty,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "outcome")]
pub enum ResolveOutcome {
    Resolved(Resolution),
    Waiting(WaitReason),
    Failed(RefProblem),
}

/// A per-regeneration count of resolutions by class (the proposal's measuring step). The
/// naming corpus asserts `geometric` stays zero for references that have names.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionStats {
    pub exact: u32,
    pub merged: u32,
    pub pieces: u32,
    pub via_alias: u32,
    pub published: u32,
    pub geometric: u32,
    pub waiting: u32,
    pub failed: u32,
}

impl ResolutionStats {
    pub fn record(&mut self, outcome: &ResolveOutcome) {
        match outcome {
            ResolveOutcome::Resolved(r) => match r.class {
                ResolutionClass::Exact => self.exact += 1,
                ResolutionClass::Merged => self.merged += 1,
                ResolutionClass::Pieces => self.pieces += 1,
                ResolutionClass::ViaAlias => self.via_alias += 1,
                ResolutionClass::Published => self.published += 1,
                ResolutionClass::Geometric => self.geometric += 1,
            },
            ResolveOutcome::Waiting(_) => self.waiting += 1,
            ResolveOutcome::Failed(_) => self.failed += 1,
        }
    }
}

/// One body as the resolver sees it.
#[derive(Clone, Copy)]
pub struct BodyView<'a> {
    pub names: &'a BodyNames,
    pub topology: &'a Topology,
}

pub struct Resolver<'a> {
    pub file: &'a FilePath,
    pub bodies: Vec<BodyView<'a>>,
    pub features: &'a FeatureTable,
    /// The consuming feature's position; `None` for items after the history (parts, regions,
    /// publish entries).
    pub consumer: Option<usize>,
    pub burials: &'a BTreeMap<(TopoKind, EntityName), FeatureId>,
    pub publishes: &'a [PublishEntry],
    pub lock: Option<&'a LockSection>,
    pub catalog: Option<&'a dyn PublishedCatalog>,
    pub policy: ResolutionPolicy,
}

enum Gate {
    Open,
    Alias { from: FeatureId, to: FeatureId },
    Wait(WaitReason),
    Fail(RefProblem),
}

type SetResult = Result<(BTreeSet<EntityHandle>, ResolutionClass, Vec<ResolveNote>), Stop>;

// `RefProblem` is the error payload on purpose; these are private and short-lived.
#[allow(clippy::large_enum_variant)]
enum Stop {
    Wait(WaitReason),
    Fail(RefProblem),
    Foreign(Box<Resolution>),
}

impl<'a> Resolver<'a> {
    /// Resolve `reference` for a site that wants `kind` entities in `cardinality`.
    pub fn resolve(
        &self,
        reference: &Reference,
        kind: Option<TopoKind>,
        cardinality: Cardinality,
    ) -> ResolveOutcome {
        match self.set(reference, 0) {
            Err(Stop::Wait(w)) => ResolveOutcome::Waiting(w),
            Err(Stop::Fail(p)) => ResolveOutcome::Failed(p),
            Err(Stop::Foreign(r)) => {
                if cardinality == Cardinality::ExactlyOne && r.len() > 1 {
                    return ResolveOutcome::Failed(RefProblem::Ambiguous {
                        count: r.len(),
                        candidates: Vec::new(),
                    });
                }
                ResolveOutcome::Resolved(*r)
            }
            Ok((set, class, notes)) => {
                if let (Some(want), Some(first)) = (kind, set.iter().next()) {
                    if first.kind != want {
                        return ResolveOutcome::Failed(RefProblem::KindMismatch {
                            expected: want,
                            found: first.kind,
                        });
                    }
                }
                if set.is_empty() && cardinality != Cardinality::Any {
                    return ResolveOutcome::Failed(RefProblem::Empty);
                }
                if cardinality == Cardinality::ExactlyOne && set.len() > 1 {
                    let candidates = self.disambiguate(reference, &set);
                    return ResolveOutcome::Failed(RefProblem::Ambiguous {
                        count: set.len(),
                        candidates,
                    });
                }
                ResolveOutcome::Resolved(Resolution {
                    target: Resolved::Local { entities: set },
                    class,
                    notes,
                })
            }
        }
    }

    /// Resolve a list site: waiting members are left out (dormant) and reported; the site waits
    /// only when every member waits.
    pub fn resolve_list(
        &self,
        references: &[Reference],
        kind: Option<TopoKind>,
    ) -> (Vec<ResolveOutcome>, ResolveOutcome) {
        let mut each = Vec::new();
        let mut set = BTreeSet::new();
        let mut waiting = Vec::new();
        let mut first_problem = None;
        let mut class = ResolutionClass::Exact;
        let mut notes = Vec::new();
        for r in references {
            let o = self.resolve(r, kind, Cardinality::AtLeastOne);
            match &o {
                ResolveOutcome::Resolved(res) => {
                    set.extend(res.entities());
                    class = class.max(res.class);
                    notes.extend(res.notes.iter().cloned());
                }
                ResolveOutcome::Waiting(w) => waiting.push(w.clone()),
                ResolveOutcome::Failed(p) => {
                    if first_problem.is_none() {
                        first_problem = Some(p.clone());
                    }
                }
            }
            each.push(o);
        }
        let combined = if let Some(p) = first_problem {
            ResolveOutcome::Failed(p)
        } else if set.is_empty() && !waiting.is_empty() {
            ResolveOutcome::Waiting(waiting[0].clone())
        } else {
            if !waiting.is_empty() {
                notes.push(ResolveNote::Dormant { waiting });
            }
            ResolveOutcome::Resolved(Resolution {
                target: Resolved::Local { entities: set },
                class,
                notes,
            })
        };
        (each, combined)
    }

    fn gate(&self, maker: &FeatureId) -> Gate {
        let Some(entry) = self.features.entries.get(maker) else {
            if let Some(lock) = self.lock {
                if let Some(to) = lock.feature_alias(self.file, maker) {
                    if self.features.entries.contains_key(&to) {
                        return Gate::Alias {
                            from: maker.clone(),
                            to,
                        };
                    }
                }
            }
            return Gate::Fail(RefProblem::MakerMissing {
                maker: maker.clone(),
            });
        };
        if let Some(c) = self.consumer {
            if entry.index >= c {
                return Gate::Fail(RefProblem::ForwardReference {
                    maker: maker.clone(),
                });
            }
        }
        let wait = |cause| {
            Gate::Wait(WaitReason {
                maker: maker.clone(),
                cause,
            })
        };
        match &entry.status {
            Some(MakerStatus::Ran) => Gate::Open,
            Some(MakerStatus::Suppressed) => wait(WaitCause::Suppressed),
            Some(MakerStatus::SuppressedInConfiguration) => {
                wait(WaitCause::SuppressedInConfiguration)
            }
            Some(MakerStatus::RolledBack) => wait(WaitCause::RolledBack),
            Some(MakerStatus::Waiting { root }) => {
                wait(WaitCause::MakerWaiting { root: root.clone() })
            }
            Some(MakerStatus::Failed) => Gate::Fail(RefProblem::Blocked {
                maker: maker.clone(),
                root: maker.clone(),
            }),
            Some(MakerStatus::Blocked { root }) => Gate::Fail(RefProblem::Blocked {
                maker: maker.clone(),
                root: root.clone(),
            }),
            // Not run yet but positioned before the consumer cannot happen during
            // regeneration; treat it as a forward reference.
            None => Gate::Fail(RefProblem::ForwardReference {
                maker: maker.clone(),
            }),
        }
    }

    #[allow(clippy::result_large_err)]
    fn set(&self, reference: &Reference, depth: u32) -> SetResult {
        if depth > 32 {
            return Err(Stop::Fail(RefProblem::Empty));
        }
        // Makers first, outermost first, so the most direct reason wins.
        if !matches!(
            reference,
            Reference::Published { .. } | Reference::WithFallback { .. }
        ) {
            for maker in reference.makers() {
                match self.gate(&maker) {
                    Gate::Open => {}
                    Gate::Wait(w) => return Err(Stop::Wait(w)),
                    Gate::Fail(p) => return Err(Stop::Fail(p)),
                    Gate::Alias { from, to } => {
                        let rewritten = reference.renamed(&from, &to);
                        let (s, _, mut notes) = self.set(&rewritten, depth + 1)?;
                        notes.insert(
                            0,
                            ResolveNote::ViaAlias {
                                from,
                                to,
                                rewritten,
                            },
                        );
                        return Ok((s, ResolutionClass::ViaAlias, notes));
                    }
                }
            }
        }
        match reference {
            Reference::Name { kind, name } => self.by_name(*kind, name),
            Reference::Piece {
                base,
                bounded_by,
                not_bounded_by,
            } => {
                let (pieces, _, notes) = self.set(base, depth + 1)?;
                let mut bounds = Vec::new();
                for b in bounded_by {
                    bounds.push(self.set(b, depth + 1)?.0);
                }
                let mut excludes = Vec::new();
                for b in not_bounded_by {
                    excludes.push(self.set(b, depth + 1)?.0);
                }
                let kept: BTreeSet<EntityHandle> = pieces
                    .iter()
                    .copied()
                    .filter(|p| {
                        let n = self.neighbours(p);
                        bounds.iter().all(|b| b.iter().any(|x| n.contains(x)))
                            && excludes.iter().all(|b| !b.iter().any(|x| n.contains(x)))
                    })
                    .collect();
                if kept.is_empty() {
                    return Err(Stop::Fail(RefProblem::NoPieceMatches {
                        base: (**base).clone(),
                        pieces: pieces.len(),
                    }));
                }
                Ok((kept, ResolutionClass::Exact, notes))
            }
            Reference::EdgesBetween { a, b } => {
                let (sa, _, _) = self.set(a, depth + 1)?;
                let (sb, _, _) = self.set(b, depth + 1)?;
                let mut out = BTreeSet::new();
                for (bi, body) in self.bodies.iter().enumerate() {
                    for (ei, e) in body.topology.edges.iter().enumerate() {
                        let fa = EntityHandle {
                            body: bi as u32,
                            kind: TopoKind::Face,
                            index: e.faces[0],
                        };
                        let fb = EntityHandle {
                            body: bi as u32,
                            kind: TopoKind::Face,
                            index: e.faces[1],
                        };
                        if (sa.contains(&fa) && sb.contains(&fb))
                            || (sa.contains(&fb) && sb.contains(&fa))
                        {
                            out.insert(EntityHandle {
                                body: bi as u32,
                                kind: TopoKind::Edge,
                                index: ei as u32,
                            });
                        }
                    }
                }
                if out.is_empty() {
                    if let Some((by, blend)) = self.blend_replacing(&sa, &sb) {
                        return Err(Stop::Fail(RefProblem::EdgeReplaced { by, blend }));
                    }
                    return Err(Stop::Fail(RefProblem::NotAdjacent {
                        a: (**a).clone(),
                        b: (**b).clone(),
                    }));
                }
                let class = if out.len() > 1 {
                    ResolutionClass::Pieces
                } else {
                    ResolutionClass::Exact
                };
                Ok((out, class, Vec::new()))
            }
            Reference::VertexOf { of } => {
                let mut sets = Vec::new();
                for r in of {
                    sets.push(self.set(r, depth + 1)?.0);
                }
                let mut out = BTreeSet::new();
                for (bi, body) in self.bodies.iter().enumerate() {
                    for (vi, v) in body.topology.vertices.iter().enumerate() {
                        let faces = body.topology.vertex_faces(vi as u32);
                        let touches = |h: &EntityHandle| {
                            h.body == bi as u32
                                && match h.kind {
                                    TopoKind::Face => faces.contains(&h.index),
                                    TopoKind::Edge => v.edges.contains(&h.index),
                                    _ => false,
                                }
                        };
                        if !sets.is_empty() && sets.iter().all(|s| s.iter().any(touches)) {
                            out.insert(EntityHandle {
                                body: bi as u32,
                                kind: TopoKind::Vertex,
                                index: vi as u32,
                            });
                        }
                    }
                }
                if out.is_empty() {
                    return Err(Stop::Fail(RefProblem::NoVertex));
                }
                Ok((out, ResolutionClass::Exact, Vec::new()))
            }
            Reference::Published {
                file,
                name,
                context,
            } => {
                let local = match file {
                    None => true,
                    Some(f) => f == self.file,
                };
                if local {
                    let entry =
                        self.publishes
                            .iter()
                            .find(|p| &p.name == name)
                            .ok_or_else(|| {
                                Stop::Fail(RefProblem::UnknownPublish { name: name.clone() })
                            })?;
                    match &entry.target {
                        PublishTarget::Entities { references, .. } => {
                            let mut out = BTreeSet::new();
                            let mut notes = Vec::new();
                            for r in references {
                                let (s, _, n) = self.set(r, depth + 1)?;
                                out.extend(s);
                                notes.extend(n);
                            }
                            Ok((out, ResolutionClass::Exact, notes))
                        }
                        _ => Err(Stop::Fail(RefProblem::KindMismatch {
                            expected: TopoKind::Face,
                            found: TopoKind::Body,
                        })),
                    }
                } else {
                    let file = file.as_ref().expect("checked above");
                    Err(self.foreign(file, name, context))
                }
            }
            Reference::WithFallback { primary, fallback } => match self.set(primary, depth + 1) {
                Ok(r) => Ok(r),
                Err(Stop::Fail(p)) if fallback_applies(&p) => {
                    if !self.policy.geometric_fallback {
                        return Err(Stop::Fail(RefProblem::FallbackDisabled {
                            inner: Box::new(p),
                        }));
                    }
                    let matches = self.geometric_matches(fallback);
                    if matches.len() == 1 {
                        Ok((
                            matches,
                            ResolutionClass::Geometric,
                            vec![ResolveNote::FallbackUsed {
                                maker: fallback.maker.clone(),
                            }],
                        ))
                    } else {
                        Err(Stop::Fail(RefProblem::FallbackFailed {
                            inner: Box::new(p),
                            matches: matches.len(),
                        }))
                    }
                }
                Err(other) => Err(other),
            },
        }
    }

    fn foreign(
        &self,
        file: &FilePath,
        name: &crate::ids::PublishName,
        context: &XrefContext,
    ) -> Stop {
        let Some(catalog) = self.catalog else {
            return Stop::Fail(RefProblem::Xref {
                problem: XrefProblem::NoCatalog,
            });
        };
        if !catalog.knows(file) {
            return Stop::Fail(RefProblem::Xref {
                problem: XrefProblem::UnknownFile { file: file.clone() },
            });
        }
        let pinned = match context {
            XrefContext::Live => {
                let here = catalog.partition(self.file);
                let there = catalog.partition(file);
                if here != there {
                    return Stop::Fail(RefProblem::Xref {
                        problem: XrefProblem::LiveAcrossPartition {
                            file: file.clone(),
                            latest_release: catalog.latest_release(file),
                        },
                    });
                }
                None
            }
            XrefContext::Pinned { revision } => Some(revision),
        };
        match catalog.resolve_published(file, name, pinned) {
            Ok(target) => {
                let mut notes = Vec::new();
                if let Some(pinned) = pinned {
                    if let Some(latest) = catalog.latest_release(file) {
                        if &latest != pinned {
                            notes.push(ResolveNote::PinBehind {
                                file: file.clone(),
                                pinned: pinned.clone(),
                                latest,
                            });
                        }
                    }
                }
                Stop::Foreign(Box::new(Resolution {
                    target: Resolved::Foreign {
                        target: Box::new(target),
                    },
                    class: ResolutionClass::Published,
                    notes,
                }))
            }
            Err(problem) => Stop::Fail(RefProblem::Xref { problem }),
        }
    }

    #[allow(clippy::result_large_err)]
    fn by_name(&self, kind: TopoKind, name: &EntityName) -> SetResult {
        let mut out = BTreeSet::new();
        let mut exact_names: BTreeMap<EntityHandle, Vec<EntityName>> = BTreeMap::new();
        match kind {
            TopoKind::Face => {
                for (bi, body) in self.bodies.iter().enumerate() {
                    for (fi, names) in body.names.faces.iter().enumerate() {
                        if names.iter().any(|n| name.matches(n)) {
                            let h = EntityHandle {
                                body: bi as u32,
                                kind,
                                index: fi as u32,
                            };
                            out.insert(h);
                            exact_names.insert(h, names.iter().cloned().collect());
                        }
                    }
                }
            }
            TopoKind::Body => {
                for (bi, body) in self.bodies.iter().enumerate() {
                    if body.names.body.iter().any(|n| name.matches(n)) {
                        let h = EntityHandle {
                            body: bi as u32,
                            kind,
                            index: 0,
                        };
                        out.insert(h);
                        exact_names.insert(h, body.names.body.iter().cloned().collect());
                    }
                }
            }
            TopoKind::Edge | TopoKind::Vertex => {
                return Err(Stop::Fail(RefProblem::KindMismatch {
                    expected: TopoKind::Face,
                    found: kind,
                }));
            }
        }
        if out.is_empty() {
            return Err(Stop::Fail(self.why_missing(kind, name)));
        }
        let mut notes = Vec::new();
        let mut class = ResolutionClass::Exact;
        if out.len() > 1 {
            // Several entities carry a name with the same ordinal: pieces of a split. Distinct
            // ordinals of one ordinal-free query are the whole logical entity, which is exact.
            let ordinals: BTreeSet<Option<u32>> = exact_names
                .values()
                .flat_map(|ns| ns.iter().filter(|n| name.matches(n)).map(ordinal_of))
                .collect();
            if ordinals.len() < out.len() {
                class = ResolutionClass::Pieces;
                notes.push(ResolveNote::Pieces {
                    name: name.clone(),
                    count: out.len(),
                });
            }
        } else if let Some(ns) = exact_names.values().next() {
            if ns.len() > 1 {
                class = ResolutionClass::Merged;
                notes.push(ResolveNote::Merged {
                    name: name.clone(),
                    also: ns.iter().filter(|n| !name.matches(n)).cloned().collect(),
                });
            }
        }
        Ok((out, class, notes))
    }

    fn why_missing(&self, kind: TopoKind, name: &EntityName) -> RefProblem {
        for ((k, n), by) in self.burials.iter() {
            if *k == kind && name.matches(n) {
                return RefProblem::Consumed {
                    name: n.clone(),
                    by: by.clone(),
                };
            }
        }
        if let EntityName::Copy {
            pattern, instance, ..
        } = name
        {
            if let Some(keys) = self.features.instances.get(pattern) {
                if !keys.contains(instance) {
                    return RefProblem::InstanceRemoved {
                        pattern: pattern.clone(),
                        instance: instance.clone(),
                        available: keys.iter().cloned().collect(),
                    };
                }
            }
        }
        RefProblem::NotGenerated {
            maker: name.maker().clone(),
            name: name.clone(),
            candidates: self.lineage_candidates(kind, name),
        }
    }

    /// Same-feature candidates for a re-pick, ranked like the proposal's lineage guess (same
    /// feature, then same role, then the same tag kind). Offered as repairs; never chosen.
    fn lineage_candidates(&self, kind: TopoKind, name: &EntityName) -> Vec<Reference> {
        let EntityName::Generated(want) = name else {
            return Vec::new();
        };
        let mut scored: Vec<(i32, EntityName)> = Vec::new();
        let mut seen = BTreeSet::new();
        let names: Vec<&EntityName> = match kind {
            TopoKind::Face => self
                .bodies
                .iter()
                .flat_map(|b| b.names.faces.iter().flatten())
                .collect(),
            TopoKind::Body => self
                .bodies
                .iter()
                .flat_map(|b| b.names.body.iter())
                .collect(),
            _ => Vec::new(),
        };
        for n in names {
            if let EntityName::Generated(g) = n {
                if g.feature != want.feature || !seen.insert(n.clone()) {
                    continue;
                }
                let mut score = 4;
                if g.role == want.role {
                    score += 2;
                }
                if std::mem::discriminant(&g.tag) == std::mem::discriminant(&want.tag) {
                    score += 1;
                }
                scored.push((-score, n.clone()));
            }
        }
        scored.sort();
        scored
            .into_iter()
            .take(4)
            .map(|(_, n)| Reference::Name { kind, name: n })
            .collect()
    }

    fn blend_replacing(
        &self,
        sa: &BTreeSet<EntityHandle>,
        sb: &BTreeSet<EntityHandle>,
    ) -> Option<(FeatureId, EntityName)> {
        // The blend's tag is the pair of primary names the edge had; match against every name
        // of the two face sets.
        let names_of = |s: &BTreeSet<EntityHandle>| -> BTreeSet<EntityName> {
            s.iter()
                .filter(|h| h.kind == TopoKind::Face)
                .flat_map(|h| {
                    self.bodies[h.body as usize].names.faces[h.index as usize]
                        .iter()
                        .cloned()
                })
                .collect()
        };
        let na = names_of(sa);
        let nb = names_of(sb);
        for body in &self.bodies {
            for names in &body.names.faces {
                for n in names {
                    if let EntityName::Generated(g) = n {
                        if let (Role::Blend | Role::Chamfer, SourceTag::Edge { between }) =
                            (g.role, &g.tag)
                        {
                            let [x, y] = &**between;
                            if (na.contains(x) && nb.contains(y))
                                || (na.contains(y) && nb.contains(x))
                            {
                                return Some((g.feature.clone(), n.clone()));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Faces next to an entity: for a face, faces sharing an edge; for an edge, faces at its
    /// ends other than its own two.
    fn neighbours(&self, h: &EntityHandle) -> BTreeSet<EntityHandle> {
        let topo = self.bodies[h.body as usize].topology;
        let faces = match h.kind {
            TopoKind::Face => topo.face_neighbours(h.index),
            TopoKind::Edge => topo.edge_neighbour_faces(h.index),
            _ => Vec::new(),
        };
        faces
            .into_iter()
            .map(|f| EntityHandle {
                body: h.body,
                kind: TopoKind::Face,
                index: f,
            })
            .collect()
    }

    fn primary_ref(&self, h: &EntityHandle) -> Option<Reference> {
        let body = self.bodies[h.body as usize];
        match h.kind {
            TopoKind::Face => body
                .names
                .primary(h.index)
                .map(|n| Reference::face(n.clone())),
            TopoKind::Edge => edge_name(body.names, body.topology, h.index)
                .map(|[a, b]| Reference::edges_between(Reference::face(a), Reference::face(b))),
            TopoKind::Body => body
                .names
                .body
                .iter()
                .next()
                .map(|n| Reference::body(n.clone())),
            TopoKind::Vertex => None,
        }
    }

    /// For each member of an ambiguous set, a reference that picks it alone, when its
    /// neighbours tell it apart.
    pub fn disambiguate(
        &self,
        reference: &Reference,
        set: &BTreeSet<EntityHandle>,
    ) -> Vec<Reference> {
        let neigh: Vec<(EntityHandle, BTreeSet<EntityName>)> = set
            .iter()
            .map(|h| {
                let names: BTreeSet<EntityName> = self
                    .neighbours(h)
                    .iter()
                    .flat_map(|n| {
                        self.bodies[n.body as usize].names.faces[n.index as usize]
                            .iter()
                            .cloned()
                    })
                    .collect();
                (*h, names)
            })
            .collect();
        let mut out = Vec::new();
        for (i, (_, mine)) in neigh.iter().enumerate() {
            let others: Vec<&BTreeSet<EntityName>> = neigh
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, (_, n))| n)
                .collect();
            if let Some(d) = mine.iter().find(|n| others.iter().all(|o| !o.contains(*n))) {
                out.push(Reference::piece(
                    reference.clone(),
                    vec![Reference::face(d.clone())],
                ));
                continue;
            }
            // Excluding: a neighbour every other piece has and this one lacks.
            let union: BTreeSet<&EntityName> = others.iter().flat_map(|o| o.iter()).collect();
            if let Some(d) = union
                .into_iter()
                .find(|n| !mine.contains(*n) && others.iter().all(|o| o.contains(*n)))
            {
                out.push(Reference::Piece {
                    base: Box::new(reference.clone()),
                    bounded_by: Vec::new(),
                    not_bounded_by: vec![Reference::face(d.clone())],
                });
            }
        }
        out
    }

    /// The canonical reference to an entity in this state (for repairs and auto-publish).
    pub fn canonical_reference(&self, h: &EntityHandle) -> Option<Reference> {
        self.primary_ref(h)
    }

    fn geometric_matches(&self, fb: &GeometricFallback) -> BTreeSet<EntityHandle> {
        let sig = &fb.signature;
        let mut out = BTreeSet::new();
        for (bi, body) in self.bodies.iter().enumerate() {
            match sig.kind {
                TopoKind::Face => {
                    for (fi, f) in body.topology.faces.iter().enumerate() {
                        let made_by_maker =
                            body.names.faces[fi].iter().any(|n| n.maker() == &fb.maker);
                        if !made_by_maker {
                            continue;
                        }
                        let g = &f.geom;
                        let d = sub(g.centroid, sig.centroid);
                        if dot(d, d).sqrt() > fb.tolerance {
                            continue;
                        }
                        if (g.area - sig.measure).abs()
                            > fb.tolerance * fb.tolerance.max(sig.measure.sqrt()) * 4.0
                        {
                            continue;
                        }
                        if let (Some(n), SurfaceKind::Plane { normal }) = (sig.normal, g.surface) {
                            if dot(n, normal) < 1.0 - 1e-9 {
                                continue;
                            }
                        }
                        out.insert(EntityHandle {
                            body: bi as u32,
                            kind: TopoKind::Face,
                            index: fi as u32,
                        });
                    }
                }
                TopoKind::Edge => {
                    for (ei, e) in body.topology.edges.iter().enumerate() {
                        let made = e.faces.iter().any(|f| {
                            body.names.faces[*f as usize]
                                .iter()
                                .any(|n| n.maker() == &fb.maker)
                        });
                        if !made {
                            continue;
                        }
                        let mid = [
                            (e.geom.start[0] + e.geom.end[0]) / 2.0,
                            (e.geom.start[1] + e.geom.end[1]) / 2.0,
                            (e.geom.start[2] + e.geom.end[2]) / 2.0,
                        ];
                        let d = sub(mid, sig.centroid);
                        if dot(d, d).sqrt() <= fb.tolerance
                            && (e.geom.length - sig.measure).abs() <= fb.tolerance
                        {
                            out.insert(EntityHandle {
                                body: bi as u32,
                                kind: TopoKind::Edge,
                                index: ei as u32,
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        out
    }
}

fn ordinal_of(n: &EntityName) -> Option<u32> {
    match n {
        EntityName::Generated(g) => g.ordinal,
        EntityName::Copy { base, .. } => ordinal_of(base),
        EntityName::CadFile { .. } => None,
    }
}

/// The fallback may stand in only where the name is genuinely gone, never where the maker is
/// waiting, blocked, missing or after the consumer.
fn fallback_applies(p: &RefProblem) -> bool {
    matches!(
        p,
        RefProblem::NotGenerated { .. }
            | RefProblem::NotAdjacent { .. }
            | RefProblem::NoPieceMatches { .. }
    )
}

//! Interference and clearance requests.
//!
//! The broad phase runs here, on R0 boxes: every leaf occurrence's configured box placed in
//! world coordinates, plus the space-allocation envelopes assemblies declare. A sweep-and-prune
//! on X (sorted, deterministic) yields candidate pairs whose boxes overlap within the requested
//! clearance. Exclusions: explicitly ignored pairs, stack members a fastener set clamps together
//! (when asked), and an envelope's own occupant subtree. Each candidate goes to the
//! [`InterferenceKernel`] seam at the requested representation (OCCT booleans and distance
//! queries at K1; [`BoxKernel`] answers from boxes and says so).

use crate::assembly::model::{DefinitionKind, DefinitionRef, RelPath, Representation};
use crate::assembly::resolve::{NodeId, ResolvedAssembly};
use crate::assembly::source::{Aabb, ConfigKey, DesignSource};
use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Severity, Site, Subject};
use crate::regen::kernel::{KernelError, Transform};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum CheckMode {
    Interference,
    /// Report pairs closer than `min` (SI).
    Clearance {
        min: f64,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InterferenceRequest {
    pub mode: CheckMode,
    /// Restrict to occurrences under these paths (empty: everything).
    #[serde(default)]
    pub within: Vec<RelPath>,
    #[serde(default = "yes")]
    pub include_envelopes: bool,
    #[serde(default)]
    pub ignore: Vec<(RelPath, RelPath)>,
    /// Pairs a fastener set clamps together are in contact by design.
    #[serde(default = "yes")]
    pub ignore_fastener_stacks: bool,
    #[serde(default = "full")]
    pub representation: Representation,
    /// Stop after this many reported pairs.
    #[serde(default = "cap")]
    pub max_reports: usize,
}

fn yes() -> bool {
    true
}
fn full() -> Representation {
    Representation::Full
}
fn cap() -> usize {
    1000
}

impl Default for InterferenceRequest {
    fn default() -> Self {
        Self {
            mode: CheckMode::Interference,
            within: Vec::new(),
            include_envelopes: true,
            ignore: Vec::new(),
            ignore_fastener_stacks: true,
            representation: Representation::Full,
            max_reports: 1000,
        }
    }
}

/// One shape in the check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlacedShape {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<NodeId>,
    /// `Some` for a declared envelope: its name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub envelope: Option<String>,
    pub label: String,
    pub definition: Option<DefinitionRef>,
    pub configuration: ConfigKey,
    pub representation: Representation,
    pub world: Transform,
    pub bbox: Aabb,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "result")]
pub enum PairResult {
    Clear { distance: f64 },
    Touching,
    Interfering { volume: f64 },
}

pub trait InterferenceKernel {
    fn check(
        &self,
        a: &PlacedShape,
        b: &PlacedShape,
        mode: CheckMode,
    ) -> Result<(PairResult, bool), KernelError>;
}

/// Answers from boxes (exact = false): interfering by the overlap volume, touching at zero
/// volume, clear by the box gap.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoxKernel;

impl InterferenceKernel for BoxKernel {
    fn check(
        &self,
        a: &PlacedShape,
        b: &PlacedShape,
        _mode: CheckMode,
    ) -> Result<(PairResult, bool), KernelError> {
        let i = a.bbox.intersection(&b.bbox);
        if !i.is_empty() {
            let v = i.volume();
            return Ok((
                if v > 0.0 {
                    PairResult::Interfering { volume: v }
                } else {
                    PairResult::Touching
                },
                false,
            ));
        }
        let mut gap2 = 0.0;
        for k in 0..3 {
            let d = (a.bbox.min[k] - b.bbox.max[k])
                .max(b.bbox.min[k] - a.bbox.max[k])
                .max(0.0);
            gap2 += d * d;
        }
        Ok((
            PairResult::Clear {
                distance: gap2.sqrt(),
            },
            false,
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairReport {
    pub a: String,
    pub b: String,
    pub result: PairResult,
    pub exact: bool,
}

#[derive(Clone, Debug, Default)]
pub struct InterferenceReport {
    pub shapes: usize,
    pub candidates: usize,
    pub pairs: Vec<PairReport>,
    pub diagnostics: DiagnosticReport,
    pub kernel_errors: usize,
}

fn under(tree: &ResolvedAssembly, n: NodeId, root: NodeId) -> bool {
    let mut cur = Some(n);
    while let Some(c) = cur {
        if c == root {
            return true;
        }
        cur = tree.node(c).parent;
    }
    false
}

pub fn check(
    tree: &ResolvedAssembly,
    source: &dyn DesignSource,
    kernel: &dyn InterferenceKernel,
    req: &InterferenceRequest,
) -> InterferenceReport {
    let mut out = InterferenceReport::default();
    let roots: Vec<NodeId> = req.within.iter().filter_map(|p| tree.find(p)).collect();
    let mut shapes: Vec<PlacedShape> = Vec::new();
    for (i, n) in tree.nodes.iter().enumerate().skip(1) {
        let id = i as NodeId;
        if n.definition.kind == DefinitionKind::Assembly && n.expanded {
            continue;
        }
        if !roots.is_empty() && !roots.iter().any(|r| under(tree, id, *r)) {
            continue;
        }
        let Ok(s) = source.configured(&n.definition, n.revision.as_ref(), &n.configuration) else {
            continue;
        };
        let Some(b) = s.bbox else { continue };
        shapes.push(PlacedShape {
            node: Some(id),
            envelope: None,
            label: tree.path(id),
            definition: Some(n.definition.clone()),
            configuration: n.configuration.clone(),
            representation: req.representation,
            world: n.world,
            bbox: b.transformed(&n.world),
        });
    }
    if req.include_envelopes {
        for e in &tree.envelopes {
            let owner = tree.path(e.declared_by);
            shapes.push(PlacedShape {
                node: None,
                envelope: Some(e.name.clone()),
                label: if owner.is_empty() {
                    format!("envelope.{}", e.name)
                } else {
                    format!("{}/envelope.{}", owner, e.name)
                },
                definition: None,
                configuration: ConfigKey::default(),
                representation: Representation::Envelope,
                world: Transform::IDENTITY,
                bbox: e.world,
            });
        }
    }
    out.shapes = shapes.len();
    // Exclusions.
    let ignore: Vec<(NodeId, NodeId)> = req
        .ignore
        .iter()
        .filter_map(|(a, b)| Some((tree.find(a)?, tree.find(b)?)))
        .collect();
    let mut stacks: BTreeSet<(NodeId, NodeId)> = BTreeSet::new();
    if req.ignore_fastener_stacks {
        for f in &tree.fasteners {
            let members: Vec<NodeId> = f
                .expansion
                .drills
                .iter()
                .filter_map(|d| {
                    let mut p = tree.rel_path(f.owner);
                    p.0.extend(d.member.0.iter().cloned());
                    tree.find(&p)
                })
                .collect();
            for (i, a) in members.iter().enumerate() {
                for b in &members[i + 1..] {
                    stacks.insert(((*a).min(*b), (*a).max(*b)));
                }
            }
        }
    }
    let excluded = |a: &PlacedShape, b: &PlacedShape| -> bool {
        match (a.node, b.node) {
            (Some(x), Some(y)) => {
                let k = (x.min(y), x.max(y));
                stacks.contains(&k)
                    || ignore.iter().any(|(p, q)| {
                        (under(tree, x, *p) && under(tree, y, *q))
                            || (under(tree, x, *q) && under(tree, y, *p))
                    })
            }
            (None, Some(n)) | (Some(n), None) => {
                // An envelope admits its own occupants.
                let env = if a.node.is_none() { a } else { b };
                tree.envelopes
                    .iter()
                    .find(|e| Some(&e.name) == env.envelope.as_ref() && e.world == env.bbox)
                    .map(|e| match e.occupant {
                        Some(o) => under(tree, n, o),
                        None => !e.keep_out,
                    })
                    .unwrap_or(false)
            }
            (None, None) => true,
        }
    };
    let margin = match req.mode {
        CheckMode::Interference => 0.0,
        CheckMode::Clearance { min } => min,
    };
    // Sweep and prune on X.
    let mut order: Vec<usize> = (0..shapes.len()).collect();
    order.sort_by(|&i, &j| {
        shapes[i].bbox.min[0]
            .partial_cmp(&shapes[j].bbox.min[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| shapes[i].label.cmp(&shapes[j].label))
    });
    let mut active: Vec<usize> = Vec::new();
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for &i in &order {
        let x = shapes[i].bbox.min[0] - margin;
        active.retain(|&j| shapes[j].bbox.max[0] >= x);
        for &j in &active {
            if shapes[i].bbox.overlaps(&shapes[j].bbox, margin) && !excluded(&shapes[i], &shapes[j])
            {
                let (a, b) = if shapes[i].label <= shapes[j].label {
                    (i, j)
                } else {
                    (j, i)
                };
                candidates.push((a, b));
            }
        }
        active.push(i);
    }
    candidates.sort_by(|a, b| {
        (&shapes[a.0].label, &shapes[a.1].label).cmp(&(&shapes[b.0].label, &shapes[b.1].label))
    });
    out.candidates = candidates.len();
    let mut reported = 0usize;
    for (a, b) in candidates {
        match kernel.check(&shapes[a], &shapes[b], req.mode) {
            Ok((result, exact)) => {
                let bad = match (result, req.mode) {
                    (PairResult::Interfering { .. }, _) => true,
                    (PairResult::Touching, CheckMode::Clearance { min }) => min > 0.0,
                    (PairResult::Clear { distance }, CheckMode::Clearance { min }) => {
                        distance < min
                    }
                    _ => false,
                };
                if !bad {
                    continue;
                }
                reported += 1;
                if reported <= req.max_reports {
                    let (code, msg) = match result {
                        PairResult::Interfering { volume } => (
                            DiagCode::AsmInterference,
                            format!(
                                "`{}` and `{}` interfere ({:.3e} m^3{})",
                                shapes[a].label,
                                shapes[b].label,
                                volume,
                                if exact { "" } else { ", from boxes" }
                            ),
                        ),
                        PairResult::Touching => (
                            DiagCode::AsmClearance,
                            format!("`{}` and `{}` touch", shapes[a].label, shapes[b].label),
                        ),
                        PairResult::Clear { distance } => (
                            DiagCode::AsmClearance,
                            format!(
                                "`{}` and `{}` are {:.4e} m apart, under the {:.4e} m clearance",
                                shapes[a].label, shapes[b].label, distance, margin
                            ),
                        ),
                    };
                    out.diagnostics.push(
                        code,
                        Severity::Error,
                        Site::of(Subject::Occurrence {
                            path: shapes[a].label.clone(),
                        }),
                        msg,
                        Cause::Root,
                        Vec::new(),
                    );
                    out.pairs.push(PairReport {
                        a: shapes[a].label.clone(),
                        b: shapes[b].label.clone(),
                        result,
                        exact,
                    });
                }
            }
            Err(e) => {
                out.kernel_errors += 1;
                out.diagnostics.push(
                    DiagCode::AsmCheckFailed,
                    Severity::Warning,
                    Site::of(Subject::Occurrence {
                        path: shapes[a].label.clone(),
                    }),
                    format!("`{}` vs `{}`: {}", shapes[a].label, shapes[b].label, e),
                    Cause::Root,
                    Vec::new(),
                );
            }
        }
    }
    if reported > req.max_reports {
        out.diagnostics.push(
            DiagCode::AsmInterference,
            Severity::Error,
            Site::of(Subject::Document),
            format!("{} more pairs not listed", reported - req.max_reports),
            Cause::Root,
            Vec::new(),
        );
    }
    out
}

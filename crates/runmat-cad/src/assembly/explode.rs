//! Exploded views: named exploded states of an assembly.
//!
//! An [`ExplodedView`] is an ordered list of steps over occurrences named by stable paths
//! ([`RelPath`]: authored names and semantic instance keys, never indices), plus explode lines
//! (trails from each occurrence's assembled position to its exploded one, with jogs). A view may
//! start from another (`base`): its steps run first, so a service view can extend the assembly
//! instructions' view.
//!
//! Steps run in order, each from the poses the previous steps left:
//! - **translate** occurrences along a direction (a vector in assembly coordinates, or an axis
//!   of an occurrence's current frame, so a bolt backs out along its own axis), by a distance,
//!   optionally spaced (`distance + i × spacing` for the i-th occurrence, the stacked-washer
//!   case);
//! - **rotate** occurrences about an axis (explicit, or an axis of an occurrence's frame);
//! - **radial**: move each occurrence away from an axis or a point by a distance, along the
//!   line from the centre to the occurrence's centre (bounding-box centre through the
//!   [`DesignSource`], else its origin).
//!
//! Moving an occurrence moves its subtree. A step that names an occurrence and one of its
//! ancestors moves the descendant twice (ancestor first); that is reported, since it is rarely
//! meant. Evaluation gives every node's exploded world transform, the moved set and the line
//! polylines, plus diagnostics with exact codes. Occurrences inactive in the resolved
//! configuration (suppressed, or another alternative of the 150% structure) are skipped with a
//! warning; renamed occurrences resolve through the lock with a rewrite repair.

use crate::assembly::model::{AssemblyDocument, PathSegment, RelPath};
use crate::assembly::resolve::{item_path, NodeId, ResolvedAssembly};
use crate::assembly::source::DesignSource;
use crate::diagnostics::{Cause, DiagCode, DiagnosticReport, Repair, Severity, Site, Subject};
use crate::ids::{FilePath, ItemUuid};
use crate::refs::lock::LockSection;
use crate::regen::kernel::Transform;
use crate::units::{Dimension, UnitAuthority};
use crate::value::{DynamicValueEvaluator, Evaluator, ValueEnv, ValueExpr};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalAxis {
    X,
    Y,
    Z,
}

impl LocalAxis {
    fn unit(self) -> [f64; 3] {
        match self {
            LocalAxis::X => [1.0, 0.0, 0.0],
            LocalAxis::Y => [0.0, 1.0, 0.0],
            LocalAxis::Z => [0.0, 0.0, 1.0],
        }
    }
}

/// A direction for a translate step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "direction")]
pub enum ExplodeDirection {
    /// In assembly (top) coordinates.
    Vector { vector: [f64; 3] },
    /// An axis of an occurrence's frame, as the previous steps left it.
    OccurrenceAxis {
        occurrence: RelPath,
        axis: LocalAxis,
        #[serde(default)]
        reversed: bool,
    },
}

/// An axis for a rotate step.
// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "about")]
pub enum ExplodeAxis {
    /// Through `point`, along `direction`, in assembly coordinates.
    Explicit {
        point: [ValueExpr; 3],
        direction: [f64; 3],
    },
    /// An axis of an occurrence's frame through its origin (plus `offset` in its frame).
    OccurrenceAxis {
        occurrence: RelPath,
        axis: LocalAxis,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        offset: Option<[ValueExpr; 3]>,
    },
}

/// What a radial step explodes away from.
// Authored data, built once per edit; variant sizes do not matter here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "center")]
pub enum RadialCenter {
    /// Perpendicular to this axis.
    Axis {
        point: [ValueExpr; 3],
        direction: [f64; 3],
    },
    /// Away from this point.
    Point { point: [ValueExpr; 3] },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "motion")]
pub enum ExplodeMotion {
    Translate {
        direction: ExplodeDirection,
        distance: ValueExpr,
        /// Each further occurrence of the step moves this much more.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        spacing: Option<ValueExpr>,
    },
    Rotate {
        axis: ExplodeAxis,
        angle: ValueExpr,
    },
    Radial {
        center: RadialCenter,
        distance: ValueExpr,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplodeStep {
    /// Stable across edits and merges.
    pub id: String,
    pub occurrences: Vec<RelPath>,
    pub motion: ExplodeMotion,
}

/// A jog: the trail leaves segment `segment` at fraction `at`, runs offset by `offset`
/// (perpendicular to the segment) and comes back at the segment's end, so the line routes
/// around what lies on the straight path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Jog {
    pub segment: u32,
    pub at: f64,
    pub offset: [ValueExpr; 3],
}

/// A trail line from an occurrence's assembled position to its exploded one, through the
/// position after every step that moved it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplodeLine {
    /// Stable across edits and merges.
    pub id: String,
    pub occurrence: RelPath,
    /// The trailed point in the occurrence's frame (default: its centre).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<[ValueExpr; 3]>,
    #[serde(default)]
    pub jogs: Vec<Jog>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplodedView {
    pub name: String,
    /// Another exploded view of this assembly whose steps run first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
    #[serde(default)]
    pub steps: Vec<ExplodeStep>,
    #[serde(default)]
    pub lines: Vec<ExplodeLine>,
    /// Sub-assemblies' own exploded views composed into this one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub include: Vec<IncludedView>,
}

/// A sub-assembly occurrence exploded by one of its own views: the view's steps run in the
/// sub-assembly's frame (as it stands when they run), moving its subtree, after the step
/// `after` of this view (`None`: before this view's own steps).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IncludedView {
    /// Stable across edits and merges.
    pub id: String,
    pub occurrence: RelPath,
    pub view: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
}

impl ExplodedView {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            base: None,
            steps: Vec::new(),
            lines: Vec::new(),
            include: Vec::new(),
        }
    }
}

// ------------------------------------------------------------------------------ results

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinePath {
    pub id: String,
    /// The occurrence's display path.
    pub occurrence: String,
    /// Assembly coordinates, assembled position first, exploded position last.
    pub points: Vec<[f64; 3]>,
}

#[derive(Clone, Debug)]
pub struct ExplodedState {
    pub view: String,
    /// Every node's exploded world transform (index = node id).
    pub world: Vec<Transform>,
    /// Nodes an exploded step moved (directly or through an ancestor), sorted.
    pub moved: Vec<NodeId>,
    pub lines: Vec<LinePath>,
    pub diagnostics: DiagnosticReport,
}

impl ExplodedState {
    /// The world-space motion of a node: `exploded ∘ assembled⁻¹`.
    pub fn offset(&self, tree: &ResolvedAssembly, node: NodeId) -> Transform {
        self.world[node as usize].compose(&tree.node(node).world.inverse())
    }
}

pub struct ExplodeContext<'a> {
    pub units: &'a dyn UnitAuthority,
    pub dynamic: &'a dyn DynamicValueEvaluator,
    /// Renamed occurrences resolve through their lock aliases.
    pub lock: Option<&'a LockSection>,
    /// Occurrence centres (bounding boxes) and the 150% structure below the top.
    pub source: Option<&'a dyn DesignSource>,
}

// ------------------------------------------------------------------------------ validation

fn site(view: &str, param: &str) -> Site {
    Site::with_param(
        Subject::Exploded {
            view: view.to_owned(),
        },
        param,
    )
}

fn push(rep: &mut DiagnosticReport, code: DiagCode, sev: Severity, s: Site, m: String) {
    rep.push(code, sev, s, m, Cause::Root, Vec::new());
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The chain of views a view runs (bases first), or the diagnostic that stops it.
fn chain<'d>(
    doc: &'d AssemblyDocument,
    name: &str,
    rep: &mut DiagnosticReport,
) -> Option<Vec<&'d ExplodedView>> {
    let mut out: Vec<&ExplodedView> = Vec::new();
    let mut cur = Some(name.to_owned());
    while let Some(n) = cur {
        let Some(v) = doc.exploded.iter().find(|v| v.name == n) else {
            let first = out.is_empty();
            push(
                rep,
                DiagCode::ExplodeUnknownView,
                Severity::Error,
                site(
                    out.last().map(|v| v.name.as_str()).unwrap_or(name),
                    if first { "name" } else { "base" },
                ),
                format!("`{}` has no exploded view `{}`", doc.path, n),
            );
            return None;
        };
        if out.iter().any(|o| o.name == v.name) {
            let mut names: Vec<&str> = out.iter().map(|o| o.name.as_str()).collect();
            names.push(&v.name);
            push(
                rep,
                DiagCode::ExplodeCycle,
                Severity::Error,
                site(name, "base"),
                format!("exploded views build on each other: {}", names.join(" → ")),
            );
            return None;
        }
        out.push(v);
        cur = v.base.clone();
    }
    out.reverse();
    Some(out)
}

/// Static checks of every exploded view of an assembly: unique names, step and line ids
/// unique along each base chain, bases that exist and do not loop, non-zero directions and
/// axes, steps that name something and name it once, jogs inside their segment.
pub fn validate_exploded(doc: &AssemblyDocument) -> DiagnosticReport {
    let mut rep = DiagnosticReport::default();
    let mut names = BTreeSet::new();
    for v in &doc.exploded {
        if !names.insert(v.name.as_str()) {
            push(
                &mut rep,
                DiagCode::ExplodeDuplicate,
                Severity::Error,
                site(&v.name, "name"),
                format!("exploded view `{}` is declared twice", v.name),
            );
        }
    }
    for v in &doc.exploded {
        check_view(doc, v, &mut rep);
    }
    rep
}

fn check_view(doc: &AssemblyDocument, v: &ExplodedView, rep: &mut DiagnosticReport) {
    let Some(views) = chain(doc, &v.name, rep) else {
        return;
    };
    let mut step_ids: BTreeMap<&str, &str> = BTreeMap::new();
    let mut line_ids: BTreeMap<&str, &str> = BTreeMap::new();
    for w in &views {
        for s in &w.steps {
            if let Some(prev) = step_ids.insert(&s.id, &w.name) {
                if w.name == v.name {
                    push(
                        rep,
                        DiagCode::ExplodeDuplicate,
                        Severity::Error,
                        site(&v.name, &format!("steps.{}", s.id)),
                        if prev == w.name {
                            format!("step `{}` appears twice in `{}`", s.id, v.name)
                        } else {
                            format!(
                                "step `{}` of `{}` repeats a step id of its base `{}`",
                                s.id, v.name, prev
                            )
                        },
                    );
                }
            }
        }
        for l in &w.lines {
            if line_ids.insert(&l.id, &w.name).is_some() && w.name == v.name {
                push(
                    rep,
                    DiagCode::ExplodeDuplicate,
                    Severity::Error,
                    site(&v.name, &format!("lines.{}", l.id)),
                    format!("explode line `{}` appears twice along `{}`", l.id, v.name),
                );
            }
        }
    }
    for s in &v.steps {
        let p = |x: &str| site(&v.name, &format!("steps.{}.{}", s.id, x));
        if s.occurrences.is_empty() {
            push(
                rep,
                DiagCode::ExplodeUnknownOccurrence,
                Severity::Error,
                p("occurrences"),
                format!("step `{}` moves no occurrence", s.id),
            );
        }
        let mut seen = BTreeSet::new();
        for o in &s.occurrences {
            if !seen.insert(o) {
                push(
                    rep,
                    DiagCode::ExplodeDuplicate,
                    Severity::Error,
                    p("occurrences"),
                    format!("step `{}` names `{}` twice", s.id, o),
                );
            }
        }
        let zero = |d: [f64; 3]| norm(d) < 1e-12 || d.iter().any(|x| !x.is_finite());
        let bad = match &s.motion {
            ExplodeMotion::Translate {
                direction: ExplodeDirection::Vector { vector },
                ..
            } => zero(*vector).then_some("direction"),
            ExplodeMotion::Rotate {
                axis: ExplodeAxis::Explicit { direction, .. },
                ..
            }
            | ExplodeMotion::Radial {
                center: RadialCenter::Axis { direction, .. },
                ..
            } => zero(*direction).then_some("axis"),
            _ => None,
        };
        if let Some(what) = bad {
            push(
                rep,
                DiagCode::ExplodeZeroDirection,
                Severity::Error,
                p(what),
                format!("step `{}` has a zero {}", s.id, what),
            );
        }
    }
    for l in &v.lines {
        let mut segs = BTreeSet::new();
        for (i, j) in l.jogs.iter().enumerate() {
            let s = site(&v.name, &format!("lines.{}.jogs[{}]", l.id, i));
            if !(j.at > 0.0 && j.at < 1.0) {
                push(
                    rep,
                    DiagCode::ExplodeBadJog,
                    Severity::Error,
                    s.clone(),
                    format!(
                        "jog {} of line `{}` is at {}; a jog sits strictly inside its segment (0 < at < 1)",
                        i, l.id, j.at
                    ),
                );
            }
            if !segs.insert(j.segment) {
                push(
                    rep,
                    DiagCode::ExplodeBadJog,
                    Severity::Error,
                    s,
                    format!(
                        "line `{}` jogs segment {} twice; one jog per segment",
                        l.id, j.segment
                    ),
                );
            }
        }
    }
}

// ------------------------------------------------------------------------------ evaluation

struct Eval<'a, 'b> {
    doc: &'a AssemblyDocument,
    tree: &'a ResolvedAssembly,
    ctx: &'a ExplodeContext<'b>,
    env: ValueEnv,
    view: String,
    rep: DiagnosticReport,
    /// Resolved paths per writing site, memoised (each site's diagnostics reported once).
    found: BTreeMap<(RelPath, String), Option<NodeId>>,
    /// Where paths start: the root, or a sub-assembly whose included view is running.
    root: NodeId,
}

impl Eval<'_, '_> {
    fn ev(&self) -> Evaluator<'_> {
        Evaluator {
            units: self.ctx.units,
            dynamic: self.ctx.dynamic,
        }
    }

    fn scalar(&mut self, e: &ValueExpr, dim: Dimension, param: &str) -> Option<f64> {
        match self.ev().eval_as(e, &self.env, dim) {
            Ok(v) => Some(v),
            Err(err) => {
                push(
                    &mut self.rep,
                    DiagCode::ValueFailed,
                    Severity::Error,
                    site(&self.view, param),
                    format!("`{}`: {}", param, err),
                );
                None
            }
        }
    }

    fn point(&mut self, p: &[ValueExpr; 3], param: &str) -> Option<[f64; 3]> {
        Some([
            self.scalar(&p[0], Dimension::LENGTH, param)?,
            self.scalar(&p[1], Dimension::LENGTH, param)?,
            self.scalar(&p[2], Dimension::LENGTH, param)?,
        ])
    }

    /// The node at a path: through lock aliases for renamed occurrences, diagnosed once.
    fn node(&mut self, path: &RelPath, param: &str) -> Option<NodeId> {
        let key = (path.clone(), param.to_owned());
        if let Some(n) = self.found.get(&key) {
            return *n;
        }
        let n = self.walk(path, param);
        self.found.insert(key, n);
        n
    }

    fn walk(&mut self, path: &RelPath, param: &str) -> Option<NodeId> {
        let tree = self.tree;
        let mut cur: NodeId = self.root;
        let mut rewritten: Vec<PathSegment> = Vec::new();
        let mut renamed = false;
        for (i, seg) in path.0.iter().enumerate() {
            let child = |name: &str, seg: &PathSegment| {
                tree.node(cur).children.iter().copied().find(|c| {
                    let s = &tree.node(*c).segment;
                    s.name == name && s.instance == seg.instance
                })
            };
            if let Some(c) = child(&seg.name, seg) {
                rewritten.push(seg.clone());
                cur = c;
                continue;
            }
            // Renamed since the step was written?
            let file = tree.node(cur).definition.file.clone();
            let alias = self
                .ctx
                .lock
                .and_then(|l| l.lookup(&item_path(&file, "occurrence", &seg.name)))
                .filter(|l| l.via_alias && l.current.item.len() == 2);
            if let Some(l) = alias {
                let to = l.current.item[1].clone();
                if let Some(c) = child(&to, seg) {
                    let mut s = seg.clone();
                    s.name = to;
                    rewritten.push(s);
                    renamed = true;
                    cur = c;
                    continue;
                }
            }
            // Present in the 150% structure but not in this configuration?
            let prefix = RelPath(path.0[..=i].to_vec()).to_string();
            if self.inactive(cur, &file, seg, &prefix) {
                push(
                    &mut self.rep,
                    DiagCode::ExplodeAbsent,
                    Severity::Warning,
                    site(&self.view, param),
                    format!(
                        "`{}` is not in this configuration; the exploded view skips it",
                        path
                    ),
                );
                return None;
            }
            push(
                &mut self.rep,
                DiagCode::ExplodeUnknownOccurrence,
                Severity::Error,
                site(&self.view, param),
                format!(
                    "exploded view `{}` names `{}`, which `{}` does not contain",
                    self.view,
                    path,
                    if prefix.contains('/') {
                        tree.path(cur)
                    } else {
                        self.doc.path.0.clone()
                    }
                ),
            );
            return None;
        }
        if renamed {
            let to = RelPath(rewritten).to_string();
            self.rep.push(
                DiagCode::AsmViaAlias,
                Severity::Warning,
                site(&self.view, param),
                format!(
                    "`{}` was renamed to `{}`; update the exploded view",
                    path, to
                ),
                Cause::Root,
                vec![Repair::RewritePath {
                    site: site(&self.view, param),
                    from: path.to_string(),
                    to,
                }],
            );
        }
        Some(cur)
    }

    /// The segment names an occurrence of the authoring assembly that this configuration
    /// leaves out (suppressed, or an inactive alternative, pattern or mirror).
    fn inactive(&self, parent: NodeId, file: &FilePath, seg: &PathSegment, display: &str) -> bool {
        if self.tree.suppressed.iter().any(|s| s == display) {
            return true;
        }
        let doc = if parent == 0 {
            Some(std::sync::Arc::new(self.doc.clone()))
        } else {
            self.ctx.source.and_then(|s| {
                s.assembly(file, self.tree.node(parent).revision.as_ref())
                    .ok()
            })
        };
        let Some(doc) = doc else { return false };
        match &seg.instance {
            None => doc.occurrences.iter().any(|o| o.name == seg.name),
            Some(i) => {
                doc.patterns.iter().any(|p| p.name == i.by)
                    || doc.mirrors.iter().any(|m| m.name == i.by)
                    || doc.fasteners.iter().any(|f| f.name == i.by)
            }
        }
    }

    /// The occurrence's centre in its own frame: its bounding-box centre, else its origin.
    fn local_center(&self, n: NodeId) -> [f64; 3] {
        let node = self.tree.node(n);
        self.ctx
            .source
            .and_then(|s| {
                s.configured(
                    &node.definition,
                    node.revision.as_ref(),
                    &node.configuration,
                )
                .ok()
            })
            .and_then(|c| c.bbox)
            .filter(|b| !b.is_empty())
            .map(|b| {
                [
                    (b.min[0] + b.max[0]) / 2.0,
                    (b.min[1] + b.max[1]) / 2.0,
                    (b.min[2] + b.max[2]) / 2.0,
                ]
            })
            .unwrap_or([0.0; 3])
    }
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let l = norm(v);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn subtree(tree: &ResolvedAssembly, n: NodeId, out: &mut Vec<NodeId>) {
    out.push(n);
    for &c in &tree.node(n).children {
        subtree(tree, c, out);
    }
}

/// The value environment of the top assembly at the tree's configuration.
fn top_env(doc: &AssemblyDocument, tree: &ResolvedAssembly, ev: &Evaluator<'_>) -> ValueEnv {
    let overrides = tree
        .node(0)
        .configuration
        .name
        .as_ref()
        .and_then(|n| doc.configuration(n))
        .map(|c| c.overrides.clone())
        .unwrap_or_default();
    ev.evaluate_env(&doc.inputs, &doc.lets, &overrides)
}

/// Evaluate the exploded view `name` of `doc` over its resolved tree.
pub fn evaluate_exploded(
    doc: &AssemblyDocument,
    tree: &ResolvedAssembly,
    name: &str,
    ctx: &ExplodeContext<'_>,
) -> ExplodedState {
    let ev = Evaluator {
        units: ctx.units,
        dynamic: ctx.dynamic,
    };
    let mut e = Eval {
        doc,
        tree,
        ctx,
        env: top_env(doc, tree, &ev),
        view: name.to_owned(),
        rep: DiagnosticReport::default(),
        found: BTreeMap::new(),
        root: 0,
    };
    let mut world: Vec<Transform> = tree.nodes.iter().map(|n| n.world).collect();
    let mut moved: BTreeSet<NodeId> = BTreeSet::new();
    let mut st = DiagnosticReport::default();
    let views = chain(doc, name, &mut st);
    if let Some(v) = doc.exploded.iter().find(|v| v.name == name) {
        check_view(doc, v, &mut st);
    }
    e.rep.append(&st);
    let Some(views) = views else {
        return ExplodedState {
            view: name.to_owned(),
            world,
            moved: Vec::new(),
            lines: Vec::new(),
            diagnostics: e.rep,
        };
    };

    // Lines: their occurrences and anchors (in the occurrence's frame), and trails.
    struct Trail {
        id: String,
        node: NodeId,
        anchor: [f64; 3],
        points: Vec<[f64; 3]>,
        line: ExplodeLine,
    }
    let mut trails: Vec<Trail> = Vec::new();
    let this = views.last().expect("the chain ends at the view itself");
    for w in &views {
        for l in &w.lines {
            let param = format!("lines.{}.occurrence", l.id);
            let Some(n) = e.node(&l.occurrence, &param) else {
                continue;
            };
            let anchor = match &l.anchor {
                Some(a) => match e.point(a, &format!("lines.{}.anchor", l.id)) {
                    Some(p) => p,
                    None => continue,
                },
                None => e.local_center(n),
            };
            trails.push(Trail {
                id: l.id.clone(),
                node: n,
                anchor,
                points: vec![world[n as usize].apply(anchor)],
                line: l.clone(),
            });
        }
    }

    let mut points: Vec<(NodeId, [f64; 3], Vec<[f64; 3]>)> = trails
        .iter()
        .map(|t| (t.node, t.anchor, t.points.clone()))
        .collect();
    run_views(
        &mut e,
        &views,
        &Transform::IDENTITY,
        &mut world,
        &mut moved,
        &mut points,
        0,
    );
    for (tr, (_, _, pts)) in trails.iter_mut().zip(points) {
        tr.points = pts;
    }

    // Jogs on the view's own lines (their segments are in the final trail).
    let mut lines = Vec::new();
    for tr in trails {
        let mut points = tr.points;
        let own = this.lines.iter().any(|l| l.id == tr.id);
        // Later segments first, so earlier segment numbers stay valid as points are inserted.
        let mut order: Vec<usize> = if own {
            (0..tr.line.jogs.len()).collect()
        } else {
            Vec::new()
        };
        order.sort_by(|a, b| tr.line.jogs[*b].segment.cmp(&tr.line.jogs[*a].segment));
        for idx in order {
            let j = &tr.line.jogs[idx];
            let param = format!("lines.{}.jogs[{}]", tr.id, idx);
            let k = j.segment as usize;
            if k + 1 >= points.len() {
                push(
                    &mut e.rep,
                    DiagCode::ExplodeBadJog,
                    Severity::Error,
                    site(name, &param),
                    format!(
                        "line `{}` has {} segment(s); jog {} is on segment {}",
                        tr.id,
                        points.len().saturating_sub(1),
                        idx,
                        k
                    ),
                );
                continue;
            }
            if !(j.at > 0.0 && j.at < 1.0) {
                continue; // reported by the static checks
            }
            let Some(o) = e.point(&j.offset, &param) else {
                continue;
            };
            let (a, b) = (points[k], points[k + 1]);
            let d = sub(b, a);
            if norm(o) < 1e-12 || dot(o, d).abs() > 1e-6 * norm(o) * norm(d) {
                push(
                    &mut e.rep,
                    DiagCode::ExplodeBadJog,
                    Severity::Error,
                    site(name, &param),
                    format!(
                        "jog {} of line `{}` must be offset perpendicular to its segment",
                        idx, tr.id
                    ),
                );
                continue;
            }
            let m = add(a, scale(d, j.at));
            points.splice(k + 1..k + 1, [m, add(m, o), add(b, o)]);
        }
        lines.push(LinePath {
            id: tr.id,
            occurrence: tree.path(tr.node),
            points,
        });
    }
    ExplodedState {
        view: name.to_owned(),
        world,
        moved: moved.into_iter().collect(),
        lines,
        diagnostics: e.rep,
    }
}

type TrailPoints = Vec<(NodeId, [f64; 3], Vec<[f64; 3]>)>;

/// Run a view chain's steps (and the sub-assembly views they include) from the poses in
/// `world`. `frame` is the view's own frame in the top assembly (identity at the top): motions
/// are evaluated in it and applied to the world poses.
fn run_views(
    e: &mut Eval<'_, '_>,
    views: &[&ExplodedView],
    frame: &Transform,
    world: &mut [Transform],
    moved: &mut BTreeSet<NodeId>,
    trails: &mut TrailPoints,
    depth: u32,
) {
    let tree = e.tree;
    let inv = frame.inverse();
    let record = |world: &[Transform], trails: &mut TrailPoints| {
        for (node, anchor, pts) in trails.iter_mut() {
            let now = world[*node as usize].apply(*anchor);
            let last = *pts.last().expect("a trail starts with a point");
            if norm(sub(now, last)) > 1e-12 {
                pts.push(now);
            }
        }
    };
    for w in views {
        run_includes(e, w, None, world, moved, trails, depth);
        for s in &w.steps {
            let p = |x: &str| format!("steps.{}.{}", s.id, x);
            let mut targets: Vec<NodeId> = Vec::new();
            for o in &s.occurrences {
                if let Some(n) = e.node(o, &p("occurrences")) {
                    if !targets.contains(&n) {
                        targets.push(n);
                    }
                }
            }
            // Ancestor and descendant in one step.
            let set: BTreeSet<NodeId> = targets.iter().copied().collect();
            for &t in &targets {
                let mut a = tree.node(t).parent;
                while let Some(x) = a {
                    if set.contains(&x) {
                        let view = e.view.clone();
                        push(
                            &mut e.rep,
                            DiagCode::ExplodeNested,
                            Severity::Info,
                            site(&view, &p("occurrences")),
                            format!(
                                "step `{}` moves `{}` and its assembly `{}`; `{}` moves twice",
                                s.id,
                                tree.path(t),
                                tree.path(x),
                                tree.path(t)
                            ),
                        );
                        break;
                    }
                    a = tree.node(x).parent;
                }
            }
            // Order: ancestors first, so a nested child's own motion composes after.
            targets.sort_by_key(|t| tree.node(*t).depth);
            // Poses in the view's frame, motions there, then back to the top frame.
            let local: Vec<Transform> = world.iter().map(|w| inv.compose(w)).collect();
            let motions = step_motions(e, &local, s, &targets, &p);
            for (t, m) in motions {
                let mw = frame.compose(&m).compose(&inv);
                let mut nodes = Vec::new();
                subtree(tree, t, &mut nodes);
                for n in nodes {
                    world[n as usize] = mw.compose(&world[n as usize]);
                    moved.insert(n);
                }
            }
            record(world, trails);
            run_includes(e, w, Some(&s.id), world, moved, trails, depth);
        }
    }
}

/// Run the views a view includes after one of its steps (`None`: before them).
fn run_includes(
    e: &mut Eval<'_, '_>,
    w: &ExplodedView,
    after: Option<&str>,
    world: &mut [Transform],
    moved: &mut BTreeSet<NodeId>,
    trails: &mut TrailPoints,
    depth: u32,
) {
    for inc in w.include.iter().filter(|i| i.after.as_deref() == after) {
        let param = format!("include.{}", inc.id);
        let Some(n) = e.node(&inc.occurrence, &param) else {
            continue;
        };
        let view = e.view.clone();
        let def = e.tree.node(n).definition.clone();
        let sub_doc = match (
            def.kind == crate::assembly::model::DefinitionKind::Assembly,
            e.ctx
                .source
                .map(|s| s.assembly(&def.file, e.tree.node(n).revision.as_ref())),
        ) {
            (true, Some(Ok(d))) => d,
            (false, _) => {
                push(
                    &mut e.rep,
                    DiagCode::ExplodeUnknownView,
                    Severity::Error,
                    site(&view, &param),
                    format!(
                        "`{}` is a part; only sub-assemblies have exploded views",
                        e.tree.path(n)
                    ),
                );
                continue;
            }
            (true, other) => {
                push(
                    &mut e.rep,
                    DiagCode::ExplodeUnknownView,
                    Severity::Error,
                    site(&view, &param),
                    format!(
                        "cannot read `{}` for its exploded view `{}`{}",
                        def.file,
                        inc.view,
                        match other {
                            Some(Err(err)) => format!(": {}", err),
                            _ => " (no design source)".into(),
                        }
                    ),
                );
                continue;
            }
        };
        if depth > 32 {
            push(
                &mut e.rep,
                DiagCode::ExplodeCycle,
                Severity::Error,
                site(&view, &param),
                "included exploded views nest too deep".into(),
            );
            continue;
        }
        let mut st = DiagnosticReport::default();
        let Some(chain_views) = chain(&sub_doc, &inc.view, &mut st) else {
            e.rep.append(&st);
            continue;
        };
        let ev = e.ev();
        let env = top_env(&sub_doc, e.tree, &ev);
        let mut child = Eval {
            doc: &sub_doc,
            tree: e.tree,
            ctx: e.ctx,
            env,
            view: format!("{}/{}:{}", view, e.tree.path(n), inc.view),
            rep: DiagnosticReport::default(),
            found: BTreeMap::new(),
            root: n,
        };
        // The sub-assembly's frame as it stands now.
        let frame = world[n as usize];
        run_views(
            &mut child,
            &chain_views,
            &frame,
            world,
            moved,
            trails,
            depth + 1,
        );
        e.rep.append(&child.rep);
    }
}

/// The world-space motion of each target of one step, at the current poses.
fn step_motions(
    e: &mut Eval<'_, '_>,
    world: &[Transform],
    s: &ExplodeStep,
    targets: &[NodeId],
    p: &dyn Fn(&str) -> String,
) -> Vec<(NodeId, Transform)> {
    let view = e.view.clone();
    let mut out = Vec::new();
    match &s.motion {
        ExplodeMotion::Translate {
            direction,
            distance,
            spacing,
        } => {
            let dir = match direction {
                ExplodeDirection::Vector { vector } => {
                    if norm(*vector) < 1e-12 {
                        return out;
                    }
                    unit(*vector)
                }
                ExplodeDirection::OccurrenceAxis {
                    occurrence,
                    axis,
                    reversed,
                } => {
                    let Some(n) = e.node(occurrence, &p("direction")) else {
                        return out;
                    };
                    let v = unit(world[n as usize].apply_vector(axis.unit()));
                    if *reversed {
                        scale(v, -1.0)
                    } else {
                        v
                    }
                }
            };
            let Some(d) = e.scalar(distance, Dimension::LENGTH, &p("distance")) else {
                return out;
            };
            let sp = match spacing {
                Some(x) => match e.scalar(x, Dimension::LENGTH, &p("spacing")) {
                    Some(v) => v,
                    None => return out,
                },
                None => 0.0,
            };
            // Spacing follows the order the step lists its occurrences.
            for (i, o) in s.occurrences.iter().enumerate() {
                let Some(n) = e
                    .found
                    .get(&(o.clone(), p("occurrences")))
                    .copied()
                    .flatten()
                else {
                    continue;
                };
                if !targets.contains(&n) || out.iter().any(|(m, _)| *m == n) {
                    continue;
                }
                out.push((n, Transform::translation(scale(dir, d + i as f64 * sp))));
            }
            // Keep the depth order the caller established.
            out.sort_by_key(|(n, _)| targets.iter().position(|t| t == n));
        }
        ExplodeMotion::Rotate { axis, angle } => {
            let (point, dir) = match axis {
                ExplodeAxis::Explicit { point, direction } => {
                    if norm(*direction) < 1e-12 {
                        return out;
                    }
                    let Some(pt) = e.point(point, &p("axis")) else {
                        return out;
                    };
                    (pt, unit(*direction))
                }
                ExplodeAxis::OccurrenceAxis {
                    occurrence,
                    axis,
                    offset,
                } => {
                    let Some(n) = e.node(occurrence, &p("axis")) else {
                        return out;
                    };
                    let local = match offset {
                        Some(o) => match e.point(o, &p("axis")) {
                            Some(v) => v,
                            None => return out,
                        },
                        None => [0.0; 3],
                    };
                    let w = world[n as usize];
                    (w.apply(local), unit(w.apply_vector(axis.unit())))
                }
            };
            let Some(a) = e.scalar(angle, Dimension::ANGLE, &p("angle")) else {
                return out;
            };
            let m = Transform::rotation_about(point, dir, a);
            out.extend(targets.iter().map(|t| (*t, m)));
        }
        ExplodeMotion::Radial { center, distance } => {
            let Some(d) = e.scalar(distance, Dimension::LENGTH, &p("distance")) else {
                return out;
            };
            let (c, axis) = match center {
                RadialCenter::Axis { point, direction } => {
                    if norm(*direction) < 1e-12 {
                        return out;
                    }
                    match e.point(point, &p("center")) {
                        Some(c) => (c, Some(unit(*direction))),
                        None => return out,
                    }
                }
                RadialCenter::Point { point } => match e.point(point, &p("center")) {
                    Some(c) => (c, None),
                    None => return out,
                },
            };
            for &t in targets {
                let at = world[t as usize].apply(e.local_center(t));
                let mut v = sub(at, c);
                if let Some(u) = axis {
                    v = sub(v, scale(u, dot(v, u)));
                }
                if norm(v) < 1e-9 {
                    push(
                        &mut e.rep,
                        DiagCode::ExplodeZeroDirection,
                        Severity::Warning,
                        site(&view, &p("center")),
                        format!(
                            "`{}` sits on the centre of radial step `{}`; it has no direction to move in and stays",
                            e.tree.path(t),
                            s.id
                        ),
                    );
                    continue;
                }
                out.push((t, Transform::translation(scale(unit(v), d))));
            }
        }
    }
    out
}

// ------------------------------------------------------------------------------ projection

pub const EXPLODED_INDEX_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplodedRow {
    pub uuid: ItemUuid,
    pub path: String,
    pub world: Transform,
}

/// The exploded state as index rows: moved occurrences by path (sorted), lines, diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplodedIndex {
    pub format_version: u32,
    pub top: FilePath,
    pub view: String,
    pub occurrences: Vec<ExplodedRow>,
    pub lines: Vec<LinePath>,
    pub diagnostics: Vec<crate::diagnostics::Diagnostic>,
}

pub fn exploded_index(tree: &ResolvedAssembly, state: &ExplodedState) -> ExplodedIndex {
    let mut occurrences: Vec<ExplodedRow> = state
        .moved
        .iter()
        .map(|n| ExplodedRow {
            uuid: tree.node(*n).uuid,
            path: tree.path(*n),
            world: state.world[*n as usize],
        })
        .collect();
    occurrences.sort_by(|a, b| a.path.cmp(&b.path));
    let mut lines = state.lines.clone();
    lines.sort_by(|a, b| a.id.cmp(&b.id));
    ExplodedIndex {
        format_version: EXPLODED_INDEX_FORMAT_VERSION,
        top: tree.top.clone(),
        view: state.view.clone(),
        occurrences,
        lines,
        diagnostics: state.diagnostics.diagnostics.clone(),
    }
}

pub fn exploded_index_json(index: &ExplodedIndex) -> String {
    serde_json::to_string_pretty(index).expect("the exploded index serialises")
}

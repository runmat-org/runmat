//! Kinematic analysis of a motion study.
//!
//! The method is the standard one of computational kinematics (Haug, *Computer-Aided
//! Kinematics and Dynamics of Mechanical Systems*, 1989, ch. 3; Nikravesh, *Computer-Aided
//! Analysis of Mechanical Systems*, 1988, ch. 4): the mates are rigid-body joint constraints
//! `Φ(q, t) = 0`, and each driven mate adds a driving constraint whose value is a known function
//! of time. At each sample time `t_k` the position problem `Φ(q, t_k) = 0` is solved by the mate
//! solver (`runmat-constraint`: damped Newton steps with a Levenberg–Marquardt fallback), started
//! from the pose solved at `t_{k-1}` so the mechanism stays on its assembly branch (continuation).
//! Velocity and acceleration come from differentiating the solved positions: second-order central
//! differences inside the run, and second-order one-sided differences at its ends.
//!
//! A sample whose solve fails (a drive past what the mechanism can reach, a conflict) ends the
//! run there; the frames before it stand and the failure is reported.

use super::{MotionDrive, MotionStudy, DEFAULT_RATE, MAX_FRAMES};
use crate::assembly::model::{
    AssemblyDocument, MateKind, OffsetExpr, Placement, RotationExpr, TargetExpr,
};
use crate::assembly::resolve::{
    resolve, MateStatus, NodeId, PoseStatus, ResolveContext, ResolveRequest, ResolvedAssembly,
};
use crate::assembly::source::{
    ConfigKey, ConfiguredSummary, DefinitionSummary, DesignSource, SourceError,
};
use crate::assembly::model::DefinitionRef;
use crate::ids::{FilePath, PartitionName, RevisionId};
use crate::regen::kernel::Transform;
use crate::units::Dimension;
use crate::value::{Evaluator, ValueEnv, ValueExpr};
use serde::Serialize;
use std::sync::Arc;

/// What a drive moves: an angle (radians) or a length (metres).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DriveQuantity {
    Angle,
    Length,
}

/// One drive's value at every solved frame (SI: radians or metres).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DriveTrace {
    pub mate: String,
    pub quantity: DriveQuantity,
    pub values: Vec<f64>,
}

/// The tracked occurrence's motion: its box centre's world position, velocity and acceleration
/// (m, m/s, m/s²), and its angular speed (rad/s), one entry per solved frame.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Track {
    pub occurrence: String,
    pub node: NodeId,
    pub position: Vec<[f64; 3]>,
    pub velocity: Vec<[f64; 3]>,
    pub acceleration: Vec<[f64; 3]>,
    pub angular_speed: Vec<f64>,
}

/// A solved study.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MotionRun {
    pub name: String,
    pub duration: f64,
    /// Sample times of the solved frames (s).
    pub times: Vec<f64>,
    pub drives: Vec<DriveTrace>,
    /// Each solved frame's world transform of every node of the tree, by node id.
    #[serde(skip)]
    pub frames: Vec<Vec<Transform>>,
    /// Nodes whose pose changes during the run (each with its subtree).
    pub moving: Vec<NodeId>,
    pub track: Option<Track>,
    /// The first sample that did not solve, with why; the run's frames end before it.
    pub stopped: Option<(f64, String)>,
    /// What is wrong with the study itself (an unknown mate, a value of the wrong kind).
    pub problems: Vec<String>,
}

/// The top assembly as a study frame sees it: `top` in place of the source's copy.
struct Overlay<'a> {
    inner: &'a dyn DesignSource,
    top: Arc<AssemblyDocument>,
}

impl DesignSource for Overlay<'_> {
    fn assembly(
        &self,
        file: &FilePath,
        revision: Option<&RevisionId>,
    ) -> Result<Arc<AssemblyDocument>, SourceError> {
        if revision.is_none() && *file == self.top.path {
            return Ok(self.top.clone());
        }
        self.inner.assembly(file, revision)
    }
    fn definition(
        &self,
        definition: &DefinitionRef,
        revision: Option<&RevisionId>,
    ) -> Result<Arc<DefinitionSummary>, SourceError> {
        self.inner.definition(definition, revision)
    }
    fn configured(
        &self,
        definition: &DefinitionRef,
        revision: Option<&RevisionId>,
        configuration: &ConfigKey,
    ) -> Result<Arc<ConfiguredSummary>, SourceError> {
        self.inner.configured(definition, revision, configuration)
    }
    fn partition(&self, file: &FilePath) -> Option<PartitionName> {
        self.inner.partition(file)
    }
    fn latest_release(&self, file: &FilePath) -> Option<RevisionId> {
        self.inner.latest_release(file)
    }
}

/// What a drive's `from` value measures, and the mate's target it sets.
fn drive_quantity(kind: &MateKind, quantity: DriveQuantity) -> bool {
    matches!(
        (kind, quantity),
        (MateKind::Revolute { .. } | MateKind::Angle { .. }, DriveQuantity::Angle)
            | (MateKind::Slider { .. } | MateKind::Distance { .. }, DriveQuantity::Length)
            | (MateKind::Cylindrical { .. }, _)
    )
}

/// Whether mate kind `kind` can be driven at all, and by what.
pub fn drivable(kind: &MateKind) -> Option<&'static str> {
    match kind {
        MateKind::Revolute { .. } | MateKind::Angle { .. } => Some("angle"),
        MateKind::Slider { .. } | MateKind::Distance { .. } => Some("distance"),
        MateKind::Cylindrical { .. } => Some("angle or distance"),
        _ => None,
    }
}

/// Sets mate `kind`'s driven value to `value` (SI).
fn set_target(kind: &mut MateKind, quantity: DriveQuantity, value: f64) {
    let exact = |v: ValueExpr| Some(TargetExpr::Exact { value: v });
    let deg = ValueExpr::deg(value.to_degrees());
    let mm = ValueExpr::mm(value * 1000.0);
    match (kind, quantity) {
        (MateKind::Revolute { angle }, _) | (MateKind::Cylindrical { angle, .. }, DriveQuantity::Angle) => {
            *angle = exact(deg)
        }
        (MateKind::Slider { distance }, _)
        | (MateKind::Cylindrical { distance, .. }, DriveQuantity::Length) => *distance = exact(mm),
        (MateKind::Distance { target, .. }, _) => *target = TargetExpr::Exact { value: mm },
        (MateKind::Angle { target, .. }, _) => *target = TargetExpr::Exact { value: deg },
        _ => {}
    }
}

/// Mate `kind`'s exact value, when it has one: the value its driving dimension is held at.
pub fn mate_value(kind: &MateKind) -> Option<&ValueExpr> {
    fn exact(t: &Option<TargetExpr>) -> Option<&ValueExpr> {
        match t {
            Some(TargetExpr::Exact { value }) => Some(value),
            _ => None,
        }
    }
    match kind {
        MateKind::Revolute { angle } => exact(angle),
        MateKind::Slider { distance } => exact(distance),
        MateKind::Cylindrical { angle, distance } => exact(angle).or(exact(distance)),
        MateKind::Distance { target, .. } | MateKind::Angle { target, .. } => match target {
            TargetExpr::Exact { value } => Some(value),
            _ => None,
        },
        _ => None,
    }
}

/// Holds mate `kind` at `value` (an angle when `angle`, else a length), as a mate position or a
/// study frame does. False when the kind has no value of that quantity.
pub fn set_mate_value(kind: &mut MateKind, value: ValueExpr, angle: bool) -> bool {
    let quantity = if angle {
        DriveQuantity::Angle
    } else {
        DriveQuantity::Length
    };
    if !drive_quantity(kind, quantity) {
        return false;
    }
    let exact = Some(TargetExpr::Exact {
        value: value.clone(),
    });
    match (kind, quantity) {
        (MateKind::Revolute { angle }, _)
        | (MateKind::Cylindrical { angle, .. }, DriveQuantity::Angle) => *angle = exact,
        (MateKind::Slider { distance }, _)
        | (MateKind::Cylindrical { distance, .. }, DriveQuantity::Length) => *distance = exact,
        (MateKind::Distance { target, .. } | MateKind::Angle { target, .. }, _) => {
            *target = TargetExpr::Exact { value }
        }
        _ => return false,
    }
    true
}

/// A drive resolved to SI end values.
struct Drive {
    mate: usize,
    quantity: DriveQuantity,
    from: f64,
    to: f64,
}

/// The value environment of the top assembly in the tree's configuration.
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

fn drive(
    doc: &AssemblyDocument,
    d: &MotionDrive,
    ev: &Evaluator<'_>,
    env: &ValueEnv,
) -> Result<Drive, String> {
    let mate = doc
        .mates
        .iter()
        .position(|m| m.name == d.mate)
        .ok_or_else(|| format!("no mate `{}` in the assembly", d.mate))?;
    let kind = &doc.mates[mate].kind;
    let Some(what) = drivable(kind) else {
        return Err(format!(
            "mate `{}` can't be driven: only revolute, slider, cylindrical, distance and angle mates have a value to drive",
            d.mate
        ));
    };
    let (quantity, from) = match ev.eval_as(&d.from, env, Dimension::ANGLE) {
        Ok(v) => (DriveQuantity::Angle, v),
        Err(_) => match ev.eval_as(&d.from, env, Dimension::LENGTH) {
            Ok(v) => (DriveQuantity::Length, v),
            Err(e) => return Err(format!("`{}` from: {e}", d.mate)),
        },
    };
    if !drive_quantity(kind, quantity) {
        return Err(format!("mate `{}` is driven by {what}", d.mate));
    }
    let dim = match quantity {
        DriveQuantity::Angle => Dimension::ANGLE,
        DriveQuantity::Length => Dimension::LENGTH,
    };
    let to = ev
        .eval_as(&d.to, env, dim)
        .map_err(|e| format!("`{}` to: {e}", d.mate))?;
    Ok(Drive {
        mate,
        quantity,
        from,
        to,
    })
}

/// The placement that starts the next solve at the previous frame's local pose.
pub(super) fn warm_placement(local: &Transform) -> Placement {
    let r = &local.rotation;
    let t = local.translation;
    Placement::Offset {
        offset: OffsetExpr {
            translation: t.map(|v| ValueExpr::mm(v * 1000.0)),
            rotation: RotationExpr::Basis {
                x: [r[0][0], r[1][0], r[2][0]],
                z: [r[0][2], r[1][2], r[2][2]],
            },
        },
    }
}

/// Why a frame's solve is not a pose of the mechanism, if it isn't: a top-level mate that solved
/// in the base pose and no longer does, or an occurrence the base placed that this frame couldn't.
pub(super) fn frame_failure(tree: &ResolvedAssembly, base: &ResolvedAssembly) -> Option<String> {
    let bad = |s: &MateStatus| {
        matches!(
            s,
            MateStatus::Conflicting { .. } | MateStatus::NotConverged | MateStatus::Failed { .. }
        )
    };
    for m in tree.mates.iter().filter(|m| m.owner == 0 && bad(&m.status)) {
        if base
            .mates
            .iter()
            .any(|b| b.owner == 0 && b.name == m.name && bad(&b.status))
        {
            continue;
        }
        return Some(match &m.status {
            MateStatus::Conflicting { with } => {
                format!("mate `{}` conflicts with {}", m.name, with.join(", "))
            }
            MateStatus::Failed { message } => format!("mate `{}`: {message}", m.name),
            _ => format!("mate `{}` did not converge", m.name),
        });
    }
    tree.nodes
        .iter()
        .zip(&base.nodes)
        .enumerate()
        .find(|(_, (n, b))| n.pose == PoseStatus::Failed && b.pose != PoseStatus::Failed)
        .map(|(i, _)| format!("`{}` could not be placed", tree.path(i as NodeId)))
}

/// Second-order derivative estimates of samples `x` spaced `h` apart: central differences inside,
/// three-point one-sided differences at the ends (first derivative); the second derivative takes
/// the three-point stencil, its end values from the neighbouring stencil.
fn derivatives(x: &[f64], h: f64) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    if n < 2 || h <= 0.0 {
        return (vec![0.0; n], vec![0.0; n]);
    }
    let mut d1 = vec![0.0; n];
    let mut d2 = vec![0.0; n];
    if n == 2 {
        let v = (x[1] - x[0]) / h;
        return (vec![v, v], vec![0.0, 0.0]);
    }
    for k in 1..n - 1 {
        d1[k] = (x[k + 1] - x[k - 1]) / (2.0 * h);
        d2[k] = (x[k + 1] - 2.0 * x[k] + x[k - 1]) / (h * h);
    }
    d1[0] = (-3.0 * x[0] + 4.0 * x[1] - x[2]) / (2.0 * h);
    d1[n - 1] = (3.0 * x[n - 1] - 4.0 * x[n - 2] + x[n - 3]) / (2.0 * h);
    d2[0] = d2[1];
    d2[n - 1] = d2[n - 2];
    (d1, d2)
}

/// The rotation angle between two rotations (rad).
fn rotation_between(a: &Transform, b: &Transform) -> f64 {
    // trace(Aᵀ B) = 1 + 2 cos θ.
    let mut tr = 0.0;
    for i in 0..3 {
        for k in 0..3 {
            tr += a.rotation[k][i] * b.rotation[k][i];
        }
    }
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
}

/// Everything a study run needs from the compile.
pub struct MotionContext<'a> {
    pub resolve: &'a ResolveContext<'a>,
    pub request: &'a ResolveRequest,
}

/// Solves study `name` of `doc` (the top assembly of `base`).
pub fn run_study(
    doc: &AssemblyDocument,
    base: &ResolvedAssembly,
    name: &str,
    ctx: &MotionContext<'_>,
) -> MotionRun {
    let mut run = MotionRun {
        name: name.to_owned(),
        ..Default::default()
    };
    let Some(study) = doc.motions.iter().find(|s| s.name == name) else {
        run.problems.push(format!("no motion study `{name}`"));
        return run;
    };
    let ev = Evaluator {
        units: ctx.resolve.units,
        dynamic: ctx.resolve.dynamic,
    };
    let env = top_env(doc, base, &ev);
    let duration = match ev.eval_as(&study.duration, &env, Dimension::TIME) {
        Ok(d) if d > 0.0 && d.is_finite() => d,
        Ok(_) => {
            run.problems.push("the duration must be more than zero".into());
            return run;
        }
        Err(e) => {
            run.problems.push(format!("duration: {e}"));
            return run;
        }
    };
    run.duration = duration;
    if study.drives.is_empty() {
        run.problems.push("the study drives no mate".into());
        return run;
    }
    let mut drives = Vec::new();
    for d in &study.drives {
        match drive(doc, d, &ev, &env) {
            Ok(d) => drives.push(d),
            Err(e) => run.problems.push(e),
        }
    }
    if !run.problems.is_empty() {
        return run;
    }
    let rate = study.rate.filter(|r| *r > 0.0 && r.is_finite()).unwrap_or(DEFAULT_RATE);
    let count = ((duration * rate).ceil() as usize + 1).clamp(2, MAX_FRAMES);
    let h = duration / (count - 1) as f64;
    run.drives = drives
        .iter()
        .map(|d| DriveTrace {
            mate: doc.mates[d.mate].name.clone(),
            quantity: d.quantity,
            values: Vec::new(),
        })
        .collect();
    // Floating occurrences of the top document, by occurrence index, with their node.
    let floating: Vec<(usize, NodeId)> = doc
        .occurrences
        .iter()
        .enumerate()
        .filter(|(_, o)| !o.fixed && !o.suppressed)
        .filter_map(|(i, o)| {
            base.node(0)
                .children
                .iter()
                .copied()
                .find(|&c| base.node(c).segment.name == o.name)
                .map(|c| (i, c))
        })
        .collect();
    let mut frame_doc = doc.clone();
    for k in 0..count {
        let t = h * k as f64;
        let s = study.profile.rise(t / duration);
        for (d, trace) in drives.iter().zip(run.drives.iter_mut()) {
            let value = d.from + (d.to - d.from) * s;
            set_target(&mut frame_doc.mates[d.mate].kind, d.quantity, value);
            trace.values.push(value);
        }
        let overlay = Overlay {
            inner: ctx.resolve.source,
            top: Arc::new(frame_doc.clone()),
        };
        let fctx = ResolveContext {
            source: &overlay,
            ..*ctx.resolve
        };
        let tree = resolve(&base.top, ctx.request, &fctx);
        let same = tree.nodes.len() == base.nodes.len()
            && tree.nodes.iter().zip(&base.nodes).all(|(a, b)| a.uuid == b.uuid);
        let failure = if same {
            frame_failure(&tree, base)
        } else {
            Some("the assembly's structure changed during the motion".into())
        };
        if let Some(why) = failure {
            for trace in &mut run.drives {
                trace.values.pop();
            }
            run.stopped = Some((t, why));
            break;
        }
        // The next solve starts where this one ended.
        for &(i, node) in &floating {
            frame_doc.occurrences[i].placement = warm_placement(&tree.node(node).local);
        }
        run.times.push(t);
        run.frames.push(tree.nodes.iter().map(|n| n.world).collect());
    }
    // What moves: nodes whose world pose differs from the first frame's, top-most only.
    if let Some(first) = run.frames.first() {
        let moved: Vec<bool> = (0..first.len())
            .map(|i| {
                run.frames
                    .iter()
                    .any(|f| f[i].max_difference(&first[i]) > 1e-9)
            })
            .collect();
        run.moving = (0..first.len())
            .filter(|&i| {
                moved[i]
                    && base.nodes[i]
                        .parent
                        .is_none_or(|p| !moved[p as usize])
            })
            .map(|i| i as NodeId)
            .collect();
    }
    run.track = track(doc, study, base, &run, ctx, h);
    run
}

/// The tracked occurrence: the study's `track`, else the first moving top-level occurrence.
fn track(
    _doc: &AssemblyDocument,
    study: &MotionStudy,
    base: &ResolvedAssembly,
    run: &MotionRun,
    ctx: &MotionContext<'_>,
    h: f64,
) -> Option<Track> {
    let node = match &study.track {
        Some(path) => (0..base.nodes.len() as NodeId).find(|&n| base.path(n) == *path)?,
        None => *run.moving.first()?,
    };
    let n = base.node(node);
    let centre = ctx
        .resolve
        .source
        .configured(&n.definition, n.revision.as_ref(), &n.configuration)
        .ok()
        .and_then(|c| c.bbox)
        .filter(|b| !b.is_empty())
        .map(|b| std::array::from_fn(|i| (b.min[i] + b.max[i]) / 2.0))
        .unwrap_or([0.0; 3]);
    let position: Vec<[f64; 3]> = run
        .frames
        .iter()
        .map(|f| f[node as usize].apply(centre))
        .collect();
    let mut velocity = vec![[0.0; 3]; position.len()];
    let mut acceleration = vec![[0.0; 3]; position.len()];
    for axis in 0..3 {
        let x: Vec<f64> = position.iter().map(|p| p[axis]).collect();
        let (d1, d2) = derivatives(&x, h);
        for k in 0..position.len() {
            velocity[k][axis] = d1[k];
            acceleration[k][axis] = d2[k];
        }
    }
    // Angular speed: the rotation between neighbouring frames over their time apart.
    let frames = &run.frames;
    let angular_speed = (0..frames.len())
        .map(|k| {
            let (a, b, steps) = match k {
                _ if frames.len() < 2 => return 0.0,
                0 => (0, 1, 1.0),
                _ if k == frames.len() - 1 => (k - 1, k, 1.0),
                _ => (k - 1, k + 1, 2.0),
            };
            rotation_between(&frames[a][node as usize], &frames[b][node as usize]) / (steps * h)
        })
        .collect();
    Some(Track {
        occurrence: base.path(node),
        node,
        position,
        velocity,
        acceleration,
        angular_speed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn central_differences_are_exact_on_a_quadratic() {
        let h = 0.1;
        let x: Vec<f64> = (0..11).map(|k| 3.0 * (k as f64 * h).powi(2)).collect();
        let (v, a) = derivatives(&x, h);
        for (k, (v, a)) in v.iter().zip(&a).enumerate() {
            assert!((v - 6.0 * k as f64 * h).abs() < 1e-9, "v[{k}] = {v}");
            assert!((a - 6.0).abs() < 1e-6, "a[{k}] = {a}");
        }
    }

    #[test]
    fn the_cycloidal_rise_starts_and_ends_at_rest() {
        let p = super::super::MotionProfile::Cycloidal;
        assert_eq!(p.rise(0.0), 0.0);
        assert!((p.rise(1.0) - 1.0).abs() < 1e-12);
        assert!((p.rise(0.5) - 0.5).abs() < 1e-12);
        let e = 1e-6;
        assert!(p.rise(e) / e < 1e-3, "zero speed at the start");
    }
}

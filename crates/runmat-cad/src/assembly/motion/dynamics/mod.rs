//! Dynamic motion studies: the assembly's mechanism moving under its loads.
//!
//! A study with `kind = dynamic` integrates the constrained Newton–Euler equations of the
//! mechanism ([`model`]): the mates are the joint constraints, each floating top-level body
//! carries the mass and inertia of the parts it holds (from their materials), and gravity,
//! springs, dampers and motors act on it. The file form is
//!
//! ```text
//! motion swing = { kind = dynamic, duration = 2 s, gravity = [0 m/s^2, -9.80665 m/s^2, 0 m/s^2],
//!                  springs = [{ mate = hinge, stiffness = 0.5 N*m/rad, damping = 0.01 N*m*s/rad, free = 0 deg }],
//!                  motors = [{ mate = hinge, torque = 0.1 N*m }] }
//! ```
//!
//! Gravity defaults to `[0, 0, −9.80665] m/s²`. Every frame is an integrated state of the
//! mechanism (no frame is a kinematic pose standing in for dynamics); each frame also records
//! every mate's reaction on the bodies it joins ([`MateReaction`]), the loads a study of one
//! part at that instant takes. Joint limits are not enforced during a dynamic run (they are
//! listed as a problem when the mechanism has them), and contact is not modelled.
//!
//! References: Haug, *Computer-Aided Kinematics and Dynamics of Mechanical Systems*, 1989;
//! Nikravesh, *Computer-Aided Analysis of Mechanical Systems*, 1988; Baumgarte, 1972.

pub mod dense;
pub mod model;
#[cfg(test)]
mod tests;

use super::kinematics::{drivable, set_moving, track, MotionContext, MotionRun};
use super::MotionStudy;
use crate::assembly::model::{AssemblyDocument, DefinitionKind, MateKind};
use crate::assembly::resolve::{resolve, MateSystem, NodeId, ResolveRequest, ResolvedAssembly};
use crate::regen::kernel::Transform;
use crate::units::Dimension;
use crate::value::{Evaluator, ValueEnv, ValueExpr};
use model::{BodyMass, MateLoad, MateMeasure, Model, Sample, State};
use runmat_types::physical::QuantityKind;
use serde::Serialize;

/// The fewest frames per second a dynamic study that doesn't say shows (smooth playback); a
/// faster motion gets more ([`frame_stride`]).
pub const DYNAMIC_RATE: f64 = 30.0;
/// The most a body turns between frames that the study picks (rad, 5°). The straight trail
/// segment between frames then strays from the arc by `r (1 − cos 2.5°)`, under 0.1 % of the
/// radius, so a fast swing's trail reads as a curve rather than crossing chords.
pub const FRAME_TURN: f64 = 5.0 * std::f64::consts::PI / 180.0;
/// The most a body's centre of mass travels between frames, as a fraction of its size (the
/// diagonal of the box with its inertia, `√(6 tr J / m)`): the same 5° step for a body turning
/// about a point half its size away.
pub const FRAME_TRAVEL: f64 = 0.05;
/// The integration steps of a study that picks its own rate come in multiples of this, so the
/// frames can be every `s`-th step for many `s` (its divisors: 1, 2, 3, 4, 5, 6, 8, 10, 12, ...).
const STEP_GRID: usize = 120;
/// Bytes one frame number takes in the index (a full-precision number as JSON text, with its
/// separator).
pub const BYTES_PER_NUMBER: usize = 24;
/// The longest integration step (s); each frame interval is split into equal steps no longer.
pub const MAX_STEP: f64 = 2e-3;
/// Standard gravity (m/s²), ISO 80000-3.
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// A mate's load on one of the components it joins, at one instant: a force and a moment
/// about the mate's connector on that component (world axes, N and N·m). It is everything the
/// mate passes to the component: the joint's reaction and any spring, damper or motor acting
/// along the mate.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JointLoad {
    pub occurrence: String,
    pub node: NodeId,
    /// The connector's origin (m, top assembly frame).
    pub point: [f64; 3],
    pub force: [f64; 3],
    pub moment: [f64; 3],
}

/// A free body's motion at one frame, world axes: the centre of mass (m), its acceleration
/// (m/s²), the angular velocity (rad/s) and the angular acceleration (rad/s²), from the same
/// solve as the frame's reactions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct BodyState {
    pub center: [f64; 3],
    pub acceleration: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub angular_acceleration: [f64; 3],
}

/// A free body of a dynamic study: its mass properties in its own frame (the occurrence's
/// axes) and its motion frame by frame.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BodyMotion {
    pub occurrence: String,
    pub node: NodeId,
    /// kg.
    pub mass: f64,
    /// The centre of mass in the body's own frame (m).
    pub center: [f64; 3],
    /// The inertia tensor about the centre of mass, body axes (kg·m²).
    pub inertia: [[f64; 3]; 3],
    pub frames: Vec<BodyState>,
}

impl BodyState {
    /// The inertia load on a body in this state (d'Alembert): the force `−m a` and the moment
    /// about the centre of mass `−(J α + ω × J ω)`, world axes, with `J` the body's inertia
    /// turned into world axes by `rotation` (body to world).
    pub fn inertia_load(&self, mass: f64, inertia: &[[f64; 3]; 3], rotation: &[[f64; 3]; 3]) -> ([f64; 3], [f64; 3]) {
        // J_world = R J Rᵀ.
        let j: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| (0..3).map(|k| (0..3).map(|l| rotation[a][k] * inertia[k][l] * rotation[b][l]).sum::<f64>()).sum())
        });
        let (w, al) = (self.angular_velocity, self.angular_acceleration);
        let m = dense::add(dense::mat3_vec(&j, al), dense::cross(w, dense::mat3_vec(&j, w)));
        (dense::scale(self.acceleration, -mass), dense::scale(m, -1.0))
    }
}

/// One mate's reaction loads, frame by frame.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MateReaction {
    pub mate: String,
    pub frames: Vec<Vec<JointLoad>>,
}

fn kind(k: QuantityKind) -> Dimension {
    Dimension::quantity_type(k)
}

/// The study's gravity (m/s²).
fn gravity(study: &MotionStudy, ev: &Evaluator<'_>, env: &ValueEnv) -> Result<[f64; 3], String> {
    let Some(g) = &study.gravity else {
        return Ok([0.0, 0.0, -STANDARD_GRAVITY]);
    };
    if g.len() != 3 {
        return Err("gravity takes three accelerations, [x, y, z]".into());
    }
    let mut out = [0.0; 3];
    for (o, e) in out.iter_mut().zip(g) {
        *o = ev
            .eval_as(e, env, kind(QuantityKind::Acceleration))
            .map_err(|e| format!("gravity: {e}"))?;
    }
    Ok(out)
}

/// The quantity a mate's load acts along, checked against the mate.
fn measure_of(doc: &AssemblyDocument, mate: &str, angle: bool) -> Result<MateMeasure, String> {
    let m = doc
        .mates
        .iter()
        .find(|m| m.name == mate)
        .ok_or_else(|| format!("no mate `{mate}` in the assembly"))?;
    let fits = matches!(
        (&m.kind, angle),
        (MateKind::Revolute { .. } | MateKind::Cylindrical { .. }, true)
            | (MateKind::Slider { .. } | MateKind::Cylindrical { .. }, false)
    );
    if !fits {
        let has = drivable(&m.kind).unwrap_or("no value");
        return Err(format!(
            "mate `{mate}` has {has} to load; springs and motors act on revolute, slider and cylindrical mates"
        ));
    }
    Ok(if angle {
        MateMeasure::Angle
    } else {
        MateMeasure::Length
    })
}

fn optional(
    ev: &Evaluator<'_>,
    env: &ValueEnv,
    e: &Option<ValueExpr>,
    dims: &[(Dimension, bool)],
    what: &str,
) -> Result<Option<(f64, bool)>, String> {
    let Some(e) = e else { return Ok(None) };
    let mut last = String::new();
    for &(d, angle) in dims {
        match ev.eval_as(e, env, d) {
            Ok(v) => return Ok(Some((v, angle))),
            Err(err) => last = err.to_string(),
        }
    }
    Err(format!("{what}: {last}"))
}

/// The springs, dampers and motors of the study, against the mate system's connectors.
fn loads(
    doc: &AssemblyDocument,
    study: &MotionStudy,
    system: &MateSystem,
    ev: &Evaluator<'_>,
    env: &ValueEnv,
) -> Result<Vec<MateLoad>, String> {
    let ends = |mate: &str| {
        system
            .mates
            .iter()
            .find(|m| m.name == mate)
            .map(|m| (m.a, m.b))
            .ok_or_else(|| format!("mate `{mate}` is not in the solved mechanism"))
    };
    let mut out = Vec::new();
    for s in &study.springs {
        let what = |f: &str| format!("spring on `{}` {f}", s.mate);
        let k = optional(ev, env, &s.stiffness, &[(kind(QuantityKind::RotationalStiffness), true), (kind(QuantityKind::Stiffness), false)], &what("stiffness"))?;
        let c = optional(ev, env, &s.damping, &[(kind(QuantityKind::TorsionalDamping), true), (kind(QuantityKind::Damping), false)], &what("damping"))?;
        let angle = match (k, c) {
            (Some((_, a)), Some((_, b))) if a != b => {
                return Err(format!("spring on `{}`: stiffness and damping measure different motions", s.mate))
            }
            (Some((_, a)), _) | (None, Some((_, a))) => a,
            (None, None) => return Err(format!("spring on `{}` has no stiffness or damping", s.mate)),
        };
        let free = match &s.free {
            None => 0.0,
            Some(e) => ev
                .eval_as(e, env, if angle { Dimension::ANGLE } else { Dimension::LENGTH })
                .map_err(|e| what(&format!("free: {e}")))?,
        };
        let (a, b) = ends(&s.mate)?;
        out.push(MateLoad {
            a,
            b,
            measure: measure_of(doc, &s.mate, angle)?,
            stiffness: k.map_or(0.0, |v| v.0),
            damping: c.map_or(0.0, |v| v.0),
            free,
            constant: 0.0,
        });
    }
    for m in &study.motors {
        let value = match (&m.torque, &m.force) {
            (Some(_), Some(_)) => return Err(format!("motor on `{}`: a torque or a force, not both", m.mate)),
            (Some(t), None) => (ev.eval_as(t, env, kind(QuantityKind::Moment)).map_err(|e| format!("motor on `{}` torque: {e}", m.mate))?, true),
            (None, Some(f)) => (ev.eval_as(f, env, kind(QuantityKind::Force)).map_err(|e| format!("motor on `{}` force: {e}", m.mate))?, false),
            (None, None) => return Err(format!("motor on `{}` has no torque or force", m.mate)),
        };
        let (a, b) = ends(&m.mate)?;
        out.push(MateLoad {
            a,
            b,
            measure: measure_of(doc, &m.mate, value.1)?,
            stiffness: 0.0,
            damping: 0.0,
            free: 0.0,
            constant: value.0,
        });
    }
    Ok(out)
}

/// The mass properties of body `node` in its own frame: every part under it (and it, if a
/// part), from the source's configured summaries.
fn body_mass(tree: &ResolvedAssembly, node: NodeId, bodies: &[NodeId], ctx: &MotionContext<'_>) -> Result<BodyMass, String> {
    let frame = tree.node(node).world.inverse();
    let mut parts: Vec<(f64, [f64; 3], [[f64; 3]; 3])> = Vec::new();
    let mut queue = vec![node];
    while let Some(n) = queue.pop() {
        let o = tree.node(n);
        queue.extend(o.children.iter().copied().filter(|c| !bodies.contains(c)));
        if o.failed || o.represented_by_ancestor || o.definition.kind != DefinitionKind::Part {
            continue;
        }
        let summary = ctx
            .resolve
            .source
            .configured(&o.definition, o.revision.as_ref(), &o.configuration)
            .map_err(|e| format!("`{}`: {e}", tree.path(n)))?;
        let (Some(m), Some(c), Some(j)) = (summary.mass, summary.center_of_mass, summary.inertia) else {
            return Err(format!("`{}` has no mass: give its part a material", tree.path(n)));
        };
        let rel = frame.compose(&o.world);
        let r = rel.rotation;
        // R J Rᵀ: the part's inertia in the body's axes.
        let jb: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| (0..3).map(|k| (0..3).map(|l| r[a][k] * j[k][l] * r[b][l]).sum::<f64>()).sum())
        });
        parts.push((m, rel.apply(c), jb));
    }
    let mass: f64 = parts.iter().map(|p| p.0).sum();
    if mass.is_nan() || mass <= 0.0 {
        return Err(format!("`{}` has no mass", tree.path(node)));
    }
    let center: [f64; 3] = std::array::from_fn(|k| parts.iter().map(|p| p.0 * p.1[k]).sum::<f64>() / mass);
    let mut inertia = [[0.0; 3]; 3];
    for (m, c, j) in &parts {
        let d = dense::sub(*c, center);
        let d2 = dense::dot(d, d);
        for a in 0..3 {
            for b in 0..3 {
                let delta = if a == b { 1.0 } else { 0.0 };
                inertia[a][b] += j[a][b] + m * (d2 * delta - d[a] * d[b]);
            }
        }
    }
    Ok(BodyMass { mass, center, inertia })
}

/// Solves dynamic study `study` of `doc` (the top assembly of `base`) into `run`, whose
/// duration is set.
pub(super) fn run_dynamic(
    doc: &AssemblyDocument,
    base: &ResolvedAssembly,
    study: &MotionStudy,
    ctx: &MotionContext<'_>,
    ev: &Evaluator<'_>,
    env: &ValueEnv,
    run: &mut MotionRun,
) {
    if !study.drives.is_empty() {
        run.problems.push("a dynamic study moves under its loads: drive mates with motors, not drives".into());
        return;
    }
    let g = match gravity(study, ev, env) {
        Ok(g) => g,
        Err(e) => return run.problems.push(e),
    };
    run.gravity = Some(g);
    let request = ResolveRequest {
        keep_mate_system: true,
        ..ctx.request.clone()
    };
    let tree = resolve(&base.top, &request, ctx.resolve);
    let Some(system) = tree.mate_system.as_deref() else {
        return run.problems.push("the assembly has no mates to move it".into());
    };
    if !system.held_pass.is_empty() {
        return run.problems.push(format!(
            "mates on patterned or mirrored instances ({}) aren't in a dynamic study yet",
            system.held_pass.join(", ")
        ));
    }
    let loads = match loads(doc, study, system, ev, env) {
        Ok(l) => l,
        Err(e) => return run.problems.push(e),
    };
    let free: Vec<NodeId> = system.bodies.iter().filter(|b| !b.ground).map(|b| b.node).collect();
    let all: Vec<NodeId> = system.bodies.iter().map(|b| b.node).collect();
    let mut masses = Vec::new();
    for b in system.bodies.iter().filter(|b| !b.ground) {
        match body_mass(&tree, b.node, &all, ctx) {
            Ok(m) => masses.push((b.entity, m)),
            Err(e) => run.problems.push(e),
        }
    }
    if !run.problems.is_empty() {
        return;
    }
    let model = match Model::new(&system.system, &masses, g, loads) {
        Ok(m) => m,
        Err(e) => return run.problems.push(e),
    };
    let limits = runmat_constraint::constraint_rows(&system.system, &system.values).limits;
    if !limits.is_empty() {
        run.problems.push("joint limits are not enforced in a dynamic study; the run ignores them".into());
    }
    // Each node rides on its nearest body (itself, or the body above it).
    let carried: Vec<Option<(usize, Transform)>> = (0..tree.nodes.len() as NodeId)
        .map(|n| {
            let mut cur = Some(n);
            while let Some(c) = cur {
                if let Some(i) = free.iter().position(|&b| b == c) {
                    let rel = tree.node(c).world.inverse().compose(&tree.node(n).world);
                    return Some((i, rel));
                }
                cur = tree.node(c).parent;
            }
            None
        })
        .collect();
    // The frames the index can carry: each holds a pose per carried node, three vectors per
    // mate end and four per body, its time and its energy.
    let numbers = 12 * carried.iter().flatten().count() + 18 * system.mates.len() + 12 * free.len() + 2;
    let most = (ctx.frame_bytes / (numbers * BYTES_PER_NUMBER)).max(2) - 1;
    let duration = run.duration;
    let rate = study.rate.filter(|r| *r > 0.0 && r.is_finite());
    // A set rate fixes the frames, each split into equal steps; else the steps come first and
    // the motion picks every how many of them is a frame.
    let (steps, per) = match rate {
        Some(rate) => {
            let intervals = ((duration * rate).ceil() as usize).clamp(1, most);
            let per = (duration / intervals as f64 / MAX_STEP).ceil().max(1.0) as usize;
            (intervals * per, Some(per))
        }
        None => (STEP_GRID * (duration / (MAX_STEP * STEP_GRID as f64)).ceil().max(1.0) as usize, None),
    };
    let h = duration / steps as f64;
    let u0 = vec![0.0; 6 * free.len()];
    let (states, failed) = model.integrate(&system.values, &u0, steps, h);
    let stride = per.unwrap_or_else(|| frame_stride(&model, &states, steps, h, most));
    let dt = stride as f64 * h;
    let mut reactions: Vec<MateReaction> = system
        .mates
        .iter()
        .map(|m| MateReaction { mate: m.name.clone(), frames: Vec::new() })
        .collect();
    let mut bodies: Vec<BodyMotion> = free
        .iter()
        .zip(&model.bodies)
        .map(|(&n, b)| BodyMotion {
            occurrence: tree.path(n),
            node: n,
            mass: b.mass.mass,
            center: b.mass.center,
            inertia: b.mass.inertia,
            frames: Vec::new(),
        })
        .collect();
    let mut energy = Vec::new();
    for (k, state) in states.iter().step_by(stride).enumerate() {
        let t = k as f64 * dt;
        let s = match model.sample(t, &state.q, &state.u) {
            Ok(s) => s,
            Err(e) => {
                run.stopped = Some((t, e));
                break;
            }
        };
        run.times.push(s.time);
        let poses: Vec<Transform> = (0..free.len()).map(|i| model.pose(&s.q, i)).collect();
        run.frames.push(
            tree.nodes
                .iter()
                .zip(&carried)
                .map(|(n, c)| match c {
                    Some((i, rel)) => poses[*i].compose(rel),
                    None => n.world,
                })
                .collect(),
        );
        energy.push(model.energy(&s.q, &s.u));
        for (mate, out) in system.mates.iter().zip(reactions.iter_mut()) {
            out.frames.push(joint_loads(&model, &tree, &free, &s, mate));
        }
        for (i, body) in bodies.iter_mut().enumerate() {
            let a = poses[i].rotation;
            let turn = |v: [f64; 3]| dense::mat3_vec(&a, v);
            body.frames.push(BodyState {
                center: model.center(&s.q, i),
                acceleration: [s.udot[6 * i], s.udot[6 * i + 1], s.udot[6 * i + 2]],
                angular_velocity: turn(model.body_rate(&s.u, i)),
                angular_acceleration: turn([s.udot[6 * i + 3], s.udot[6 * i + 4], s.udot[6 * i + 5]]),
            });
        }
    }
    if run.stopped.is_none() {
        run.stopped = failed;
    }
    run.bodies = bodies;
    run.energy = energy;
    run.reactions = reactions;
    set_moving(run, base);
    run.track = track(study, base, run, ctx, dt);
}

/// Every how many of `states` (steps of `h`) a study that doesn't set its rate takes a frame:
/// enough frames that no body turns more than [`FRAME_TURN`] or moves its centre more than
/// [`FRAME_TRAVEL`] of its size between frames at its fastest, and at least [`DYNAMIC_RATE`] a
/// second, up to `most` frame intervals. The stride divides `steps`, so the frames stay evenly
/// spaced and end at the study's end.
fn frame_stride(model: &Model<'_>, states: &[State], steps: usize, h: f64, most: usize) -> usize {
    let duration = steps as f64 * h;
    let norm = |v: [f64; 3]| dense::dot(v, v).sqrt();
    let mut need = (duration * DYNAMIC_RATE).ceil();
    for (i, b) in model.bodies.iter().enumerate() {
        let j = b.mass.inertia;
        let size = (6.0 * (j[0][0] + j[1][1] + j[2][2]) / b.mass.mass).sqrt();
        for s in states {
            need = need.max((duration * norm(model.body_rate(&s.u, i)) / FRAME_TURN).ceil());
            if size > 0.0 {
                need = need.max((duration * norm(model.center_velocity(&s.u, i)) / (FRAME_TRAVEL * size)).ceil());
            }
        }
    }
    let need = (need.min(usize::MAX as f64) as usize).clamp(1, most);
    let divisors: Vec<usize> = (1..=steps).filter(|&d| steps.is_multiple_of(d)).collect();
    // The longest stride giving `need` intervals, unless that is more than the index can carry.
    let fine = divisors.iter().rev().copied().find(|&d| steps / d >= need).unwrap_or(1);
    if steps / fine <= most {
        fine
    } else {
        divisors.iter().copied().find(|&d| steps / d <= most).unwrap_or(steps)
    }
}

/// Mate `mate`'s load on each free component it joins at sample `s`.
fn joint_loads(model: &Model<'_>, tree: &ResolvedAssembly, free: &[NodeId], s: &Sample, mate: &crate::assembly::resolve::MateRows) -> Vec<JointLoad> {
    let rows: Vec<usize> = (0..s.owners.len()).filter(|&r| mate.constraints.contains(&s.owners[r])).collect();
    let mut out = Vec::new();
    for connector in [mate.a, mate.b] {
        let runmat_constraint::EntityKind::Connector { body, .. } = &model.sys.entity(connector).kind else {
            continue;
        };
        let Some(i) = model.body_index(*body) else { continue };
        // The joint's reaction, then the mate's springs, dampers and motors: a force at the
        // centre of mass and a moment about it.
        let (mut force, mut moment) = model.reaction_on(s, &rows, i);
        let center = model.center(&s.q, i);
        for load in model.loads.iter().filter(|l| (l.a, l.b) == (mate.a, mate.b)) {
            if let Some((at, f, t)) = model.mate_load_on(&s.q, &s.u, load, i) {
                force = dense::add(force, f);
                moment = dense::add(moment, dense::add(t, dense::cross(dense::sub(at, center), f)));
            }
        }
        let point = model.connector(&s.q, connector).translation;
        let arm = dense::sub(point, center);
        out.push(JointLoad {
            occurrence: tree.path(free[i]),
            node: free[i],
            point,
            force,
            moment: dense::sub(moment, dense::cross(arm, force)),
        });
    }
    out
}

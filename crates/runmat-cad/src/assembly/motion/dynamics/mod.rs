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
use super::{MotionStudy, MAX_FRAMES};
use crate::assembly::model::{AssemblyDocument, DefinitionKind, MateKind};
use crate::assembly::resolve::{resolve, MateSystem, NodeId, ResolveRequest, ResolvedAssembly};
use crate::regen::kernel::Transform;
use crate::units::Dimension;
use crate::value::{Evaluator, ValueEnv, ValueExpr};
use model::{BodyMass, MateLoad, MateMeasure, Model, Sample};
use runmat_types::physical::QuantityKind;
use serde::Serialize;

/// Frames per second of a dynamic study that doesn't say.
pub const DYNAMIC_RATE: f64 = 30.0;
/// The longest integration step (s); each frame interval is split into equal steps no longer.
pub const MAX_STEP: f64 = 2e-3;
/// Standard gravity (m/s²), ISO 80000-3.
pub const STANDARD_GRAVITY: f64 = 9.80665;

/// A mate's load on one of the components it joins, at one instant: a force and a moment
/// about the mate's connector on that component (world axes, N and N·m).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JointLoad {
    pub occurrence: String,
    pub node: NodeId,
    /// The connector's origin (m, top assembly frame).
    pub point: [f64; 3],
    pub force: [f64; 3],
    pub moment: [f64; 3],
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
    let rate = study.rate.filter(|r| *r > 0.0 && r.is_finite()).unwrap_or(DYNAMIC_RATE);
    let count = ((run.duration * rate).ceil() as usize + 1).clamp(2, MAX_FRAMES);
    let dt = run.duration / (count - 1) as f64;
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
    let u0 = vec![0.0; 6 * free.len()];
    let mut reactions: Vec<MateReaction> = system
        .mates
        .iter()
        .map(|m| MateReaction { mate: m.name.clone(), frames: Vec::new() })
        .collect();
    let mut energy = Vec::new();
    let result = model.simulate(&system.values, &u0, count, dt, MAX_STEP, |s: &Sample| {
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
            out.frames.push(joint_loads(&model, &tree, &free, s, mate));
        }
    });
    if let Err((t, why)) = result {
        run.stopped = Some((t, why));
    }
    run.energy = energy;
    run.reactions = reactions;
    set_moving(run, base);
    run.track = track(study, base, run, ctx, dt);
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
        let (force, moment) = model.reaction_on(s, &rows, i);
        let point = model.connector(&s.q, connector).translation;
        let arm = dense::sub(point, model.center(&s.q, i));
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

//! Closed-form starting poses for fastened bodies.
//!
//! A fasten's rotation rows are the vector part of the connectors' relative quaternion,
//! `sin(θ/2)·axis`. Their slope falls to zero as the relative turn `θ` nears half a turn, so
//! Newton steps from a start more than a quarter turn away stall (a foot that starts upright
//! and is fastened with its connector's Z pointing down). Before iterating, each body reached
//! from a grounded body through fastens is placed where its fasten puts it exactly (the body's
//! rotation is the target connector's rotation times the inverse of its own connector's local
//! rotation, and its position follows), when its start is more than a quarter turn from that
//! pose. Smaller gaps keep the caller's start. A fasten that contradicts another one is left
//! to the solve, which reports it.

use crate::system::{Constraint, ConstraintKindRecord, EntityKind, Frame, Mate, System};

/// Relative turns above this (`|cos(θ/2)| < cos 45°`, so `θ > 90°`) are seeded.
const SEED_BELOW_COS_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2 - 1e-9;

struct Side {
    body: usize,
    local: Frame,
}

/// Seeds `x` (the variables' starting values) in place.
pub(crate) fn seed_fastens(sys: &System, x: &mut [f64], is_free: &[bool]) {
    let bodies = |e: usize| match &sys.entities[e].kind {
        EntityKind::Body {
            position,
            orientation,
        } => Some((*position, *orientation)),
        _ => None,
    };
    let side = |c: crate::EntityId| match &sys.entities[c.index()].kind {
        EntityKind::Connector { body, local } => Some(Side {
            body: body.index(),
            local: *local,
        }),
        _ => None,
    };
    let fastens: Vec<(Side, Side)> = sys
        .constraints
        .iter()
        .filter(|r| !r.reference)
        .filter_map(|r| match &r.constraint {
            ConstraintKindRecord::User(Constraint::Mate(Mate::Fasten { a, b })) => {
                Some((side(*a)?, side(*b)?))
            }
            _ => None,
        })
        .collect();
    if fastens.is_empty() {
        return;
    }
    let pose_of = |x: &[f64], e: usize| -> Option<Frame> {
        let (p, q) = bodies(e)?;
        Some(Frame {
            origin: p.map(|v| x[v.index()]),
            rotation: normalize(q.map(|v| x[v.index()])),
        })
    };
    // A body moves only when all seven of its variables are free; one that is held in part
    // (or grounded) is a fixed reference.
    let movable = |e: usize| {
        bodies(e).is_some_and(|(p, q)| p.iter().chain(q.iter()).all(|v| is_free[v.index()]))
    };
    // Each placed body's pose: grounded and held bodies first; then the fasten tree.
    let mut placed: Vec<Option<Frame>> = vec![None; sys.entities.len()];
    for (a, b) in &fastens {
        for s in [a, b] {
            if !movable(s.body) {
                placed[s.body] = pose_of(x, s.body);
            }
        }
    }
    loop {
        let mut grew = true;
        while grew {
            grew = false;
            for (a, b) in &fastens {
                for (from, to) in [(a, b), (b, a)] {
                    let Some(base) = placed[from.body] else {
                        continue;
                    };
                    if placed[to.body].is_some() || !movable(to.body) {
                        continue;
                    }
                    let Some(start) = pose_of(x, to.body) else {
                        continue;
                    };
                    let target = fastened_pose(&base, &from.local, &to.local);
                    let turn = qmul(conj(target.rotation), start.rotation)[0].abs();
                    if turn < SEED_BELOW_COS_HALF {
                        write(x, bodies(to.body).expect("a body"), &target, &start);
                    }
                    placed[to.body] = Some(target);
                    grew = true;
                }
            }
        }
        // Fastens among free bodies only: the first unplaced one keeps its start and the rest
        // follow it.
        let root = fastens
            .iter()
            .find_map(|(a, _)| placed[a.body].is_none().then_some(a.body));
        match root {
            Some(r) => placed[r] = pose_of(x, r),
            None => break,
        }
    }
}

/// The pose of the body whose connector (`to_local`) coincides with the connector
/// `from_local` on a body at `base`.
fn fastened_pose(base: &Frame, from_local: &Frame, to_local: &Frame) -> Frame {
    let world = compose(base, from_local);
    let rotation = normalize(qmul(world.rotation, conj(to_local.rotation)));
    let o = rotate(rotation, to_local.origin);
    Frame {
        origin: [
            world.origin[0] - o[0],
            world.origin[1] - o[1],
            world.origin[2] - o[2],
        ],
        rotation,
    }
}

fn write(
    x: &mut [f64],
    (p, q): ([crate::VarId; 3], [crate::VarId; 4]),
    target: &Frame,
    start: &Frame,
) {
    // Keep the quaternion's sign nearest the start.
    let dot: f64 = (0..4).map(|k| target.rotation[k] * start.rotation[k]).sum();
    let s = if dot < 0.0 { -1.0 } else { 1.0 };
    for k in 0..3 {
        x[p[k].index()] = target.origin[k];
    }
    for k in 0..4 {
        x[q[k].index()] = s * target.rotation[k];
    }
}

fn compose(a: &Frame, b: &Frame) -> Frame {
    let o = rotate(a.rotation, b.origin);
    Frame {
        origin: [a.origin[0] + o[0], a.origin[1] + o[1], a.origin[2] + o[2]],
        rotation: normalize(qmul(a.rotation, b.rotation)),
    }
}

fn normalize(q: [f64; 4]) -> [f64; 4] {
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    if n > 0.0 {
        q.map(|v| v / n)
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

fn conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn rotate(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let r = qmul(qmul(q, [0.0, v[0], v[1], v[2]]), conj(q));
    [r[1], r[2], r[3]]
}

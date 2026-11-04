//! The sketch solver seam and its two implementations.
//!
//! - [`ExplicitSketchSolver`] takes coordinates as written (a stub for tests that do not
//!   exercise constraints).
//! - [`ConstraintSketchSolver`] adapts `runmat-constraint` (A1). It honours that crate's
//!   contract: internal constraints (arc radius) are skipped when mapping back, the later
//!   constraint in statement order is the one reported redundant, starting values come from the
//!   `solved { }` cache when present (so branches stay put), and the report is mapped here
//!   because it has no serde.

use crate::ids::SketchEntityId;
use crate::model::sketch::{
    PointExpr, PointRole, Sketch, SketchConstraintKind as K, SketchGeometry as G,
    SketchSolveFailure, SketchSolveStatus, SketchTarget, SolvedCache, SolvedKey, SolvedSketch,
};
use crate::units::Dimension;
use crate::value::{Evaluator, ValueEnv};
use std::collections::BTreeMap;

pub trait SketchSolver {
    /// An id and version for cache keys.
    fn id(&self) -> String;
    fn solve(
        &self,
        sketch: &Sketch,
        eval: &Evaluator<'_>,
        env: &ValueEnv,
    ) -> Result<SolvedSketch, SketchSolveFailure>;
}

fn fail(message: impl Into<String>) -> SketchSolveFailure {
    SketchSolveFailure {
        message: message.into(),
        constraints: Vec::new(),
    }
}

type StartingGeometry = (
    BTreeMap<(SketchEntityId, PointRole), [f64; 2]>,
    BTreeMap<SketchEntityId, f64>,
);

/// Starting geometry: every entity point evaluated from its expression, shared points
/// (`l1.end`) resolved, and the `solved { }` cache preferred where it has a value.
fn starting_points(
    sketch: &Sketch,
    eval: &Evaluator<'_>,
    env: &ValueEnv,
) -> Result<StartingGeometry, SketchSolveFailure> {
    let by_id: BTreeMap<&SketchEntityId, &G> = sketch
        .entities
        .iter()
        .map(|e| (&e.id, &e.geometry))
        .collect();
    fn expr_of(g: &G, role: PointRole) -> Option<&PointExpr> {
        match (g, role) {
            (G::Point { at }, PointRole::At) => Some(at),
            (G::Line { start, .. }, PointRole::Start) => Some(start),
            (G::Line { end, .. }, PointRole::End) => Some(end),
            (G::Arc { center, .. }, PointRole::Center) => Some(center),
            (G::Arc { start, .. }, PointRole::Start) => Some(start),
            (G::Arc { end, .. }, PointRole::End) => Some(end),
            (G::Circle { center, .. }, PointRole::Center) => Some(center),
            _ => None,
        }
    }
    fn point(
        by_id: &BTreeMap<&SketchEntityId, &G>,
        id: &SketchEntityId,
        role: PointRole,
        eval: &Evaluator<'_>,
        env: &ValueEnv,
        depth: u32,
    ) -> Result<[f64; 2], SketchSolveFailure> {
        if depth > 64 {
            return Err(fail(format!("point references loop at {}.{:?}", id, role)));
        }
        let g = by_id
            .get(id)
            .ok_or_else(|| fail(format!("unknown sketch entity `{}`", id)))?;
        let e =
            expr_of(g, role).ok_or_else(|| fail(format!("`{}` has no {:?} point", id, role)))?;
        match e {
            PointExpr::Coords { x, y } => {
                let x = eval
                    .eval_as(x, env, Dimension::LENGTH)
                    .map_err(|e| fail(e.to_string()))?;
                let y = eval
                    .eval_as(y, env, Dimension::LENGTH)
                    .map_err(|e| fail(e.to_string()))?;
                Ok([x, y])
            }
            PointExpr::Of { entity, role } => point(by_id, entity, *role, eval, env, depth + 1),
        }
    }
    let mut points = BTreeMap::new();
    let mut radii = BTreeMap::new();
    for e in &sketch.entities {
        let roles: &[PointRole] = match &e.geometry {
            G::Point { .. } => &[PointRole::At],
            G::Line { .. } => &[PointRole::Start, PointRole::End],
            G::Arc { .. } => &[PointRole::Center, PointRole::Start, PointRole::End],
            G::Circle { radius, .. } => {
                let r = sketch
                    .solved
                    .scalars
                    .get(&SolvedKey::Radius {
                        entity: e.id.clone(),
                    })
                    .copied()
                    .map(Ok)
                    .unwrap_or_else(|| {
                        eval.eval_as(radius, env, Dimension::LENGTH)
                            .map_err(|e| fail(e.to_string()))
                    })?;
                radii.insert(e.id.clone(), r);
                &[PointRole::Center]
            }
        };
        for &role in roles {
            let p = match sketch.solved.points.get(&SolvedKey::Point {
                entity: e.id.clone(),
                role,
            }) {
                Some(p) => *p,
                None => point(&by_id, &e.id, role, eval, env, 0)?,
            };
            points.insert((e.id.clone(), role), p);
        }
    }
    Ok((points, radii))
}

fn cache_of(
    points: &BTreeMap<(SketchEntityId, PointRole), [f64; 2]>,
    radii: &BTreeMap<SketchEntityId, f64>,
) -> SolvedCache {
    SolvedCache {
        points: points
            .iter()
            .map(|((e, r), p)| {
                (
                    SolvedKey::Point {
                        entity: e.clone(),
                        role: *r,
                    },
                    *p,
                )
            })
            .collect(),
        scalars: radii
            .iter()
            .map(|(e, r)| (SolvedKey::Radius { entity: e.clone() }, *r))
            .collect(),
    }
}

/// Takes the written coordinates; checks nothing.
#[derive(Clone, Debug, Default)]
pub struct ExplicitSketchSolver;

impl SketchSolver for ExplicitSketchSolver {
    fn id(&self) -> String {
        "explicit/1".into()
    }

    fn solve(
        &self,
        sketch: &Sketch,
        eval: &Evaluator<'_>,
        env: &ValueEnv,
    ) -> Result<SolvedSketch, SketchSolveFailure> {
        // Written coordinates win over the cache: this solver solves nothing.
        let mut plain = sketch.clone();
        plain.solved = SolvedCache::default();
        let (points, radii) = starting_points(&plain, eval, env)?;
        let cache = cache_of(&points, &radii);
        Ok(SolvedSketch {
            points,
            radii,
            status: SketchSolveStatus::Explicit,
            entity_dof: BTreeMap::new(),
            dof: None,
            redundant: Vec::new(),
            conflicting: Vec::new(),
            cache,
        })
    }
}

/// `runmat-constraint` behind the seam.
#[derive(Clone, Debug, Default)]
pub struct ConstraintSketchSolver;

impl SketchSolver for ConstraintSketchSolver {
    fn id(&self) -> String {
        format!("runmat-constraint/{}", env!("CARGO_PKG_VERSION"))
    }

    fn solve(
        &self,
        sketch: &Sketch,
        eval: &Evaluator<'_>,
        env: &ValueEnv,
    ) -> Result<SolvedSketch, SketchSolveFailure> {
        use runmat_constraint::{
            solve, Constraint as C, ConstraintId, EntityId, SolveOptions, Status, System,
        };
        let (start, radii0) = starting_points(sketch, eval, env)?;
        let mut sys = System::new();
        let build = |e: runmat_constraint::BuildError| fail(format!("sketch build: {:?}", e));

        // One solver point per distinct authored point; `Of` references share the target's.
        let by_id: BTreeMap<&SketchEntityId, &G> = sketch
            .entities
            .iter()
            .map(|e| (&e.id, &e.geometry))
            .collect();
        let mut point_ids: BTreeMap<(SketchEntityId, PointRole), EntityId> = BTreeMap::new();
        fn canonical(
            by_id: &BTreeMap<&SketchEntityId, &G>,
            id: &SketchEntityId,
            role: PointRole,
            depth: u32,
        ) -> (SketchEntityId, PointRole) {
            if depth > 64 {
                return (id.clone(), role);
            }
            let e = by_id.get(id).and_then(|g| match (g, role) {
                (G::Line { start, .. }, PointRole::Start)
                | (G::Arc { start, .. }, PointRole::Start) => Some(start),
                (G::Line { end, .. }, PointRole::End) | (G::Arc { end, .. }, PointRole::End) => {
                    Some(end)
                }
                (G::Arc { center, .. }, PointRole::Center)
                | (G::Circle { center, .. }, PointRole::Center) => Some(center),
                (G::Point { at }, PointRole::At) => Some(at),
                _ => None,
            });
            match e {
                Some(PointExpr::Of { entity, role }) => canonical(by_id, entity, *role, depth + 1),
                _ => (id.clone(), role),
            }
        }
        let mut get_point = |sys: &mut System, id: &SketchEntityId, role: PointRole| -> EntityId {
            let key = canonical(&by_id, id, role, 0);
            if let Some(p) = point_ids.get(&key) {
                let p = *p;
                point_ids.insert((id.clone(), role), p);
                return p;
            }
            let at = start.get(&key).copied().unwrap_or([0.0, 0.0]);
            let p = sys.point2(at[0], at[1]);
            point_ids.insert(key, p);
            point_ids.insert((id.clone(), role), p);
            p
        };
        let mut curves: BTreeMap<SketchEntityId, EntityId> = BTreeMap::new();
        let mut is_round: BTreeMap<SketchEntityId, bool> = BTreeMap::new();
        for e in &sketch.entities {
            let id = match &e.geometry {
                G::Point { .. } => get_point(&mut sys, &e.id, PointRole::At),
                G::Line { .. } => {
                    let a = get_point(&mut sys, &e.id, PointRole::Start);
                    let b = get_point(&mut sys, &e.id, PointRole::End);
                    sys.line(a, b).map_err(build)?
                }
                G::Arc { .. } => {
                    let c = get_point(&mut sys, &e.id, PointRole::Center);
                    let a = get_point(&mut sys, &e.id, PointRole::Start);
                    let b = get_point(&mut sys, &e.id, PointRole::End);
                    is_round.insert(e.id.clone(), true);
                    sys.arc(c, a, b).map_err(build)?
                }
                G::Circle { .. } => {
                    let c = get_point(&mut sys, &e.id, PointRole::Center);
                    is_round.insert(e.id.clone(), true);
                    sys.circle(c, radii0.get(&e.id).copied().unwrap_or(1.0))
                        .map_err(build)?
                }
            };
            if e.construction {
                sys.set_construction(id, true);
            }
            curves.insert(e.id.clone(), id);
        }
        let target = |t: &SketchTarget,
                      point_ids: &BTreeMap<(SketchEntityId, PointRole), EntityId>|
         -> Result<EntityId, SketchSolveFailure> {
            match t {
                SketchTarget::Entity { entity } => curves
                    .get(entity)
                    .copied()
                    .ok_or_else(|| fail(format!("unknown entity `{}`", entity))),
                SketchTarget::Point { entity, role } => point_ids
                    .get(&(entity.clone(), *role))
                    .copied()
                    .ok_or_else(|| fail(format!("unknown point `{}.{:?}`", entity, role))),
            }
        };
        let len = |v: &crate::value::ValueExpr| {
            eval.eval_as(v, env, Dimension::LENGTH)
                .map_err(|e| fail(e.to_string()))
        };
        let mut ours: BTreeMap<ConstraintId, usize> = BTreeMap::new();
        for (i, c) in sketch.constraints.iter().enumerate() {
            let t = |x: &SketchTarget| target(x, &point_ids);
            let round = |x: &SketchTarget| matches!(x, SketchTarget::Entity { entity } if is_round.get(entity).copied().unwrap_or(false));
            let cc = match &c.kind {
                K::Coincident { a, b } => C::Coincident(t(a)?, t(b)?),
                K::Horizontal { line } => C::Horizontal(t(line)?),
                K::Vertical { line } => C::Vertical(t(line)?),
                K::Parallel { a, b } => C::Parallel(t(a)?, t(b)?),
                K::Perpendicular { a, b } => C::Perpendicular(t(a)?, t(b)?),
                K::Tangent { a, b } => C::Tangent(t(a)?, t(b)?),
                K::Equal { a, b } => {
                    if round(a) {
                        C::EqualRadius(t(a)?, t(b)?)
                    } else {
                        C::EqualLength(t(a)?, t(b)?)
                    }
                }
                K::Fix { target: x } => C::Fix(t(x)?),
                K::Distance { a, b, value } => C::Distance {
                    a: t(a)?,
                    b: t(b)?,
                    value: len(value)?,
                },
                K::Length { line, value } => C::Length {
                    line: t(line)?,
                    value: len(value)?,
                },
                K::Radius { curve, value } => C::Radius {
                    curve: t(curve)?,
                    value: len(value)?,
                },
                K::Angle { a, b, value } => C::Angle {
                    a: t(a)?,
                    b: t(b)?,
                    value: eval
                        .eval_as(value, env, Dimension::ANGLE)
                        .map_err(|e| fail(e.to_string()))?,
                },
                K::Limit {
                    dimension,
                    min,
                    max,
                } => {
                    let dim = if matches!(**dimension, K::Angle { .. }) {
                        Dimension::ANGLE
                    } else {
                        Dimension::LENGTH
                    };
                    let range = |v: &crate::value::ValueExpr| {
                        eval.eval_as(v, env, dim).map_err(|e| fail(e.to_string()))
                    };
                    let (lo, hi) = (range(min)?, range(max)?);
                    let inner = match &**dimension {
                        K::Distance { a, b, .. } => C::Distance {
                            a: t(a)?,
                            b: t(b)?,
                            value: lo,
                        },
                        K::Length { line, .. } => C::Length {
                            line: t(line)?,
                            value: lo,
                        },
                        K::Radius { curve, .. } => C::Radius {
                            curve: t(curve)?,
                            value: lo,
                        },
                        K::Angle { a, b, .. } => C::Angle {
                            a: t(a)?,
                            b: t(b)?,
                            value: lo,
                        },
                        _ => {
                            return Err(fail(
                                "a range limit applies to a distance, length, radius or angle",
                            ))
                        }
                    };
                    C::limit(inner, lo, hi)
                }
            };
            let id = sys.constrain(cc).map_err(build)?;
            ours.insert(id, i);
        }

        let report = solve(&sys, &SolveOptions::default());
        let map_ids = |ids: Vec<ConstraintId>| -> Vec<usize> {
            let mut v: Vec<usize> = ids.iter().filter_map(|c| ours.get(c).copied()).collect();
            v.sort();
            v.dedup();
            v
        };
        match report.status {
            Status::Conflicting => {
                return Err(SketchSolveFailure {
                    message: "sketch constraints conflict".into(),
                    constraints: map_ids(report.conflict_set()),
                })
            }
            Status::NotConverged => return Err(fail("sketch solve did not converge")),
            _ => {}
        }
        let mut points = BTreeMap::new();
        for ((e, r), p) in &point_ids {
            let vars = sys.entity_vars(*p);
            points.insert(
                (e.clone(), *r),
                [report.value(vars[0]), report.value(vars[1])],
            );
        }
        let mut radii = BTreeMap::new();
        for e in &sketch.entities {
            if let G::Circle { .. } = e.geometry {
                let vars = sys.entity_vars(curves[&e.id]);
                radii.insert(
                    e.id.clone(),
                    report.value(*vars.last().expect("circle has a radius")),
                );
            }
        }
        let entity_dof = curves
            .iter()
            .filter_map(|(e, id)| report.entity_dof(*id).map(|d| (e.clone(), d as u32)))
            .collect();
        let status = match report.status {
            Status::WellDefined => SketchSolveStatus::WellDefined,
            Status::UnderDefined => SketchSolveStatus::UnderDefined,
            _ => SketchSolveStatus::OverDefined,
        };
        let cache = cache_of(&points, &radii);
        Ok(SolvedSketch {
            points,
            radii,
            status,
            entity_dof,
            dof: Some(report.dof as u32),
            redundant: map_ids(report.redundant_ids()),
            conflicting: Vec::new(),
            cache,
        })
    }
}

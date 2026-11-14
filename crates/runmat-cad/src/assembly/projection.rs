//! Converted sketch entities in an assembly (Convert entities, Offset of a component edge): an
//! entity written `from = published(occurrence = upper, name = upper_e1)` takes its geometry from
//! the edges component `upper` publishes as `upper_e1`, carried into assembly coordinates by the
//! occurrence's placement and projected onto the sketch's plane, on every regeneration. The
//! solver holds it fixed, so the rest of the sketch is dimensioned and related to it, and it
//! follows the component when the component moves or its part changes.
//!
//! The projected polyline is read as the simplest curve that runs through all of its points
//! (to [`FIT`]): a line, then a circle (a closed edge) or an arc, then a spline through them.

use crate::assembly::model::{PathSegment, RelPath};
use crate::assembly::resolve::Resolver;
use crate::assembly::source::PublishedGeometry;
use crate::ids::SketchEntityId;
use crate::model::datum::Frame;
use crate::model::sketch::{
    PointExpr, Sketch, SketchConstraint, SketchConstraintKind, SketchGeometry, SketchTarget,
};
use crate::regen::kernel::Transform;
use crate::value::ValueExpr;

/// How far (millimetres) a projected point may sit from the curve read through it.
pub const FIT: f64 = 0.01;

/// The most points a projected spline keeps.
const SPLINE_POINTS: usize = 24;

impl<'a, 'b> Resolver<'a, 'b> {
    /// `sketch` with each converted entity re-read from what it was converted from and fixed,
    /// in the sketch's `frame` (assembly coordinates at node `owner`).
    pub(in crate::assembly) fn with_projections(
        &mut self,
        owner: u32,
        frame: &Frame,
        sketch: &Sketch,
    ) -> Result<Sketch, String> {
        let mut out = sketch.clone();
        for entity in out.entities.iter_mut() {
            let Some(from) = entity.projected.clone() else {
                continue;
            };
            let chains = self.projection_chains(owner, &from.occurrence, &from.published)?;
            let [chain] = chains.as_slice() else {
                return Err(format!(
                    "`{}`'s `{}` is {} separate edge chains; convert one connected chain per entity",
                    from.occurrence,
                    from.published,
                    chains.len()
                ));
            };
            let local: Vec<[f64; 2]> = chain.iter().map(|p| in_plane(frame, *p)).collect();
            entity.geometry = fit(&local).ok_or_else(|| {
                format!(
                    "`{}`'s `{}` projects to a point on this sketch's plane",
                    from.occurrence, from.published
                )
            })?;
        }
        for entity in &sketch.entities {
            if entity.projected.is_some() {
                out.constraints.push(SketchConstraint {
                    id: None,
                    kind: SketchConstraintKind::Fix {
                        target: SketchTarget::Entity {
                            entity: entity.id.clone(),
                        },
                    },
                });
            }
        }
        Ok(out)
    }

    /// The edge chains component `occurrence` (a dotted path below `owner`) publishes as
    /// `name`, in `owner`'s coordinates (metres).
    fn projection_chains(
        &mut self,
        owner: u32,
        occurrence: &str,
        name: &str,
    ) -> Result<Vec<Vec<[f64; 3]>>, String> {
        let path = RelPath(
            occurrence
                .split('.')
                .map(|n| PathSegment {
                    name: n.to_owned(),
                    instance: None,
                })
                .collect(),
        );
        let host = self
            .walk(owner, &path)
            .map_err(|_| format!("the assembly has no component `{occurrence}`"))?;
        let node = self.nodes[host as usize].clone();
        let summary = self
            .configured_summary(&node.definition, node.revision.as_ref(), &node.configuration)
            .map_err(|e| e.to_string())?;
        let chains = match summary.publishes.get(&crate::ids::PublishName::new(name)) {
            Some(PublishedGeometry::Path { chains }) => chains.clone(),
            Some(_) => {
                return Err(format!(
                    "`{occurrence}`'s `{name}` isn't an edge set to convert"
                ))
            }
            None => {
                return Err(format!(
                    "`{occurrence}` ({}) publishes no `{name}`",
                    node.definition
                ))
            }
        };
        let to_owner: Transform = self.chain(owner, host);
        Ok(chains
            .into_iter()
            .map(|c| c.points.into_iter().map(|p| to_owner.apply(p)).collect())
            .collect())
    }
}

/// A point in assembly coordinates (metres), in `frame`'s plane (millimetres).
pub fn in_plane(frame: &Frame, p: [f64; 3]) -> [f64; 2] {
    let d = [
        p[0] - frame.origin[0],
        p[1] - frame.origin[1],
        p[2] - frame.origin[2],
    ];
    let dot = |a: [f64; 3]| (d[0] * a[0] + d[1] * a[1] + d[2] * a[2]) * 1000.0;
    [dot(frame.x), dot(frame.y)]
}

fn mm(v: f64) -> ValueExpr {
    ValueExpr::mm((v * 1e4).round() / 1e4)
}

fn at(p: [f64; 2]) -> PointExpr {
    PointExpr::Coords {
        x: mm(p[0]),
        y: mm(p[1]),
    }
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// The circle through three points, if they aren't in a line: its center and radius.
fn circle_through(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<([f64; 2], f64)> {
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-12 {
        return None;
    }
    let sq = |p: [f64; 2]| p[0] * p[0] + p[1] * p[1];
    let x = (sq(a) * (b[1] - c[1]) + sq(b) * (c[1] - a[1]) + sq(c) * (a[1] - b[1])) / d;
    let y = (sq(a) * (c[0] - b[0]) + sq(b) * (a[0] - c[0]) + sq(c) * (b[0] - a[0])) / d;
    Some(([x, y], dist([x, y], a)))
}

/// The simplest curve through a projected polyline (millimetres): a line, a circle (closed), an
/// arc, else a spline. `None` when it projects to a point.
pub fn fit(points: &[[f64; 2]]) -> Option<SketchGeometry> {
    let (&first, &last) = (points.first()?, points.last()?);
    let closed = points.len() > 2 && dist(first, last) < FIT;
    let far = points
        .iter()
        .copied()
        .max_by(|a, b| dist(first, *a).total_cmp(&dist(first, *b)))?;
    if dist(first, far) < FIT {
        return None;
    }
    // A line: every point on the chord.
    if !closed {
        let (dx, dy) = (last[0] - first[0], last[1] - first[1]);
        let len = dx.hypot(dy);
        if len > FIT
            && points
                .iter()
                .all(|p| ((p[0] - first[0]) * dy - (p[1] - first[1]) * dx).abs() / len < FIT)
        {
            return Some(SketchGeometry::Line {
                start: at(first),
                end: at(last),
            });
        }
    }
    // A circle or an arc: every point at one distance from the center.
    let mid = points[points.len() / 3];
    let third = points[(2 * points.len()) / 3];
    if let Some((center, radius)) = circle_through(first, mid, third) {
        if points
            .iter()
            .all(|p| (dist(*p, center) - radius).abs() < FIT.max(radius * 1e-4))
        {
            if closed {
                return Some(SketchGeometry::Circle {
                    center: at(center),
                    radius: mm(radius),
                });
            }
            // Counter-clockwise from start to end: a polyline that turns left runs counter-clockwise.
            let cross = (mid[0] - first[0]) * (last[1] - first[1])
                - (mid[1] - first[1]) * (last[0] - first[0]);
            let (start, end) = if cross > 0.0 { (first, last) } else { (last, first) };
            return Some(SketchGeometry::Arc {
                center: at(center),
                start: at(start),
                end: at(end),
            });
        }
    }
    let step = (points.len() as f64 / SPLINE_POINTS as f64).ceil().max(1.0) as usize;
    let mut kept: Vec<[f64; 2]> = points.iter().step_by(step).copied().collect();
    if closed {
        if kept.len() > 1 && dist(*kept.last()?, first) < FIT {
            kept.pop();
        }
    } else if kept.last() != Some(&last) {
        kept.push(last);
    }
    Some(SketchGeometry::Spline {
        points: kept.into_iter().map(at).collect(),
        closed,
    })
}

/// The entity IDs of a sketch that are converted geometry.
pub fn projected_ids(sketch: &Sketch) -> Vec<SketchEntityId> {
    sketch
        .entities
        .iter()
        .filter(|e| e.projected.is_some())
        .map(|e| e.id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_straight_polyline_reads_as_a_line_and_a_round_one_as_an_arc_or_circle() {
        let line = fit(&[[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]).unwrap();
        assert!(matches!(line, SketchGeometry::Line { .. }), "{line:?}");
        let ring: Vec<[f64; 2]> = (0..=36)
            .map(|i| {
                let t = i as f64 / 36.0 * std::f64::consts::TAU;
                [10.0 + 4.0 * t.cos(), 5.0 + 4.0 * t.sin()]
            })
            .collect();
        match fit(&ring).unwrap() {
            SketchGeometry::Circle { radius, .. } => assert_eq!(radius, ValueExpr::mm(4.0)),
            other => panic!("{other:?}"),
        }
        let half: Vec<[f64; 2]> = ring[..19].to_vec();
        assert!(matches!(fit(&half).unwrap(), SketchGeometry::Arc { .. }));
        let wiggle = [[0.0, 0.0], [3.0, 2.0], [6.0, -1.0], [9.0, 3.0], [12.0, 0.0]];
        assert!(matches!(fit(&wiggle).unwrap(), SketchGeometry::Spline { closed: false, .. }));
        assert!(fit(&[[1.0, 1.0], [1.0, 1.0]]).is_none());
    }
}

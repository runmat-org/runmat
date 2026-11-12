//! `sketch.ellipse`: an ellipse, or an elliptical arc, with its two semi-axes dimensioned.
//!
//! Values, in sketch-plane millimetres and degrees:
//!
//! - `center`: a point value (`{ x, y }`, or `{ x, y, of }` when it snapped to a sketch point);
//! - `major` and `minor`: the semi-axes;
//! - `angle`: the major axis's direction from the sketch's x axis (0 when left out);
//! - `type`: `full` (the default) or `partial`. A partial ellipse also takes `start` and `end`,
//!   points that land on the ellipse at their direction from the centre; the arc runs
//!   counter-clockwise from `start` to `end`.
//!
//! Parabolas and general conics aren't in the sketch model; those forms say so.

use super::{constraint, mm, model, num, point, xy, EditFailure, K};
use runmat_cad::ids::SketchEntityId;
use runmat_cad::model::sketch::{
    PointExpr, SketchConstraint, SketchEntity, SketchGeometry as G, SketchTarget,
};
use runmat_cad::value::ValueExpr;
use serde_json::{Map, Value};

/// The point of the ellipse at the direction of `(x, y)` from its centre (millimetres).
fn on_ellipse(c: (f64, f64), a: f64, b: f64, turn: f64, (x, y): (f64, f64)) -> PointExpr {
    let (s, k) = turn.sin_cos();
    let (dx, dy) = (x - c.0, y - c.1);
    // The direction in the ellipse's own axes, then the eccentric angle of the point of the
    // ellipse along it.
    let (u, v) = (k * dx + s * dy, -s * dx + k * dy);
    let t = (a * v).atan2(b * u);
    let (u, v) = (a * t.cos(), b * t.sin());
    let round = |v: f64| (v * 1e6).round() / 1e6;
    PointExpr::Coords {
        x: mm(round(c.0 + k * u - s * v)),
        y: mm(round(c.1 + s * u + k * v)),
    }
}

pub(super) fn draw(
    values: &Map<String, Value>,
    kind: &str,
    name: &mut dyn FnMut(&str) -> String,
) -> Result<(Vec<SketchEntity>, Vec<SketchConstraint>), EditFailure> {
    match kind {
        "" | "full" | "partial" => {}
        "parabola" | "conic" => {
            return Err(model(format!(
                "{} aren't in the sketch model yet; draw an ellipse or a partial ellipse",
                if kind == "parabola" {
                    "parabolas"
                } else {
                    "conics"
                }
            )))
        }
        other => return Err(model(format!("an ellipse has no `{other}` form"))),
    }
    let center = point(values, "center")?;
    let axis = |key: &str| {
        values
            .get(key)
            .and_then(num)
            .filter(|v| *v > 0.0)
            .ok_or_else(|| model(format!("give the ellipse its {key} radius")))
    };
    let (major, minor) = (axis("major")?, axis("minor")?);
    let angle = values.get("angle").and_then(num).unwrap_or(0.0);
    let (start, end) = if kind == "partial" {
        let c =
            xy(values, "center").ok_or_else(|| model("pick the ellipse's centre on the sketch"))?;
        let (Some(s), Some(e)) = (xy(values, "start"), xy(values, "end")) else {
            return Err(model("pick where the partial ellipse starts and ends"));
        };
        let turn = angle.to_radians();
        let (s, e) = (
            on_ellipse(c, major, minor, turn, s),
            on_ellipse(c, major, minor, turn, e),
        );
        if s == e {
            return Err(model(
                "a partial ellipse starts and ends at different points",
            ));
        }
        (Some(s), Some(e))
    } else {
        (None, None)
    };
    let id = name("e");
    let curve = SketchTarget::entity(&id);
    Ok((
        vec![SketchEntity {
            id: SketchEntityId::new(&id),
            geometry: G::Ellipse {
                center,
                major: mm(major),
                minor: mm(minor),
                angle: (angle != 0.0).then(|| ValueExpr::deg((angle * 1e4).round() / 1e4)),
                start,
                end,
            },
            construction: false,
        }],
        vec![
            constraint(K::MajorRadius {
                curve: curve.clone(),
                value: mm(major),
            }),
            constraint(K::MinorRadius {
                curve,
                value: mm(minor),
            }),
        ],
    ))
}

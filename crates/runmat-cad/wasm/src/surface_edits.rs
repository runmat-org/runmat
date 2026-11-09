//! The Surfaces tab's edit commands as `surface.*` features (CAD-36), with faces, edges and
//! bodies referenced by their canonical names:
//!
//! - `surface.loft` → `surface.loft(profiles, guides)`: a sheet through two or more sketches;
//! - `surface.boundary` → `surface.boundary(first, second)`: a surface through edges in one
//!   direction, or a patch bounded by edges in two;
//! - `surface.fill` → `surface.fill(boundary)`: a patch over a closed loop of edges, each at the
//!   continuity asked for (contact, tangent or curvature with the face it bounds);
//! - `surface.trim` → `surface.trim(bodies, tool, keep)`: sheets cut by a plane, one side kept;
//! - `surface.knit` → `surface.knit(bodies, solid)`: sheets sewn along shared edges, closed into
//!   a solid when they are watertight;
//! - `surface.thicken` → `surface.thicken(bodies, thickness, both_sides)`: a sheet grown into a
//!   solid on its normal side, the other side (a negative thickness) or both.
//!
//! - `surface.extrude` → `surface.extrude(profile, start, end)`: a sketch's curves swept along
//!   its normal, `depth` long (centred on the sketch with `symmetric`);
//! - `surface.revolve` → `surface.revolve(profile, axis, angle)`;
//! - `surface.sweep` → `surface.sweep(profile, path)`: a sketch's curves swept along another's;
//! - `surface.planar` → `surface.planar(profile)`: a flat patch inside a closed sketch;
//! - `surface.ruled` → `surface.ruled(edges, distance, angle)`: strips along edges, turned
//!   `angle` outward from the faces they bound;
//! - `surface.offset` → `surface.offset(faces, distance)`: faces copied along their normals;
//! - `surface.extend` → `surface.extend(edges, distance)`: a sheet grown past its edges;
//! - `surface.delete-hole` → `surface.delete_hole(edges)`: a sheet's hole closed;
//! - `surface.cut` → `surface.cut_with_surface(body, surface, keep)`: a solid cut by a sheet,
//!   one side of the sheet's normal kept;
//! - `surface.mid-surface` → `surface.mid_surface(body, pairs)`: a thin solid's mid-surface
//!   from opposite faces, picked in pairs.
//!
//! A face picked where a body is wanted stands for the body that holds it.

use crate::edit::model;
use crate::features::{picks_of, value};
use crate::picks::Picks;
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{json, Map, Value};

type Values = Map<String, Value>;

fn some(values: &Values, id: &str, what: &str) -> Result<Vec<String>, EditFailure> {
    let all = picks_of(values, id);
    if all.is_empty() {
        Err(model(format!("pick {what}")))
    } else {
        Ok(all)
    }
}

fn text<'a>(values: &'a Values, id: &str, default: &'a str) -> &'a str {
    values.get(id).and_then(Value::as_str).unwrap_or(default)
}

/// `feature.patch.face.top` → `feature.patch.body`: the body a picked face belongs to.
pub(crate) fn body_of(pick: &str) -> String {
    match pick
        .strip_prefix("feature.")
        .and_then(|rest| rest.split_once('.'))
    {
        Some((id, rest)) if rest.starts_with("face") || rest.starts_with("edge") => {
            format!("feature.{id}.body")
        }
        _ => pick.to_owned(),
    }
}

fn bodies(values: &Values, id: &str, picks: &Picks) -> Result<Vec<Value>, EditFailure> {
    let mut seen: Vec<String> = Vec::new();
    for pick in some(values, id, "the surfaces")? {
        let body = body_of(&pick);
        if !seen.contains(&body) {
            seen.push(body);
        }
    }
    picks.references(&seen)
}

fn edges(picked: &[String], continuity: &str, picks: &Picks) -> Result<Vec<Value>, EditFailure> {
    picked
        .iter()
        .map(|p| Ok(json!({ "edge": picks.reference(p)?, "continuity": continuity })))
        .collect()
}

/// `values.continuity` as the model writes it: `g0`, `g1` or `g2`.
fn continuity(values: &Values) -> Result<&str, EditFailure> {
    match text(values, "continuity", "g0") {
        c @ ("g0" | "g1" | "g2") => Ok(c),
        other => Err(model(format!("continuity is g0, g1 or g2, not `{other}`"))),
    }
}

/// The `surface.*` feature a Surfaces command adds, with the stem its name takes; `None` for a
/// command this module doesn't lower.
pub fn surface_kind(
    command: &str,
    values: &Values,
    picks: &Picks,
) -> Result<Option<(&'static str, Value)>, EditFailure> {
    let kind = match command {
        "surface.loft" => {
            let profiles = some(values, "profiles", "two or more profiles")?
                .iter()
                .map(|p| Ok(json!({ "sketch": picks.sketch(p)? })))
                .collect::<Result<Vec<_>, EditFailure>>()?;
            if profiles.len() < 2 {
                return Err(model("a lofted surface takes two or more profiles"));
            }
            (
                "loft",
                json!({ "feature": "surface", "kind": "loft", "profiles": profiles, "guides": picks.references(&picks_of(values, "guides"))? }),
            )
        }
        "surface.boundary" => {
            let first = some(values, "direction1", "the edges in the first direction")?;
            let second = picks_of(values, "direction2");
            let c = continuity(values)?;
            (
                "boundary",
                json!({ "feature": "surface", "kind": "boundary", "first": edges(&first, c, picks)?, "second": edges(&second, c, picks)? }),
            )
        }
        "surface.fill" => {
            let boundary = some(values, "boundary", "the boundary edges")?;
            (
                "fill",
                json!({ "feature": "surface", "kind": "fill", "boundary": edges(&boundary, continuity(values)?, picks)? }),
            )
        }
        "surface.trim" => {
            let tool = some(values, "tool", "the trim plane")?.remove(0);
            let keep = match text(values, "keep", "positive") {
                k @ ("positive" | "negative") => k,
                other => return Err(model(format!(
                    "keep the side above (positive) or below (negative) the plane, not `{other}`"
                ))),
            };
            (
                "trim",
                json!({ "feature": "surface", "kind": "trim", "bodies": bodies(values, "bodies", picks)?, "tool": picks.plane(&tool)?, "keep": keep }),
            )
        }
        "surface.knit" => (
            "knit",
            json!({ "feature": "surface", "kind": "knit", "bodies": bodies(values, "bodies", picks)?, "solid": values.get("solid").and_then(Value::as_bool).unwrap_or(true) }),
        ),
        "surface.thicken" => {
            let mut thickness = value(values, "thickness", "mm")?;
            let side = text(values, "side", "one");
            if side == "other" {
                // The other side is a negative thickness.
                match thickness.get("value").and_then(Value::as_f64) {
                    Some(t) if thickness.get("form").and_then(Value::as_str) == Some("quantity") => thickness["value"] = json!(-t),
                    _ => return Err(model("thicken to the other side takes a number; write a negative expression in the source")),
                }
            }
            (
                "thicken",
                json!({ "feature": "surface", "kind": "thicken", "bodies": bodies(values, "surface", picks)?, "thickness": thickness, "both_sides": side == "both" }),
            )
        }
        "surface.extrude" => {
            let profile =
                json!({ "sketch": picks.sketch(&some(values, "profile", "the sketch")?[0])? });
            let depth = value(values, "depth", "mm")?;
            let (start, end) = if values
                .get("symmetric")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                match depth.get("value").and_then(Value::as_f64) {
                    Some(d) if depth.get("form").and_then(Value::as_str) == Some("quantity") => {
                        (json!({ "form": "quantity", "value": -d / 2.0, "unit": "mm" }), json!({ "form": "quantity", "value": d / 2.0, "unit": "mm" }))
                    }
                    _ => return Err(model("a symmetric extruded surface takes a number; write the start and end in the source")),
                }
            } else {
                (
                    json!({ "form": "quantity", "value": 0, "unit": "mm" }),
                    depth,
                )
            };
            (
                "extrude",
                json!({ "feature": "surface", "kind": "extrude", "profile": profile, "start": start, "end": end }),
            )
        }
        "surface.revolve" => (
            "revolve",
            json!({
                "feature": "surface",
                "kind": "revolve",
                "profile": { "sketch": picks.sketch(&some(values, "profile", "the sketch")?[0])? },
                "axis": picks.axis(&some(values, "axis", "the axis")?[0])?,
                "angle": value(values, "angle", "deg")?,
            }),
        ),
        "surface.sweep" => (
            "sweep",
            json!({
                "feature": "surface",
                "kind": "sweep",
                "profile": { "sketch": picks.sketch(&some(values, "profile", "the profile sketch")?[0])? },
                "path": { "sketch": picks.sketch(&some(values, "path", "the path sketch")?[0])? },
            }),
        ),
        "surface.planar" => (
            "planar",
            json!({ "feature": "surface", "kind": "planar", "profile": { "sketch": picks.sketch(&some(values, "profile", "the closed sketch")?[0])? } }),
        ),
        "surface.ruled" => (
            "ruled",
            json!({
                "feature": "surface",
                "kind": "ruled",
                "edges": picks.references(&some(values, "edges", "the edges")?)?,
                "distance": value(values, "distance", "mm")?,
                "angle": value(values, "angle", "deg")?,
            }),
        ),
        "surface.offset" => (
            "offset",
            json!({ "feature": "surface", "kind": "offset", "faces": picks.references(&some(values, "faces", "the faces")?)?, "distance": value(values, "distance", "mm")? }),
        ),
        "surface.extend" => (
            "extend",
            json!({ "feature": "surface", "kind": "extend", "edges": picks.references(&some(values, "edges", "the edges")?)?, "distance": value(values, "distance", "mm")? }),
        ),
        "surface.delete-hole" => (
            "delete_hole",
            json!({ "feature": "surface", "kind": "delete_hole", "edges": picks.references(&some(values, "edges", "the hole's edges")?)? }),
        ),
        "surface.cut" => {
            let surface = body_of(&some(values, "surface", "the cutting surface")?[0]);
            let body = body_of(&some(values, "body", "the solid to cut")?[0]);
            if surface == body {
                return Err(model("a body can't be cut with itself"));
            }
            let keep = match text(values, "keep", "positive") {
                k @ ("positive" | "negative") => k,
                other => return Err(model(format!("keep the side the surface faces (positive) or the other (negative), not `{other}`"))),
            };
            (
                "cut_with_surface",
                json!({ "feature": "surface", "kind": "cut_with_surface", "body": picks.reference(&body)?, "surface": picks.reference(&surface)?, "keep": keep }),
            )
        }
        "surface.mid-surface" => {
            let faces = some(values, "pairs", "opposite faces in pairs")?;
            if faces.len() % 2 != 0 {
                return Err(model(
                    "pick the faces in pairs: each face, then the face opposite it",
                ));
            }
            let body = body_of(&faces[0]);
            if faces.iter().any(|f| body_of(f) != body) {
                return Err(model("every face of a mid-surface is on one body"));
            }
            let refs = picks.references(&faces)?;
            let pairs: Vec<Value> = refs.chunks(2).map(|p| json!([p[0], p[1]])).collect();
            (
                "mid_surface",
                json!({ "feature": "surface", "kind": "mid_surface", "body": picks.reference(&body)?, "pairs": pairs }),
            )
        }
        _ => return Ok(None),
    };
    Ok(Some(kind))
}

#[cfg(test)]
mod tests {
    use super::body_of;

    #[test]
    fn a_picked_face_stands_for_its_body() {
        assert_eq!(body_of("feature.patch.face.top"), "feature.patch.body");
        assert_eq!(
            body_of("feature.patch.edge.boundary(s.b)"),
            "feature.patch.body"
        );
        assert_eq!(body_of("feature.patch.body"), "feature.patch.body");
    }
}

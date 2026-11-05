//! Part catalog commands as feature models: each command's field values (numbers in the field's
//! unit, value edits, picks) become the `FeatureKind` its source text would parse to.
//!
//! [`feature_kind`] answers `Ok(None)` for a command that adds no feature here, and a message
//! naming the missing pick or the option the model lacks otherwise.

use crate::picks::Picks;
use runmat_cad::syntax::edit::EditFailure;
use serde_json::{json, Map, Value};

fn model(message: impl Into<String>) -> EditFailure {
    EditFailure::Model(message.into())
}

type Values = Map<String, Value>;

/// A length or angle as a value expression: a number in the field's unit, a value edit
/// (`{ kind: "value", value }`), an expression, or a number as text.
pub fn value(values: &Values, field: &str, unit: &str) -> Result<Value, EditFailure> {
    match values.get(field) {
        Some(Value::Number(n)) => Ok(json!({ "form": "quantity", "value": n, "unit": unit })),
        Some(Value::Object(o)) if o.get("kind").and_then(Value::as_str) == Some("value") => o
            .get("value")
            .cloned()
            .ok_or_else(|| model(format!("`{field}` has no value"))),
        Some(Value::Object(o)) if o.contains_key("form") => Ok(Value::Object(o.clone())),
        Some(Value::String(s)) => s
            .trim()
            .parse::<f64>()
            .map(|n| json!({ "form": "quantity", "value": n, "unit": unit }))
            .map_err(|_| model(format!("`{field}` isn't a number: {s}"))),
        _ => Err(model(format!("`{field}` needs a value"))),
    }
}

fn number(values: &Values, field: &str) -> Result<Value, EditFailure> {
    match values.get(field) {
        Some(Value::Number(n)) => Ok(json!({ "form": "number", "value": n })),
        Some(Value::String(s)) => s
            .trim()
            .parse::<f64>()
            .map(|n| json!({ "form": "number", "value": n }))
            .map_err(|_| model(format!("`{field}` isn't a number: {s}"))),
        Some(Value::Object(o)) if o.get("kind").and_then(Value::as_str) == Some("value") => o
            .get("value")
            .cloned()
            .ok_or_else(|| model(format!("`{field}` has no value"))),
        _ => Err(model(format!("`{field}` needs a count"))),
    }
}

fn text<'a>(values: &'a Values, id: &str, default: &'a str) -> &'a str {
    values.get(id).and_then(Value::as_str).unwrap_or(default)
}

fn flag(values: &Values, id: &str) -> bool {
    values.get(id).and_then(Value::as_bool).unwrap_or(false)
}

fn flag_or(values: &Values, id: &str, default: bool) -> bool {
    values.get(id).and_then(Value::as_bool).unwrap_or(default)
}

/// A pick field's picks, in order.
pub fn picks_of(values: &Values, id: &str) -> Vec<String> {
    match values.get(id) {
        Some(Value::String(s)) if !s.is_empty() => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .flat_map(|v| match v {
                Value::String(s) => vec![s.clone()],
                Value::Array(inner) => inner
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect(),
                _ => Vec::new(),
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn one(values: &Values, id: &str, what: &str) -> Result<String, EditFailure> {
    picks_of(values, id)
        .into_iter()
        .next()
        .ok_or_else(|| model(format!("pick {what}")))
}

fn some(values: &Values, id: &str, what: &str) -> Result<Vec<String>, EditFailure> {
    let all = picks_of(values, id);
    if all.is_empty() {
        Err(model(format!("pick {what}")))
    } else {
        Ok(all)
    }
}

/// How a body feature combines: the `op` field, with the `merge` targets.
fn mode(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let targets = picks.references(&picks_of(values, "merge"))?;
    Ok(match text(values, "op", "add") {
        "add" => json!({ "mode": "add", "targets": targets }),
        "remove" => json!({ "mode": "cut", "targets": targets }),
        "intersect" => json!({ "mode": "intersect", "targets": targets }),
        _ => json!({ "mode": "new" }),
    })
}

fn profile(values: &Values, field: &str, picks: &Picks) -> Result<Value, EditFailure> {
    let pick = one(values, field, "the sketch")?;
    Ok(json!({ "sketch": picks.sketch(&pick)? }))
}

/// One extrude direction's end, with the geometry it names.
fn extent(values: &Values, prefix: &str, picks: &Picks) -> Result<Value, EditFailure> {
    let id = |s: &str| format!("{prefix}{s}");
    let up_to = || one(values, &id("up_to"), "what the extrude goes up to");
    Ok(match text(values, &id("end"), "blind") {
        "blind" => json!({ "extent": "blind", "distance": value(values, &id("distance"), "mm")? }),
        "mid_plane" => {
            json!({ "extent": "symmetric", "distance": value(values, &id("distance"), "mm")? })
        }
        "through_all" => json!({ "extent": "through_all" }),
        "up_to_next" => json!({ "extent": "up_to_next" }),
        "up_to_face" => json!({ "extent": "up_to", "face": picks.reference(&up_to()?)? }),
        "up_to_vertex" => json!({ "extent": "up_to_vertex", "vertex": picks.point(&up_to()?)? }),
        "up_to_body" => json!({ "extent": "up_to_body", "body": picks.reference(&up_to()?)? }),
        "offset_from_face" => json!({
            "extent": "offset_from_face",
            "face": picks.reference(&up_to()?)?,
            "offset": value(values, &id("distance"), "mm")?,
        }),
        other => {
            return Err(model(format!(
                "an extrude can't end {}",
                other.replace('_', " ")
            )))
        }
    })
}

fn extrude(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let mut kind = json!({
        "feature": "extrude",
        "profile": profile(values, "profile", picks)?,
        "extent": extent(values, "", picks)?,
        "reverse": flag(values, "flip"),
        "mode": mode(values, picks)?,
    });
    match text(values, "from", "sketch_plane") {
        "sketch_plane" => {}
        "offset" => {
            let field = if values.contains_key("from_offset") {
                "from_offset"
            } else {
                "offset"
            };
            kind["from"] = json!({ "from": "offset", "distance": value(values, field, "mm")? });
        }
        "surface" => {
            let face = one(values, "from_face", "the face the extrude starts from")
                .or_else(|_| one(values, "up_to", "the face the extrude starts from"))?;
            kind["from"] = json!({ "from": "surface", "face": picks.reference(&face)? });
        }
        "vertex" => {
            let at = one(values, "from_vertex", "the vertex the extrude starts from")?;
            kind["from"] = json!({ "from": "vertex", "vertex": picks.point(&at)? });
        }
        other => return Err(model(format!("an extrude can't start from {other}"))),
    }
    if flag(values, "second_direction") {
        kind["second"] = extent(values, "second_", picks)?;
    }
    if flag(values, "draft") {
        kind["draft"] = json!({ "angle": value(values, "draft_angle", "deg")?, "outward": flag(values, "draft_outward") });
    }
    if flag(values, "thin") {
        kind["thin"] = json!({ "thickness": value(values, "thin_thickness", "mm")?, "side": text(values, "thin_side", "one") });
    }
    Ok(kind)
}

fn seeds(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let all = some(values, "seeds", "the features or bodies to repeat")?;
    let features: Vec<&str> = all
        .iter()
        .filter_map(|p| p.strip_prefix("feature:"))
        .collect();
    if features.len() == all.len() {
        return Ok(json!({ "seeds": "features", "features": features }));
    }
    Ok(json!({ "seeds": "bodies", "bodies": picks.references(&all)? }))
}

fn chamfer_form(values: &Values) -> Result<Value, EditFailure> {
    Ok(match text(values, "type", "distance_angle") {
        "equal" => json!({ "chamfer": "equal", "distance": value(values, "distance", "mm")? }),
        "distance_distance" => {
            json!({ "chamfer": "distances", "distance": value(values, "distance", "mm")?, "distance2": value(values, "distance2", "mm")? })
        }
        "vertex" => {
            let (a, b) = (
                value(values, "distance", "mm")?,
                value(values, "distance2", "mm")?,
            );
            json!({ "chamfer": "vertex", "distances": [a, b.clone(), b] })
        }
        _ => {
            json!({ "chamfer": "distance_angle", "distance": value(values, "distance", "mm")?, "angle": value(values, "angle", "deg")? })
        }
    })
}

fn fillet(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let all = some(values, "edges", "the edges or faces to round")?;
    let profile = match text(values, "profile", "circular") {
        "conic" => json!({ "profile": "conic", "rho": { "form": "number", "value": 0.5 } }),
        "curvature" => json!({ "profile": "curvature" }),
        _ => json!({ "profile": "circular" }),
    };
    match text(values, "type", "constant") {
        "face" | "full_round" => {
            let faces = picks.references(&all)?;
            if text(values, "type", "") == "face" {
                if faces.len() < 2 {
                    return Err(model("a face fillet takes two faces"));
                }
                let (first, second) = faces.split_at(1);
                Ok(
                    json!({ "feature": "face_fillet", "first": first, "second": second, "radius": value(values, "radius", "mm")?, "profile": profile }),
                )
            } else {
                let [a, c, b] = faces.as_slice() else {
                    return Err(model(
                        "a full round takes three faces: one side, the centre, the other side",
                    ));
                };
                Ok(json!({ "feature": "full_round", "first": [a], "center": [c], "second": [b] }))
            }
        }
        kind => {
            let (vertices, edges): (Vec<String>, Vec<String>) =
                all.into_iter().partition(|p| p.starts_with("vertex_of("));
            let mut out = json!({
                "feature": "fillet",
                "edges": picks.references(&edges)?,
                "radius": value(values, "radius", "mm")?,
                "profile": profile,
                "propagate": flag_or(values, "tangent_propagation", true),
            });
            if kind == "variable" {
                let radii: Vec<Value> = match values.get("radii") {
                    Some(Value::Array(rs)) => rs.clone(),
                    _ => Vec::new(),
                };
                let n = radii.len().max(1);
                let mut variable = Vec::new();
                for (i, r) in radii.iter().enumerate() {
                    let mut m = Map::new();
                    m.insert("r".into(), r.clone());
                    let at = if n == 1 {
                        0.5
                    } else {
                        i as f64 / (n - 1) as f64
                    };
                    variable.push(json!({ "at": { "form": "number", "value": at }, "radius": value(&m, "r", "mm")? }));
                }
                out["variable"] = Value::Array(variable);
            }
            if kind == "setback" {
                let distance = value(values, "setback", "mm")?;
                out["setbacks"] = vertices
                    .iter()
                    .map(|v| Ok(json!({ "vertex": picks.reference(v)?, "distance": distance })))
                    .collect::<Result<Vec<_>, EditFailure>>()?
                    .into();
            }
            Ok(out)
        }
    }
}

fn hole(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let face = one(values, "face", "the face the holes start on")?;
    let positions: Vec<Value> = picks_of(values, "positions")
        .iter()
        .enumerate()
        .map(|(i, p)| Ok(json!({ "key": format!("p{}", i + 1), "at": picks.point(p)? })))
        .collect::<Result<_, EditFailure>>()?;
    if positions.is_empty() {
        return Err(model("pick where the holes go"));
    }
    let kind = text(values, "type", "simple");
    let form = match kind {
        "counterbore" => {
            json!({ "hole": "counterbore", "diameter": value(values, "cbore_diameter", "mm")?, "depth": value(values, "cbore_depth", "mm")? })
        }
        "countersink" => {
            json!({ "hole": "countersink", "diameter": value(values, "csink_diameter", "mm")?, "angle": value(values, "csink_angle", "deg")? })
        }
        "tapped" => {
            json!({ "hole": "tapped", "thread_depth": value(values, "thread_depth", "mm")?, "modelled": !flag_or(values, "cosmetic_thread", true) })
        }
        "pipe_tap" => {
            json!({ "hole": "pipe_tap", "thread_depth": value(values, "thread_depth", "mm")?, "modelled": !flag_or(values, "cosmetic_thread", true) })
        }
        "slot" => json!({ "hole": "slot", "length": value(values, "slot_length", "mm")? }),
        _ => json!({ "hole": "simple" }),
    };
    let end = match text(values, "end", "blind") {
        "through_all" => json!({ "end": "through_all" }),
        "up_to_next" => json!({ "end": "up_to_next" }),
        "up_to_face" => {
            json!({ "end": "up_to_face", "face": picks.reference(&one(values, "up_to", "the face the hole goes to")?)? })
        }
        _ => json!({ "end": "blind", "depth": value(values, "depth", "mm")? }),
    };
    let mut out = json!({ "feature": "hole", "face": picks.reference(&face)?, "positions": positions, "end": end, "form": form });
    // A simple hole with a diameter drills it; every other hole takes its sizes from the standard.
    let standard = match text(values, "standard", "iso") {
        "ansi_metric" => "ansi_metric",
        "ansi_inch" => "ansi_inch",
        _ if kind == "pipe_tap" => "npt",
        _ => "iso",
    };
    if kind == "simple" && values.get("diameter").is_some() {
        out["diameter"] = value(values, "diameter", "mm")?;
    } else {
        out["spec"] = json!({ "standard": standard, "size": text(values, "size", "M6"), "fit": text(values, "fit", "normal") });
    }
    Ok(out)
}

fn pattern(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let layout = match text(values, "layout", "linear") {
        "linear" => {
            let first = picks.axis(&one(values, "direction", "the pattern's direction")?)?;
            if flag(values, "second_direction") {
                json!({
                    "layout": "grid",
                    "first": first,
                    "first_spacing": value(values, "spacing", "mm")?,
                    "first_count": number(values, "instances")?,
                    "second": picks.axis(&one(values, "direction2", "the second direction")?)?,
                    "second_spacing": value(values, "spacing2", "mm")?,
                    "second_count": number(values, "instances2")?,
                })
            } else {
                json!({ "layout": "linear", "direction": first, "spacing": value(values, "spacing", "mm")?, "count": number(values, "instances")? })
            }
        }
        "circular" => json!({
            "layout": "circular",
            "axis": picks.axis(&one(values, "axis", "the pattern's axis")?)?,
            "count": number(values, "instances")?,
            "angle": value(values, "angle", "deg")?,
            "equal_spacing": flag_or(values, "equal_spacing", true),
        }),
        "curve" => {
            let pick = one(values, "curve", "the curve to follow")?;
            let curve = match picks.sketch(&pick) {
                Ok(sketch) => json!({ "curve": "sketch", "sketch": sketch }),
                Err(_) => json!({ "curve": "edge", "edge": picks.reference(&pick)? }),
            };
            json!({ "layout": "curve", "curve": curve, "count": number(values, "instances")? })
        }
        "sketch" => {
            json!({ "layout": "sketch", "sketch": picks.sketch(&one(values, "sketch", "the sketch of points")?)? })
        }
        "fill" => json!({
            "layout": "fill",
            "boundary": { "sketch": picks.sketch(&one(values, "boundary", "the region to fill")?)? },
            "spacing": value(values, "spacing", "mm")?,
        }),
        other => {
            return Err(model(format!(
                "a {other} pattern is set up in the source editor"
            )))
        }
    };
    if picks_of(values, "skip").len()
        + values
            .get("skip")
            .and_then(Value::as_array)
            .map_or(0, Vec::len)
        > 0
    {
        return Err(model("skipping pattern instances is picked on the canvas in a later build; skip them in the source editor"));
    }
    Ok(json!({ "feature": "pattern", "seeds": seeds(values, picks)?, "layout": layout }))
}

/// The feature a part command adds, as its model JSON, and the stem its name takes.
pub fn feature_kind(
    command: &str,
    values: &Values,
    picks: &Picks,
) -> Result<Option<(&'static str, Value)>, EditFailure> {
    Ok(Some(match command {
        "part.extrude" => ("extrude", extrude(values, picks)?),
        "part.revolve" => {
            match text(values, "type", "one") {
                "one" => {}
                other => {
                    return Err(model(format!(
                        "a {} revolve isn't in the part model yet; revolve one way",
                        if other == "mid" {
                            "mid-plane"
                        } else {
                            "two-direction"
                        }
                    )))
                }
            }
            if flag(values, "thin") {
                return Err(model("a thin revolve isn't in the part model yet"));
            }
            (
                "revolve",
                json!({
                    "feature": "revolve",
                    "profile": profile(values, "profile", picks)?,
                    "axis": picks.axis(&one(values, "axis", "the axis to revolve about")?)?,
                    "angle": value(values, "angle", "deg")?,
                    "mode": mode(values, picks)?,
                }),
            )
        }
        "part.sweep" => (
            "sweep",
            json!({
                "feature": "sweep",
                "profile": profile(values, "profile", picks)?,
                "path": profile(values, "path", picks)?,
                "mode": mode(values, picks)?,
            }),
        ),
        "part.loft" | "part.boundary" => {
            let field = if command == "part.loft" {
                "profiles"
            } else {
                "direction1"
            };
            let profiles: Vec<Value> = some(values, field, "two or more profiles")?
                .iter()
                .map(|p| Ok(json!({ "sketch": picks.sketch(p)? })))
                .collect::<Result<_, EditFailure>>()?;
            if profiles.len() < 2 {
                return Err(model("a loft takes two or more profiles"));
            }
            (
                "loft",
                json!({ "feature": "loft", "profiles": profiles, "guides": picks.references(&picks_of(values, "guides"))?, "mode": mode(values, picks)? }),
            )
        }
        "part.fillet" => ("fillet", fillet(values, picks)?),
        "part.chamfer" => (
            "chamfer",
            json!({
                "feature": "chamfer",
                "edges": picks.references(&some(values, "edges", "the edges to bevel")?)?,
                "form": chamfer_form(values)?,
                "flip": flag(values, "flip"),
            }),
        ),
        "part.shell" => {
            let open = picks_of(values, "open_faces");
            let body = match picks_of(values, "body").first() {
                Some(b) => picks.reference(b)?,
                None => {
                    let first = open
                        .first()
                        .ok_or_else(|| model("pick the body to hollow, or the faces to open"))?;
                    picks
                        .body_of_face(first)
                        .ok_or_else(|| model("pick the body to hollow"))?
                }
            };
            (
                "shell",
                json!({
                    "feature": "shell",
                    "body": body,
                    "open_faces": picks.references(&open)?,
                    "thickness": value(values, "thickness", "mm")?,
                    "outward": flag(values, "outward"),
                }),
            )
        }
        "part.draft" => {
            let kind = match text(values, "type", "neutral_plane") {
                "parting_line" => {
                    json!({ "draft": "parting_line", "edges": picks.references(&some(values, "parting", "the parting line")?)? })
                }
                "step" => {
                    json!({ "draft": "step", "edges": picks.references(&some(values, "parting", "the parting line")?)? })
                }
                _ => json!({ "draft": "neutral_plane" }),
            };
            let neutral = match picks_of(values, "neutral").first() {
                Some(n) => picks.plane(n)?,
                None if text(values, "type", "neutral_plane") != "neutral_plane" => {
                    json!({ "on": "datum", "datum": { "datum": "origin_plane", "plane": "xy" } })
                }
                None => return Err(model("pick the neutral plane")),
            };
            (
                "draft",
                json!({
                    "feature": "draft",
                    "faces": picks.references(&some(values, "faces", "the faces to draft")?)?,
                    "neutral": neutral,
                    "angle": value(values, "angle", "deg")?,
                    "kind": kind,
                    "flip": flag(values, "flip"),
                }),
            )
        }
        "part.hole" => ("hole", hole(values, picks)?),
        "part.pattern" => ("pattern", pattern(values, picks)?),
        "part.mirror" => (
            "mirror",
            json!({
                "feature": "mirror",
                "seeds": seeds(values, picks)?,
                "plane": picks.plane(&one(values, "plane", "the mirror plane")?)?,
            }),
        ),
        "part.boolean" => (
            "boolean",
            json!({
                "feature": "boolean",
                "kind": text(values, "kind", "union"),
                "target": picks.reference(&one(values, "target", "the target body")?)?,
                "tools": picks.references(&some(values, "tools", "the tool bodies")?)?,
                "keep_tools": flag(values, "keep_tools"),
            }),
        ),
        "part.body-combine" => {
            let bodies = picks.references(&some(values, "bodies", "the bodies to combine")?)?;
            let (target, tools) = bodies
                .split_first()
                .ok_or_else(|| model("pick the bodies to combine"))?;
            if tools.is_empty() {
                return Err(model("combining takes two or more bodies"));
            }
            (
                "combine",
                json!({ "feature": "boolean", "kind": "union", "target": target, "tools": tools, "keep_tools": false }),
            )
        }
        "part.split" => (
            "split",
            json!({
                "feature": "split",
                "body": picks.reference(&one(values, "body", "the body to split")?)?,
                "tool": picks.plane(&one(values, "tool", "the plane or face to split with")?)?,
            }),
        ),
        "part.body-move" => {
            if values
                .get("angle")
                .and_then(Value::as_f64)
                .is_some_and(|a| a != 0.0)
                || flag(values, "copy")
            {
                return Err(model("moving bodies by rotation or as copies isn't in the part model yet; move them by translation"));
            }
            (
                "move",
                json!({
                    "feature": "body_op",
                    "op": "translate",
                    "bodies": picks.references(&some(values, "bodies", "the bodies to move")?)?,
                    "dx": value(values, "dx", "mm")?,
                    "dy": value(values, "dy", "mm")?,
                    "dz": value(values, "dz", "mm")?,
                }),
            )
        }
        "part.body-delete" => {
            if text(values, "keep", "delete") == "keep" {
                return Err(model(
                    "keeping only some bodies isn't in the part model yet; delete the others",
                ));
            }
            (
                "delete",
                json!({ "feature": "body_op", "op": "delete", "bodies": picks.references(&some(values, "bodies", "the bodies to delete")?)? }),
            )
        }
        "part.plane" => {
            let def = match text(values, "type", "offset") {
                "offset" => {
                    json!({ "plane": "offset", "base": picks.plane(&one(values, "base", "the reference plane")?)?, "distance": value(values, "distance", "mm")? })
                }
                "angle" => {
                    json!({ "plane": "angle", "base": picks.plane(&one(values, "base", "the reference plane")?)?, "axis": picks.axis(&one(values, "axis", "the axis")?)?, "angle": value(values, "angle", "deg")? })
                }
                "three_point" => {
                    let pts = some(values, "points", "three points")?;
                    let [a, b, c] = pts.as_slice() else {
                        return Err(model("a plane through points takes three"));
                    };
                    json!({ "plane": "three_point", "a": picks.point(a)?, "b": picks.point(b)?, "c": picks.point(c)? })
                }
                "mid" => {
                    json!({ "plane": "mid", "a": picks.plane(&one(values, "base", "the first plane")?)?, "b": picks.plane(&one(values, "other", "the second plane")?)? })
                }
                "tangent_to_face" => {
                    json!({ "plane": "tangent_to_face", "face": picks.reference(&one(values, "face", "the surface")?)?, "at": picks.point(&one(values, "at", "the point")?)? })
                }
                other => {
                    return Err(model(format!(
                        "a {} plane is set up in the source editor",
                        other.replace('_', " ")
                    )))
                }
            };
            let mut def = def;
            def["feature"] = json!("plane");
            ("plane", def)
        }
        "sheet-metal.base-flange" => {
            let mut rules = json!({
                "thickness": value(values, "thickness", "mm")?,
                "bend_radius": value(values, "radius", "mm")?,
                "rule": match text(values, "bend_rule", "k_factor") {
                    "k_factor" => json!({ "rule": "k_factor", "k": number(values, "k").unwrap_or(json!({ "form": "number", "value": 0.44 })) }),
                    other => return Err(model(format!("a {} rule is set up in the source editor (it needs its table)", other.replace('_', " ")))),
                },
            });
            if rules["rule"].is_null() {
                rules["rule"] =
                    json!({ "rule": "k_factor", "k": { "form": "number", "value": 0.44 } });
            }
            let mut kind = json!({ "feature": "sheet_metal", "sheet_metal": "base_flange", "profile": profile(values, "profile", picks)?, "rules": rules });
            if values.get("depth").is_some() {
                kind["depth"] = value(values, "depth", "mm")?;
            }
            ("flange", kind)
        }
        "sheet-metal.edge-flange" => (
            "flange",
            json!({
                "feature": "sheet_metal",
                "sheet_metal": "edge_flange",
                "edges": picks.references(&some(values, "edges", "the edges to flange")?)?,
                "length": value(values, "length", "mm")?,
                "angle": value(values, "angle", "deg")?,
                "position": match text(values, "position", "bend_outside") { "inside" => "material_inside", "outside" => "material_outside", _ => "bend_outside" },
                "radius": value(values, "radius", "mm")?,
                "relief": match text(values, "relief", "tear") {
                    "rectangular" => json!({ "relief": "rectangular", "width": { "form": "quantity", "value": 1.0, "unit": "mm" }, "depth": { "form": "quantity", "value": 1.0, "unit": "mm" } }),
                    "obround" => json!({ "relief": "obround", "width": { "form": "quantity", "value": 1.0, "unit": "mm" }, "depth": { "form": "quantity", "value": 1.0, "unit": "mm" } }),
                    _ => json!({ "relief": "tear" }),
                },
            }),
        ),
        "sheet-metal.hem" => (
            "hem",
            json!({
                "feature": "sheet_metal",
                "sheet_metal": "hem",
                "edges": picks.references(&some(values, "edges", "the edges to hem")?)?,
                "kind": match text(values, "hem", "closed") {
                    "open" => json!({ "hem": "open", "gap": value(values, "gap", "mm")? }),
                    "closed" => json!({ "hem": "closed" }),
                    other => return Err(model(format!("a {other} hem is set up in the source editor (it needs its roll radius and angle)"))),
                },
                "length": value(values, "length", "mm")?,
            }),
        ),
        "sheet-metal.relief" => {
            let kind = match text(values, "relief", "rectangular") {
                "tear" => json!({ "relief": "tear" }),
                k => {
                    json!({ "relief": if k == "round" { "round" } else { "rectangular" }, "size": value(values, "size", "mm")? })
                }
            };
            (
                "relief",
                json!({
                    "feature": "sheet_metal",
                    "sheet_metal": "corner_relief",
                    "corners": picks.references(&some(values, "corners", "the corners to relieve")?)?,
                    "kind": kind,
                }),
            )
        }
        "sheet-metal.closed-corner" => (
            "corner",
            json!({
                "feature": "sheet_metal",
                "sheet_metal": "closed_corner",
                "faces": picks.references(&some(values, "faces", "the flange faces to extend")?)?,
                "gap": value(values, "gap", "mm")?,
                "overlap": match text(values, "type", "overlap") { "underlap" => "underlap", "butt" => "butt", _ => "overlap" },
            }),
        ),
        "sheet-metal.sketched-bend" | "sheet-metal.jog" => {
            let (sketch, entity) = picks.sketch_entity(&one(values, "line", "the bend line")?)?;
            let face = match picks_of(values, "fixed").first() {
                Some(f) => picks.reference(f)?,
                None => return Err(model("pick the face that stays fixed")),
            };
            if command == "sheet-metal.jog" {
                (
                    "jog",
                    json!({ "feature": "sheet_metal", "sheet_metal": "jog", "face": face, "line": [sketch, entity], "offset": value(values, "offset", "mm")?, "angle": value(values, "angle", "deg")? }),
                )
            } else {
                (
                    "bend",
                    json!({ "feature": "sheet_metal", "sheet_metal": "sketched_bend", "face": face, "line": [sketch, entity], "angle": value(values, "angle", "deg")?, "radius": value(values, "radius", "mm")?, "fixed": "positive" }),
                )
            }
        }
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests {
    use crate::edit::edit_json;
    use crate::runtime::{compile_json, Files};
    use serde_json::{json, Value};

    /// A 40 x 20 plate, 10 mm thick, from sketch s1.
    const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n}\n";

    const END: &str = "feature.body.face.end";
    const TOP_EDGE: &str = "edges_between(feature.body.face.side(s1.l1), feature.body.face.end)";

    /// Runs a command from the canvas and checks the change: one inserted statement that names
    /// the new feature, and a part that still regenerates with a body and no error.
    fn added(command: &str, values: Value) -> String {
        let request = json!({ "command": command, "values": values });
        let out: Value = serde_json::from_str(&edit_json(
            "/p/plate.part",
            PLATE,
            &Files::new(),
            &request.to_string(),
        ))
        .unwrap();
        assert_eq!(out["ok"], true, "{command}: {out}");
        let file = &out["files"][0];
        assert_eq!(file["reprinted"], false, "{command}");
        let edits = file["edits"].as_array().unwrap();
        assert_eq!(edits.len(), 1, "{command}: {edits:?}");
        assert_eq!(
            edits[0]["start"], edits[0]["end"],
            "{command}: an insertion"
        );
        let text = file["text"].as_str().unwrap().to_owned();
        let compiled: Value =
            serde_json::from_str(&compile_json("/p/plate.part", &text, &Files::new())).unwrap();
        let errors: Vec<&Value> = compiled["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|d| d["severity"] == "error")
            .collect();
        assert!(errors.is_empty(), "{command}: {errors:?}\n{text}");
        assert!(
            !compiled["display"]["data"]["bodies"]
                .as_array()
                .unwrap()
                .is_empty(),
            "{command}"
        );
        text
    }

    #[test]
    fn dress_up_features_take_their_edges_and_faces_from_the_canvas() {
        let t = added(
            "part.fillet",
            json!({ "type": "constant", "edges": [TOP_EDGE], "radius": 2 }),
        );
        assert!(
            t.contains("feature fillet1 = fillet(") && t.contains("2 mm"),
            "{t}"
        );
        let t = added(
            "part.chamfer",
            json!({ "type": "equal", "edges": [TOP_EDGE], "distance": 1 }),
        );
        assert!(t.contains("feature chamfer1 = chamfer("), "{t}");
        let t = added(
            "part.shell",
            json!({ "open_faces": [END], "thickness": 1.5 }),
        );
        assert!(
            t.contains("feature shell1 = shell(") && t.contains("feature.body.body"),
            "the body comes from the face: {t}"
        );
        let t = added(
            "part.draft",
            json!({ "type": "neutral_plane", "faces": ["feature.body.face.side(s1.l1)"], "neutral": ["feature.body.face.start"], "angle": 3 }),
        );
        assert!(t.contains("feature draft1 = draft("), "{t}");
    }

    #[test]
    fn holes_patterns_mirrors_and_planes_take_their_picks() {
        let t = added(
            "part.hole",
            json!({ "type": "simple", "face": [END], "positions": ["sketch.s1.l1"], "diameter": 3, "end": "through_all" }),
        );
        assert!(t.contains("feature hole1 = hole("), "{t}");
        let t = added(
            "part.hole",
            json!({ "type": "counterbore", "standard": "iso", "size": "M4", "face": [END], "positions": ["sketch.s1.l2"], "cbore_diameter": 8, "cbore_depth": 4, "end": "blind", "depth": 6 }),
        );
        assert!(t.contains("M4"), "the standard size is written: {t}");
        let t = added(
            "part.pattern",
            json!({ "layout": "linear", "seeds": ["feature:body"], "direction": [TOP_EDGE], "spacing": 50, "instances": 3 }),
        );
        assert!(t.contains("feature pattern1 = pattern("), "{t}");
        let t = added(
            "part.pattern",
            json!({ "layout": "circular", "seeds": ["feature:body"], "axis": ["origin:z"], "instances": 4, "angle": 360 }),
        );
        assert!(t.contains("origin.z"), "{t}");
        let t = added(
            "part.mirror",
            json!({ "seeds": ["feature:body"], "plane": ["origin:right"] }),
        );
        assert!(t.contains("origin.yz"), "{t}");
        let t = added(
            "part.plane",
            json!({ "type": "offset", "base": ["origin:top"], "distance": 25 }),
        );
        assert!(t.contains("plane1"), "{t}");
    }

    #[test]
    fn profile_features_and_extrude_ends_that_name_geometry() {
        let t = added(
            "part.revolve",
            json!({ "profile": ["feature:s1"], "axis": ["origin:y"], "op": "new", "type": "one", "angle": 90 }),
        );
        assert!(t.contains("feature revolve1 = revolve("), "{t}");
        let t = added(
            "part.extrude",
            json!({ "profile": ["feature:s1"], "op": "add", "end": "up_to_face", "up_to": [END] }),
        );
        assert!(
            t.contains("feature extrude1 = extrude(") && t.contains("feature.body.face.end"),
            "{t}"
        );
        let t = added(
            "part.extrude",
            json!({ "profile": ["feature:s1"], "op": "new", "from": "offset", "offset": 20, "end": "blind", "distance": 5 }),
        );
        assert!(t.contains("feature extrude1"), "{t}");
    }

    #[test]
    fn what_the_model_lacks_is_said_plainly() {
        let request = json!({ "command": "part.revolve", "values": { "profile": ["feature:s1"], "axis": ["origin:y"], "type": "mid", "angle": 90 } });
        let out: Value = serde_json::from_str(&edit_json(
            "/p/plate.part",
            PLATE,
            &Files::new(),
            &request.to_string(),
        ))
        .unwrap();
        assert_eq!(out["ok"], false);
        assert!(
            out["message"].as_str().unwrap().contains("mid-plane"),
            "{out}"
        );
        let request =
            json!({ "command": "part.fillet", "values": { "type": "constant", "radius": 1 } });
        let out: Value = serde_json::from_str(&edit_json(
            "/p/plate.part",
            PLATE,
            &Files::new(),
            &request.to_string(),
        ))
        .unwrap();
        assert!(out["message"].as_str().unwrap().contains("pick"), "{out}");
    }
}

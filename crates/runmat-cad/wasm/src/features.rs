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

/// A region pick (`sketch.<s>.region(<curves>;<curves>)`) as its sketch and loops: each loop
/// by its curves, the outer loop first, then the holes inside it.
pub fn region_pick(pick: &str) -> Option<(String, Vec<Vec<String>>)> {
    let rest = pick.strip_prefix("sketch.")?;
    let (sketch, inner) = rest.split_once(".region(")?;
    let inner = inner.strip_suffix(')')?;
    let loops: Vec<Vec<String>> = inner
        .split(';')
        .map(|l| {
            l.split(',')
                .map(|c| c.trim().to_owned())
                .filter(|c| !c.is_empty())
                .collect()
        })
        .filter(|l: &Vec<String>| !l.is_empty())
        .collect();
    (!loops.is_empty()).then(|| (sketch.to_owned(), loops))
}

/// Every pick of a profile field as the profiles features take, in pick order: a planar face
/// (its reference), a whole sketch, or a sketch's picked regions together (their loops). Each
/// profile becomes its own feature.
fn profiles(values: &Values, field: &str, picks: &Picks) -> Result<Vec<Value>, EditFailure> {
    enum P {
        Face(Value),
        Whole(String),
        Regions(String, Vec<Vec<String>>),
    }
    let all = some(
        values,
        field,
        "the profile: a sketch, a region of one or a flat face",
    )?;
    let mut out: Vec<P> = Vec::new();
    for pick in &all {
        if let Some((sketch, loops)) = region_pick(pick) {
            let sketch = picks.sketch(&format!("sketch:{sketch}"))?;
            let at = out
                .iter_mut()
                .find(|p| matches!(p, P::Whole(s) | P::Regions(s, _) if *s == sketch));
            match at {
                Some(P::Regions(_, ls)) => {
                    for l in loops {
                        if !ls.contains(&l) {
                            ls.push(l);
                        }
                    }
                }
                Some(_) => {}
                None => out.push(P::Regions(sketch, loops)),
            }
        } else if let Ok(sketch) = picks.sketch(pick) {
            match out
                .iter_mut()
                .find(|p| matches!(p, P::Whole(s) | P::Regions(s, _) if *s == sketch))
            {
                Some(p) => *p = P::Whole(sketch),
                None => out.push(P::Whole(sketch)),
            }
        } else {
            let face = picks.reference(pick)?;
            if !out.iter().any(|p| matches!(p, P::Face(f) if *f == face)) {
                out.push(P::Face(face));
            }
        }
    }
    Ok(out
        .into_iter()
        .map(|p| match p {
            P::Face(face) => face,
            P::Whole(sketch) => json!({ "sketch": sketch }),
            P::Regions(sketch, loops) => json!({ "sketch": sketch, "loops": loops }),
        })
        .collect())
}

/// Extrude, revolve and sweep: one feature per profile picked (see [`profiles`]). `None` for
/// other commands.
pub fn profile_features(
    command: &str,
    values: &Values,
    picks: &Picks,
) -> Result<Option<Vec<(&'static str, Value)>>, EditFailure> {
    let stem = match command {
        "part.extrude" => "extrude",
        "part.revolve" => "revolve",
        "part.sweep" => "sweep",
        _ => return Ok(None),
    };
    let mut out = Vec::new();
    for profile in profiles(values, "profile", picks)? {
        let kind = match command {
            "part.extrude" => extrude(values, picks, profile)?,
            "part.revolve" => revolve(values, picks, profile)?,
            _ => json!({
                "feature": "sweep",
                "profile": profile,
                "path": self::profile(values, "path", picks)?,
                "mode": mode(values, picks)?,
            }),
        };
        out.push((stem, kind));
    }
    Ok(Some(out))
}

fn revolve(values: &Values, picks: &Picks, profile: Value) -> Result<Value, EditFailure> {
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
    Ok(json!({
        "feature": "revolve",
        "profile": profile,
        "axis": picks.axis(&one(values, "axis", "the axis to revolve about")?)?,
        "angle": value(values, "angle", "deg")?,
        "mode": mode(values, picks)?,
    }))
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
        "up_to_plane" => json!({ "extent": "up_to_plane", "plane": picks.plane(&up_to()?)? }),
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

fn extrude(values: &Values, picks: &Picks, profile: Value) -> Result<Value, EditFailure> {
    let mut kind = json!({
        "feature": "extrude",
        "profile": profile,
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

use layouts::seeds;

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
            if !vertices.is_empty() {
                return Err(model(
                    "a fillet rounds edges, faces or a feature's edges; pick those, not vertices",
                ));
            }
            Ok(out)
        }
    }
}

/// A shell's walls with their own thickness: the picked faces, at the form's other thickness.
fn shell_faces(values: &Values, picks: &Picks) -> Result<Vec<Value>, EditFailure> {
    let faces = picks_of(values, "thick_faces");
    if faces.is_empty() {
        return Ok(Vec::new());
    }
    Ok(vec![json!({
        "faces": picks.references(&faces)?,
        "thickness": value(values, "face_thickness", "mm")?,
    })])
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
    if flag(values, "series") {
        // Through every body the hole's axis crosses, in order, ending in the same form.
        out["series"] = json!({ "bodies": [], "end": out["form"].clone() });
    }
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
        "table" => layouts::table_layout(values, false)?,
        "variable" => layouts::table_layout(values, true)?,
        other => {
            return Err(model(format!(
                "a {other} pattern is set up in the source editor"
            )))
        }
    };
    let skip = layouts::skip_keys(values, &layout)?;
    let mut kind =
        json!({ "feature": "pattern", "seeds": seeds(values, picks)?, "layout": layout });
    if !skip.is_empty() {
        kind["skip"] = Value::Array(skip);
    }
    Ok(kind)
}

/// The feature a part command adds, as its model JSON, and the stem its name takes.
pub fn feature_kind(
    command: &str,
    values: &Values,
    picks: &Picks,
) -> Result<Option<(&'static str, Value)>, EditFailure> {
    Ok(Some(match command {
        "part.extrude" | "part.revolve" | "part.sweep" => {
            let mut all = profile_features(command, values, picks)?.unwrap_or_default();
            if all.is_empty() {
                return Err(model("pick the profile"));
            }
            all.remove(0)
        }
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
                    "faces": shell_faces(values, picks)?,
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
                    "relief": kind,
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
                "tool": picks.plane(&one(values, "tool", "the plane, face or body to split with")?)?,
            }),
        ),
        "part.body-move" => layouts::body_move(values, picks)?,
        "part.body-indent" => ("indent", layouts::body_indent(values, picks)?),
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
            let sketch = picks.sketch(&one(values, "profile", "the sketch")?)?;
            let mut kind = json!({ "feature": "sheet_metal", "sheet_metal": "base_flange", "profile": { "sketch": sketch }, "rules": rules });
            // Depth sweeps an open chain; a closed profile is a flat plate (with a depth it would
            // sweep its outline into walls).
            if values.get("depth").is_some()
                && !crate::sheet_profile::closed_profile(picks, &sketch)
            {
                kind["depth"] = value(values, "depth", "mm")?;
            }
            ("flange", kind)
        }
        "sheet-metal.flat-pattern" => (
            "flat",
            json!({ "feature": "sheet_metal", "sheet_metal": "flat_pattern", "fixed_face": crate::sheet_profile::flat_fixed_face(picks)? }),
        ),
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
                    "relief": kind,
                }),
            )
        }
        "sheet-metal.form-tool" => {
            let q = |v: f64| json!({ "form": "quantity", "value": v, "unit": "mm" });
            let tool = match text(values, "tool", "louver") {
                "dimple" => json!({ "form": "dimple", "diameter": q(10.0), "height": q(3.0) }),
                "bead" => {
                    json!({ "form": "bead", "length": q(30.0), "width": q(8.0), "height": q(3.0) })
                }
                "louver" => {
                    json!({ "form": "louver", "length": q(20.0), "width": q(5.0), "height": q(3.0) })
                }
                other => {
                    return Err(model(format!(
                        "there is no `{other}` form tool (louver, dimple or bead)"
                    )))
                }
            };
            (
                "form",
                json!({
                    "feature": "sheet_metal",
                    "sheet_metal": "form_tool",
                    "tool": tool,
                    "face": picks.reference(&one(values, "face", "the face to stamp")?)?,
                    "at": picks.point(&one(values, "position", "where to stamp")?)?,
                    "rotation": value(values, "rotation", "deg")?,
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
        _ => match details::feature_kind(command, values, picks)? {
            Some(kind) => kind,
            None => return curves::feature_kind(command, values, picks),
        },
    }))
}

mod curves;
mod details;
mod layouts;
#[cfg(test)]
mod picked_tests;
#[cfg(test)]
mod layout_tests;

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
        inserted(command, values, 1)
    }

    /// As [`added`], for a command that inserts `count` statements (one feature per profile).
    fn inserted(command: &str, values: Value, count: usize) -> String {
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
        assert_eq!(edits.len(), count, "{command}: {edits:?}");
        for edit in edits {
            assert_eq!(edit["start"], edit["end"], "{command}: an insertion");
        }
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

    /// A face picked as the profile extrudes as a prism of the face: Add thickens the plate,
    /// Remove pushes into it, New body makes a second body, and the end conditions apply.
    #[test]
    fn a_face_profile_extrudes_a_prism_of_the_face() {
        let compiled = |text: &str| -> Value {
            serde_json::from_str(&compile_json("/p/plate.part", text, &Files::new())).unwrap()
        };
        let height = |text: &str| -> f64 {
            let b = &compiled(text)["display"]["data"]["bounds"];
            (b["max"][2].as_f64().unwrap() - b["min"][2].as_f64().unwrap()) * 1e3
        };
        let t = added(
            "part.extrude",
            json!({ "profile": [END], "op": "add", "from": "sketch_plane", "end": "blind", "distance": 5 }),
        );
        assert!(
            t.contains("feature extrude1 = extrude(profile = feature.body.face.end")
                && t.contains("distance = 5 mm"),
            "{t}"
        );
        assert!((height(&t) - 15.0).abs() < 1e-6, "{t}");
        // Remove cuts inward: 4 mm off the whole top face leaves the plate 6 mm thick.
        let t = added(
            "part.extrude",
            json!({ "profile": [END], "op": "remove", "distance": 4 }),
        );
        assert!((height(&t) - 6.0).abs() < 1e-6, "{}: {t}", height(&t));
        let t = added(
            "part.extrude",
            json!({ "profile": [END], "op": "new", "distance": 5, "draft": false }),
        );
        assert_eq!(
            compiled(&t)["display"]["data"]["bodies"]
                .as_array()
                .unwrap()
                .len(),
            2,
            "{t}"
        );
        // Every profile picked is written: a face and a sketch make two extrudes.
        let t = inserted(
            "part.extrude",
            json!({ "profile": [END, "sketch:s1"], "op": "new", "distance": 5 }),
            2,
        );
        assert!(
            t.contains("extrude1 = extrude(profile = feature.body.face.end")
                && t.contains("extrude2 = extrude(profile = sketch.s1"),
            "{t}"
        );
    }

    /// A region picked in the view extrudes its own loops; regions of one sketch go together.
    #[test]
    fn region_picks_write_their_loops() {
        let t = added(
            "part.extrude",
            json!({ "profile": ["sketch.s1.region(l1,l2,l3,l4)"], "op": "new", "distance": 3 }),
        );
        assert!(t.contains("extrude1 = extrude(") && t.contains("l4"), "{t}");
        assert!(
            super::region_pick("sketch.s1.region(l1,l2;c1)")
                == Some((
                    "s1".into(),
                    vec![vec!["l1".into(), "l2".into()], vec!["c1".into()]]
                ))
        );
    }

    /// Up to plane, and the start fields for From = surface, vertex and offset.
    #[test]
    fn extrude_up_to_a_plane_and_from_the_start_fields() {
        let t = added(
            "part.extrude",
            json!({ "profile": ["sketch:s1"], "op": "new", "end": "up_to_plane", "up_to": ["origin:top"], "from": "offset", "from_offset": -5 }),
        );
        assert!(t.contains("up_to_plane"), "{t}");
        let t = added(
            "part.extrude",
            json!({ "profile": ["sketch:s1"], "op": "add", "from": "surface", "from_face": [END], "distance": 2 }),
        );
        assert!(t.contains("from = surface("), "{t}");
    }

    #[test]
    fn sheet_metal_plate_flange_and_flat_pattern_from_the_canvas() {
        let sketch_only = PLATE.replace(
            "    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n",
            "",
        );
        let edit = |text: &str, command: &str, values: Value| -> String {
            let request = json!({ "command": command, "values": values });
            let out: Value = serde_json::from_str(&edit_json(
                "/p/plate.part",
                text,
                &Files::new(),
                &request.to_string(),
            ))
            .unwrap();
            assert_eq!(out["ok"], true, "{command}: {out}");
            out["files"][0]["text"].as_str().unwrap().to_owned()
        };
        let t = edit(
            &sketch_only,
            "sheet-metal.base-flange",
            json!({ "profile": ["sketch:s1"], "thickness": 1.5, "radius": 1, "depth": 20, "bend_rule": "k_factor", "k": 0.44 }),
        );
        assert!(
            t.contains("feature flange1 = sheet_metal.base_flange(") && !t.contains("depth"),
            "{t}"
        );
        let t = edit(&t, "sheet-metal.flat-pattern", json!({}));
        assert!(t.contains("feature flat1 = sheet_metal.flat_pattern(fixed_face = feature.flange1.face.sheet(plate"), "{t}");
        let compiled: Value =
            serde_json::from_str(&compile_json("/p/plate.part", &t, &Files::new())).unwrap();
        let errors: Vec<&Value> = compiled["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|d| d["severity"] == "error")
            .collect();
        assert!(errors.is_empty(), "{errors:?}\n{t}");
    }

    /// A lofted surface through two sketches, thickened: the features the canvas writes.
    #[test]
    fn a_lofted_surface_from_two_sketches_thickens_into_a_solid() {
        let two = "runmat 0.7\nstd 1\n\npart skin {\n    sketch a on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = 10 mm)\n    }\n    feature up = plane.offset(base = origin.xy, distance = 20 mm)\n    sketch b on plane.up {\n        c = circle(center = [0 mm, 0 mm], radius = 6 mm)\n    }\n}\n";
        let edit = |text: &str, command: &str, values: Value| -> String {
            let request = json!({ "command": command, "values": values });
            let out: Value = serde_json::from_str(&edit_json(
                "/p/skin.part",
                text,
                &Files::new(),
                &request.to_string(),
            ))
            .unwrap();
            assert_eq!(out["ok"], true, "{command}: {out}");
            out["files"][0]["text"].as_str().unwrap().to_owned()
        };
        let t = edit(
            two,
            "surface.loft",
            json!({ "profiles": ["sketch:a", "sketch:b"] }),
        );
        assert!(
            t.contains("feature loft1 = surface.loft(profiles = [sketch.a, sketch.b]"),
            "{t}"
        );
        let t = edit(
            &t,
            "surface.thicken",
            json!({ "surface": ["feature:loft1"], "thickness": 1.5, "side": "one" }),
        );
        assert!(
            t.contains("feature thicken1 = surface.thicken(bodies = [feature.loft1.body]"),
            "{t}"
        );
        // The grid test kernel doesn't loft; the solid is checked on OCCT
        // (geometry wasm host, cad_loft_thicken).
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

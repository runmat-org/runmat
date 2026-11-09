//! Canvas appliers for reference curves and mate connectors. Each returns the feature's model
//! JSON and the stem its name takes, like the other arms of [`super::feature_kind`].
//!
//! - **Helix**: about a picked axis or linear edge, starting at a picked point (its distance
//!   from the axis is the radius), `revolutions` turns of `pitch`; clockwise turns left-handed,
//!   and a taper changes the radius by a length per turn.
//! - **Spiral**: flat in a picked plane or planar face, from a start radius growing by `pitch`
//!   per turn.
//! - **Composite**: picked edges joined end to end.
//! - **Intersection**: where a planar face meets a plane.
//! - **Through points**: picked points in order, optionally closed.
//! - **Equation**: `x(t)`, `y(t)`, `z(t)` as value expressions of `t` (lengths, such as
//!   `cos(t) * 10 mm`; a quantity written before `*` reads as a compound unit), over `[t0, t1]`.
//! - **Mate connector**: on a face (its middle, or the axis of a round face), an edge (its
//!   midpoint, Z along it) or a vertex; flipped and turned about its own Z. Its name is the
//!   form's Name.

use super::{flag, model, number, one, picks_of, text, value, EditFailure, Picks, Values};
use runmat_design_syntax::ast::Statement;
use runmat_design_syntax::{parse_source, Dialect};
use serde_json::{json, Value};

/// Samples along an equation curve.
const EQUATION_SAMPLES: u32 = 120;

fn curve(kind: &str, def: Value) -> Value {
    let mut def = def;
    def["feature"] = json!("curve");
    def["curve"] = json!(kind);
    def
}

/// `written` with each bare `t` spelled as the curve's input, `input.t`, as the source reads it.
fn with_input(written: &str) -> String {
    let mut out = String::with_capacity(written.len() + 8);
    let mut word = String::new();
    let mut after_dot = false;
    let flush = |word: &mut String, out: &mut String, after_dot: bool| {
        if word == "t" && !after_dot {
            out.push_str("input.t");
        } else {
            out.push_str(word);
        }
        word.clear();
    };
    for c in written.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            if word.is_empty() {
                after_dot = out.ends_with('.');
            }
            word.push(c);
        } else {
            flush(&mut word, &mut out, after_dot);
            out.push(c);
        }
    }
    flush(&mut word, &mut out, after_dot);
    out
}

/// An equation field as a value expression of `t`.
fn expression(values: &Values, field: &str) -> Result<Value, EditFailure> {
    let written = text(values, field, "").trim();
    if written.is_empty() {
        return Err(model(format!("write {field}(t)")));
    }
    let wrong = |why: String| model(format!("{field}(t) isn't an expression of t: {written} ({why})"));
    let source = format!("runmat 0.7\nstd 1\n\npart probe {{\n    let probe = {}\n}}\n", with_input(written));
    let syntax = parse_source(Dialect::Part, source);
    if let Some(d) = syntax.diagnostics.first() {
        return Err(wrong(d.message.clone()));
    }
    let expr = syntax
        .syntax()
        .descendants()
        .into_iter()
        .filter_map(Statement::cast)
        .find_map(|s| match s {
            Statement::Value(v) if v.keyword() == "let" => v.value(),
            _ => None,
        })
        .ok_or_else(|| wrong("no expression".into()))?;
    let lowered = runmat_cad::syntax::value::lower_value(&expr).map_err(|e| wrong(e.message))?;
    serde_json::to_value(lowered).map_err(|e| wrong(e.to_string()))
}

/// What a lowered reference names: `edge`, `vertex` or `face`.
fn topo(reference: &Value) -> &'static str {
    match reference["ref"].as_str() {
        Some("edges_between") => "edge",
        Some("vertex_of") => "vertex",
        Some("piece") => topo(&reference["base"]),
        _ if reference["kind"] == "edge" => "edge",
        _ if reference["kind"] == "vertex" => "vertex",
        _ => "face",
    }
}

fn connector(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let on = picks.reference(&one(values, "on", "the face, edge or vertex it sits on")?)?;
    let placement = match (topo(&on), text(values, "origin", "centroid")) {
        ("vertex", _) => json!({ "on": "vertex", "vertex": on }),
        ("edge", "center") => return Err(model("the axis of a round face takes a face; pick one, or place the connector at the edge's middle")),
        ("edge", _) => json!({ "on": "edge_midpoint", "edge": on }),
        // On a face, the middle: a planar face's centroid, a round face's axis at mid-length.
        _ => json!({ "on": "center", "of": on }),
    };
    let rotation = match values.get("rotate") {
        Some(Value::Number(n)) if n.as_f64() == Some(0.0) => None,
        None | Some(Value::Null) => None,
        Some(_) => Some(value(values, "rotate", "deg")?),
    };
    // Offsets along the connector's own X, Y and Z; written only when one is set.
    let zero = |k: &str| matches!(values.get(k), None | Some(Value::Null)) || values.get(k).and_then(Value::as_f64) == Some(0.0);
    let axis = |k: &str| if zero(k) { Ok(json!({ "form": "quantity", "value": 0, "unit": "mm" })) } else { value(values, k, "mm") };
    let offset = if ["offset_x", "offset_y", "offset_z"].iter().all(|k| zero(k)) {
        None
    } else {
        Some(json!([axis("offset_x")?, axis("offset_y")?, axis("offset_z")?]))
    };
    Ok(json!({
        "feature": "mate_connector",
        "placement": placement,
        "offset": offset,
        "rotation": rotation,
        "flip": flag(values, "flip"),
    }))
}

fn helix(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    if text(values, "type", "helix") == "spiral" {
        return Ok(curve(
            "spiral",
            json!({
                "plane": picks.plane(&one(values, "plane", "the plane the spiral lies in")?)?,
                "start_radius": value(values, "start_radius", "mm")?,
                "pitch": value(values, "pitch", "mm")?,
                "turns": number(values, "revolutions")?,
            }),
        ));
    }
    let taper = if flag(values, "taper") { Some(value(values, "taper_step", "mm")?) } else { None };
    Ok(curve(
        "helix",
        json!({
            "axis": picks.axis(&one(values, "axis", "the axis the helix turns about")?)?,
            "start": picks.point(&one(values, "start", "the point the helix starts at")?)?,
            "pitch": value(values, "pitch", "mm")?,
            "turns": number(values, "revolutions")?,
            "left_hand": flag(values, "clockwise"),
            "taper": taper,
        }),
    ))
}

/// The curve and connector commands; `None` for other commands.
pub(super) fn feature_kind(command: &str, values: &Values, picks: &Picks) -> Result<Option<(&'static str, Value)>, EditFailure> {
    Ok(Some(match command {
        "part.mate-connector" => ("connector", connector(values, picks)?),
        "part.curve-helix" => {
            let def = helix(values, picks)?;
            (if def["curve"] == "spiral" { "spiral" } else { "helix" }, def)
        }
        "part.curve-composite" => {
            let edges = picks_of(values, "curves");
            if edges.len() < 2 {
                return Err(model("pick two or more edges that join end to end"));
            }
            ("composite", curve("composite", json!({ "edges": picks.references(&edges)? })))
        }
        "part.curve-intersection" => (
            "intersection",
            curve(
                "intersection",
                json!({
                    "a": picks.reference(&one(values, "face", "the planar face")?)?,
                    "b": picks.plane(&one(values, "plane", "the plane that cuts it")?)?,
                }),
            ),
        ),
        "part.curve-through-points" => {
            let points = picks_of(values, "points");
            if points.len() < 2 {
                return Err(model("pick two or more points"));
            }
            let points = points.iter().map(|p| picks.point(p)).collect::<Result<Vec<_>, _>>()?;
            ("spline", curve("through_points", json!({ "points": points, "closed": flag(values, "closed") })))
        }
        "part.curve-equation" => (
            "equation",
            curve(
                "equation",
                json!({
                    "parameter": "t",
                    "x": expression(values, "x")?,
                    "y": expression(values, "y")?,
                    "z": expression(values, "z")?,
                    "t0": number(values, "t0")?,
                    "t1": number(values, "t1")?,
                    "samples": EQUATION_SAMPLES,
                }),
            ),
        ),
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests {
    use crate::edit::edit_json;
    use crate::runtime::{compile_json, Files};
    use serde_json::{json, Value};

    /// A 40 x 20 x 10 plate with a 6 mm bore.
    const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n        c1 = circle(center = [20 mm, 10 mm], radius = 3 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n}\n";

    const TOP: &str = "feature.body.face.end";
    const BORE: &str = "feature.body.face.side(s1.c1)";
    const E1: &str = "edges_between(feature.body.face.side(s1.l1), feature.body.face.end)";
    const E2: &str = "edges_between(feature.body.face.side(s1.l2), feature.body.face.end)";
    const V: &str = "vertex_of(feature.body.face.side(s1.l1), feature.body.face.side(s1.l2), feature.body.face.end)";

    /// Applies `command` and returns the new text, which must regenerate with no error, and the
    /// compiled display's references.
    fn added(command: &str, values: Value) -> (String, Vec<Value>) {
        let request = json!({ "command": command, "values": values });
        let out: Value = serde_json::from_str(&edit_json("/p/plate.part", PLATE, &Files::new(), &request.to_string())).unwrap();
        assert_eq!(out["ok"], true, "{command}: {out}");
        let text = out["files"][0]["text"].as_str().unwrap().to_owned();
        let compiled: Value = serde_json::from_str(&compile_json("/p/plate.part", &text, &Files::new())).unwrap();
        let errors: Vec<&Value> = compiled["diagnostics"].as_array().unwrap().iter().filter(|d| d["severity"] == "error").collect();
        assert!(errors.is_empty(), "{command}: {errors:?}\n{text}");
        let refs = compiled["display"]["data"]["references"].as_array().cloned().unwrap_or_default();
        (text, refs)
    }

    fn refuses(command: &str, values: Value) -> String {
        let request = json!({ "command": command, "values": values });
        let out: Value = serde_json::from_str(&edit_json("/p/plate.part", PLATE, &Files::new(), &request.to_string())).unwrap();
        assert_eq!(out["ok"], false, "{command}: {out}");
        out["message"].as_str().unwrap_or_default().to_owned()
    }

    #[test]
    fn a_bare_t_reads_as_the_curves_input() {
        assert_eq!(super::with_input("10 mm * cos(t) + t2 + a.t"), "10 mm * cos(input.t) + t2 + a.t");
    }

    #[test]
    fn helix_and_spiral_from_the_canvas_draw_as_curves() {
        let (t, refs) = added("part.curve-helix", json!({ "type": "helix", "axis": ["origin:z"], "start": [V], "pitch": 2, "revolutions": 3, "clockwise": true, "taper": true, "taper_step": 0.5 }));
        assert!(t.contains("feature helix1 = curve.helix(") && t.contains("pitch = 2 mm") && t.contains("left_hand,") && t.contains("taper = 0.5 mm"), "{t}");
        assert_eq!(refs.len(), 1, "{refs:?}");
        assert_eq!(refs[0]["kind"], "curve");
        // 3 turns at 48 samples a turn: 145 points.
        assert_eq!(refs[0]["lines"][0].as_array().unwrap().len(), 145 * 3);
        let (t, refs) = added("part.curve-helix", json!({ "type": "spiral", "plane": [TOP], "start_radius": 2, "pitch": 1, "revolutions": 2 }));
        assert!(t.contains("feature spiral1 = curve.spiral(") && t.contains("start_radius = 2 mm"), "{t}");
        assert_eq!(refs[0]["id"], "spiral1");
        assert!(refuses("part.curve-helix", json!({ "type": "helix", "start": [V], "pitch": 2, "revolutions": 3 })).contains("axis"));
    }

    #[test]
    fn composite_intersection_points_and_equation_curves_from_the_canvas() {
        let (t, refs) = added("part.curve-composite", json!({ "curves": [E1, E2] }));
        assert!(t.contains("feature composite1 = curve.composite("), "{t}");
        assert_eq!(refs[0]["lines"][0].as_array().unwrap().len(), 9);
        let (t, _) = added("part.curve-intersection", json!({ "face": [TOP], "plane": ["origin:right"] }));
        assert!(t.contains("feature intersection1 = curve.intersection("), "{t}");
        let (t, refs) = added("part.curve-through-points", json!({ "points": ["origin:point", V], "closed": false }));
        assert!(t.contains("feature spline1 = curve.through_points("), "{t}");
        assert_eq!(refs[0]["kind"], "curve");
        let (t, refs) = added("part.curve-equation", json!({ "x": "cos(t) * 10 mm", "y": "sin(t) * 10 mm", "z": "t * 2 mm", "t0": 0, "t1": 6.283 }));
        assert!(t.contains("feature equation1 = curve.equation(") && t.contains("cos(input.t)"), "{t}");
        assert_eq!(refs[0]["lines"][0].as_array().unwrap().len(), 121 * 3);
        assert!(refuses("part.curve-composite", json!({ "curves": [E1] })).contains("two or more"));
        // Edges in any order: the chain grows from whichever end the next edge meets.
        let e3 = "edges_between(feature.body.face.side(s1.l3), feature.body.face.end)";
        let (_, refs) = added("part.curve-composite", json!({ "curves": [E1, e3, E2] }));
        assert_eq!(refs[0]["lines"][0].as_array().unwrap().len(), 4 * 3);
    }

    #[test]
    fn mate_connectors_from_the_canvas_take_their_name() {
        let (t, refs) = added("part.mate-connector", json!({ "name": "pivot", "on": [TOP], "origin": "centroid", "flip": false, "rotate": 0 }));
        assert!(t.contains("feature pivot = mate_connector(") && t.contains("on = center"), "{t}");
        // The middle of the 40 x 20 top face, 10 mm up (display in metres).
        let o = &refs[0]["lines"][0];
        assert!([0.02, 0.01, 0.01].iter().enumerate().all(|(k, v)| (o[k].as_f64().unwrap() - v).abs() < 1e-6), "{o}");
        assert_eq!(refs[0]["kind"], "connector");
        assert_eq!(refs[0]["lines"].as_array().unwrap().len(), 3);
        let (t, _) = added("part.mate-connector", json!({ "name": "bore", "on": [BORE], "origin": "center", "rotate": 30, "flip": true }));
        assert!(t.contains("feature bore = mate_connector(") && t.contains("on = center") && t.contains("rotation = 30 deg") && t.contains(", flip)"), "{t}");
        let (t, _) = added("part.mate-connector", json!({ "name": "", "on": [E1] }));
        assert!(t.contains("feature connector1 = mate_connector(") && t.contains("edge_midpoint"), "{t}");
        let (t, _) = added("part.mate-connector", json!({ "on": [V] }));
        assert!(t.contains("on = vertex") && !t.contains("offset"), "{t}");
        // Offsets move the origin along the connector's own axes: 5 mm up the face normal.
        let (t, refs) = added("part.mate-connector", json!({ "name": "lifted", "on": [TOP], "offset_x": 0, "offset_y": 0, "offset_z": 5 }));
        assert!(t.contains("offset = [0 mm, 0 mm, 5 mm]"), "{t}");
        let (x0, z0) = (refs[0]["lines"][0][0].as_f64().unwrap(), refs[0]["lines"][0][2].as_f64().unwrap());
        assert!((x0 - 0.02).abs() < 1e-6, "connector origin x {x0}");
        // The display is in metres: the 10 mm top face plus 5 mm.
        assert!((z0 - 0.015).abs() < 1e-6, "connector origin z {z0}");
        assert!(refuses("part.mate-connector", json!({ "name": "body", "on": [TOP] })).contains("body"));
        assert!(refuses("part.mate-connector", json!({ "name": "2x", "on": [TOP] })).contains("name"));
    }
}

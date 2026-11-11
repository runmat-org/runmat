//! Canvas appliers for detail features and reference geometry: rib, thread, emboss, and the
//! datum axis, point and coordinate system. Each returns the feature's model JSON and the stem
//! its name takes, like the other arms of [`super::feature_kind`].
//!
//! - **Rib**: an open sketch chain, thickened in its plane by `thickness` and extended `depth`
//!   along the sketch normal into the body (`reverse` flips the side).
//! - **Thread**: a cylindrical face takes the size's pitch and thread depth from the ISO metric
//!   or unified inch tables. The thread is cosmetic: an attribute for drawings and
//!   PMI. (Modelled helical threads wait on a kernel helical cut; the form doesn't offer one.)
//! - **Emboss**: a sketch on a face raises (emboss) or sinks (deboss) by the depth.
//! - **Axis**: along a linear edge, where two planes meet, through two points, the axis of a
//!   cylindrical face, or through a point normal to a plane or face.
//! - **Point**: coordinates, the centre of a circular edge or of a face, where an axis meets a
//!   plane, or a fraction of the way along an edge or curve.
//! - **Coordinate system**: an origin, Z along a picked axis or edge, X towards a second one;
//!   or the frame of a planar face.

use super::{flag, model, one, picks_of, some, text, value, EditFailure, Picks, Values};
use runmat_cad::model::hole_tables::{thread, HoleSpec, HoleStandard};
use serde_json::{json, Value};

fn mm(v: f64) -> Value {
    json!({ "form": "quantity", "value": v, "unit": "mm" })
}

/// The picks of `references`, at least `n` of them.
fn refs(values: &Values, n: usize, what: &str) -> Result<Vec<String>, EditFailure> {
    let all = picks_of(values, "references");
    if all.len() < n {
        return Err(model(format!("pick {what}")));
    }
    Ok(all)
}

/// A curve pick: a reference-curve or sketch feature, else an edge.
fn curve(pick: &str, picks: &Picks) -> Result<Value, EditFailure> {
    if let Some(id) = pick.strip_prefix("feature:") {
        return Ok(json!({ "curve": "feature", "feature": id }));
    }
    match picks.sketch(pick) {
        Ok(sketch) => Ok(json!({ "curve": "sketch", "sketch": sketch })),
        Err(_) => Ok(json!({ "curve": "edge", "edge": picks.reference(pick)? })),
    }
}

fn axis(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    Ok(match text(values, "type", "edge") {
        "edge" => {
            let r = refs(values, 1, "a linear edge")?;
            json!({ "axis": "of_edge", "edge": picks.reference(&r[0])? })
        }
        "two_planes" => {
            let r = refs(values, 2, "two planes or planar faces")?;
            json!({ "axis": "two_planes", "a": picks.plane(&r[0])?, "b": picks.plane(&r[1])? })
        }
        "two_points" => {
            let r = refs(values, 2, "two points or vertices")?;
            json!({ "axis": "through", "a": picks.point(&r[0])?, "b": picks.point(&r[1])? })
        }
        "cylinder" => {
            let r = refs(values, 1, "a cylindrical or conical face")?;
            json!({ "axis": "of_face", "face": picks.reference(&r[0])? })
        }
        "point_face" => {
            let r = refs(values, 2, "a point and a plane or planar face")?;
            // Either order: the one that lowers as a point is the point.
            let (p, f) = match picks.point(&r[0]) {
                Ok(p) => (p, picks.plane(&r[1])?),
                Err(_) => (picks.point(&r[1])?, picks.plane(&r[0])?),
            };
            json!({ "axis": "normal", "plane": f, "through": p })
        }
        other => return Err(model(format!("there is no `{other}` axis"))),
    })
}

fn point(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    Ok(match text(values, "type", "coordinates") {
        "coordinates" => json!({
            "point_def": "at",
            "at": { "point": "coords", "x": value(values, "x", "mm")?, "y": value(values, "y", "mm")?, "z": value(values, "z", "mm")? },
        }),
        "center" | "face_center" => {
            let r = refs(values, 1, "a circular edge or a face")?;
            json!({ "point_def": "center_of", "of": picks.reference(&r[0])? })
        }
        "intersection" => {
            let r = refs(values, 2, "an axis or edge and a plane or face")?;
            let (a, p) = match picks.axis(&r[0]) {
                Ok(a) => (a, picks.plane(&r[1])?),
                Err(_) => (picks.axis(&r[1])?, picks.plane(&r[0])?),
            };
            json!({ "point_def": "intersection", "axis": a, "plane": p })
        }
        "along_curve" => {
            let r = refs(values, 1, "the edge or curve")?;
            let t = values
                .get("parameter")
                .and_then(Value::as_f64)
                .unwrap_or(0.5);
            if !(0.0..=1.0).contains(&t) {
                return Err(model("the point sits a fraction 0 to 1 of the way along"));
            }
            json!({ "point_def": "on_curve", "curve": curve(&r[0], picks)?, "parameter": { "form": "number", "value": t } })
        }
        other => return Err(model(format!("there is no `{other}` point"))),
    })
}

fn frame(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    Ok(match text(values, "type", "axes") {
        "face" => {
            json!({ "frame": "on_face", "face": picks.reference(&one(values, "face", "the planar face")?)? })
        }
        _ => {
            let x = match picks_of(values, "x").first() {
                Some(p) => Some(picks.axis(p)?),
                None => None,
            };
            json!({
                "frame": "at",
                "origin": picks.point(&one(values, "origin", "the origin")?)?,
                "z": picks.axis(&one(values, "z", "the Z axis")?)?,
                "x": x,
            })
        }
    })
}

fn thread_feature(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let standard = match text(values, "standard", "iso") {
        "unified" => HoleStandard::AnsiInch,
        _ => HoleStandard::Iso,
    };
    let size = text(values, "designation", "M6x1").trim();
    let data = thread(&HoleSpec {
        standard,
        size: size.into(),
        fit: Default::default(),
        class: None,
    })
    .map_err(|e| model(format!("the thread size: {e}")))?;
    let length = match values.get("length") {
        Some(v) if !v.is_null() && v != "" => Some(value(values, "length", "mm")?),
        _ => None,
    };
    Ok(json!({
        "feature": "thread",
        "face": picks.reference(&one(values, "face", "the cylindrical face")?)?,
        "spec": {
            "designation": data.designation,
            "pitch": mm(data.pitch * 1000.0),
            "depth": mm((data.depth() * 1e6).round() / 1000.0),
            "left_hand": flag(values, "left_hand"),
            "class": data.class,
        },
        "length": length,
        // Cosmetic: the kernel has no helical thread cut yet, so the canvas makes none.
        "modelled": false,
    }))
}

fn emboss(values: &Values, picks: &Picks) -> Result<Value, EditFailure> {
    let mut height = value(values, "depth", "mm")?;
    if text(values, "mode", "emboss") == "deboss" {
        match height.get_mut("value") {
            Some(Value::Number(n)) => {
                let v = -n.as_f64().unwrap_or(0.0);
                height["value"] = json!(v);
            }
            _ => return Err(model("a deboss depth is a number of millimetres")),
        }
    }
    Ok(json!({
        "feature": "emboss",
        "profile": { "sketch": picks.sketch(&one(values, "profile", "the sketch")?)? },
        "face": picks.reference(&one(values, "face", "the face")?)?,
        "height": height,
        "wrap": false,
    }))
}

/// The detail and reference features; `None` for other commands.
pub(super) fn feature_kind(
    command: &str,
    values: &Values,
    picks: &Picks,
) -> Result<Option<(&'static str, Value)>, EditFailure> {
    let with = |kind: &str, mut def: Value| {
        def["feature"] = json!(kind);
        def
    };
    Ok(Some(match command {
        "part.rib" => (
            "rib",
            json!({
                "feature": "rib",
                "profile": { "sketch": picks.sketch(&some(values, "profile", "the open sketch")?[0])? },
                "thickness": value(values, "thickness", "mm")?,
                "depth": value(values, "depth", "mm")?,
                "reverse": flag(values, "reverse"),
            }),
        ),
        "part.thread" => ("thread", thread_feature(values, picks)?),
        "part.emboss" => ("emboss", emboss(values, picks)?),
        "part.axis" => ("axis", with("axis", axis(values, picks)?)),
        "part.point" => ("point", with("point", point(values, picks)?)),
        "part.coordinate-system" => ("frame", with("frame", frame(values, picks)?)),
        _ => return Ok(None),
    }))
}

#[cfg(test)]
mod tests {
    use crate::edit::edit_json;
    use crate::runtime::{compile_json, Files};
    use serde_json::{json, Value};

    /// A 40 x 20 x 10 plate with a 6 mm bore, an open rib sketch on its top and a logo sketch.
    const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [40 mm, 0 mm])\n        l2 = line(start = [40 mm, 0 mm], end = [40 mm, 20 mm])\n        l3 = line(start = [40 mm, 20 mm], end = [0 mm, 20 mm])\n        l4 = line(start = [0 mm, 20 mm], end = [0 mm, 0 mm])\n        c1 = circle(center = [20 mm, 10 mm], radius = 3 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    sketch logo on feature.body.face.end {\n        g1 = line(start = [4 mm, 4 mm], end = [10 mm, 4 mm])\n        g2 = line(start = [10 mm, 4 mm], end = [10 mm, 8 mm])\n        g3 = line(start = [10 mm, 8 mm], end = [4 mm, 8 mm])\n        g4 = line(start = [4 mm, 8 mm], end = [4 mm, 4 mm])\n    }\n    sketch web on origin.zx {\n        w1 = line(start = [10 mm, 5 mm], end = [20 mm, 30 mm])\n    }\n}\n";

    /// Applies `command` and returns the new text, which must regenerate with no error.
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
        let text = out["files"][0]["text"].as_str().unwrap().to_owned();
        let compiled: Value =
            serde_json::from_str(&compile_json("/p/plate.part", &text, &Files::new())).unwrap();
        let errors: Vec<&Value> = compiled["diagnostics"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|d| d["severity"] == "error")
            .collect();
        assert!(errors.is_empty(), "{command}: {errors:?}\n{text}");
        text
    }

    const BORE: &str = "feature.body.face.side(s1.c1)";
    const TOP: &str = "feature.body.face.end";
    const EDGE: &str = "edges_between(feature.body.face.side(s1.l1), feature.body.face.end)";

    #[test]
    fn rib_thread_and_emboss_from_the_canvas() {
        let t = added(
            "part.rib",
            json!({ "profile": ["feature:web"], "thickness": 2, "depth": 5 }),
        );
        assert!(
            t.contains("feature rib1 = rib(") && t.contains("thickness = 2 mm"),
            "{t}"
        );
        let t = added(
            "part.thread",
            json!({ "face": [BORE], "standard": "iso", "designation": "M6", "length": 8 }),
        );
        assert!(
            t.contains("feature thread1 = thread(") && t.contains("M6x1"),
            "{t}"
        );
        // Cosmetic: a regenerating thread with no helical cut.
        assert!(!t.contains("modelled"), "{t}");
        let t = added(
            "part.emboss",
            json!({ "profile": ["feature:logo"], "face": [TOP], "depth": 0.5, "mode": "deboss" }),
        );
        assert!(
            t.contains("feature emboss1 = emboss(") && t.contains("-0.5 mm"),
            "{t}"
        );
    }

    #[test]
    fn reference_axes_points_and_frames_from_the_canvas() {
        let t = added("part.axis", json!({ "type": "edge", "references": [EDGE] }));
        assert!(t.contains("feature axis1 = "), "{t}");
        let t = added(
            "part.axis",
            json!({ "type": "two_planes", "references": ["origin:top", "origin:front"] }),
        );
        assert!(t.contains("feature axis1 = "), "{t}");
        let t = added(
            "part.axis",
            json!({ "type": "cylinder", "references": [BORE] }),
        );
        assert!(t.contains("feature axis1 = "), "{t}");
        let t = added(
            "part.point",
            json!({ "type": "coordinates", "x": 5, "y": 6, "z": 7 }),
        );
        assert!(t.contains("feature point1 = ") && t.contains("7 mm"), "{t}");
        let t = added(
            "part.point",
            json!({ "type": "along_curve", "references": [EDGE], "parameter": 0.25 }),
        );
        assert!(t.contains("feature point1 = ") && t.contains("0.25"), "{t}");
        let t = added(
            "part.coordinate-system",
            json!({ "type": "face", "face": [TOP] }),
        );
        assert!(t.contains("feature frame1 = "), "{t}");
    }
}

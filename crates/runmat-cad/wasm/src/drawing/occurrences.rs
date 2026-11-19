//! What an assembly view's annotations read from its occurrences: an occurrence's faces where the
//! assembly places them, the datum planes its part publishes, and a door occurrence's swing.
//!
//! - [`scoped`]: the faces of one occurrence's part as a part display, its points, normals and
//!   axes moved by the occurrence's world transform. Dimensions and level marks on an assembly
//!   view resolve their references against it, as a part view resolves against its part.
//! - [`datum_planes`]: the datum planes an occurrence's part draws (a skeleton's grids), each
//!   its centre and box in the assembly (mm).
//! - [`door_swing`]: a door occurrence's leaf, open 90° from its hinge, and the arc to its
//!   closed position, in the assembly (mm).

use runmat_cad::drawing::{Hinge, Swing};
use serde_json::{json, Value};

type P3 = [f64; 3];

fn floats(v: &Value) -> Vec<f64> {
    runmat_cad::display::wire::numbers(v)
}

/// An occurrence's prototype and world transform (the rotation's rows, then the translation
/// in metres).
fn placed<'v>(display: &'v Value, path: &str) -> Option<(&'v Value, [f64; 12])> {
    if display["kind"] != "assembly" {
        return None;
    }
    let data = &display["data"];
    let occ = data["occurrences"]
        .as_array()?
        .iter()
        .find(|o| o["path"] == path)?;
    let proto = data["prototypes"].get(occ["prototype"].as_u64()? as usize)?;
    let t = floats(&occ["transform"]);
    (t.len() == 12).then(|| (proto, std::array::from_fn(|i| t[i])))
}

fn rotate(t: &[f64; 12], v: P3) -> P3 {
    [0, 1, 2].map(|r| t[3 * r] * v[0] + t[3 * r + 1] * v[1] + t[3 * r + 2] * v[2])
}

/// A part point (metres) in the assembly (metres).
fn place(t: &[f64; 12], p: P3) -> P3 {
    let r = rotate(t, p);
    [0, 1, 2].map(|i| r[i] + t[9 + i])
}

fn vec3(v: &Value) -> Option<P3> {
    let f = floats(v);
    (f.len() == 3).then(|| [f[0], f[1], f[2]])
}

/// The prototype's file, as the assembly display names it.
pub fn file_of(display: &Value, path: &str) -> Option<String> {
    placed(display, path).and_then(|(proto, _)| proto["file"].as_str().map(str::to_owned))
}

/// The faces of the occurrence `path`'s part as a part display (`{ kind: "part", data: {
/// bodies } }`), placed in the assembly: positions, plane normals and cylinder axes moved by the
/// occurrence's transform. `None` when the assembly has no such occurrence or draws it as a box.
pub fn scoped(display: &Value, path: &str) -> Option<Value> {
    let (proto, t) = placed(display, path)?;
    if proto["shape"] != "mesh" {
        return None;
    }
    let mut bodies = Vec::new();
    for body in proto["part"]["bodies"].as_array().into_iter().flatten() {
        let mut faces = Vec::new();
        for f in body["faces"].as_array().into_iter().flatten() {
            let positions: Vec<f64> = floats(&f["positions"])
                .chunks_exact(3)
                .flat_map(|c| place(&t, [c[0], c[1], c[2]]))
                .collect();
            let mut surface = f["surface"].clone();
            if let Value::Object(o) = &mut surface {
                for key in ["normal", "axis_direction"] {
                    if let Some(v) = o.get(key).and_then(vec3) {
                        o.insert(key.into(), json!(rotate(&t, v)));
                    }
                }
                if let Some(p) = o.get("axis_point").and_then(vec3) {
                    o.insert("axis_point".into(), json!(place(&t, p)));
                }
            }
            faces.push(json!({
                "reference": f["reference"],
                "names": f["names"],
                "positions": positions,
                "surface": surface,
            }));
        }
        bodies.push(json!({ "faces": faces }));
    }
    Some(json!({ "kind": "part", "data": { "bodies": bodies } }))
}

/// Whether a scoped display has a face carrying this reference text.
pub fn has_face(scoped: &Value, text: &str) -> bool {
    super::values::face(scoped, text).is_some()
}

/// The datum planes `features` of the occurrence `path`'s part, in order: each plane's centre
/// and its box `[min, max]` in the assembly (mm), from the outline the part's display draws it
/// with. `None` for a feature the part doesn't draw as a plane.
pub fn datum_planes(
    display: &Value,
    path: &str,
    features: &[String],
) -> Vec<Option<(P3, [P3; 2])>> {
    let Some((proto, t)) = placed(display, path) else {
        return features.iter().map(|_| None).collect();
    };
    let references = proto["part"]["references"].as_array();
    features
        .iter()
        .map(|feature| {
            let r = references?
                .iter()
                .find(|r| r["id"] == feature.as_str() && r["kind"] == "plane")?;
            let points: Vec<P3> = r["lines"]
                .as_array()?
                .iter()
                .flat_map(|l| {
                    floats(l)
                        .chunks_exact(3)
                        .map(|c| place(&t, [c[0], c[1], c[2]]).map(|x| x * 1000.0))
                        .collect::<Vec<_>>()
                })
                .collect();
            if points.is_empty() {
                return None;
            }
            let lo = [0, 1, 2].map(|i| points.iter().map(|p| p[i]).fold(f64::MAX, f64::min));
            let hi = [0, 1, 2].map(|i| points.iter().map(|p| p[i]).fold(f64::MIN, f64::max));
            Some(([0, 1, 2].map(|i| (lo[i] + hi[i]) / 2.0), [lo, hi]))
        })
        .collect()
}

/// A door occurrence's swing in the assembly (mm): `{ hinge, open, closed, arc }`, the leaf from
/// `hinge` to `open` (square to the wall) and the quarter arc from `open` to `closed` (where the
/// leaf's end sits shut). The leaf is the door's thinnest body along its y (the slab, not the
/// frame), as wide as that body runs along its x; it hangs at the `hinge` end of that run, on the
/// `swing` face of the body.
pub fn door_swing(display: &Value, path: &str, hinge: Hinge, swing: Swing) -> Option<Value> {
    let (proto, t) = placed(display, path)?;
    let boxes: Vec<(P3, P3)> = match proto["shape"].as_str() {
        Some("mesh") => proto["part"]["bodies"]
            .as_array()?
            .iter()
            .filter_map(|b| {
                let pts: Vec<f64> = b["faces"]
                    .as_array()?
                    .iter()
                    .flat_map(|f| floats(&f["positions"]))
                    .collect();
                let pts: Vec<P3> = pts.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
                (!pts.is_empty()).then(|| {
                    (
                        [0, 1, 2].map(|i| pts.iter().map(|p| p[i]).fold(f64::MAX, f64::min)),
                        [0, 1, 2].map(|i| pts.iter().map(|p| p[i]).fold(f64::MIN, f64::max)),
                    )
                })
            })
            .collect(),
        _ => vec![(
            vec3(&proto["bounds"]["min"])?,
            vec3(&proto["bounds"]["max"])?,
        )],
    };
    let (lo, hi) = boxes
        .into_iter()
        .min_by(|a, b| (a.1[1] - a.0[1]).total_cmp(&(b.1[1] - b.0[1])))?;
    let width = hi[0] - lo[0];
    if width <= 0.0 {
        return None;
    }
    let (x_hinge, x_free) = match hinge {
        Hinge::Start => (lo[0], hi[0]),
        Hinge::End => (hi[0], lo[0]),
    };
    let (y, out) = match swing {
        Swing::Positive => (hi[1], 1.0),
        Swing::Negative => (lo[1], -1.0),
    };
    let z = lo[2];
    let world = |p: P3| place(&t, p).map(|x| x * 1000.0);
    let h = [x_hinge, y, z];
    let open = [x_hinge, y + out * width, z];
    let closed = [x_free, y, z];
    // The arc about the hinge, from open to closed, in 16 steps.
    let arc: Vec<P3> = (0..=16)
        .map(|k| {
            let a = std::f64::consts::FRAC_PI_2 * k as f64 / 16.0;
            let (along, across) = (a.sin(), a.cos());
            world([
                x_hinge + (x_free - x_hinge) * along,
                y + out * width * across,
                z,
            ])
        })
        .collect();
    Some(json!({ "hinge": world(h), "open": world(open), "closed": world(closed), "arc": arc }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-face prototype placed by a quarter turn about z and moved 1 m along x.
    fn assembly() -> Value {
        json!({ "kind": "assembly", "data": {
            "prototypes": [{ "file": "slab.part", "shape": "mesh", "part": {
                "bodies": [
                    { "faces": [{ "reference": "feature.f.face.end", "names": [],
                        "positions": [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.8, 0.04, 0.0, 0.0, 0.04, 0.0],
                        "surface": { "surface": "plane", "normal": [0.0, 1.0, 0.0] } }] },
                    { "faces": [{ "reference": "feature.g.face.end", "names": [],
                        "positions": [0.0, -0.05, 0.0, 0.9, -0.05, 0.0, 0.9, 0.1, 0.0],
                        "surface": { "surface": "plane", "normal": [0.0, 0.0, 1.0] } }] }
                ],
                "references": [{ "id": "grid_2", "kind": "plane",
                    "lines": [[0.5, -1.0, -1.0, 0.5, 1.0, -1.0, 0.5, 1.0, 1.0, 0.5, -1.0, 1.0]] }]
            } }],
            "occurrences": [{ "path": "set/door", "prototype": 0,
                "transform": [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0] }]
        } })
    }

    #[test]
    fn an_occurrence_s_faces_sit_where_the_assembly_places_them() {
        let s = scoped(&assembly(), "set/door").expect("scoped");
        let (points, surface) = super::super::values::face(&s, "feature.f.face.end").unwrap();
        // Part (0.8, 0, 0) m turned a quarter about z is (0, 0.8, 0), then moved 1 m along x.
        assert!(points
            .iter()
            .any(|p| (p[0] - 1000.0).abs() < 1e-6 && (p[1] - 800.0).abs() < 1e-6));
        let n = vec3(&surface["normal"]).unwrap();
        assert!((n[0] + 1.0).abs() < 1e-9 && n[1].abs() < 1e-9);
        assert!(scoped(&assembly(), "set/missing").is_none());
    }

    #[test]
    fn a_datum_plane_reads_its_centre_and_box_in_the_assembly() {
        let planes = datum_planes(&assembly(), "set/door", &["grid_2".into(), "nope".into()]);
        let (c, [lo, hi]) = planes[0].expect("grid_2");
        // x = 0.5 m in the part is y = 0.5 m in the assembly.
        assert!((c[1] - 500.0).abs() < 1e-6 && (lo[1] - hi[1]).abs() < 1e-6);
        assert!(planes[1].is_none());
    }

    #[test]
    fn a_door_swing_opens_its_thinnest_body_square_to_the_wall() {
        let swing = door_swing(&assembly(), "set/door", Hinge::Start, Swing::Positive).unwrap();
        let p = |k: &str| vec3(&swing[k]).unwrap();
        // The leaf is the 40 mm body (0.8 m wide), hinged at part x = 0 on its +y face (y = 40 mm).
        let (h, open, closed) = (p("hinge"), p("open"), p("closed"));
        let len = |a: P3, b: P3| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        assert!((len(h, open) - 800.0).abs() < 1e-6);
        assert!((len(h, closed) - 800.0).abs() < 1e-6);
        // Square to the wall: part +y is assembly -x.
        assert!((open[0] - (h[0] - 800.0)).abs() < 1e-6);
        let arc = swing["arc"].as_array().unwrap();
        assert_eq!(arc.len(), 17);
        assert!(arc
            .iter()
            .all(|q| (len(vec3(q).unwrap(), h) - 800.0).abs() < 1e-6));
    }
}

//! Use edges in a part: Project, Intersection, Silhouette and Offset on surface bring the
//! part's own edges and faces into a sketch, and each converted entity follows the model.

use super::tests::{applied, run};
use crate::runtime::compile_part;
use serde_json::json;

/// A 40 x 20 x 5 mm plate centred on the origin, with an empty sketch on its top face.
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        l1 = line(start = [-20 mm, -10 mm], end = [20 mm, -10 mm])\n        l2 = line(start = l1.end, end = [20 mm, 10 mm])\n        l3 = line(start = l2.end, end = [-20 mm, 10 mm])\n        l4 = line(start = l3.end, end = l1.start)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 5 mm)\n    sketch top on feature.body.face.end {\n    }\n}\n";

/// The solved entities of sketch `id`, in millimetres: (kind, start, end) of each.
fn solved(text: &str, id: &str) -> Vec<(String, [f64; 2], [f64; 2])> {
    let compiled = compile_part("/p/plate.part", text).unwrap_or_else(|d| panic!("{d:?}\n{text}"));
    let sketch = compiled
        .display
        .sketches
        .iter()
        .find(|s| s.id == id)
        .unwrap();
    sketch
        .entities
        .iter()
        .map(|e| {
            let mm = |k: &str| {
                e.points
                    .get(k)
                    .map(|p| {
                        [
                            (p[0] as f64 * 1e3 * 1e3).round() / 1e3,
                            (p[1] as f64 * 1e3 * 1e3).round() / 1e3,
                        ]
                    })
                    .unwrap_or([f64::NAN; 2])
            };
            (e.kind.clone(), mm("start"), mm("end"))
        })
        .collect()
}

/// The longest line of a sketch, in millimetres.
fn longest(lines: &[(String, [f64; 2], [f64; 2])]) -> f64 {
    lines
        .iter()
        .filter(|l| l.0 == "line")
        .map(|(_, a, b)| (a[0] - b[0]).hypot(a[1] - b[1]))
        .fold(0.0, f64::max)
}

#[test]
fn project_brings_a_face_s_edges_and_they_follow_the_model() {
    let text = applied(
        PLATE,
        "sketch.use-edges",
        json!({ "sketch": "top", "type": "project", "items": ["feature.body.face.end"] }),
    );
    let from: Vec<&str> = text
        .lines()
        .filter(|l| l.contains("from = project("))
        .collect();
    assert_eq!(from.len(), 4, "{text}");
    let lines = solved(&text, "top");
    assert_eq!(
        lines.iter().filter(|l| l.0 == "line").count(),
        4,
        "{lines:?}"
    );
    assert!((longest(&lines) - 40.0).abs() < 1e-3, "{lines:?}");
    // The plate grows to 60 mm: the converted lines follow it on the next rebuild.
    let wider = text.replace("-20 mm", "-30 mm").replace("[20 mm", "[30 mm");
    assert!(
        (longest(&solved(&wider, "top")) - 60.0).abs() < 1e-3,
        "{wider}"
    );
}

#[test]
fn offset_on_surface_insets_a_face_s_edges_into_a_closed_profile() {
    let text = applied(
        PLATE,
        "sketch.use-edges",
        json!({ "sketch": "top", "type": "offset_on_surface", "items": ["feature.body.face.end"], "distance": 4 }),
    );
    assert_eq!(
        text.matches("from = offset_on_surface(").count(),
        4,
        "{text}"
    );
    assert!(text.contains("distance = 4 mm"), "{text}");
    let lines = solved(&text, "top");
    let mut lengths: Vec<f64> = lines
        .iter()
        .map(|(_, a, b)| ((a[0] - b[0]).hypot(a[1] - b[1]) * 1e3).round() / 1e3)
        .collect();
    lengths.sort_by(f64::total_cmp);
    assert_eq!(lengths, vec![12.0, 12.0, 32.0, 32.0], "{lines:?}");
    // The inset is a closed profile: a pocket cuts through it.
    let pocket = text.replace(
        "    }\n}\n",
        "    }\n    feature pocket = extrude(profile = sketch.top, distance = 2 mm, mode = cut, reverse)\n}\n",
    );
    let compiled = compile_part("/p/plate.part", &pocket).unwrap_or_else(|d| panic!("{d:?}"));
    let errors: Vec<&str> = compiled
        .diagnostics
        .iter()
        .filter(|d| d.severity == "error")
        .map(|d| d.message.as_str())
        .collect();
    assert!(errors.is_empty(), "{errors:?}\n{pocket}");
}

#[test]
fn intersection_cuts_the_body_with_the_sketch_plane() {
    let text = PLATE.replace(
        "    sketch top on feature.body.face.end {\n    }\n",
        "    sketch cut on origin.zx {\n    }\n",
    );
    let out = applied(
        &text,
        "sketch.use-edges",
        json!({ "sketch": "cut", "type": "intersection", "items": ["feature.body.body"] }),
    );
    assert_eq!(out.matches("from = intersection(").count(), 4, "{out}");
    let lines = solved(&out, "cut");
    assert_eq!(
        lines.iter().filter(|l| l.0 == "line").count(),
        4,
        "{lines:?}"
    );
    assert!((longest(&lines) - 40.0).abs() < 1e-3, "{lines:?}");
    // An edge is refused, naming what Intersection reads.
    let edge = run(
        &text,
        "sketch.use-edges",
        json!({ "sketch": "cut", "type": "intersection", "items": ["edges_between(feature.body.face.end, feature.body.face.side(s1.l1))"] }),
    );
    assert_eq!(edge["ok"], false, "{edge}");
    assert!(
        edge["message"].as_str().unwrap().contains("reads faces"),
        "{edge}"
    );
}

#[test]
fn silhouette_outlines_a_pin_seen_from_the_side_and_refuses_a_flat_face() {
    let pin = "runmat 0.7\nstd 1\n\npart plate {\n    sketch s1 on origin.xy {\n        c1 = circle(center = [0 mm, 0 mm], radius = 5 mm)\n    }\n    feature body = extrude(profile = sketch.s1, distance = 30 mm)\n    sketch side on origin.zx {\n    }\n}\n";
    let out = applied(
        pin,
        "sketch.use-edges",
        json!({ "sketch": "side", "type": "silhouette", "items": ["feature.body.face.side(s1.c1)"] }),
    );
    assert_eq!(out.matches("from = silhouette(").count(), 2, "{out}");
    let lines = solved(&out, "side");
    assert_eq!(
        lines.iter().filter(|l| l.0 == "line").count(),
        2,
        "{lines:?}"
    );
    assert!((longest(&lines) - 30.0).abs() < 1e-2, "{lines:?}");
    let flat = run(
        pin,
        "sketch.use-edges",
        json!({ "sketch": "side", "type": "silhouette", "items": ["feature.body.face.end"] }),
    );
    assert_eq!(flat["ok"], false, "{flat}");
    assert!(
        flat["message"].as_str().unwrap().contains("no silhouette"),
        "{flat}"
    );
}

/// Edit plane moves the selected sketch to another plane or face made before it; its converted
/// entities read the model again on the new plane.
#[test]
fn edit_plane_moves_a_sketch_and_refuses_a_face_made_after_it() {
    let text = applied(
        PLATE,
        "sketch.use-edges",
        json!({ "sketch": "top", "type": "project", "items": ["feature.body.face.end"] }),
    );
    let moved = applied(
        &text,
        "part.edit-sketch-plane",
        json!({ "selection": ["sketch:top"], "plane": "origin:top" }),
    );
    assert!(moved.contains("sketch top on origin.xy {"), "{moved}");
    // On the base plane, 5 mm below the face, the face's edges still read as the 40 x 20 mm
    // outline.
    let lines = solved(&moved, "top");
    assert_eq!(
        lines.iter().filter(|l| l.0 == "line").count(),
        4,
        "{lines:?}"
    );
    assert!((longest(&lines) - 40.0).abs() < 1e-3, "{lines:?}");
    let out = run(
        PLATE,
        "part.edit-sketch-plane",
        json!({ "selection": ["sketch:s1"], "plane": "feature.body.face.end" }),
    );
    assert_eq!(out["ok"], false, "{out}");
    assert!(
        out["message"].as_str().unwrap().contains("comes after"),
        "{out}"
    );
    let none = run(
        PLATE,
        "part.edit-sketch-plane",
        json!({ "plane": "origin:front" }),
    );
    assert!(
        none["message"]
            .as_str()
            .unwrap()
            .contains("select the sketch"),
        "{none}"
    );
}

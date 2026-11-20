//! `sketch.ellipse` through the dispatcher: an ellipse or a partial ellipse with its semi-axes
//! dimensioned, and the forms the sketch model lacks refused by name.

use super::tests::{added, applied, run, PLATE};
use crate::runtime::regenerate_part;
use serde_json::json;

#[test]
fn an_ellipse_lands_with_both_semi_axes_dimensioned_and_draws() {
    let text = applied(
        PLATE,
        "sketch.ellipse",
        json!({ "sketch": "s1", "type": "full", "center": { "x": 20, "y": 10 }, "major": 12, "minor": 5, "angle": 15 }),
    );
    assert_eq!(
        added(PLATE, &text),
        vec![
            "        e1 = ellipse(center = [20 mm, 10 mm], major = 12 mm, minor = 5 mm, angle = 15 deg)",
            "        constrain major_radius(e1) = 12 mm",
            "        constrain minor_radius(e1) = 5 mm",
        ]
    );
    let part = regenerate_part("/p/plate.part", &text).unwrap_or_else(|d| panic!("{d:?}"));
    let sketch = part.display.sketches.iter().find(|s| s.id == "s1").unwrap();
    let e = sketch.entities.iter().find(|e| e.id == "e1").unwrap();
    assert_eq!(e.kind, "ellipse");
    let [a, b, turn] = e.ellipse.expect("an ellipse's axes");
    assert!(
        (a - 0.012).abs() < 1e-6
            && (b - 0.005).abs() < 1e-6
            && (turn.to_degrees() - 15.0).abs() < 1e-3
    );
    // The curve is sampled all the way round, from the end of the major axis back to it.
    assert!(e.samples.len() > 32);
    assert_eq!(e.samples.first(), e.samples.last());
    // The dimension tool on an ellipse dimensions its major axis, or its minor on request.
    let dims = applied(
        &text,
        "sketch.dimension",
        json!({ "sketch": "s1", "entities": ["sketch.s1.e1"], "value": 4, "axis": "minor" }),
    );
    assert_eq!(
        added(&text, &dims),
        vec!["        constrain minor_radius(e1) = 4 mm"]
    );
}

#[test]
fn a_partial_ellipse_puts_its_ends_on_the_ellipse() {
    let text = applied(
        PLATE,
        "sketch.ellipse",
        json!({ "sketch": "s1", "type": "partial", "center": { "x": 20, "y": 10 }, "major": 10, "minor": 4,
                "start": { "x": 40, "y": 10 }, "end": { "x": 20, "y": 30 } }),
    );
    let new = added(PLATE, &text);
    assert_eq!(
        new[0],
        "        e1 = ellipse(center = [20 mm, 10 mm], major = 10 mm, minor = 4 mm, start = [30 mm, 10 mm], end = [20 mm, 14 mm])",
        "{new:?}"
    );
}

#[test]
fn parabolas_and_conics_say_they_are_not_in_the_sketch_model() {
    for kind in ["parabola", "conic"] {
        let out = run(
            PLATE,
            "sketch.ellipse",
            json!({ "sketch": "s1", "type": kind, "center": { "x": 0, "y": 0 }, "major": 3, "minor": 2 }),
        );
        assert_eq!(out["ok"], false, "{out}");
        assert!(
            out["message"]
                .as_str()
                .unwrap()
                .contains("aren't in the sketch model yet"),
            "{out}"
        );
    }
    let out = run(
        PLATE,
        "sketch.ellipse",
        json!({ "sketch": "s1", "center": { "x": 0, "y": 0 }, "major": 3 }),
    );
    assert!(
        out["message"].as_str().unwrap().contains("minor radius"),
        "{out}"
    );
}

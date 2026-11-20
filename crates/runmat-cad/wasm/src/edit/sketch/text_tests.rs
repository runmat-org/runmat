//! `sketch.text` through the dispatcher: lettering lands in the single-line font and draws as
//! closed outlines; fonts the sketch doesn't have and text along a curve say so.

use super::tests::{added, applied, run, PLATE};
use crate::runtime::regenerate_part;
use serde_json::json;

#[test]
fn text_lands_in_the_single_line_font_and_draws_its_outline() {
    let text = applied(
        PLATE,
        "sketch.text",
        json!({ "sketch": "s1", "text": "AB", "at": { "x": 5, "y": 5 }, "height": 6, "angle": 0, "font": "single_line", "stroke": 1 }),
    );
    assert_eq!(
        added(PLATE, &text),
        vec!["        t1 = text(text = \"AB\", at = [5 mm, 5 mm], height = 6 mm, font = single_line, stroke = 1 mm)"]
    );
    let part = regenerate_part("/p/plate.part", &text).unwrap_or_else(|d| panic!("{d:?}"));
    let sketch = part.display.sketches.iter().find(|s| s.id == "s1").unwrap();
    let t = sketch.entities.iter().find(|e| e.id == "t1").unwrap();
    assert_eq!((t.kind.as_str(), t.text.as_deref()), ("text", Some("AB")));
    // A and B are an outline each, with one counter in A and two in B.
    assert_eq!(t.outlines.len(), 5, "{:?}", t.outlines.len());
    for ring in &t.outlines {
        assert_eq!(ring.first(), ring.last());
    }
}

#[test]
fn other_fonts_text_along_a_curve_and_empty_text_say_why() {
    let other = run(
        PLATE,
        "sketch.text",
        json!({ "sketch": "s1", "text": "A", "at": { "x": 5, "y": 5 }, "height": 6, "font": "Inter" }),
    );
    assert!(
        other["message"]
            .as_str()
            .unwrap()
            .contains("single-line font"),
        "{other}"
    );
    let along = run(
        PLATE,
        "sketch.text",
        json!({ "sketch": "s1", "text": "A", "at": { "x": 5, "y": 5 }, "height": 6, "along": ["sketch.s1.l1"] }),
    );
    assert!(
        along["message"].as_str().unwrap().contains("along a curve"),
        "{along}"
    );
    let empty = run(
        PLATE,
        "sketch.text",
        json!({ "sketch": "s1", "text": " ", "at": { "x": 5, "y": 5 }, "height": 6 }),
    );
    assert!(
        empty["message"].as_str().unwrap().contains("type the text"),
        "{empty}"
    );
}

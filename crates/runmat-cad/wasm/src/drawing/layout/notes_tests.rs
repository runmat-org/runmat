//! Notes as text blocks on the sheet: long notes wrap to their width or to the room they have,
//! the general notes read as a numbered list with hanging indents above the title block, and no
//! note overprints the title block, the border, a table or another note. The fixture is the L
//! bracket an agent drew with its notes as long single lines along the bottom of the sheet.

use super::solver::{text_box, BoxR};
use super::tests::{compile_ring, prims, ring_drawing, sheet};
use super::*;
use crate::runtime::Files;

const PART: &str = include_str!("../../../../tests/fixtures/notes/l_bracket.part");
const AGENT: &str = include_str!("../../../../tests/fixtures/notes/l_bracket.drawing");
const LISTED: &str = include_str!("../../../../tests/fixtures/notes/l_bracket_notes.drawing");

/// The drawing as `/p/<name>.drawing` beside the L bracket, compiled with no error.
fn compile(name: &str, drawing: &str) -> Value {
    let path = format!("/p/{name}.drawing");
    let files: Files = [
        ("/p/l_bracket.part".to_owned(), PART.to_owned()),
        (path.clone(), drawing.to_owned()),
    ]
    .into();
    let out: Value = serde_json::from_str(&crate::drawing::compile_json(
        &path,
        drawing,
        &files,
    ))
    .unwrap();
    let errors: Vec<&Value> = out["diagnostics"]
        .as_array()
        .into_iter()
        .chain(out["projection"]["data"]["diagnostics"].as_array())
        .flatten()
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:#?}");
    out
}

/// Each text line an item draws: (text, x, baseline y, box).
fn lines_of(p: &[Prim], id: &str) -> Vec<(String, f64, f64, BoxR)> {
    p.iter()
        .filter(|q| q.id == id)
        .filter_map(|q| match &q.shape {
            Shape::Text { text, x, y, .. } => Some((text.clone(), *x, *y, text_box(q)?)),
            _ => None,
        })
        .collect()
}

fn union(boxes: impl IntoIterator<Item = BoxR>) -> Option<BoxR> {
    boxes.into_iter().reduce(|a, b| a.union(&b))
}

/// The title block's box: everything drawn for `title`.
fn title_box(p: &[Prim]) -> BoxR {
    union(p.iter().filter(|q| q.id == "title").filter_map(|q| match &q.shape {
        Shape::Rect { x, y, w, h } => Some(BoxR::new(*x, *y, x + w, y + h)),
        _ => text_box(q),
    }))
    .expect("a title block")
}

fn words(t: &str) -> Vec<String> {
    t.split_whitespace().map(str::to_owned).collect()
}

#[test]
fn an_agent_s_long_notes_wrap_and_keep_clear_of_the_title_block_the_border_and_each_other() {
    let out = compile("l_bracket", AGENT);
    let layout = sheet(&out, "1");
    let p = prims(&layout);
    let title = title_box(&p);
    let frame = super::border::frame_box([420.0, 297.0], true);
    let texts = &out["projection"]["data"]["text"];
    let mut boxes: Vec<(String, BoxR)> = Vec::new();
    for n in ["n1", "n2", "n3"] {
        let id = format!("annotation:{n}");
        let lines = lines_of(&p, &id);
        assert!(lines.len() > 1, "{n} wraps onto more lines: {lines:?}");
        // Every word is drawn, in order.
        let drawn: Vec<String> = lines.iter().flat_map(|l| words(&l.0)).collect();
        assert_eq!(drawn, words(texts[n].as_str().unwrap()), "{n}");
        // Lines start at one x and step down by the standard's pitch (ISO: 1.7 x 3.5 mm).
        for w in lines.windows(2) {
            assert!((w[0].1 - w[1].1).abs() < 1e-9, "{n}: {w:?}");
            assert!((w[0].2 - w[1].2 - 1.7 * 3.5).abs() < 1e-6, "{n}: {w:?}");
        }
        let b = union(lines.iter().map(|l| l.3)).unwrap();
        assert!(b.inside(&frame), "{n} {b:?} runs past the border {frame:?}");
        assert!(!b.hits(&title), "{n} {b:?} overprints the title block {title:?}");
        boxes.push((n.to_owned(), b));
    }
    for (i, (a, ba)) in boxes.iter().enumerate() {
        for (b, bb) in &boxes[i + 1..] {
            assert!(!ba.hits(bb), "{a} {ba:?} overprints {b} {bb:?}");
        }
    }
    // No note text sits on any other text on the sheet.
    for (n, b) in &boxes {
        let id = format!("annotation:{n}");
        for q in p.iter().filter(|q| q.id != id && matches!(q.shape, Shape::Text { .. })) {
            let Some(qb) = text_box(q) else { continue };
            assert!(!b.hits(&qb), "{n} {b:?} overprints {} {:?}", q.id, q.shape);
        }
    }
    // n2 and n3 were both put at [20, 30], on the border's edge under the views: at least one
    // moved, and the layout says where and why.
    let moved: Vec<&Value> = layout["findings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["code"] == "note_moved")
        .collect();
    assert!(!moved.is_empty(), "{:#}", layout["findings"]);
    for f in &moved {
        let m = f["message"].as_str().unwrap();
        assert!(m.contains("has no room there") && m.contains("`at`") && m.contains("table notes = notes("), "{m}");
    }
}

#[test]
fn the_general_notes_list_numbers_each_note_with_a_hanging_indent_above_the_title_block() {
    let out = compile("l_bracket_notes", LISTED);
    let layout = sheet(&out, "1");
    assert_eq!(layout["findings"], serde_json::json!([]), "{:#}", layout["findings"]);
    let p = prims(&layout);
    let lines = lines_of(&p, "table:notes");
    assert_eq!(lines[0].0, "NOTES:");
    let x0 = lines[0].1;
    let numbers: Vec<&(String, f64, f64, BoxR)> = lines.iter().filter(|l| l.1 == x0).skip(1).collect();
    assert_eq!(
        numbers.iter().map(|l| l.0.as_str()).collect::<Vec<_>>(),
        ["1.", "2.", "3."]
    );
    // Each note's lines, wrapped, all at the indent: the hanging indent under its number.
    let body: Vec<&(String, f64, f64, BoxR)> = lines.iter().filter(|l| l.1 > x0).collect();
    let indent = body[0].1;
    assert!(body.iter().all(|l| (l.1 - indent).abs() < 1e-9), "{body:?}");
    assert!((indent - x0 - Std::text_width("3. ", 3.5)).abs() < 1e-9);
    assert!(body.len() > 3, "the notes wrap: {body:?}");
    // The number sits on its note's first line.
    for n in &numbers {
        assert!(body.iter().any(|l| (l.2 - n.2).abs() < 1e-9), "{n:?}");
    }
    // `{material}` is filled from the part; every word of every note is drawn.
    let drawn: Vec<String> = body.iter().flat_map(|l| words(&l.0)).collect();
    assert!(drawn.join(" ").contains("MATERIAL: Aluminum 6061."), "{drawn:?}");
    for row in out["projection"]["data"]["notes"]["notes"].as_array().unwrap() {
        for w in words(row.as_str().unwrap()) {
            assert!(drawn.contains(&w), "{w} isn't drawn");
        }
    }
    // As wide as the title block, sitting on it, clear of it.
    let title = title_box(&p);
    let b = union(lines.iter().map(|l| l.3)).unwrap();
    assert!(b.x1 - b.x0 <= runmat_cad::drawing::title_layout::BLOCK_WIDTH + 1e-6, "{b:?}");
    assert!(b.y0 - title.y1 >= 2.0 && b.y0 - title.y1 < 15.0, "{b:?} above {title:?}");
    assert!(!b.hits(&title));
    // No ruled grid: the list is text only.
    assert!(p.iter().filter(|q| q.id == "table:notes").all(|q| matches!(q.shape, Shape::Text { .. })));
}

#[test]
fn a_note_wraps_to_its_width_and_breaks_where_the_text_says() {
    let drawing = |args: &str| {
        let d = format!("runmat 0.7\nstd 1\n\ndrawing ring {{\n    format a3 = {{ size = {{ size = iso, name = A3 }}, title_block = [] }}\n    sheet \"1\" = {{ format = a3, scale = [1, 2], projection = third_angle }}\n    view top = standard(orientation = top, sheet = \"1\", model = {{ part = ring }}, at = [200, 150], display = visible_lines)\n    annotation n = note({args}, view = top)\n}}\n");
        lines_of(&prims(&sheet(&compile_ring(&d), "1")), "annotation:n")
    };
    let long = "DEBURR ALL HOLES BOTH SIDES AND BREAK EVERY SHARP EDGE 0.2 TO 0.5 MM BEFORE ANODIZING";
    let lines = drawing(&format!("text = \"{long}\", width = 60 mm, at = [30, 270]"));
    assert!(lines.len() >= 3, "{lines:?}");
    assert!(lines.iter().all(|l| Std::text_width(&l.0, 3.5) <= 60.0 + 1e-9), "{lines:?}");
    assert_eq!(lines.iter().flat_map(|l| words(&l.0)).collect::<Vec<_>>(), words(long));
    assert_eq!(lines[0].1, 30.0, "a note with room stays where it was put");
    // A width in inches.
    let inches = drawing(&format!("text = \"{long}\", width = 2 in, at = [30, 270]"));
    assert!(inches.iter().all(|l| Std::text_width(&l.0, 3.5) <= 50.8 + 1e-9), "{inches:?}");
    // `\n` breaks a line; every line is the standard's height.
    let broken = drawing("text = \"6X ⌀5.5 THRU\\nM5 CLEARANCE\", at = [30, 270]");
    assert_eq!(
        broken.iter().map(|l| l.0.as_str()).collect::<Vec<_>>(),
        ["6X ⌀5.5 THRU", "M5 CLEARANCE"]
    );
    // A leadered note wraps under its shoulder.
    let leadered = drawing(&format!("text = \"{long}\", width = 50 mm, leader = feature.body.face.end"));
    assert!(leadered.len() >= 3, "{leadered:?}");
}

#[test]
fn a_note_put_past_the_sheet_edge_moves_onto_the_sheet_and_says_so() {
    let moved = |at: &str| {
        let layout = sheet(&ring_drawing(at), "1");
        let found = layout["findings"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["code"] == "note_moved")
            .count();
        let b = union(lines_of(&prims(&layout), "annotation:notes").iter().map(|l| l.3)).unwrap();
        (found, b.inside(&super::border::frame_box([420.0, 297.0], true)))
    };
    assert_eq!(moved("[320, 120]"), (0, true), "a note on free sheet");
    assert_eq!(moved("[-60, 40]"), (1, true), "a note at a negative x");
    assert_eq!(moved("[300, 500]"), (1, true), "a note above the A3 sheet");
}

#[test]
fn a_note_with_no_room_on_the_sheet_is_reported_with_what_to_change() {
    let huge = "WORD ".repeat(1500);
    let d = format!("runmat 0.7\nstd 1\n\ndrawing ring {{\n    format a4 = {{ size = {{ size = iso, name = A4 }}, title_block = [] }}\n    sheet \"1\" = {{ format = a4, scale = [1, 4], projection = third_angle }}\n    view top = standard(orientation = top, sheet = \"1\", model = {{ part = ring }}, at = [100, 150], display = visible_lines)\n    annotation big = note(text = \"{huge}\", view = top)\n}}\n");
    let layout = sheet(&compile_ring(&d), "1");
    let f: Vec<&Value> = layout["findings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["code"] == "note_no_room")
        .collect();
    assert_eq!(f.len(), 1, "{:#}", layout["findings"]);
    let m = f[0]["message"].as_str().unwrap();
    assert!(m.starts_with("Note big doesn't fit on the sheet: it needs"), "{m}");
    assert!(m.contains("`width`") && m.contains("larger sheet"), "{m}");
}

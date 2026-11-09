//! The hand-built drawing (every view kind and annotation kind on one A3 sheet) laid out: no
//! two labels overlap, no label sits on a view's lines, every view and table shows, the title
//! block is there, a repeated dimension is drawn once, and the layout is the same every time.

use super::solver::{text_box, BoxR};
use super::*;
use crate::runtime::Files;

const DRAWING: &str = include_str!("../../../../tests/fixtures/hand-built/hand.drawing");
const PART: &str = include_str!("../../../../tests/fixtures/hand-built/bracket.part");

fn compiled() -> Value {
    let files: Files = [("/p/bracket.part".to_owned(), PART.to_owned()), ("/p/bracket.drawing".to_owned(), DRAWING.to_owned())].into();
    serde_json::from_str(&crate::drawing::compile_json("/p/bracket.drawing", DRAWING, &files)).unwrap()
}

fn sheet(out: &Value, name: &str) -> Value {
    out["projection"]["data"]["layouts"][name].clone()
}

fn prims(layout: &Value) -> Vec<Prim> {
    layout["primitives"].as_array().unwrap().iter().map(prim_of).collect()
}

fn prim_of(v: &Value) -> Prim {
    let s = &v["shape"];
    let n = |k: &str| s[k].as_f64().unwrap_or(0.0);
    let shape = match s["type"].as_str().unwrap() {
        "text" => Shape::Text { x: n("x"), y: n("y"), text: s["text"].as_str().unwrap().to_owned(), size: n("size"), anchor: s["anchor"].as_str().map(|a| match a { "middle" => "middle", "end" => "end", _ => "start" }), rotation: s["rotation"].as_f64() },
        "rect" => Shape::Rect { x: n("x"), y: n("y"), w: n("w"), h: n("h") },
        "circle" => Shape::Circle { x: n("x"), y: n("y"), r: n("r") },
        "polygon" => Shape::Polygon { rings: s["rings"].as_array().unwrap().iter().map(floats).collect() },
        _ => Shape::Polyline { points: floats(&s["points"]), closed: s["closed"].as_bool().unwrap_or(false) },
    };
    let leak = |s: &str| -> &'static str { Box::leak(s.to_owned().into_boxed_str()) };
    Prim { id: v["id"].as_str().unwrap().to_owned(), kind: leak(v["kind"].as_str().unwrap()), layer: leak(v["layer"].as_str().unwrap()), style: leak(v["style"].as_str().unwrap()), shape, width: v["width"].as_f64() }
}

fn texts(p: &[Prim]) -> Vec<String> {
    p.iter().filter_map(|p| if let Shape::Text { text, .. } = &p.shape { Some(text.clone()) } else { None }).collect()
}

/// Each view's lines' box on the sheet.
fn view_boxes(p: &[Prim]) -> Vec<(String, BoxR)> {
    let mut out: Vec<(String, BoxR)> = Vec::new();
    for q in p.iter().filter(|q| q.kind == "view" && q.style == "visible") {
        let Shape::Polyline { points, .. } = &q.shape else { continue };
        for c in points.chunks_exact(2) {
            let b = BoxR::new(c[0], c[1], c[0], c[1]);
            match out.iter_mut().find(|(n, _)| n == &q.id) {
                Some((_, x)) => *x = x.union(&b),
                None => out.push((q.id.clone(), b)),
            }
        }
    }
    out
}

/// Labels: annotation figures and callouts, and view labels.
fn labels(p: &[Prim]) -> Vec<(String, String, BoxR)> {
    p.iter()
        .filter(|q| q.id.starts_with("annotation:") || q.kind == "viewlabel")
        .filter_map(|q| Some((q.id.clone(), if let Shape::Text { text, .. } = &q.shape { text.clone() } else { String::new() }, text_box(q)?)))
        .collect()
}

#[test]
fn the_hand_built_sheet_lays_out_with_no_label_over_another_or_over_a_view() {
    let out = compiled();
    let layout = sheet(&out, "1");
    // With `RUNMAT_LAYOUT_OUT` set to a file, the sheet's layout is written there (for review).
    if let Ok(path) = std::env::var("RUNMAT_LAYOUT_OUT") {
        std::fs::write(path, layout.to_string()).unwrap();
    }
    let p = prims(&layout);
    let ls = labels(&p);
    assert!(ls.len() > 20, "{} labels", ls.len());
    for (i, (a, ta, ba)) in ls.iter().enumerate() {
        for (b, tb, bb) in &ls[i + 1..] {
            assert!(a == b || !ba.hits(bb), "{a} {ta:?} {ba:?} overlaps {b} {tb:?} {bb:?}");
        }
    }
    let views = view_boxes(&p);
    assert!(views.len() >= 9, "{views:?}");
    for (id, t, b) in &ls {
        for (view, vb) in &views {
            assert!(!b.hits(vb), "{id} {t:?} {b:?} sits on {view} {vb:?}");
        }
    }
    // No two views overlap.
    for (i, (a, ba)) in views.iter().enumerate() {
        for (b, bb) in &views[i + 1..] {
            assert!(!ba.hits(bb), "{a} {ba:?} overlaps {b} {bb:?}");
        }
    }
    // Everything stays in the frame (A3, ISO-style margins).
    let frame = BoxR::new(10.0, 10.0, 410.0, 287.0);
    for (id, t, b) in &ls {
        assert!(b.inside(&frame), "{id} {t:?} {b:?} runs past the frame");
    }
}

#[test]
fn every_view_is_labelled_and_every_table_and_the_title_block_show() {
    let out = compiled();
    let p = prims(&sheet(&out, "1"));
    let all = texts(&p);
    for want in ["SECTION A-A", "DETAIL B", "SCALE 4:1", "VIEW C", "BROKEN-OUT SECTION", "ISOMETRIC VIEW", "SCALE 1:2", "BOTTOM VIEW", "HOLE TABLE", "REVISIONS", "Initial release"] {
        assert!(all.iter().any(|t| t == want), "no {want:?} in {all:?}");
    }
    // The hole table lists the hole from its origin (the left face): 60 mm along, as the part has it.
    let hole_rows: Vec<String> = p.iter().filter(|q| q.id == "table:holes").filter_map(|q| if let Shape::Text { text, .. } = &q.shape { Some(text.clone()) } else { None }).collect();
    assert!(hole_rows.iter().any(|t| t.contains("THRU")), "{hole_rows:?}");
    // The title block is always there: with no fields named, the standard ones.
    let title: Vec<String> = p.iter().filter(|q| q.id == "title").filter_map(|q| if let Shape::Text { text, .. } = &q.shape { Some(text.clone()) } else { None }).collect();
    for want in ["TITLE", "PART NUMBER", "MATERIAL", "SCALE", "SHEET"] {
        assert!(title.iter().any(|t| t == want), "no {want} in the title block {title:?}");
    }
    // Zones along the border.
    assert!(p.iter().any(|q| q.kind == "label" && q.layer == "Frame" && matches!(&q.shape, Shape::Text { text, .. } if text == "A")));
}

#[test]
fn a_repeated_dimension_is_drawn_once_and_reported() {
    let out = compiled();
    let layout = sheet(&out, "1");
    let findings: Vec<&str> = layout["findings"].as_array().unwrap().iter().filter(|f| f["code"] == "duplicate_dimension").map(|f| f["item"].as_str().unwrap()).collect();
    assert!(findings.contains(&"chain") && findings.contains(&"ordinate2"), "{layout:#}");
    // The chain draws its other pair; the 60 it repeats shows once (the ordinate's).
    let p = prims(&layout);
    let sixties = p.iter().filter(|q| q.id.starts_with("annotation:") && matches!(&q.shape, Shape::Text { text, .. } if text == "60")).count();
    assert_eq!(sixties, 1);
    assert!(p.iter().any(|q| q.id == "annotation:chain" && matches!(&q.shape, Shape::Text { text, .. } if text == "20")));
}

#[test]
fn the_layout_is_the_same_every_time_and_draws_every_annotation_kind() {
    let a = sheet(&compiled(), "1");
    let b = sheet(&compiled(), "1");
    assert_eq!(a, b);
    let p = prims(&a);
    let doc = &compiled()["projection"]["data"]["document"];
    for ann in doc["annotations"].as_array().unwrap() {
        let id = format!("annotation:{}", ann["id"].as_str().unwrap());
        if ann["id"] == "ordinate2" {
            continue; // repeats `dim`
        }
        assert!(p.iter().any(|q| q.id == id), "{id} isn't drawn");
    }
    // Line weights from the sheet's standard (ASME Y14.2: 0.6 mm visible).
    assert!(p.iter().filter(|q| q.style == "visible").all(|q| q.width == Some(0.6)));
}

#[test]
fn a_moved_callout_and_a_placed_view_stay_where_they_were_put() {
    let moved = DRAWING.replace("annotation hole = hole_callout(hole = feature.h.face.wall(a), view = top)", "annotation hole = hole_callout(hole = feature.h.face.wall(a), view = top, at = [60, 250])");
    assert_ne!(moved, DRAWING);
    let files: Files = [("/p/bracket.part".to_owned(), PART.to_owned())].into();
    let out: Value = serde_json::from_str(&crate::drawing::compile_json("/p/bracket.drawing", &moved, &files)).unwrap();
    let p = prims(&sheet(&out, "1"));
    let elbow = p.iter().find_map(|q| match &q.shape {
        Shape::Polyline { points, .. } if q.id == "annotation:hole" && q.style == "dimink" => Some([points[2], points[3]]),
        _ => None,
    });
    let hole: Vec<&Prim> = p.iter().filter(|q| q.id == "annotation:hole").collect();
    assert_eq!(elbow, Some([60.0, 250.0]), "{hole:?} {}", out["diagnostics"]);
    // The front view sits where the drawing put it.
    let front = view_boxes(&p).into_iter().find(|(n, _)| n == "view:front").unwrap().1;
    assert!((front.center()[0] - 100.0).abs() < 1e-6 && (front.center()[1] - 95.0).abs() < 1e-6, "{front:?}");
}

#[test]
fn rows_pack_shorter_dimensions_nearer_and_chain_ones_that_touch() {
    // Two touching spans (a chain) share the first row; the span over both goes outside them.
    assert_eq!(super::dims::pack_rows(&[(0.0, 20.0), (20.0, 50.0), (0.0, 50.0)]), vec![0, 0, 1]);
    // Overlapping spans stack, shortest nearest.
    assert_eq!(super::dims::pack_rows(&[(0.0, 60.0), (0.0, 10.0), (0.0, 30.0)]), vec![2, 0, 1]);
}

#[test]
fn a_tight_dimension_puts_its_arrows_outside_and_its_figure_past_the_end() {
    let st = Std::of(&Value::Null);
    let (wide, at) = st.linear_dimension("d", [0.0, 0.0], [40.0, 0.0], 0, -10.0, "40");
    assert!((at[0] - 20.0).abs() < 1e-9, "centred: {at:?}");
    let (tight, at) = st.linear_dimension("d", [0.0, 0.0], [4.0, 0.0], 0, -10.0, "4.5");
    assert!(at[0] > 4.0 + st.arrow, "past the far end: {at:?}");
    // The dimension line runs past both extension lines.
    let line_of = |p: &[Prim]| p.iter().find_map(|q| match &q.shape { Shape::Polyline { points, .. } if points[1] == -10.0 && points[3] == -10.0 => Some(points.clone()), _ => None }).unwrap();
    assert_eq!(line_of(&wide)[..1], [0.0]);
    assert!(line_of(&tight)[0] < 0.0 && line_of(&tight)[2] > 4.0);
}

const SHOWCASE_DRAWING: &str = include_str!("../../../../tests/fixtures/showcase/angle_bracket.drawing");
const SHOWCASE_PART: &str = include_str!("../../../../tests/fixtures/showcase/angle_bracket.part");

#[test]
fn the_showcase_bracket_lays_out_on_a3_with_nothing_over_anything_else() {
    // The angle bracket's production drawing: three views, a section, a detail and an isometric,
    // dimensioned and toleranced, with GD&T, finishes, notes, the title block and revisions.
    if let Err(d) = crate::runtime::compile_part("/p/angle_bracket.part", SHOWCASE_PART) {
        panic!("the part doesn't compile: {d:#?}");
    }
    let files: Files = [("/p/angle_bracket.part".to_owned(), SHOWCASE_PART.to_owned())].into();
    let out: Value = serde_json::from_str(&crate::drawing::compile_json("/p/angle_bracket.drawing", SHOWCASE_DRAWING, &files)).unwrap();
    let layout = sheet(&out, "1");
    // With `RUNMAT_SHOWCASE_OUT` set to a file, the whole compile is written there (for review).
    if let Ok(path) = std::env::var("RUNMAT_SHOWCASE_OUT") {
        std::fs::write(path, out.to_string()).unwrap();
    }
    assert_eq!(out["diagnostics"].as_array().map(|d| d.iter().filter(|d| d["severity"] == "error").count()), Some(0), "{}", out["diagnostics"]);
    let errors: Vec<&Value> = out["projection"]["data"]["diagnostics"].as_array().into_iter().flatten().collect();
    assert!(errors.is_empty(), "{errors:#?}");
    assert_eq!(layout["findings"].as_array().map(Vec::len), Some(0), "{}", layout["findings"]);
    let p = prims(&layout);
    let doc = &out["projection"]["data"]["document"];
    for ann in doc["annotations"].as_array().unwrap() {
        let id = format!("annotation:{}", ann["id"].as_str().unwrap());
        assert!(p.iter().any(|q| q.id == id), "{id} isn't drawn");
    }
    let ls = labels(&p);
    for (i, (a, ta, ba)) in ls.iter().enumerate() {
        for (b, tb, bb) in &ls[i + 1..] {
            assert!(a == b || !ba.hits(bb), "{a} {ta:?} {ba:?} overlaps {b} {tb:?} {bb:?}");
        }
    }
    let views = view_boxes(&p);
    assert_eq!(views.len(), 6, "{views:?}");
    for (id, t, b) in &ls {
        for (view, vb) in &views {
            assert!(!b.hits(vb), "{id} {t:?} {b:?} sits on {view} {vb:?}");
        }
    }
    for (i, (a, ba)) in views.iter().enumerate() {
        for (b, bb) in &views[i + 1..] {
            assert!(!ba.hits(bb), "{a} {ba:?} overlaps {b} {bb:?}");
        }
    }
    let frame = BoxR::new(10.0, 10.0, 410.0, 287.0);
    for (id, t, b) in &ls {
        assert!(b.inside(&frame), "{id} {t:?} {b:?} runs past the frame");
    }
    let all = texts(&p);
    for want in ["SECTION A-A", "DETAIL B", "SCALE 3:1", "REVISIONS", "AB-6061-100", "Angle bracket"] {
        assert!(all.iter().any(|t| t.contains(want)), "no {want:?} in {all:?}");
    }
}

#[test]
fn no_other_dimension_line_runs_through_an_ordinate_figure() {
    let p = prims(&sheet(&compiled(), "1"));
    let ords: Vec<String> = compiled()["projection"]["data"]["document"]["annotations"].as_array().unwrap().iter().filter(|a| a["kind"]["style"] == "ordinate").map(|a| format!("annotation:{}", a["id"].as_str().unwrap())).collect();
    assert!(!ords.is_empty());
    let figures: Vec<(&str, BoxR)> = p.iter().filter(|q| ords.contains(&q.id) && q.style == "dimtext").filter_map(|q| Some((q.id.as_str(), text_box(q)?))).collect();
    assert!(!figures.is_empty());
    for q in p.iter().filter(|q| q.style == "dimink" && q.kind == "annotation") {
        let Shape::Polyline { points, .. } = &q.shape else { continue };
        for (owner, f) in &figures {
            if *owner == q.id {
                continue;
            }
            for c in points.chunks_exact(2).collect::<Vec<_>>().windows(2) {
                let (a, b) = (c[0], c[1]);
                let mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
                let inside = |x: &[f64]| f.hits(&BoxR::new(x[0], x[1], x[0], x[1]));
                assert!(!(inside(a) || inside(b) || inside(&mid)), "{} runs through {owner}'s figure {f:?}: {a:?}-{b:?}", q.id);
            }
        }
    }
}

#[test]
fn a_sheet_too_full_for_its_views_suggests_a_larger_sheet_or_a_second_one() {
    let layout = sheet(&compiled(), "1");
    let full: Vec<&str> = layout["findings"].as_array().unwrap().iter().filter(|f| f["code"] == "sheet_full").filter_map(|f| f["message"].as_str()).collect();
    assert_eq!(full.len(), 1, "{layout:#}");
    assert!(full[0].contains("larger sheet (A2)") && full[0].contains("second sheet"), "{}", full[0]);
    assert_eq!(super::views::larger_sheet([431.8, 279.4]), Some("C"));
    assert_eq!(super::views::larger_sheet([1189.0, 841.0]), None);
}

#[test]
fn an_angular_dimension_draws_as_an_arc_between_its_faces_lines() {
    let p = prims(&sheet(&compiled(), "1"));
    let arc = p.iter().filter(|q| q.id == "annotation:angle" && q.style == "dimink").find_map(|q| match &q.shape {
        Shape::Polyline { points, .. } if points.len() >= 10 => Some(points.clone()),
        _ => None,
    });
    let arc = arc.expect("an arc");
    // Every point of the arc sits the same distance from one centre: fit it from three points.
    let pt = |i: usize| [arc[2 * i], arc[2 * i + 1]];
    let n = arc.len() / 2;
    let (a, b, c) = (pt(0), pt(n / 2), pt(n - 1));
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    let sq = |p: [f64; 2]| p[0] * p[0] + p[1] * p[1];
    let ctr = [(sq(a) * (b[1] - c[1]) + sq(b) * (c[1] - a[1]) + sq(c) * (a[1] - b[1])) / d, (sq(a) * (c[0] - b[0]) + sq(b) * (a[0] - c[0]) + sq(c) * (b[0] - a[0])) / d];
    let r = ((a[0] - ctr[0]).powi(2) + (a[1] - ctr[1]).powi(2)).sqrt();
    for i in 0..n {
        let q = pt(i);
        assert!((((q[0] - ctr[0]).powi(2) + (q[1] - ctr[1]).powi(2)).sqrt() - r).abs() < 1e-6);
    }
    // Two arrowheads and the figure.
    assert_eq!(p.iter().filter(|q| q.id == "annotation:angle" && q.style == "arrow").count(), 2);
    assert!(p.iter().any(|q| q.id == "annotation:angle" && matches!(&q.shape, Shape::Text { text, .. } if text.contains('°'))));
}

#[test]
fn a_custom_sheet_lays_out_at_its_own_width_and_height() {
    for (size, want) in [("width = 500 mm, height = 350 mm", [500.0, 350.0]), ("width = 17 in, height = 11 in", [431.8, 279.4])] {
        let drawing = DRAWING.replace("size = { size = iso, name = A3 }", &format!("size = {{ size = custom, {size} }}"));
        assert_ne!(drawing, DRAWING);
        let files: Files = [("/p/bracket.part".to_owned(), PART.to_owned()), ("/p/bracket.drawing".to_owned(), drawing.clone())].into();
        let out: Value = serde_json::from_str(&crate::drawing::compile_json("/p/bracket.drawing", &drawing, &files)).unwrap();
        let layout = sheet(&out, "1");
        let laid: Vec<f64> = layout["size"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        assert!((laid[0] - want[0]).abs() < 1e-6 && (laid[1] - want[1]).abs() < 1e-6, "{size}: laid out at {laid:?}");
        // The frame and the title block follow the sheet: the block's right edge sits on the frame, 10 mm in.
        let p = prims(&layout);
        let right = p.iter().filter(|q| q.id == "title").filter_map(|q| match &q.shape { Shape::Rect { x, w, .. } => Some(x + w), _ => None }).fold(f64::MIN, f64::max);
        assert!((right - (want[0] - 10.0)).abs() < 1e-6, "{size}: title block ends at {right}");
    }
}

#[test]
fn general_notes_number_and_wrap_above_the_title_block_clear_of_the_centring_marks() {
    let long = "Deburr and break all sharp edges 0.2 to 0.5 mm unless stated, on every machined, laser-cut and molded part of the assembly, before finishing";
    let notes = format!("    table notes = notes(rows = [\"Dimensions in millimeters.\", \"{long}\"], sheet = \"1\", at = [0, 0])\n}}");
    let at = DRAWING.rfind('}').unwrap();
    let drawing = format!("{}{notes}{}", &DRAWING[..at], &DRAWING[at + 1..]);
    let files: Files = [("/p/bracket.part".to_owned(), PART.to_owned()), ("/p/bracket.drawing".to_owned(), drawing.clone())].into();
    let out: Value = serde_json::from_str(&crate::drawing::compile_json("/p/bracket.drawing", &drawing, &files)).unwrap();
    let layout = sheet(&out, "1");
    let p = prims(&layout);
    let cells: Vec<String> = p.iter().filter(|q| q.id == "table:notes").filter_map(|q| if let Shape::Text { text, .. } = &q.shape { Some(text.clone()) } else { None }).collect();
    for want in ["GENERAL NOTES", "1", "2", "Dimensions in millimeters."] {
        assert!(cells.iter().any(|t| t == want), "no {want:?} in {cells:?}");
    }
    // The long note wraps onto more lines; every word is drawn.
    let words: Vec<&str> = cells.iter().flat_map(|c| c.split(' ')).collect();
    assert!(long.split(' ').all(|w| words.contains(&w)), "{cells:?}");
    assert!(!cells.iter().any(|c| c == long), "{cells:?}");
    // As wide as the title block, its box clear of the centring marks and of everything else.
    let rects: Vec<BoxR> = p
        .iter()
        .filter(|q| q.id == "table:notes")
        .filter_map(|q| match &q.shape {
            Shape::Rect { x, y, w, h } => Some(BoxR::new(*x, *y, x + w, y + h)),
            Shape::Polyline { points, .. } => {
                let (xs, ys): (Vec<f64>, Vec<f64>) = points.chunks(2).map(|c| (c[0], c[1])).unzip();
                Some(BoxR::new(xs.iter().cloned().fold(f64::MAX, f64::min), ys.iter().cloned().fold(f64::MAX, f64::min), xs.iter().cloned().fold(f64::MIN, f64::max), ys.iter().cloned().fold(f64::MIN, f64::max)))
            }
            _ => None,
        })
        .collect();
    let b = rects.iter().skip(1).fold(rects[0], |a, r| BoxR::new(a.x0.min(r.x0), a.y0.min(r.y0), a.x1.max(r.x1), a.y1.max(r.y1)));
    assert!((b.x1 - b.x0 - runmat_cad::drawing::title_layout::BLOCK_WIDTH).abs() < 1e-6, "{b:?}");
    let size = [420.0, 297.0];
    let frame = super::border::frame_box(size, false);
    for (a, c) in super::border::centring_marks(size, &frame) {
        assert!(!b.crossed_by(a, c), "notes {b:?} cross the centring mark {a:?}-{c:?}");
    }
    let before = sheet(&compiled(), "1")["findings"].as_array().unwrap().len();
    assert_eq!(layout["findings"].as_array().unwrap().len(), before, "{:#}", layout["findings"]);
}

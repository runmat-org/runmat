//! Tangent edges in drawing views, on OCCT's exact hidden-line projection, and dimension figures
//! kept off the model's lines.
//!
//! The bracket is a 5 mm L, 50 × 60 mm, with three 6.5 mm through holes in each leg and a 0.5 mm
//! round on every edge, the hole rims included. With every edge rounded no sharp edge is left:
//! each edge is a fillet's boundary, where the fillet meets its face tangent-continuously. A view
//! that leaves tangent edges out draws only the fillets' outlines, so an isometric view showed
//! each hole as a short arc and lost most of the L. Pictorial views draw tangent edges thin by
//! default, so the rims close into ellipses; orthographic views leave them out.

use serde_json::Value;

const PART: &str = include_str!("fixtures/drawings/rounded_bracket.part");
const DRAWING: &str = include_str!("fixtures/drawings/rounded_bracket.drawing");

fn files(part: &str, drawing: &str) -> runmat_cad_wasm::runtime::Files {
    [
        ("/p/bracket.part".to_string(), part.to_string()),
        ("/p/bracket.drawing".to_string(), drawing.to_string()),
    ]
    .into_iter()
    .collect()
}

fn compile(part: &str, drawing: &str) -> Value {
    runmat_geometry_wasm_host::cad::install_occt();
    serde_json::from_str(&runmat_cad_wasm::drawing::compile_json(
        "/p/bracket.drawing",
        drawing,
        &files(part, drawing),
    ))
    .expect("json")
}

/// The bracket without its rounds: every edge sharp.
fn sharp_part() -> String {
    let start = PART.find("    // Round the bracket").expect("the rounds");
    let end = PART.find("    output bracket").expect("the output");
    format!("{}{}", &PART[..start], &PART[end..])
}

fn lines(entry: &Value, key: &str) -> Vec<Vec<f64>> {
    serde_json::from_value(entry[key].clone()).unwrap_or_default()
}

fn dot(a: [f64; 3], b: &Value) -> f64 {
    (0..3).map(|i| a[i] * b[i].as_f64().unwrap()).sum()
}

/// Each hole's rim on the face the isometric view looks from (+x, −y, +z): the bottom leg's on
/// its outer face (y = 0), the upright's on its inner face (x = 5), in model mm.
const RIMS: [[f64; 3]; 6] = [
    [25.0, 0.0, 15.0],
    [25.0, 0.0, 30.0],
    [25.0, 0.0, 45.0],
    [5.0, 25.0, 15.0],
    [5.0, 25.0, 30.0],
    [5.0, 25.0, 45.0],
];

/// Whether one polyline closes around `c` (view mm) as a rim does: every point between 1 and
/// 5 mm from it, and no gap of more than 20° in the directions its points lie in.
fn closes_around(line: &[f64], c: [f64; 2]) -> bool {
    let pts: Vec<[f64; 2]> = line.chunks_exact(2).map(|p| [p[0], p[1]]).collect();
    if pts.len() < 8 {
        return false;
    }
    let near = pts.iter().all(|p| {
        let d = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt();
        (1.0..=5.0).contains(&d)
    });
    if !near {
        return false;
    }
    let mut angles: Vec<f64> = pts
        .iter()
        .map(|p| (p[1] - c[1]).atan2(p[0] - c[0]).to_degrees())
        .collect();
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let wrap = angles[0] + 360.0 - angles[angles.len() - 1];
    let widest = angles.windows(2).map(|w| w[1] - w[0]).fold(wrap, f64::max);
    widest < 20.0
}

/// The rims (by index in [`RIMS`]) that close into an ellipse among `drawn` lines of the view
/// `entry`.
fn closed_rims(entry: &Value, drawn: &[Vec<f64>]) -> Vec<usize> {
    (0..RIMS.len())
        .filter(|&i| {
            let c = [
                dot(RIMS[i], &entry["axes"]["u"]),
                dot(RIMS[i], &entry["axes"]["v"]),
            ];
            drawn.iter().any(|l| closes_around(l, c))
        })
        .collect()
}

/// The rims the sharp bracket's isometric view draws closed: every rim it shows whole.
fn sharp_rims() -> Vec<usize> {
    let sharp = compile(&sharp_part(), DRAWING);
    let iso = &sharp["projection"]["data"]["hlr"]["iso"];
    assert!(
        lines(iso, "tangent").is_empty(),
        "a sharp part has no tangent edges"
    );
    let closed = closed_rims(iso, &lines(iso, "visible"));
    assert!(closed.len() >= 3, "sharp: {closed:?} {iso:#}");
    closed
}

/// The hypothesis, drawn both ways: the sharp bracket's isometric view closes every rim with its
/// visible edges; the rounded one's, with tangent edges left out, closes none (only the rounds'
/// outlines are left, the arcs the owner saw).
#[test]
fn rounding_every_edge_leaves_only_outlines_when_tangent_edges_are_hidden() {
    sharp_rims();
    let hidden = DRAWING.replace(
        "view iso = standard(orientation = isometric,",
        "view iso = standard(orientation = isometric, tangent_edges = hide,",
    );
    let rounded = compile(PART, &hidden);
    let iso = &rounded["projection"]["data"]["hlr"]["iso"];
    assert_eq!(iso["tangent_edges"], "hide");
    assert!(lines(iso, "tangent").is_empty());
    assert_eq!(
        closed_rims(iso, &lines(iso, "visible")),
        Vec::<usize>::new(),
        "{iso:#}"
    );
}

/// The fix: the rounded bracket's isometric view draws its tangent edges thin by default, so each
/// hole's rounded rim closes into an ellipse, and the sheet draws them in the thin continuous
/// line. The orthographic views keep leaving them out.
#[test]
fn a_filleted_plates_hole_rims_draw_as_closed_ellipses_in_an_isometric_view() {
    let out = compile(PART, DRAWING);
    let data = &out["projection"]["data"];
    let iso = &data["hlr"]["iso"];
    assert_eq!(iso["tangent_edges"], "thin");
    let tangent = lines(iso, "tangent");
    // Every rim the sharp bracket shows whole closes again, in the thin lines.
    assert_eq!(closed_rims(iso, &tangent), sharp_rims(), "{iso:#}");
    // The L reads whole: the thin lines span the view as its outline does.
    let span = |ls: &[Vec<f64>]| {
        let v: Vec<f64> = ls.iter().flat_map(|l| l.iter().skip(1).step_by(2).copied()).collect();
        v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
    };
    let outline = span(&lines(iso, "visible"));
    assert!(
        span(&tangent) > 0.9 * outline,
        "thin lines span {} of the outline's {outline}",
        span(&tangent)
    );
    for view in ["front", "top", "side"] {
        assert_eq!(data["hlr"][view]["tangent_edges"], "hide", "{view}");
        assert!(lines(&data["hlr"][view], "tangent").is_empty(), "{view}");
    }
    // On the sheet: the iso view's tangent lines, narrow (ISO 128-2 type 01.1).
    let prims = data["layouts"]["1"]["primitives"].as_array().unwrap();
    let thin: Vec<&Value> = prims
        .iter()
        .filter(|p| p["id"] == "view:iso" && p["style"] == "tangent")
        .collect();
    assert!(
        thin.len() >= tangent.len().min(6),
        "{} thin lines",
        thin.len()
    );
    let visible = prims
        .iter()
        .find(|p| p["style"] == "visible")
        .and_then(|p| p["width"].as_f64())
        .unwrap();
    assert!(thin.iter().all(|p| p["width"].as_f64().unwrap() < visible));
}

/// The text box the layout gives a figure: 0.6 of its height wide per character, from just below
/// the baseline to just above the cap height.
fn figure_box(shape: &Value) -> [f64; 4] {
    let (x, y, size) = (
        shape["x"].as_f64().unwrap(),
        shape["y"].as_f64().unwrap(),
        shape["size"].as_f64().unwrap(),
    );
    let w = shape["text"].as_str().unwrap().chars().count() as f64 * size * 0.6;
    let start = match shape["anchor"].as_str() {
        Some("middle") => -w / 2.0,
        Some("end") => -w,
        _ => 0.0,
    };
    let (lo, hi) = (-0.2 * size, 1.05 * size);
    if shape["rotation"]
        .as_f64()
        .is_some_and(|r| (r - 90.0).abs() < 1.0)
    {
        [x - hi, y + start, x - lo, y + start + w]
    } else {
        [x + start, y + lo, x + start + w, y + hi]
    }
}

fn crosses(a: [f64; 2], b: [f64; 2], g: [f64; 4]) -> bool {
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for (p, d, lo, hi) in [
        (a[0], b[0] - a[0], g[0], g[2]),
        (a[1], b[1] - a[1], g[1], g[3]),
    ] {
        if d.abs() < 1e-12 {
            if p < lo || p > hi {
                return false;
            }
        } else {
            let (u, v) = ((lo - p) / d, (hi - p) / d);
            t0 = t0.max(u.min(v));
            t1 = t1.min(u.max(v));
        }
    }
    t1 - t0 > 1e-9
}

/// The bracket drawing's `height` dimension is placed by hand across the right view, its line
/// through the middle hole. Its figure moves along the line, off the hole and the outline, and
/// Check drawing finds no text over the model's lines.
#[test]
fn a_dimension_placed_across_a_view_keeps_its_figure_off_the_models_lines() {
    let out = compile(PART, DRAWING);
    let data = &out["projection"]["data"];
    let prims = data["layouts"]["1"]["primitives"].as_array().unwrap();
    let figure = prims
        .iter()
        .find(|p| p["id"] == "annotation:height" && p["style"] == "dimtext")
        .expect("the 50 figure");
    assert_eq!(figure["shape"]["text"], "50");
    let b = figure_box(&figure["shape"]);
    let g = [b[0] + 0.2, b[1] + 0.2, b[2] - 0.2, b[3] - 0.2];
    for p in prims.iter().filter(|p| {
        p["kind"] == "view" && matches!(p["style"].as_str(), Some("visible" | "hidden" | "tangent"))
    }) {
        let pts: Vec<f64> =
            serde_json::from_value(p["shape"]["points"].clone()).unwrap_or_default();
        let over = pts
            .chunks_exact(2)
            .collect::<Vec<_>>()
            .windows(2)
            .any(|w| crosses([w[0][0], w[0][1]], [w[1][0], w[1][1]], g));
        assert!(!over, "the 50 at {b:?} is over a line of {}", p["id"]);
    }
    let files = files(PART, DRAWING);
    let (found, _) = runmat_cad_wasm::analyze::drawing_check::findings(
        "/p/bracket.drawing",
        DRAWING,
        &files,
        &serde_json::Map::new(),
    )
    .expect("check");
    // The 50 is clear; the front view's 15s, placed by hand over its hidden lines, are found.
    let over = |item: &str| {
        found
            .iter()
            .any(|f| f.item == item && f.message.contains("drawn over the"))
    };
    assert!(!over("annotation:height"), "{found:#?}");
    assert!(over("annotation:bottom_pitch"), "{found:#?}");
}

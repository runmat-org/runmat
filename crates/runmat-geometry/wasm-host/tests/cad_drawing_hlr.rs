//! Drawing views on OCCT's exact hidden-line projection: a lug's rounded end, seen along the
//! flat sides it runs into, keeps its outline (the tangent edges there are the silhouette).

const LUG: &str = r#"runmat 0.7
std 1

part lug {
    sketch s on origin.xy {
        l1 = line(start = [-14 mm, -7.5 mm], end = [0 mm, -7.5 mm])
        a1 = arc(center = [0 mm, 0 mm], start = [0 mm, -7.5 mm], end = [0 mm, 7.5 mm])
        l2 = line(start = [0 mm, 7.5 mm], end = [-14 mm, 7.5 mm])
        l3 = line(start = [-14 mm, 7.5 mm], end = [-14 mm, -7.5 mm])
        pin = circle(center = [0 mm, 0 mm], radius = 2 mm)
    }
    feature body = extrude(profile = sketch.s, distance = 3.4 mm)
}
"#;

const DRAWING: &str = r#"runmat 0.7
std 1

drawing lug_views {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = []
    }
    sheet "1" = { format = a3, scale = [2, 1], projection = third_angle }
    view front = standard(orientation = front, sheet = "1", model = { part = lug }, at = [150, 200], display = hidden_lines)
    view right = projected(from = front, direction = right, sheet = "1", model = { part = lug }, at = [300, 200], display = hidden_lines)
}
"#;

/// A vertical visible line at `u` spanning `v` 0 to 3.4 mm.
fn has_side(lines: &[Vec<f64>], u: f64) -> bool {
    lines.iter().any(|l| {
        let us: Vec<f64> = l.chunks_exact(2).map(|p| p[0]).collect();
        let vs: Vec<f64> = l.chunks_exact(2).map(|p| p[1]).collect();
        let lo = vs.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = vs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        us.iter().all(|x| (x - u).abs() < 1e-3) && lo < 1e-3 && (hi - 3.4).abs() < 1e-3
    })
}

#[test]
fn a_lugs_rounded_end_keeps_its_outline_in_the_side_view() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files: runmat_cad_wasm::runtime::Files =
        [("/p/lug.part".to_string(), LUG.to_string()), ("/p/lug_views.drawing".to_string(), DRAWING.to_string())].into_iter().collect();
    let out: serde_json::Value =
        serde_json::from_str(&runmat_cad_wasm::drawing::compile_json("/p/lug_views.drawing", DRAWING, &files)).expect("json");
    let right = &out["projection"]["data"]["hlr"]["right"];
    let visible: Vec<Vec<f64>> = serde_json::from_value(right["visible"].clone()).expect("visible lines");
    // Seen along x, the lug is a 15 x 3.4 mm rectangle: its sides at y = ±7.5 mm are where the
    // rounded end turns away from the viewer, on its tangent edges with the flat sides.
    assert!(has_side(&visible, -7.5) && has_side(&visible, 7.5), "right view: {visible:?}");
}

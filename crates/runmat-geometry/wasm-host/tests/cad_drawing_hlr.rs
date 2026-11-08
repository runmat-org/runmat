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

/// The Desktop's standard drawing of its bracket fixture, on OCCT: the front view's width and
/// thickness measure the part (80 and 10 mm) and show the cut's walls and the hole dashed.
#[test]
fn the_standard_drawing_measures_the_bracket_on_occt() {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../runmat-private-cad/desktop/src/app/components/cad/fixtures");
    let (Ok(part), Ok(drawing)) = (std::fs::read_to_string(dir.join("bracket.part")), std::fs::read_to_string(dir.join("standard.drawing"))) else {
        eprintln!("the Desktop checkout isn't beside runmat-cad; skipped");
        return;
    };
    let files = serde_json::json!({ "/p/bracket.part": part }).to_string();
    let request = serde_json::json!({ "fn": "compile", "args": ["/p/standard.drawing", drawing, files, "{}"] });
    let v: serde_json::Value = serde_json::from_str(&runmat_geometry_wasm_host::cad::call(&request.to_string())).unwrap();
    let data = &v["projection"]["data"];
    if let Ok(out) = std::env::var("RUNMAT_STANDARD_OUT") {
        std::fs::write(out, v.to_string()).unwrap();
    }
    assert_eq!(data["text"]["width"], "80", "{}", data["text"]);
    assert_eq!(data["text"]["thickness"], "10", "{}", data["text"]);
    // Seen from the front, the 4 mm hole at x = 60 draws both its outlines dashed, x = 58 and
    // 62; OCCT puts the hole's seam at x = 62, where the hidden-line algorithm draws no outline.
    let hidden: Vec<Vec<f64>> = data["hlr"]["front"]["hidden"].as_array().unwrap().iter().map(|l| l.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect();
    for x in [20.0, 40.0, 58.0, 62.0] {
        assert!(hidden.iter().any(|l| l.chunks(2).all(|p| (p[0] - x).abs() < 1e-3)), "no hidden line at x = {x}: {hidden:?}");
    }
    // The browser's framed call writes the models' meshes as bytes beside the JSON; the drawing
    // still measures their faces (it once read 0 for every face of 16 or more mesh values).
    let (framed, bytes) = runmat_geometry_wasm_host::cad::call_framed(&request.to_string());
    let framed: serde_json::Value = serde_json::from_str(&framed).unwrap();
    assert!(!bytes.is_empty());
    assert_eq!(framed["projection"]["data"]["text"], data["text"]);
    if let Ok(out) = std::env::var("RUNMAT_STANDARD_OUT") {
        std::fs::write(format!("{out}.framed"), framed.to_string()).unwrap();
    }
    // The same lines and sheet, to the meshes' f32 precision.
    for key in ["hlr", "layouts", "anchors"] {
        assert!(close(&framed["projection"]["data"][key], &data[key]), "{key} differs when framed");
    }
}

/// Two JSON values equal but for numbers within 1e-4 of each other (f32 mesh points at 8:1).
fn close(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() < 1e-4,
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| close(p, q)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, p)| y.get(k).is_some_and(|q| close(p, q))),
        _ => a == b,
    }
}

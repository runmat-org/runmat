//! A bent plate with a hole and a window, flattened on OCCT: the flat pattern's layout (what
//! drawings, DXF and DWG draw) carries the cuts at their developed places, the hole as a
//! circle.

use runmat_cad::regen::kernel::SegmentKind;
use runmat_cad_wasm::runtime::regenerate_part;

const CUT_PLATE: &str = r#"runmat 0.7
std 1

part plate_sm {
    sketch s_plate on origin.xy {
        e1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        e2 = line(start = [80 mm, 0 mm], end = [80 mm, 40 mm])
        e3 = line(start = [80 mm, 40 mm], end = [0 mm, 40 mm])
        e4 = line(start = [0 mm, 40 mm], end = [0 mm, 0 mm])
    }
    feature base = sheet_metal.base_flange(
        profile = sketch.s_plate,
        rules = { thickness = 2 mm, bend_radius = 3 mm, rule = k_factor(k = 0.44) }
    )
    sketch s_line on origin.xy {
        bl = line(start = [50 mm, -5 mm], end = [50 mm, 45 mm])
    }
    feature bend = sheet_metal.sketched_bend(
        face = feature.base.face.sheet(plate, ordinal = 1),
        line = [s_line, bl],
        angle = 90 deg,
        fixed = positive
    )
    feature h = hole(
        face = feature.base.face.sheet(plate, ordinal = 1),
        at = { h1 = point(20 mm, 20 mm, 2 mm) },
        diameter = 8 mm,
        through_all
    )
    sketch s_window on origin.yz {
        w1 = line(start = [15 mm, 12 mm], end = [25 mm, 12 mm])
        w2 = line(start = [25 mm, 12 mm], end = [25 mm, 22 mm])
        w3 = line(start = [25 mm, 22 mm], end = [15 mm, 22 mm])
        w4 = line(start = [15 mm, 22 mm], end = [15 mm, 12 mm])
    }
    feature window = extrude(profile = sketch.s_window, distance = 400 mm, symmetric, mode = cut)
    feature flat = sheet_metal.flat_pattern(fixed_face = feature.base.face.sheet(plate, ordinal = 1))
}
"#;

#[test]
fn a_cut_plates_flat_layout_draws_its_hole_and_window_where_they_develop() {
    runmat_geometry_wasm_host::cad::install_occt();
    let part = regenerate_part("/p/plate_sm.part", CUT_PLATE).expect("compiles");
    let errors: Vec<_> = part
        .result
        .diagnostics
        .diagnostics
        .iter()
        .filter(|d| d.severity == runmat_cad::diagnostics::Severity::Error)
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let (_, layout) = runmat_cad_wasm::exchange::flat::flat_layout(&part).expect("a flat pattern");
    let segs: Vec<_> = layout.loops.iter().flatten().collect();
    // The hole: one circle, Ø8 at (20, 20) mm in the plate's frame.
    let circles: Vec<_> = segs
        .iter()
        .filter_map(|s| match s.kind {
            SegmentKind::Circle { center, radius } => Some((center, radius)),
            _ => None,
        })
        .collect();
    assert_eq!(circles.len(), 1, "{circles:?}");
    let (c, r) = circles[0];
    assert!(
        (r - 0.004).abs() < 1e-9 && (c[0] - 0.02).abs() < 1e-9 && (c[1] - 0.02).abs() < 1e-9,
        "{c:?} {r}"
    );
    // The window's sides across the strip: at the moved flat's start (50 mm + BA/2) plus their
    // height above where the flat starts (t + R = 5 mm).
    let ba = std::f64::consts::FRAC_PI_2 * (0.003 + 0.44 * 0.002);
    let start = 0.05 + ba / 2.0;
    let want = [start + 0.007, start + 0.017];
    let across: Vec<f64> = segs
        .iter()
        .filter(|s| matches!(s.kind, SegmentKind::Line) && (s.start[0] - s.end[0]).abs() < 1e-12)
        .map(|s| s.start[0])
        .filter(|x| *x > start + 1e-4 && *x < 0.08 - 1e-4)
        .collect();
    for w in want {
        assert!(
            across.iter().any(|x| (x - w).abs() < 1e-9),
            "{across:?} vs {want:?}"
        );
    }
    // The outline keeps the developed plate's extents.
    let (lo, hi) = layout.extents();
    assert!(
        (hi[0] - lo[0] - 0.08).abs() < 1e-9 && (hi[1] - lo[1] - 0.04).abs() < 1e-9,
        "{lo:?} {hi:?}"
    );
}

/// The plate with an edge flange on its side x = 80 mm and a hole, flattened.
const FLANGED_PLATE: &str = r#"runmat 0.7
std 1

part plate_sm {
    sketch s_plate on origin.xy {
        e1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        e2 = line(start = [80 mm, 0 mm], end = [80 mm, 40 mm])
        e3 = line(start = [80 mm, 40 mm], end = [0 mm, 40 mm])
        e4 = line(start = [0 mm, 40 mm], end = [0 mm, 0 mm])
    }
    feature base = sheet_metal.base_flange(
        profile = sketch.s_plate,
        rules = { thickness = 2 mm, bend_radius = 3 mm, rule = k_factor(k = 0.44) }
    )
    feature lip = sheet_metal.edge_flange(
        edges = [edges_between(feature.base.face.sheet(plate, ordinal = 1), feature.base.face.sheet_edge(e2))],
        length = 20 mm,
        angle = 90 deg,
        length_reference = tangent
    )
    feature h = hole(
        face = feature.base.face.sheet(plate, ordinal = 1),
        at = { h1 = point(20 mm, 20 mm, 2 mm) },
        diameter = 8 mm,
        through_all
    )
    feature flat = sheet_metal.flat_pattern(fixed_face = feature.base.face.sheet(plate, ordinal = 1))
}
"#;

const FLAT_DRAWING: &str = r#"runmat 0.7
std 1

drawing plate_flat {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = []
    }
    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }
    view flat = flat_pattern(feature = flat, sheet = "1", model = { file = "plate_sm.part" }, at = [150, 150], display = visible_lines)
}
"#;

#[test]
fn a_drawings_flat_pattern_view_carries_the_developed_sheet_and_its_bend_table() {
    runmat_geometry_wasm_host::cad::install_occt();
    let files: runmat_cad_wasm::runtime::Files = [
        ("/p/plate_sm.part".to_string(), FLANGED_PLATE.to_string()),
        ("/p/plate_flat.drawing".to_string(), FLAT_DRAWING.to_string()),
    ]
    .into_iter()
    .collect();
    let out: serde_json::Value = serde_json::from_str(&runmat_cad_wasm::drawing::compile_json(
        "/p/plate_flat.drawing",
        FLAT_DRAWING,
        &files,
    ))
    .expect("json");
    let data = &out["projection"]["data"];
    let errors: Vec<_> = data["diagnostics"]["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(out["diagnostics"].as_array().into_iter().flatten())
        .filter(|d| d["severity"] == "error")
        .collect();
    assert!(errors.is_empty(), "{errors:?}");
    let sheet = &data["hlr"]["flat"]["flat"];
    // The plate and the flange's strip, BA + 20 mm past x = 80 mm, in millimetres.
    let ba = std::f64::consts::FRAC_PI_2 * (3.0 + 0.44 * 2.0);
    let max = sheet["max"].as_array().expect("extents");
    assert!((max[0].as_f64().unwrap() - (80.0 + ba + 20.0)).abs() < 1e-6, "{sheet}");
    assert!((max[1].as_f64().unwrap() - 40.0).abs() < 1e-6, "{sheet}");
    // The hole is a circle of the outline; one bend line and one row of the bend table.
    let circles = sheet["outline"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|l| l.as_array().cloned().unwrap_or_default())
        .filter(|c| c["curve"] == "circle")
        .count();
    assert_eq!(circles, 1, "{sheet}");
    let lines = sheet["bend_lines"].as_array().expect("bend lines");
    assert_eq!(lines.len(), 1, "{sheet}");
    assert!((lines[0]["start"][0].as_f64().unwrap() - (80.0 + ba / 2.0)).abs() < 1e-6, "{sheet}");
    assert!(lines[0]["note"].as_str().unwrap_or("").contains("UP"), "{sheet}");
    let table = sheet["bend_table"].as_array().expect("bend table");
    assert_eq!(table.len(), 1, "{sheet}");
    assert!((table[0]["angle"].as_f64().unwrap() - 90.0).abs() < 1e-9, "{sheet}");
}

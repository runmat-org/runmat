//! DWG R2018: sheets written and read back entity by entity, the same sheet's DXF read back
//! to the same records, and drawings and flat patterns exported through `design.export`.

use base64::Engine;
use runmat_design_tools::exchange::dxf;
use runmat_design_tools::formats::dwg;
use runmat_design_tools::formats::sheet::{canonical, standard_linetypes, Block, Dimension, DimensionKind, Layer, Shape, Sheet};
use runmat_design_tools::{call, Files};
use serde_json::json;

/// A sheet with every entity kind the writers write, a block and a dimension block.
fn sample() -> Sheet {
    let mut s = Sheet {
        paper: [420.0, 297.0],
        layers: vec![
            Layer { name: "VISIBLE".into(), color: 7, linetype: "CONTINUOUS".into() },
            Layer { name: "HIDDEN".into(), color: 8, linetype: "DASHED".into() },
            Layer { name: "NOTES".into(), color: 3, linetype: "CONTINUOUS".into() },
        ],
        linetypes: standard_linetypes(),
        ..Sheet::default()
    };
    s.line("VISIBLE", [10.0, 10.0], [410.0, 10.0]);
    s.line("HIDDEN", [20.5, 30.25], [20.5, 130.75]);
    s.add("VISIBLE", Shape::Arc { center: [100.0, 100.0], radius: 12.5, start: 0.25, end: 2.5 });
    s.add("VISIBLE", Shape::Circle { center: [150.0, 120.0], radius: 2.0 });
    s.rect("VISIBLE", 50.0, 50.0, 80.0, 50.0);
    s.add("VISIBLE", Shape::Polyline { points: vec![[0.0, 0.0], [1.0, 2.0], [3.0, 2.0]], closed: false });
    s.text("NOTES", [30.0, 20.0], 3.0, "FRONT  SCALE 2:1");
    s.add("NOTES", Shape::Text { at: [60.0, 20.0], height: 2.5, rotation: std::f64::consts::FRAC_PI_2, text: "⌀4 THRU ±0.1 90°".into() });
    s.add("NOTES", Shape::MText { at: [200.0, 200.0], height: 2.5, width: 40.0, text: "∥ | 0.05 | A".into() });
    s.add("VISIBLE", Shape::Hatch { loops: vec![vec![[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]]], pattern: "ANSI31".into(), angle: 0.0, scale: 1.0 });
    s.add("VISIBLE", Shape::Hatch { loops: vec![vec![[20.0, 0.0], [30.0, 0.0], [25.0, 5.0]]], pattern: "SOLID".into(), angle: 0.0, scale: 1.0 });
    let mut tb = Sheet::default();
    tb.rect("NOTES", -120.0, 0.0, 120.0, 7.0);
    tb.text("NOTES", [-118.0, 2.2], 2.2, "Title");
    s.blocks.push(Block { name: "TITLE_BLOCK".into(), base: [0.0, 0.0], entities: tb.entities });
    s.add("NOTES", Shape::Insert { block: "TITLE_BLOCK".into(), at: [410.0, 10.0], scale: 1.0, rotation: 0.0 });
    s.add("NOTES", Shape::Insert { block: "TITLE_BLOCK".into(), at: [300.0, 250.0], scale: 0.5, rotation: 0.0 });
    s.dimension(
        "NOTES",
        Dimension {
            kind: DimensionKind::Linear { from: [50.0, 50.0], to: [130.0, 50.0], line: [50.0, 40.0], rotation: 0.0 },
            text_at: [90.0, 41.5],
            measurement: 80.0,
            text: Some("80".into()),
            block: String::new(),
        },
    );
    s.dimension(
        "NOTES",
        Dimension { kind: DimensionKind::Radius { center: [100.0, 100.0], on: [112.5, 100.0] }, text_at: [115.0, 102.0], measurement: 12.5, text: None, block: String::new() },
    );
    s.dimension(
        "NOTES",
        Dimension { kind: DimensionKind::Diameter { from: [148.0, 120.0], to: [152.0, 120.0] }, text_at: [155.0, 122.0], measurement: 4.0, text: None, block: String::new() },
    );
    s
}

#[test]
fn a_sheet_round_trips_through_dwg() {
    let sheet = sample();
    let bytes = dwg::write(&sheet);
    assert_eq!(&bytes[..6], b"AC1032");
    let (back, report) = dwg::read(&bytes).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(report.version, "AC1032");
    assert_eq!(report.classes, ["LWPOLYLINE", "HATCH", "LAYOUT"]);
    for name in ["AcDb:Header", "AcDb:Classes", "AcDb:Handles", "AcDb:AcDbObjects", "AcDb:AuxHeader", "AcDb:Template", "AcDb:SummaryInfo"] {
        assert!(report.sections.iter().any(|s| s == name), "{name}");
    }
    assert_eq!(back.entities, sheet.entities, "model space comes back entity by entity");
    assert_eq!(back.blocks, sheet.blocks);
    assert_eq!(back.layers, sheet.layers);
    assert_eq!(back.linetypes, sheet.linetypes);
    assert_eq!(dwg::write(&sheet), bytes, "the same sheet writes the same bytes");
}

#[test]
fn a_damaged_dwg_is_refused() {
    let bytes = dwg::write(&sample());
    let mut bad = bytes.clone();
    let n = bad.len();
    bad[n / 2] ^= 0x5A;
    assert!(dwg::read(&bad).is_err());
}

#[test]
fn dwg_and_dxf_of_one_sheet_carry_the_same_records() {
    let sheet = sample();
    let (from_dwg, _) = dwg::read(&dwg::write(&sheet)).unwrap();
    let from_dxf = dxf::read(&dxf::write(&sheet)).unwrap();
    assert_eq!(canonical(&from_dwg), canonical(&from_dxf));
    assert_eq!(canonical(&from_dwg), canonical(&sheet));
}

const BRACKET: &str = r#"runmat 0.7
std 1

part bracket {
    sketch s1 on origin.xy {
        l1 = line(start = [0 mm, 0 mm], end = [80 mm, 0 mm])
        l2 = line(start = [80 mm, 0 mm], end = [80 mm, 50 mm])
        l3 = line(start = [80 mm, 50 mm], end = [0 mm, 50 mm])
        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])
    }
    feature body = extrude(profile = sketch.s1, distance = 10 mm)
    feature h = hole(face = feature.body.face.end, at = { a = point(60 mm, 10 mm, 10 mm) }, diameter = 4 mm, through_all)
    output bracket = { body = feature.body.body, part_number = "BR-100" }
    datum A = { on = feature.body.face.start }
    tolerance top_flat = { characteristic = flatness, on = [feature.body.face.end], tolerance = 0.05 mm }
    tolerance hole_square = { characteristic = perpendicularity, on = [feature.h.face.wall(a)], tolerance = 0.02 mm, diametral = true, datums = [{ label = "A" }] }
    finish top = { on = [feature.body.face.end], ra = 1.6 um }
}
"#;

/// The bracket's production drawing: frames, a datum, a finish mark, a hole callout, a
/// diameter dimension, a notes block and a revision table.
const PRODUCTION: &str = r#"runmat 0.7
std 1

drawing production {
    format a3 = { size = { size = iso, name = A3 }, title_block = [{ label = "Title", source = { source = literal, text = "Bracket" } }] }
    sheet "1" = { format = a3, scale = [2, 1], projection = third_angle }
    view front = standard(orientation = front, sheet = "1", model = { part = bracket }, at = [115, 85], display = hidden_lines)
    view top = projected(from = front, direction = up, sheet = "1", model = { part = bracket }, at = [115, 185], display = hidden_lines)
    annotation hole = hole_callout(hole = feature.h.face.wall(a), view = top)
    annotation hole_size = dimension(style = diameter, references = [feature.h.face.wall(a)], view = top)
    annotation flat = gdt(tolerance = top_flat, view = front)
    annotation square = gdt(tolerance = hole_square, view = top)
    annotation datum_a = datum_symbol(label = A, view = front)
    annotation top_finish = surface_finish(on = feature.body.face.end, view = front)
    annotation notes = note(text = "NOTES:\n1. BREAK SHARP EDGES.", view = front, at = [250, 120])
    table revisions = revision(rows = [{ rev = "A", description = "Initial release", date = "2026-11-05", by = "RM" }], sheet = "1", at = [410, 287])
}
"#;

const DRAWING: &str = r#"runmat 0.7
std 1

drawing standard {
    format a3 = {
        size = { size = iso, name = A3 },
        title_block = [
            { label = "Title", source = { source = literal, text = "Bracket" } },
            { label = "Part number", source = { source = property, name = part_number } }
        ]
    }
    sheet "1" = { format = a3, scale = [2, 1], projection = third_angle }
    view front = standard(orientation = front, sheet = "1", model = { part = bracket }, at = [115, 85], display = hidden_lines)
    view top = projected(from = front, direction = up, sheet = "1", model = { part = bracket }, at = [115, 185], display = hidden_lines)
    annotation width = dimension(style = linear, references = [feature.body.face.side(s1.l4), feature.body.face.side(s1.l2)], view = front)
    annotation hole = hole_callout(hole = feature.h.face.wall(a), view = top)
}
"#;

fn files() -> Files {
    [("/p/bracket.part", BRACKET), ("/p/standard.drawing", DRAWING), ("/p/production.drawing", PRODUCTION)].into_iter().map(|(k, v)| (k.to_owned(), v.to_owned())).collect()
}

#[test]
fn a_drawing_exports_as_dwg_with_the_content_of_its_dxf() {
    let out = call("design.export", json!({ "path": "/p/standard.drawing", "format": "dwg" }), &files()).unwrap_or_else(|e| panic!("{e}"));
    let file = &out["files"][0];
    assert_eq!(file["name"], "standard-1.dwg");
    let bytes = base64::engine::general_purpose::STANDARD.decode(file["content"].as_str().unwrap()).unwrap();
    let (sheet, _) = dwg::read(&bytes).unwrap_or_else(|e| panic!("{e}"));
    let kinds = |k: fn(&Shape) -> bool| sheet.entities.iter().filter(|e| k(&e.shape)).count();
    assert!(kinds(|s| matches!(s, Shape::Polyline { .. } | Shape::Line { .. })) > 10, "frame and view edges");
    assert_eq!(kinds(|s| matches!(s, Shape::Insert { .. })), 1, "the title block");
    // The title block as the compile lays it out (ISO 7200): 180 mm wide in the frame's bottom
    // right corner (A3: x 230 to 410 mm, rows up from y 10 mm), values in 3.5 mm text.
    let block = sheet.blocks.iter().find(|b| b.name == "TITLE_BLOCK").expect("the title block");
    let value = block.entities.iter().find_map(|e| match &e.shape { Shape::Text { at, height, text, .. } if text == "Bracket" => Some((*at, *height)), _ => None });
    let (at, height) = value.unwrap_or_else(|| panic!("no title text in {:?}", block.entities));
    assert!((230.0..410.0).contains(&at[0]) && (10.0..40.0).contains(&at[1]) && (height - 3.5).abs() < 1e-9, "{at:?} {height}");
    let dims: Vec<&Dimension> = sheet.entities.iter().filter_map(|e| if let Shape::Dimension(d) = &e.shape { Some(d) } else { None }).collect();
    assert_eq!(dims.len(), 1, "{:?}", sheet.entities.iter().map(|e| &e.layer).collect::<Vec<_>>());
    assert_eq!(dims[0].text.as_deref(), Some("80"));
    // The views' edges come from the display tessellation, which holds positions as f32
    // (2e-6 mm at this size).
    assert!((dims[0].measurement - 80.0).abs() < 1e-4, "{}", dims[0].measurement);
    assert!(sheet.entities.iter().any(|e| matches!(&e.shape, Shape::Text { text, .. } if text.contains("THRU"))), "the hole callout");
    let dxf_out = call("design.export", json!({ "path": "/p/standard.drawing", "format": "dxf" }), &files()).unwrap();
    let from_dxf = dxf::read(dxf_out["files"][0]["content"].as_str().unwrap()).unwrap();
    assert_eq!(canonical(&sheet), canonical(&from_dxf));
}

/// DXF carries a production drawing's callouts as the Desktop's sheet draws them: the feature
/// control frames' cells boxed on leaders, the datum's boxed letter, the finish mark's value, the
/// hole callout and diameter on leaders, the notes block's lines and the revision table's rows.
#[test]
fn a_production_drawing_exports_its_callouts_as_dxf() {
    let out = call("design.export", json!({ "path": "/p/production.drawing", "format": "dxf" }), &files()).unwrap_or_else(|e| panic!("{e}"));
    let sheet = dxf::read(out["files"][0]["content"].as_str().unwrap()).unwrap();
    let texts: Vec<&str> = sheet.entities.iter().filter_map(|e| if let Shape::Text { text, .. } = &e.shape { Some(text.as_str()) } else { None }).collect();
    for want in ["⏥", "0.05", "⊥", "⌀0.02", "A", "Ra 1.6", "⌀4 THRU", "⌀4", "NOTES:", "1. BREAK SHARP EDGES.", "REVISIONS", "DESCRIPTION", "APPROVED", "Initial release", "2026-11-05"] {
        assert!(texts.contains(&want), "no {want:?} in {texts:?}");
    }
    // Arrowheads and the datum triangle are solid fills; R12 DXF carries a fill as its closed
    // boundary, a triangle.
    let solids = sheet.entities.iter().filter(|e| matches!(&e.shape, Shape::Polyline { points, closed: true } if points.len() == 3)).count();
    assert!(solids >= 5, "{solids} filled triangles");
    // The revision table's top-right corner is where the drawing puts it.
    let corner = sheet.entities.iter().any(|e| matches!(&e.shape, Shape::Polyline { points, closed: true } if points.iter().any(|p| (p[0] - 410.0).abs() < 1e-6 && (p[1] - 287.0).abs() < 1e-6)));
    assert!(corner, "the revision table's frame");
    // The PDF of the same sheet carries the same callouts.
    let out = call("design.export", json!({ "path": "/p/production.drawing", "format": "pdf" }), &files()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(out["files"][0]["name"], "production-1.pdf");
    assert_eq!(out["files"][0]["mimeType"], "application/pdf");
    let pdf = base64::engine::general_purpose::STANDARD.decode(out["files"][0]["content"].as_str().unwrap()).unwrap();
    let body = String::from_utf8_lossy(&pdf);
    for want in ["(Ra 1.6) Tj", "(REVISIONS) Tj", "(Initial release) Tj", "(NOTES:) Tj", "(0.05) Tj", "(4 THRU) Tj"] {
        assert!(body.contains(want), "no {want} in the PDF");
    }
}

#[test]
fn a_part_without_a_flat_pattern_says_so() {
    let err = call("design.export", json!({ "path": "/p/bracket.part", "format": "dwg" }), &files()).unwrap_err().to_string();
    assert!(err.contains("flat pattern"), "{err}");
}

/// A weldment's drawing exports its cut list where the drawing places it: the header and one
/// row per member group, with the cut length and end angles.
#[test]
fn a_weldment_drawing_exports_its_cut_list() {
    let frame = include_str!("../../runmat-cad/tests/corpus/documents/part07.part");
    let drawing = r#"runmat 0.7
std 1

drawing frame {
    format a3 = { size = { size = iso, name = A3 }, title_block = [] }
    sheet "1" = { format = a3, scale = [1, 1], projection = third_angle }
    view top = standard(orientation = top, sheet = "1", model = { part = frame }, at = [150, 180], display = visible_lines)
    table cuts = cut_list(view = top, sheet = "1", at = [20, 100])
}
"#;
    let files: Files = [("/p/frame.part", frame), ("/p/frame.drawing", drawing)].into_iter().map(|(k, v)| (k.to_owned(), v.to_owned())).collect();
    let out = call("design.export", json!({ "path": "/p/frame.drawing", "format": "dxf" }), &files).unwrap_or_else(|e| panic!("{e}"));
    let sheet = dxf::read(out["files"][0]["content"].as_str().unwrap()).unwrap();
    let texts: Vec<&str> = sheet.entities.iter().filter_map(|e| if let Shape::Text { text, .. } = &e.shape { Some(text.as_str()) } else { None }).collect();
    for want in ["ITEM", "QTY", "LENGTH", "ANGLE 1", "220.0", "120.0", "45.0°"] {
        assert!(texts.contains(&want), "no {want:?} in {texts:?}");
    }
    let corner = sheet.entities.iter().any(|e| matches!(&e.shape, Shape::Polyline { points, closed: true } if points.iter().any(|p| (p[0] - 20.0).abs() < 1e-6 && (p[1] - 100.0).abs() < 1e-6)));
    assert!(corner, "the cut list's frame at its top-left corner");
}

/// The hand-built drawing (every view and annotation kind on one sheet) exports what the
/// Desktop's sheet draws, from the same layout: angular, ordinate, baseline and chain
/// dimensions, center marks and centerlines, the weld symbol, the datum target, the leadered
/// note, and the hole table, in DXF and PDF alike.
#[test]
fn the_hand_built_drawing_exports_every_annotation_kind_from_the_one_layout() {
    let files: Files = [
        ("/p/bracket.part", include_str!("../../runmat-cad/tests/fixtures/hand-built/bracket.part")),
        ("/p/bracket.drawing", include_str!("../../runmat-cad/tests/fixtures/hand-built/hand.drawing")),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let out = call("design.export", json!({ "path": "/p/bracket.drawing", "format": "dxf" }), &files).unwrap_or_else(|e| panic!("{e}"));
    let sheet = dxf::read(out["files"][0]["content"].as_str().unwrap()).unwrap();
    // Dimension figures are in their dimension's block.
    let texts: Vec<&str> = sheet.entities.iter().chain(sheet.blocks.iter().flat_map(|b| &b.entities)).filter_map(|e| if let Shape::Text { text, .. } = &e.shape { Some(text.as_str()) } else { None }).collect();
    for want in ["90°", "0", "60", "10", "50", "20", "FILLET 3", "A1", "DEBURR", "R2", "⌀4 H7", "⌀4 THRU", "Ra 0.8", "HOLE TABLE", "⌀4.00 THRU", "SECTION A-A", "VIEW C", "BROKEN-OUT SECTION"] {
        assert!(texts.contains(&want), "no {want:?} in {texts:?}");
    }
    // Center marks and the centerline on the center layer; the linear dimension as an entity.
    assert!(sheet.entities.iter().filter(|e| e.layer == "CENTER").count() >= 4);
    assert!(sheet.entities.iter().any(|e| matches!(&e.shape, Shape::Dimension(d) if d.text.as_deref() == Some("80 ±0.2"))));
    // The title block, filled from the part with the standard fields.
    let block = sheet.blocks.iter().find(|b| b.name == "TITLE_BLOCK").expect("the title block");
    assert!(block.entities.iter().any(|e| matches!(&e.shape, Shape::Text { text, .. } if text == "BR-100")));
    let pdf = call("design.export", json!({ "path": "/p/bracket.drawing", "format": "pdf" }), &files).unwrap();
    let body = base64::engine::general_purpose::STANDARD.decode(pdf["files"][0]["content"].as_str().unwrap()).unwrap();
    let body = String::from_utf8_lossy(&body);
    for want in ["(FILLET 3) Tj", "(DEBURR) Tj", "(HOLE TABLE) Tj", "(VIEW C) Tj"] {
        assert!(body.contains(want), "no {want} in the PDF");
    }
}

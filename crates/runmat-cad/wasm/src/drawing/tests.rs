//! The drawing compile: displays, measured values, sections, details and auto-drawing.

use super::*;

const BRACKET: &str = include_str!("../../../tests/corpus/documents/bracket.part");
pub(crate) const DRAWING: &str = "runmat 0.7\nstd 1\n\ndrawing bracket {\n    format a3 = {\n        size = { size = iso, name = A3 },\n        title_block = [\n            { label = \"Part number\", source = { source = property, name = part_number } }\n        ]\n    }\n\n    sheet \"1\" = { format = a3, scale = [1, 1], projection = third_angle }\n\n    view front = standard(\n        orientation = front,\n        sheet = \"1\",\n        model = { part = bracket },\n        at = [100, 100],\n        display = hidden_lines\n    )\n    view top = projected(\n        from = front,\n        direction = up,\n        sheet = \"1\",\n        model = { part = bracket },\n        at = [100, 200],\n        display = hidden_lines\n    )\n}\n";

/// Export compiled fixture data only when the caller selects a destination.
/// Public tests use repository-owned designs and require no sibling checkout.
fn desktop_fixture(name: &str, json: &str) {
    if let Ok(dir) = std::env::var("RUNMAT_DESKTOP_FIXTURES") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(name), json).unwrap();
    }
}

#[test]
fn each_sheet_carries_its_standards_drafting_style() {
    let files: Files = [("/p/bracket.part".to_owned(), BRACKET.to_owned())].into();
    let out: Value =
        serde_json::from_str(&compile_json("/p/bracket.drawing", DRAWING, &files)).unwrap();
    let style = &out["projection"]["data"]["styles"]["1"];
    assert_eq!(style["standard"], "iso", "{style}");
    assert!(style["text_height"].as_f64().unwrap() > 0.0, "{style}");
    assert!(
        style["lines"]["visible"].as_f64().unwrap() > style["lines"]["dimension"].as_f64().unwrap(),
        "{style}"
    );
}

#[test]
fn a_drawing_compiles_from_its_text_with_the_displays_of_its_models() {
    let deps = dependencies("/p/bracket.drawing", DRAWING);
    // The model is named by symbol, so the host lists the drawing's folder.
    assert_eq!(deps, vec!["/p/".to_owned()]);
    // The host reads the folder's designs.
    let files: Files = [("/p/bracket.part".to_owned(), BRACKET.to_owned())].into();
    let out: Value =
        serde_json::from_str(&compile_json("/p/bracket.drawing", DRAWING, &files)).unwrap();
    assert_eq!(out["projection"]["kind"], "drawing", "{out}");
    let data = &out["projection"]["data"];
    assert_eq!(data["document"]["views"].as_array().map(Vec::len), Some(2));
    // Models are keyed by their workspace path.
    let model = &out["models"]["/p/bracket.part"];
    assert_eq!(model["kind"], "part", "{out}");
    assert!(model["data"]["bodies"]
        .as_array()
        .is_some_and(|b| !b.is_empty()));
    assert_eq!(
        out["diagnostics"].as_array().map(Vec::len),
        Some(0),
        "{out}"
    );
}

/// A standard drawing of the public bracket fixture: three views at 2:1, section
/// A–A through the hole, detail B of the round, model dimensions, a hole callout and a
/// GD&T frame from the part's PMI.
#[test]
fn the_standard_drawing_measures_its_model_and_cuts_its_section() {
    const PART: &str = include_str!("../../../tests/fixtures/drawing/bracket.part");
    const STANDARD: &str = include_str!("../../../tests/fixtures/drawing/standard.drawing");
    let files: Files = [("/p/bracket.part".to_owned(), PART.to_owned())]
        .into_iter()
        .collect();
    let compiled = compile_json("/p/standard.drawing", STANDARD, &files);
    desktop_fixture("standard.drawing.compiled.json", &compiled);
    let out: Value = serde_json::from_str(&compiled).unwrap();
    let data = &out["projection"]["data"];
    assert_eq!(
        out["diagnostics"].as_array().map(Vec::len),
        Some(0),
        "{}",
        out["diagnostics"]
    );
    assert_eq!(
        data["diagnostics"].as_array().map(Vec::len),
        Some(0),
        "{}",
        data["diagnostics"]
    );
    let text = &data["text"];
    assert_eq!(text["width"], "80");
    assert_eq!(text["thickness"], "10");
    assert_eq!(text["depth"], "50");
    assert_eq!(text["hole"], "⌀4 THRU");
    assert_eq!(text["parallel"], "∥|0.05|A");
    // The width's references: the two side faces, 80 mm apart along x.
    let w = &data["anchors"]["width"];
    assert!(
        (w[1][0].as_f64().unwrap() - w[0][0].as_f64().unwrap() - 80.0).abs() < 1e-3,
        "{w}"
    );
    // Every feature located and sized (ASME Y14.5): the cut's 20 x 10 and its position
    // from the datum edges, and the hole's position to its axis.
    for (id, value) in [
        ("cut_width", "20"),
        ("cut_height", "10"),
        ("cut_x", "20"),
        ("cut_y", "20"),
        ("hole_x", "60"),
        ("hole_y", "10"),
    ] {
        assert_eq!(text[id], value, "{id}: {text}");
    }
    // A planar reference carries its box; the hole (a cylinder) locates by its axis.
    let e = &data["extents"]["hole_x"];
    assert!(e[0].is_array() && e[1].is_null(), "{e}");
    let rows = data["title_blocks"]["1"].to_string();
    assert!(
        rows.contains("Aluminum") && rows.contains("BR-100") && rows.contains("2:1"),
        "{rows}"
    );
    // Section A–A: the cut through the hole's axis, hatched in two pieces either side of
    // the hole, and its cutting line on the top view at y = 10 mm.
    let section = &data["hlr"]["a_a"];
    let hatch = section["hatch"].as_array().map(Vec::len).unwrap_or(0);
    assert!(hatch >= 2, "{} hatch loops", hatch);
    assert_eq!(section["cut"]["view"], "top");
    assert!(
        (section["cut"]["at"].as_f64().unwrap() - 10.0).abs() < 1e-6,
        "{}",
        section["cut"]
    );
    assert!(section["hidden"].as_array().is_some_and(|h| h.is_empty()));
    // Detail B: centred on the rounded edge as the right view sees it (y = 0, z = 10).
    let c = &data["hlr"]["b"]["detail"]["center"];
    let (u, v) = (c[0].as_f64().unwrap(), c[1].as_f64().unwrap());
    assert!(u.abs() < 1.6 && (v - 10.0).abs() < 1.6, "{c}");
}

/// A model no file of the folder declares is an error on the view that names it.
#[test]
fn a_model_no_file_declares_is_an_error_that_names_it() {
    let out: Value =
        serde_json::from_str(&compile_json("/p/bracket.drawing", DRAWING, &Files::new())).unwrap();
    let messages: Vec<&str> = out["diagnostics"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|d| d["severity"] == "error")
        .filter_map(|d| d["message"].as_str())
        .collect();
    assert!(messages.iter().any(|m| m.contains("part bracket")), "{out}");
}

#[test]
fn an_empty_file_is_an_empty_drawing() {
    let out: Value =
        serde_json::from_str(&compile_json("/p/new.drawing", "\n  \n", &Files::new())).unwrap();
    assert_eq!(out["projection"]["kind"], "drawing", "{out}");
    let document = &out["projection"]["data"]["document"];
    assert_eq!(
        document["sheets"].as_array().map(Vec::len),
        Some(0),
        "{out}"
    );
    assert!(out["diagnostics"]
        .as_array()
        .is_some_and(|d| d.iter().all(|d| d["severity"] != "error")));
}

#[test]
fn syntax_errors_are_diagnostics() {
    let out: Value =
        serde_json::from_str(&compile_json("/p/x.drawing", "drawing x {", &Files::new())).unwrap();
    assert!(out.get("projection").is_none());
    assert!(!out["diagnostics"].as_array().unwrap().is_empty());
}

#[test]
fn auto_drawing_lays_out_views_a_section_and_hole_callouts() {
    const PART: &str = include_str!("../../../tests/fixtures/drawing/bracket.part");
    let files: Files = [("/p/bracket.part".to_owned(), PART.to_owned())]
        .into_iter()
        .collect();
    let values = serde_json::json!({ "model": "bracket.part" });
    let (change, summary) =
        auto::auto("/p/auto.drawing", "", &files, values.as_object().unwrap()).unwrap();
    assert!(
        summary.contains("2:1") && summary.contains("section A-A"),
        "{summary}"
    );
    let out: Value =
        serde_json::from_str(&compile_json("/p/auto.drawing", &change.source, &files)).unwrap();
    let data = &out["projection"]["data"];
    assert_eq!(
        data["diagnostics"].as_array().map(Vec::len),
        Some(0),
        "{}\n{}",
        data["diagnostics"],
        change.source
    );
    assert_eq!(data["document"]["views"].as_array().map(Vec::len), Some(4));
    assert_eq!(data["text"]["width"], "80");
    assert_eq!(data["text"]["depth"], "50");
    assert_eq!(data["text"]["thickness"], "10");
    assert_eq!(data["text"]["hole1"], "⌀4 THRU");
    assert!(data["hlr"]["a_a"]["hatch"]
        .as_array()
        .is_some_and(|h| !h.is_empty()));
    // Run again: the drawing keeps its sheet and gains fresh names.
    let (again, _) = auto::auto(
        "/p/auto.drawing",
        &change.source,
        &files,
        values.as_object().unwrap(),
    )
    .unwrap();
    assert!(
        again.source.contains("view front2 = standard("),
        "{}",
        again.source
    );
    assert_eq!(again.source.matches("sheet \"1\" =").count(), 1);
}

/// A parts list counts a part by its part number: four brackets set to two widths are one line,
/// quantity 4. A part with no `description` reads its output's name as words.
#[test]
fn a_parts_list_on_the_title_block_steps_left_of_the_right_centring_mark() {
    // Ten pins of different sizes on A4: the parts list on the title block reaches past the
    // right centring mark at mid-height.
    let pin = "runmat 0.7\nstd 1\n\npart pin {\n    input radius : length = 2 mm\n    sketch s on origin.xy {\n        c = circle(center = [0 mm, 0 mm], radius = input.radius)\n    }\n    feature body = extrude(profile = sketch.s, distance = 4 mm)\n    output pin = { body = feature.body.body }\n}\n";
    let uses: String = (1..=10)
        .map(|k| format!("    use part pin as p{k} {{\n        configuration = {{ select = inputs, values = {{ radius = {k} mm }} }}\n        placement = {{ placement = offset, offset = {{ translation = [{} mm, 0 mm, 0 mm] }} }}\n    }}\n", 30 * k))
        .collect();
    let rig = format!("runmat 0.7\nstd 1\n\nassembly rig {{\n{uses}}}\n");
    let drawing = "runmat 0.7\nstd 1\n\ndrawing rig {\n    format a4 = { size = { size = iso, name = A4 }, title_block = [] }\n    sheet \"1\" = { format = a4, scale = [1, 5], projection = third_angle }\n    view iso = standard(orientation = isometric, sheet = \"1\", model = { assembly = rig }, at = [70, 120], display = visible_lines)\n    table parts = bom(view = iso, sheet = \"1\", at = [0, 0])\n}\n";
    let files: Files = [
        ("/p/pin.part".to_owned(), pin.to_owned()),
        ("/p/rig.assembly".to_owned(), rig),
    ]
    .into();
    let out: Value =
        serde_json::from_str(&compile_json("/p/rig.drawing", drawing, &files)).unwrap();
    let layout = &out["projection"]["data"]["layouts"]["1"];
    let lines = out["projection"]["data"]["bom_tables"]["parts"]["lines"]
        .as_array()
        .map_or(0, |l| l.len());
    assert_eq!(lines, 10, "{}", out["projection"]["data"]["bom_tables"]);
    assert!(
        layout["findings"].as_array().is_some_and(|f| f.is_empty()),
        "{}",
        layout["findings"]
    );
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in layout["primitives"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["id"] == "table:parts")
    {
        let s = &p["shape"];
        let pts: Vec<f64> = match s["type"].as_str() {
            Some("rect") => {
                let (x, y, w, h) = (
                    s["x"].as_f64().unwrap(),
                    s["y"].as_f64().unwrap(),
                    s["w"].as_f64().unwrap(),
                    s["h"].as_f64().unwrap(),
                );
                vec![x, y, x + w, y + h]
            }
            Some("polyline") => s["points"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(Value::as_f64)
                .collect(),
            _ => vec![],
        };
        for c in pts.chunks(2) {
            b = [
                b[0].min(c[0]),
                b[1].min(c[1]),
                b[2].max(c[0]),
                b[3].max(c[1]),
            ];
        }
    }
    // A4: the frame's right edge is at 287 mm and the mark reaches 5 mm in at 105 mm.
    assert!(
        b[1] < 105.0 && b[3] > 105.0,
        "the table does not reach the mark's height: {b:?}"
    );
    assert!(
        b[2] <= 287.0 - 5.0 - 1.0,
        "the table runs into the centring mark: {b:?}"
    );
    assert!(
        b[2] >= 287.0 - 5.0 - 4.0,
        "the table left the title block's corner: {b:?}"
    );
}

#[test]
fn a_parts_list_groups_one_part_number_and_fills_its_description() {
    let rig = "runmat 0.7\nstd 1\n\nassembly rig {\n    use part bracket as a\n    use part bracket as b {\n        placement = { placement = offset, offset = { translation = [100 mm, 0 mm, 0 mm] } }\n    }\n    use part bracket as c {\n        configuration = { select = inputs, values = { width = 90 mm } }\n        placement = { placement = offset, offset = { translation = [0 mm, 100 mm, 0 mm] } }\n    }\n    use part bracket as d {\n        configuration = { select = inputs, values = { width = 70 mm } }\n        placement = { placement = offset, offset = { translation = [100 mm, 100 mm, 0 mm] } }\n    }\n}\n";
    let drawing = "runmat 0.7\nstd 1\n\ndrawing rig {\n    format a3 = { size = { size = iso, name = A3 }, title_block = [] }\n    sheet \"1\" = { format = a3, scale = [1, 5], projection = third_angle }\n    view iso = standard(orientation = isometric, sheet = \"1\", model = { assembly = rig }, at = [150, 150], display = visible_lines)\n    annotation b1 = balloon(occurrence = [{ name = c }], table = parts, view = iso)\n    table parts = bom(view = iso, sheet = \"1\", at = [0, 0])\n}\n";
    let files: Files = [
        ("/p/bracket.part".to_owned(), BRACKET.to_owned()),
        ("/p/rig.assembly".to_owned(), rig.to_owned()),
    ]
    .into();
    let out: Value =
        serde_json::from_str(&compile_json("/p/rig.drawing", drawing, &files)).unwrap();
    let table = &out["projection"]["data"]["bom_tables"]["parts"];
    let lines = table["lines"]
        .as_array()
        .unwrap_or_else(|| panic!("{}", out["diagnostics"]));
    assert_eq!(lines.len(), 1, "{table}");
    assert_eq!(lines[0]["quantity"], 4, "{table}");
    assert_eq!(
        lines[0]["cells"],
        serde_json::json!(["1", "BR-100", "Bracket", "Aluminum", "4"]),
        "{table}"
    );
    assert!(
        table["diagnostics"]["diagnostics"]
            .as_array()
            .is_none_or(|d| d.is_empty()),
        "{table}"
    );
    // The balloon on the widened bracket reads the line's number.
    let layout = &out["projection"]["data"]["layouts"]["1"];
    assert!(
        layout["findings"].as_array().is_some_and(|f| f.is_empty()),
        "{}",
        layout["findings"]
    );
}

/// A section's plane can be an origin plane: the cutting line crosses the parent view through
/// the model's origin. A face reference there still works, and a datum feature is an error that
/// says what a section takes.
#[test]
fn a_section_cuts_through_an_origin_plane() {
    let files: Files = [("/p/bracket.part".to_owned(), BRACKET.to_owned())].into();
    let with = |plane: &str| {
        let text = DRAWING.replacen(
            "\n}\n",
            &format!("\n    view a_a = section(from = top, plane = {plane}, label = A, sheet = \"1\", model = {{ part = bracket }}, at = [250, 200], display = visible_lines)\n}}\n"),
            1,
        );
        let out: Value =
            serde_json::from_str(&compile_json("/p/bracket.drawing", &text, &files)).unwrap();
        out
    };
    // The compile's own problems, and the drawing check's (in the projection's data).
    let findings = |out: &Value| {
        let mut all: Vec<Value> = Vec::new();
        for list in [
            &out["diagnostics"],
            &out["projection"]["data"]["diagnostics"],
        ] {
            all.extend(list.as_array().into_iter().flatten().cloned());
        }
        all
    };
    let out = with("origin.zx");
    assert_eq!(findings(&out), Vec::<Value>::new());
    let cut = &out["projection"]["data"]["hlr"]["a_a"]["cut"];
    assert_eq!(cut["view"], "top", "{cut}");
    assert!(
        cut["at"].as_f64().is_some_and(|at| at.abs() < 1e-6),
        "{cut}"
    );
    let out = with("plane.nowhere");
    let messages = Value::Array(findings(&out)).to_string();
    assert!(
        messages.contains("a section cuts through a face or an origin plane"),
        "{messages}"
    );
}

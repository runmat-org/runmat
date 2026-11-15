//! Annotate on a part, command by command on a plate with a hole: each annotation reads back,
//! shows in the model where its geometry is, edits in place, follows a rebuild and deletes
//! with the PMI it shows.

use crate::edit::{edit, EditRequest};
use crate::runtime::{compile_part, Files};
use runmat_cad::drawing::AnnotationKind;
use runmat_cad::syntax::parse_part;
use serde_json::{json, Value};

const PATH: &str = "/p/plate.part";
const PLATE: &str = "runmat 0.7\nstd 1\n\npart plate {\n    input width : length = 80 mm\n\n    sketch s1 on origin.xy {\n        l1 = line(start = [0 mm, 0 mm], end = [input.width, 0 mm])\n        l2 = line(start = [input.width, 0 mm], end = [input.width, 50 mm])\n        l3 = line(start = [input.width, 50 mm], end = [0 mm, 50 mm])\n        l4 = line(start = [0 mm, 50 mm], end = [0 mm, 0 mm])\n    }\n    feature body = extrude(profile = sketch.s1, distance = 10 mm)\n    feature h = hole(\n        face = feature.body.face.end,\n        at = { a = point(60 mm, 20 mm, 10 mm) },\n        diameter = 8 mm,\n        through_all\n    )\n}\n";

const TOP: &str = "feature.body.face.end";
const BOTTOM: &str = "feature.body.face.start";
const LEFT: &str = "feature.body.face.side(s1.l4)";
const RIGHT: &str = "feature.body.face.side(s1.l2)";
const NEAR: &str = "feature.body.face.side(s1.l1)";
const HOLE: &str = "feature.h.face.wall(a)";
const EDGE: &str = "edges_between(feature.body.face.side(s1.l1), feature.body.face.end)";

fn try_run(text: &str, request: Value) -> Result<(String, String), String> {
    let request: EditRequest = serde_json::from_value(request).unwrap();
    let out = edit(PATH, text, &Files::new(), &request).map_err(|e| e.to_string())?;
    let file = out.files.into_iter().next().expect("a changed file");
    // The written text parses and keeps every annotation's id unique.
    assert!(
        parse_part(PATH, &file.text).document.is_some(),
        "{}",
        file.text
    );
    Ok((file.text, out.summary))
}

fn run(text: &str, request: Value) -> String {
    let command = request["command"].clone();
    try_run(text, request)
        .unwrap_or_else(|e| panic!("{command}: {e}\n{text}"))
        .0
}

fn shown(text: &str) -> Vec<runmat_cad::display::AnnotationDisplay> {
    compile_part(PATH, text)
        .expect("compiles")
        .display
        .annotations
}

fn rows(text: &str) -> Vec<runmat_cad::index::AnnotationRow> {
    compile_part(PATH, text)
        .expect("compiles")
        .index
        .annotations
}

fn errors(text: &str) -> Vec<String> {
    compile_part(PATH, text)
        .expect("compiles")
        .diagnostics
        .iter()
        .filter(|d| d.severity == "error")
        .map(|d| d.message.clone())
        .collect()
}

fn document(text: &str) -> runmat_cad::model::PartDocument {
    parse_part(PATH, text).document.expect("parses")
}

#[test]
fn every_annotate_command_writes_pmi_and_an_annotation_the_model_shows() {
    let mut t = PLATE.to_owned();
    t = run(
        &t,
        json!({ "command": "annotate.datum", "values": { "label": "a", "target": [BOTTOM] } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.datum", "values": { "label": "B", "target": [LEFT] } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.gdt", "values": { "characteristic": "position", "tolerance": 0.2, "diametral": true, "modifier": "mmc", "datums": "A|B", "targets": [HOLE] } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.dimension", "values": { "targets": [HOLE], "tolerance": "fit", "fit": "H7" } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.dimension", "values": { "targets": [LEFT, RIGHT], "tolerance": "bilateral", "plus": 0.1, "minus": 0.05 } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.surface-finish", "values": { "faces": [TOP], "ra": 0.8, "removal": "removal_required", "method": "ground", "lay": "=" } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.coating", "values": { "spec": "anodize type II, black" } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.weld-symbol", "values": { "edges": [EDGE], "type": "fillet", "size": 5, "side": "both", "all_around": true, "intermittent": true, "length": 30, "pitch": 60, "tail": "GMAW" } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.note", "values": { "text": "Break sharp edges\n0.2 max", "anchor": [NEAR] } }),
    );
    assert!(errors(&t).is_empty(), "{:?}\n{t}", errors(&t));
    let d = document(&t);
    assert_eq!(d.pmi.datums.len(), 2);
    assert_eq!(d.pmi.datums[0].label, "A", "labels are upper case");
    assert_eq!(d.pmi.tolerances.len(), 1);
    assert_eq!(d.pmi.sizes.len(), 1);
    assert_eq!(d.pmi.locations.len(), 1);
    assert_eq!(d.pmi.finishes.len(), 1);
    assert_eq!(d.pmi.treatments.len(), 1);
    assert_eq!(d.pmi.annotations.len(), 9, "{t}");
    // The file says it in the drawing's statement form.
    assert!(
        t.contains("annotation gdt1 = gdt(tolerance = position1)"),
        "{t}"
    );
    assert!(
        t.contains("annotation datum1 = datum_symbol(label = A)"),
        "{t}"
    );
    let shown = shown(&t);
    assert_eq!(shown.len(), 9);
    assert!(shown.iter().all(|a| a.ok), "{shown:#?}");
    let cells = |id: &str| shown.iter().find(|a| a.id == id).unwrap().cells.clone();
    assert_eq!(cells("gdt1"), ["⌖", "⌀0.2 Ⓜ", "A", "B"]);
    assert_eq!(cells("dim1"), ["⌀8 H7"]);
    assert_eq!(cells("dim2"), ["80 +0.1/-0.05"]);
    assert_eq!(cells("finish1")[..3], ["required", "Ra 0.8", "ground"]);
    assert_eq!(cells("coating1"), ["anodize type II, black"]);
    assert_eq!(
        cells("weld1"),
        ["fillet", "5", "both", "all_around", "", "30-60", "GMAW"]
    );
    assert_eq!(cells("note1"), ["Break sharp edges", "0.2 max"]);
    // A hole's frame hangs off the hole's axis (60, 20); the near side's note off y = 0.
    let gdt = shown.iter().find(|a| a.id == "gdt1").unwrap();
    assert!(
        (gdt.anchor[0] - 0.060).abs() < 1e-4 && (gdt.anchor[1] - 0.020).abs() < 1e-4,
        "{gdt:?}"
    );
    let note = shown.iter().find(|a| a.id == "note1").unwrap();
    assert!(note.anchor[1].abs() < 1e-4, "{note:?}");
    assert!(shown.iter().find(|a| a.id == "dim2").unwrap().to.is_some());
    // Rows carry each annotation's command and values for Properties.
    let rows = rows(&t);
    let row = rows.iter().find(|r| r.id == "gdt1").unwrap();
    assert_eq!(row.command, "annotate.gdt");
    assert_eq!(row.label, "Position");
    assert_eq!(row.values["datums"], "A|B");
    assert_eq!(
        rows.iter().find(|r| r.id == "finish1").unwrap().values["ra"],
        0.8
    );
}

#[test]
fn weld_symbols_of_each_type_go_on_each_picked_edge() {
    let other = "edges_between(feature.body.face.side(s1.l3), feature.body.face.end)";
    for kind in ["fillet", "groove", "plug", "spot"] {
        let t = run(
            PLATE,
            json!({ "command": "annotate.weld-symbol", "values": { "edges": [EDGE, other], "type": kind, "size": 3 } }),
        );
        let d = document(&t);
        assert_eq!(d.pmi.annotations.len(), 2, "{kind}: one per edge");
        assert!(
            matches!(&d.pmi.annotations[1].kind, AnnotationKind::Weld { symbol, .. } if symbol == kind)
        );
        assert!(shown(&t).iter().all(|a| a.ok), "{kind}");
    }
    let refused = try_run(
        PLATE,
        json!({ "command": "annotate.weld-symbol", "values": { "edges": [EDGE], "type": "spot", "side": "both" } }),
    );
    assert!(refused.unwrap_err().contains("one side"));
}

#[test]
fn an_annotation_edits_in_place_moves_and_deletes_with_its_pmi() {
    let mut t = run(
        PLATE,
        json!({ "command": "annotate.datum", "values": { "label": "A", "target": [BOTTOM] } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.gdt", "values": { "characteristic": "perpendicularity", "tolerance": 0.05, "datums": "A", "targets": [HOLE] } }),
    );
    // Properties: one field changes; the frame keeps its id and tolerance name.
    t = run(
        &t,
        json!({ "command": "part.set-property", "selection": ["annotation:gdt1"], "values": { "tolerance": 0.02 } }),
    );
    let d = document(&t);
    assert_eq!(d.pmi.tolerances[0].name, "perpendicularity1");
    assert_eq!(
        runmat_cad::units::length_mm(&d.pmi.tolerances[0].tolerance),
        Some(0.02)
    );
    assert_eq!(
        d.pmi.annotations[1].id, "gdt1",
        "the edited frame keeps its place"
    );
    // The label dragged in the view.
    t = run(
        &t,
        json!({ "command": "part.set-property", "selection": ["annotation:gdt1"], "values": { "at": [0, 0, 25] } }),
    );
    assert_eq!(document(&t).pmi.annotations[1].at, Some([0.0, 0.0, 25.0]));
    let gdt = shown(&t).into_iter().find(|a| a.id == "gdt1").unwrap();
    assert!((gdt.at[2] - gdt.anchor[2] - 0.025).abs() < 1e-5, "{gdt:?}");
    // The datum a frame uses stays until the frame goes.
    let refused = try_run(
        &t,
        json!({ "command": "part.delete", "selection": ["annotation:datum1"] }),
    );
    assert!(refused.unwrap_err().contains("used by perpendicularity1"));
    t = run(
        &t,
        json!({ "command": "part.delete", "selection": ["annotation:gdt1"] }),
    );
    t = run(
        &t,
        json!({ "command": "part.delete", "selection": ["annotation:datum1"] }),
    );
    assert!(document(&t).pmi.is_empty(), "{t}");
    // A form tolerance takes no datums; a location tolerance needs them.
    let flat = try_run(
        PLATE,
        json!({ "command": "annotate.gdt", "values": { "characteristic": "flatness", "tolerance": 0.05, "datums": "A", "targets": [TOP] } }),
    );
    assert!(flat.unwrap_err().contains("takes no datums"));
    let pos = try_run(
        PLATE,
        json!({ "command": "annotate.gdt", "values": { "characteristic": "position", "tolerance": 0.1, "targets": [HOLE] } }),
    );
    assert!(pos.unwrap_err().contains("datum"));
}

#[test]
fn annotations_follow_the_geometry_on_rebuild() {
    let t = run(
        PLATE,
        json!({ "command": "annotate.surface-finish", "values": { "faces": [RIGHT], "ra": 1.6 } }),
    );
    let before = shown(&t)[0].anchor;
    let wider = t.replace(
        "input width : length = 80 mm",
        "input width : length = 120 mm",
    );
    let after = shown(&wider)[0].anchor;
    assert!(
        (after[0] - before[0] - 0.040).abs() < 1e-5,
        "{before:?} → {after:?}"
    );
}

#[test]
fn auto_dimension_schemes_dimension_each_feature_once() {
    let t = run(
        PLATE,
        json!({ "command": "annotate.auto-dimension", "values": { "scheme": "plus_minus", "tolerance": 0.1 } }),
    );
    let d = document(&t);
    assert_eq!(
        d.pmi.datums.len(),
        3,
        "A, B and C inferred from the largest faces\n{t}"
    );
    assert!(
        d.pmi
            .sizes
            .iter()
            .any(|s| s.on[0].to_string().contains("wall")),
        "{t}"
    );
    assert!(!d.pmi.locations.is_empty(), "{t}");
    assert!(errors(&t).is_empty(), "{:?}", errors(&t));
    assert!(shown(&t).iter().all(|a| a.ok));
    let again = try_run(
        &t,
        json!({ "command": "annotate.auto-dimension", "values": { "scheme": "plus_minus" } }),
    );
    assert!(again.unwrap_err().contains("dimensioned already"));
    let g = run(
        PLATE,
        json!({ "command": "annotate.auto-dimension", "values": { "scheme": "geometric", "datums": [BOTTOM, LEFT, NEAR] } }),
    );
    let d = document(&g);
    assert_eq!(
        d.pmi
            .datums
            .iter()
            .map(|d| d.label.as_str())
            .collect::<Vec<_>>(),
        ["A", "B", "C"]
    );
    assert!(d
        .pmi
        .tolerances
        .iter()
        .any(|t| t.characteristic == runmat_cad::pmi::Characteristic::Position));
    assert!(d
        .pmi
        .tolerances
        .iter()
        .any(|t| t.characteristic == runmat_cad::pmi::Characteristic::Flatness));
    assert!(errors(&g).is_empty(), "{:?}", errors(&g));
}

#[test]
fn annotation_views_group_annotations_and_take_the_canvas_camera() {
    let mut t = run(
        PLATE,
        json!({ "command": "annotate.note", "values": { "text": "A" } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.note", "values": { "text": "B" } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.annotation-view", "values": { "name": "Front PMI", "orientation": "front", "annotations": ["annotation:note1"] } }),
    );
    let d = document(&t);
    assert_eq!(d.pmi.views[0].name, "front_pmi");
    assert_eq!(d.pmi.views[0].from, [0.0, -1.0, 0.0]);
    assert_eq!(d.pmi.annotations[0].view.as_deref(), Some("front_pmi"));
    assert_eq!(d.pmi.annotations[1].view, None);
    let current = try_run(
        &t,
        json!({ "command": "annotate.annotation-view", "values": { "name": "mine", "orientation": "current" } }),
    );
    assert!(current.unwrap_err().contains("canvas"));
    t = run(
        &t,
        json!({ "command": "annotate.annotation-view", "values": { "name": "mine", "orientation": "current", "camera": { "from": [1, 2, 3], "up": [0, 0, 1] } } }),
    );
    let d = document(&t);
    assert_eq!(d.pmi.views[1].from, [1.0, 2.0, 3.0]);
    assert_eq!(
        d.pmi.annotations[1].view.as_deref(),
        Some("mine"),
        "the rest go in"
    );
    assert!(errors(&t).is_empty());
    // Deleting a view keeps its annotations, in no view.
    t = run(
        &t,
        json!({ "command": "part.delete", "selection": ["annotation_view:front_pmi"] }),
    );
    let d = document(&t);
    assert_eq!(d.pmi.views.len(), 1);
    assert_eq!(d.pmi.annotations[0].view, None);
    t = run(
        &t,
        json!({ "command": "part.set-property", "selection": ["annotation_view:mine"], "values": { "orientation": "top" } }),
    );
    assert_eq!(document(&t).pmi.views[0].from, [0.0, 0.0, 1.0]);
}

#[test]
fn a_face_annotation_sits_on_the_face_clear_of_its_hole() {
    // The top's mesh crowds round the hole at (60, 20); the datum still lands on the top itself.
    let t = run(
        PLATE,
        json!({ "command": "annotate.datum", "values": { "label": "A", "target": [TOP] } }),
    );
    let a = shown(&t)[0].anchor;
    assert!((a[2] - 0.010).abs() < 1e-5, "{a:?}");
    let from_hole = ((a[0] - 0.060).powi(2) + (a[1] - 0.020).powi(2)).sqrt();
    assert!(from_hole > 0.004 + 1e-4, "{a:?} is in the hole");
    // Near the face's area center (about (39.8, 25.0) mm), not pulled toward the hole's
    // crowded mesh.
    let from_center = ((a[0] - 0.0398).powi(2) + (a[1] - 0.025).powi(2)).sqrt();
    assert!(from_center < 0.008, "{a:?} is off the face's center");
    assert!(
        a[0] > 0.0 && a[0] < 0.080 && a[1] > 0.0 && a[1] < 0.050,
        "{a:?}"
    );
}

#[test]
fn a_dimension_to_a_face_runs_along_the_face_normal_and_labels_keep_apart() {
    // The hole's location from the left side: the line runs along x, from the side to the axis.
    let mut t = run(
        PLATE,
        json!({ "command": "annotate.dimension", "values": { "targets": [LEFT, HOLE] } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.dimension", "values": { "targets": [LEFT, RIGHT] } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.datum", "values": { "label": "A", "target": [BOTTOM] } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.datum", "values": { "label": "B", "target": [TOP] } }),
    );
    let all = shown(&t);
    let location = &all[0];
    let to = location.to.expect("a dimension has two ends");
    assert!(
        location.anchor[0].abs() < 1e-5 && (to[0] - 0.060).abs() < 1e-4,
        "{location:?}"
    );
    assert!(
        (location.anchor[1] - to[1]).abs() < 1e-5 && (location.anchor[2] - to[2]).abs() < 1e-5,
        "{location:?}"
    );
    let width = &all[1];
    let to = width.to.expect("a dimension has two ends");
    assert!(
        (to[0] - width.anchor[0] - 0.080).abs() < 1e-5 && (to[1] - width.anchor[1]).abs() < 1e-5,
        "{width:?}"
    );
    // No two labels sit on one another seen from the top, the front or the side.
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            for skip in 0..3 {
                let d: f32 = (0..3)
                    .filter(|&k| k != skip)
                    .map(|k| (a.at[k] - b.at[k]).powi(2))
                    .sum::<f32>()
                    .sqrt();
                assert!(
                    d > 0.003,
                    "{} and {} overlap along axis {skip}: {:?} {:?}",
                    a.id,
                    b.id,
                    a.at,
                    b.at
                );
            }
        }
    }
}

#[test]
fn a_note_without_a_leader_and_a_whole_part_coating_are_free() {
    let mut t = run(
        PLATE,
        json!({ "command": "annotate.note", "values": { "text": "Inspect per plan" } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.coating", "values": { "spec": "anodize type II, black" } }),
    );
    t = run(
        &t,
        json!({ "command": "annotate.note", "values": { "text": "Deburr", "anchor": [NEAR] } }),
    );
    let free: Vec<bool> = shown(&t).iter().map(|a| a.free).collect();
    assert_eq!(free, [true, true, false]);
}

#[test]
fn a_weld_symbol_points_at_the_middle_of_its_edge() {
    // The near top edge runs from (0, 0, 10) to (80, 0, 10) mm.
    let t = run(
        PLATE,
        json!({ "command": "annotate.weld-symbol", "values": { "edges": [EDGE], "type": "fillet", "size": 5 } }),
    );
    let a = shown(&t)[0].anchor;
    assert!(
        (a[0] - 0.040).abs() < 1e-4 && a[1].abs() < 1e-5 && (a[2] - 0.010).abs() < 1e-5,
        "{a:?}"
    );
    let swapped = "edges_between(feature.body.face.end, feature.body.face.side(s1.l1))";
    let t = run(
        PLATE,
        json!({ "command": "annotate.weld-symbol", "values": { "edges": [swapped], "type": "fillet", "size": 5 } }),
    );
    let a = shown(&t)[0].anchor;
    assert!((a[0] - 0.040).abs() < 1e-4, "{a:?}");
}

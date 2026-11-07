//! Properties edits on a circuit's items (`circuit.set-property`): the Desktop sends one field and
//! its value with the item's Outline id, and the runtime writes the item's own source: the
//! component statement, the footprint, the net statement, or the track, via or pour in the
//! layout. Fields that are only shown are refused.

use runmat_circuit::runtime::{CompileOptions, EditError, EditOutput, EditRequest, Files, Session};
use serde_json::{json, Value};

const CIRCUIT: &str = "/p/io.circuit";
const LAYOUT: &str = "/p/io.circuit.layout";

fn files() -> Files {
    ["io.circuit", "io.circuit.layout", "io.circuit.lock"]
        .iter()
        .map(|name| {
            let text = std::fs::read_to_string(format!("{}/tests/projects/io/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap();
            (format!("/p/{name}"), text)
        })
        .collect()
}

fn set(files: &Files, item: &str, field: &str, value: Value, options: CompileOptions) -> Result<EditOutput, EditError> {
    let mut session = Session::new();
    let out = session.compile(CIRCUIT, &files[CIRCUIT], files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let request = EditRequest {
        command: "circuit.set-property".into(),
        values: [(field.to_owned(), value)].into_iter().collect(),
        selection: vec![item.to_owned()],
        options,
    };
    session.edit(CIRCUIT, &files[CIRCUIT], files, &request)
}

fn ok(files: &Files, item: &str, field: &str, value: Value) -> Files {
    let out = set(files, item, field, value, CompileOptions::default()).unwrap_or_else(|e| panic!("{item} {field}: {}", e.message));
    assert!(!out.files.is_empty(), "{item} {field} changed nothing: {}", out.summary);
    let mut next = files.clone();
    for f in out.files {
        next.insert(f.path, f.text);
    }
    // The edited project still compiles clean.
    let mut session = Session::new();
    let compiled = session.compile(CIRCUIT, &next[CIRCUIT], &next, &CompileOptions::default());
    assert!(compiled.diagnostics.is_empty(), "{item} {field}: {:?}", compiled.diagnostics);
    next
}

fn line<'a>(text: &'a str, start: &str) -> &'a str {
    text.lines().map(str::trim).find(|l| l.starts_with(start)).unwrap_or_else(|| panic!("no line starting `{start}`"))
}

#[test]
fn a_components_value_and_part_are_written_in_its_statement() {
    let files = files();
    let next = ok(&files, "component:C900", "value", json!("4.7 uF ± 10 %"));
    let statement = line(&next[CIRCUIT], "component C900");
    assert!(statement.contains("part = \"C0805-10U\""), "{statement}");
    assert!(statement.contains("value = 4.7 uF"), "{statement}");
    // Only that statement changed.
    assert_eq!(next[CIRCUIT].lines().count(), files[CIRCUIT].lines().count());
    let next = ok(&files, "component:C900", "part", json!("C0402-100N"));
    assert!(line(&next[CIRCUIT], "component C900").contains("part = \"C0402-100N\""));
}

#[test]
fn a_new_designator_renames_the_statement_and_keeps_the_board_name() {
    let files = files();
    let next = ok(&files, "component:C900", "designator", json!("C901"));
    let statement = line(&next[CIRCUIT], "component C901");
    assert!(statement.contains("renamed_from = C900"), "{statement}");
    assert!(!next[CIRCUIT].contains("component C900 "));
    let e = set(&files, "component:C900", "designator", json!("J1"), CompileOptions::default()).expect_err("J1 is taken");
    assert_eq!(e.message, "J1 is already in the design");
}

#[test]
fn placement_fields_move_lock_and_flip_the_footprint() {
    let files = files();
    let next = ok(&files, "component:C900", "x", json!(90));
    assert!(line(&next[LAYOUT], "footprint C900").contains("at = [90, 24]"), "{}", line(&next[LAYOUT], "footprint C900"));
    let next = ok(&files, "component:C900", "y", json!("30 mm"));
    assert!(line(&next[LAYOUT], "footprint C900").contains("at = [97, 30]"), "{}", line(&next[LAYOUT], "footprint C900"));
    let next = ok(&files, "component:C900", "locked", json!(true));
    assert!(line(&next[LAYOUT], "footprint C900").contains("locked"), "{}", line(&next[LAYOUT], "footprint C900"));
    let next = ok(&files, "component:C900", "side", json!("bottom"));
    assert_ne!(line(&next[LAYOUT], "footprint C900"), line(&files[LAYOUT], "footprint C900"));
    let same = set(&files, "component:C900", "side", json!("top"), CompileOptions::default()).unwrap();
    assert!(same.files.is_empty(), "already on top: {}", same.summary);
}

#[test]
fn a_tracks_width_and_a_vias_sizes_are_written_on_that_item_only() {
    let files = files();
    let parsed = runmat_circuit::syntax::board::parse_board(LAYOUT, &files[LAYOUT]);
    let board = parsed.board.expect("the io layout reads");
    let track = *board.tracks.keys().next().expect("a track");
    let widths = |text: &str| text.lines().filter(|l| l.trim_start().starts_with("track ") && l.contains("width = 0.5,")).count();
    let next = ok(&files, &format!("track:{}", track.0), "width", json!("0.5 mm"));
    assert_eq!(widths(&next[LAYOUT]), widths(&files[LAYOUT]) + 1);
    let e = set(&files, &format!("track:{}", track.0), "width", json!(0), CompileOptions::default()).expect_err("zero width");
    assert_eq!(e.message, "Width must be more than zero");
    if let Some(via) = board.vias.keys().next() {
        let e = set(&files, &format!("via:{}", via.0), "drill", json!("5 mm"), CompileOptions::default()).expect_err("drill past the pad");
        assert_eq!(e.message, "the via's drill must be smaller than its diameter");
    }
}

#[test]
fn a_nets_class_is_written_in_its_net_statement() {
    let files = files();
    let next = ok(&files, "net:SHIELD", "class", json!("analog"));
    assert!(line(&next[CIRCUIT], "net SHIELD").contains("class = analog"), "{}", line(&next[CIRCUIT], "net SHIELD"));
    let e = set(&files, "net:SHIELD", "class", json!("nonesuch"), CompileOptions::default()).expect_err("no such class");
    assert!(e.message.starts_with("no net class `nonesuch`"), "{}", e.message);
}

#[test]
fn a_global_rail_is_not_renamed_from_properties() {
    let e = set(&files(), "net:GND", "name", json!("GND2"), CompileOptions::default()).expect_err("a rail");
    assert!(e.message.contains("global rail"), "{}", e.message);
}

#[test]
fn fitting_needs_the_variant_and_shown_fields_are_refused() {
    let files = files();
    let e = set(&files, "component:C900", "fitting", json!("not_fitted"), CompileOptions::default()).expect_err("no variant");
    assert_eq!(e.message, "choose the variant first; fitting is set per variant");
    let lite = CompileOptions { variant: Some("lite".into()), ..CompileOptions::default() };
    let out = set(&files, "component:C900", "fitting", json!("not_fitted"), lite).unwrap_or_else(|e| panic!("{}", e.message));
    let text = &out.files.iter().find(|f| f.path == CIRCUIT).expect("the circuit").text;
    assert!(line(text, "component C900").contains("not_fitted"), "{}", line(text, "component C900"));
    let e = set(&files, "component:C900", "stock", json!("12"), CompileOptions::default()).expect_err("shown only");
    assert_eq!(e.message, "stock is shown, not set, on this item");
    let e = set(&files, "component:C900", "value", json!("4.7 uF"), CompileOptions::default()).and_then(|_| set(&files, "component:C900", "value", json!(""), CompileOptions::default())).expect_err("empty");
    assert_eq!(e.message, "Value can't be empty");
}

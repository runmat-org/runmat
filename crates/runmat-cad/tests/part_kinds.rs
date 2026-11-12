//! The part-kind classifier (`index::part_file_state`), the one source the Guide tab, the
//! agent's `design.overview` and the panel's suggestions read: each kind from a part of that
//! kind. The piping, electrical and framing parts are excerpts of the house example's.

use runmat_cad::index::{part_file_state, FileState};
use runmat_cad::syntax::parse_part;

fn state_of(path: &str, text: &str) -> FileState {
    let parsed = parse_part(path, text);
    let doc = parsed
        .document
        .unwrap_or_else(|| panic!("{path} doesn't parse: {:#?}", parsed.diagnostics));
    part_file_state(&doc)
}

fn fixture(name: &str) -> FileState {
    let path = format!("{}/tests/fixtures/part_kinds/{name}", env!("CARGO_MANIFEST_DIR"));
    state_of(name, &std::fs::read_to_string(&path).expect(&path))
}

/// A corpus document, read as `as_path` (the file a part lives in is named after the part).
fn corpus(name: &str, as_path: &str) -> FileState {
    let path = format!("{}/tests/corpus/documents/{name}", env!("CARGO_MANIFEST_DIR"));
    state_of(as_path, &std::fs::read_to_string(&path).expect(&path))
}

#[test]
fn a_pipe_run_is_piping() {
    // One sweep in PEX, and a PEX manifold and a device box beside it.
    assert_eq!(fixture("supply_piping.part"), FileState::Piping);
}

#[test]
fn three_sweeps_are_runs_whatever_their_material() {
    let text = r#"runmat 0.7
std 1

part ducts {
    sketch p on origin.xy {
        c = circle(center = [0 mm, 0 mm], radius = 5 mm)
    }
    sketch a on origin.zx {
        l = line(start = [0 mm, 0 mm], end = [0 mm, 100 mm])
    }
    feature r1 = sweep(profile = sketch.p, path = sketch.a)
    feature r2 = sweep(profile = sketch.p, path = sketch.a)
    feature r3 = sweep(profile = sketch.p, path = sketch.a)
}
"#;
    assert_eq!(state_of("ducts.part", text), FileState::Piping);
}

#[test]
fn boxes_and_devices_are_electrical() {
    assert_eq!(fixture("electrical_devices.part"), FileState::Electrical);
}

#[test]
fn lumber_plates_are_structural() {
    assert_eq!(fixture("wall_framing.part"), FileState::Structural);
}

#[test]
fn a_machined_bracket_is_a_written_part() {
    assert_eq!(corpus("bracket.part", "bracket.part"), FileState::Written);
}

#[test]
fn a_part_with_sheet_metal_features_is_sheet_metal() {
    assert_eq!(corpus("part01.part", "bracket_sm.part"), FileState::SheetMetal);
    assert_eq!(corpus("part08.part", "bracket_sm.part"), FileState::SheetMetal);
}

#[test]
fn datums_alone_are_a_reference_skeleton() {
    let text = r#"runmat 0.7
std 1

part skeleton {
    feature deck = plane.offset(base = origin.xy, distance = 40 in)
    feature wall = plane.offset(base = origin.yz, distance = 120 in)
}
"#;
    assert_eq!(state_of("skeleton.part", text), FileState::Reference);
}

#[test]
fn a_new_part_is_empty_and_an_input_makes_it_written() {
    // What the Files menu writes for a new part.
    assert_eq!(state_of("new.part", "part new {\n}\n"), FileState::Empty);
    let text = "part new {\n    input width : length = 80 mm\n}\n";
    assert_eq!(state_of("new.part", text), FileState::Written);
}

#[test]
fn file_states_serialize_as_the_catalog_names_them() {
    let names: Vec<String> = FileState::ALL
        .iter()
        .map(|s| serde_json::to_string(s).unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "\"empty\"",
            "\"written\"",
            "\"sheet_metal\"",
            "\"reference\"",
            "\"piping\"",
            "\"electrical\"",
            "\"structural\""
        ]
    );
}

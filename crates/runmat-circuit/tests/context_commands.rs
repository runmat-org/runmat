//! The right-click commands on a circuit's components and nets, run through the shared
//! dispatcher from the schematic and from the board: Edit value, Rename (a net) and Delete.

use runmat_circuit::runtime::{CompileOptions, EditError, EditOutput, EditRequest, Files, Session};
use serde_json::{json, Value};

const CIRCUIT: &str = "/p/io.circuit";
const LAYOUT: &str = "/p/io.circuit.layout";

fn files() -> Files {
    ["io.circuit", "io.circuit.layout", "io.circuit.lock"]
        .iter()
        .map(|name| {
            let text = std::fs::read_to_string(format!(
                "{}/tests/projects/io/{name}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap();
            (format!("/p/{name}"), text)
        })
        .collect()
}

fn run(
    from: &str,
    command: &str,
    selection: &[&str],
    values: Value,
) -> Result<EditOutput, EditError> {
    let files = files();
    let mut session = Session::new();
    let out = session.compile(from, &files[from], &files, &CompileOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    let request = EditRequest {
        command: command.into(),
        values: values.as_object().cloned().unwrap_or_default(),
        selection: selection.iter().map(|s| s.to_string()).collect(),
        options: CompileOptions::default(),
    };
    session.edit(from, &files[from], &files, &request)
}

/// The project after the edit, checked to compile clean.
fn applied(from: &str, command: &str, selection: &[&str], values: Value) -> (Files, String) {
    let out = run(from, command, selection, values)
        .unwrap_or_else(|e| panic!("{command}: {}", e.message));
    assert!(
        !out.files.is_empty(),
        "{command} changed nothing: {}",
        out.summary
    );
    let mut next = files();
    for f in out.files {
        next.insert(f.path, f.text);
    }
    let mut session = Session::new();
    let compiled = session.compile(CIRCUIT, &next[CIRCUIT], &next, &CompileOptions::default());
    assert!(
        compiled.diagnostics.is_empty(),
        "{command}: {:?}",
        compiled.diagnostics
    );
    (next, out.summary)
}

fn has_line(text: &str, start: &str) -> bool {
    text.lines().any(|l| l.trim().starts_with(start))
}

#[test]
fn edit_value_writes_the_components_statement_from_either_view() {
    for from in [CIRCUIT, LAYOUT] {
        let (next, _) = applied(
            from,
            "circuit.edit-value",
            &["component:C900"],
            json!({ "value": "4.7 uF ± 10 %" }),
        );
        let line = next[CIRCUIT]
            .lines()
            .find(|l| l.trim().starts_with("component C900"))
            .unwrap();
        assert!(line.contains("4.7"), "{from}: {line}");
    }
    let e = run(
        CIRCUIT,
        "circuit.edit-value",
        &["net:SHIELD"],
        json!({ "value": "1 k" }),
    )
    .unwrap_err();
    assert_eq!(e.message, "select one component");
    let e = run(
        CIRCUIT,
        "circuit.edit-value",
        &["component:C900"],
        json!({ "value": " " }),
    )
    .unwrap_err();
    assert_eq!(e.message, "Value can't be empty");
}

#[test]
fn rename_net_renames_the_declared_net_and_refuses_a_rail() {
    for from in [CIRCUIT, LAYOUT] {
        let (next, _) = applied(
            from,
            "circuit.rename-net",
            &["net:SHIELD"],
            json!({ "name": "CHASSIS" }),
        );
        assert!(
            has_line(&next[CIRCUIT], "net CHASSIS"),
            "{from}: {}",
            next[CIRCUIT]
        );
        assert!(!has_line(&next[CIRCUIT], "net SHIELD"), "{from}");
    }
    let e = run(
        CIRCUIT,
        "circuit.rename-net",
        &["net:GND"],
        json!({ "name": "GND2" }),
    )
    .unwrap_err();
    assert!(e.message.contains("global rail"), "{}", e.message);
}

#[test]
fn delete_takes_a_component_off_the_schematic_and_the_board() {
    for from in [CIRCUIT, LAYOUT] {
        let (next, summary) = applied(from, "circuit.delete", &["component:C900"], json!({}));
        assert!(!next[CIRCUIT].contains("C900"), "{from}: {}", next[CIRCUIT]);
        assert!(!has_line(&next[LAYOUT], "footprint C900"), "{from}");
        assert_eq!(summary, "Deleted C900");
    }
    // A component of a repeated sheet is in every channel: one copy alone is refused.
    let e = run(CIRCUIT, "circuit.delete", &["component:C1"], json!({})).unwrap_err();
    assert!(e.message.contains("delete them together"), "{}", e.message);
    let e = run(CIRCUIT, "circuit.delete", &["net:SHIELD"], json!({})).unwrap_err();
    assert!(e.message.contains("can't be deleted here"), "{}", e.message);
}

#[test]
fn delete_takes_board_items_and_rooms_off_the_layout() {
    let parsed = runmat_circuit::syntax::board::parse_board(LAYOUT, &files()[LAYOUT]);
    let board = parsed.board.expect("the io layout reads");
    let track = *board.tracks.keys().next().expect("a track");
    let tracks = |text: &str| {
        text.lines()
            .filter(|l| l.trim_start().starts_with("track "))
            .count()
    };
    let (next, summary) = applied(
        LAYOUT,
        "circuit.delete",
        &[&format!("track:{}", track.0), "room:ch[0]"],
        json!({}),
    );
    assert_eq!(tracks(&next[LAYOUT]), tracks(&files()[LAYOUT]) - 1);
    assert!(!next[LAYOUT].contains("room \"ch[0]\""), "{}", next[LAYOUT]);
    assert!(next[LAYOUT].contains("room \"ch[1]\""));
    assert_eq!(summary, "Deleted 1 board item and room ch[0]");
    assert_eq!(
        next[CIRCUIT],
        files()[CIRCUIT],
        "the schematic is untouched"
    );
}

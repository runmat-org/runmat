//! The Desktop's `design-commands.json`: the CAD catalog merged with the circuit catalog.
//!
//! The Desktop checks it in at `desktop/src/generated/design-commands.json`. Set
//! `RUNMAT_DESIGN_COMMANDS_OUT` to that path to write it; with it unset, the test compares
//! against the sibling Desktop checkout when there is one.

use runmat_design_commands::{Catalog, EditorKind};

/// CAD plus circuit: the file the Desktop loads.
fn design_catalog() -> Catalog {
    let mut c = runmat_cad::commands::catalog();
    c.merge(runmat_circuit::commands::catalog());
    c
}

#[test]
fn merged_catalog_is_sound() {
    let c = design_catalog();
    let problems = c.lint();
    assert!(problems.is_empty(), "{problems:#?}");
    for editor in [
        EditorKind::Part,
        EditorKind::Assembly,
        EditorKind::Drawing,
        EditorKind::Circuit,
        EditorKind::CircuitLayout,
        EditorKind::Fea,
    ] {
        assert!(
            c.rows.iter().any(|r| r.editor == editor),
            "{} has rows",
            editor.as_str()
        );
    }
    // The shared commands are kept once and placed in every editor's rows that use them.
    let measure = c.command("view.measure").expect("shared measure");
    assert_eq!(
        c.commands.iter().filter(|x| x.id == "view.measure").count(),
        1
    );
    assert!(c
        .rows
        .iter()
        .filter(|r| r
            .view_cluster
            .iter()
            .any(|i| i.command_ids().contains(&"view.measure")))
        .all(|r| measure.editors.contains(&r.editor)));
}

#[test]
fn design_commands_json_is_current() {
    let json = design_catalog().to_json();
    assert_eq!(json, design_catalog().to_json(), "deterministic");
    if let Ok(out) = std::env::var("RUNMAT_DESIGN_COMMANDS_OUT") {
        std::fs::write(&out, &json).expect("write design-commands.json");
        return;
    }
    let sibling = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../runmat-private-cad/desktop/src/generated/design-commands.json");
    if let Ok(checked_in) = std::fs::read_to_string(&sibling) {
        assert!(
            checked_in == json,
            "{} is stale; regenerate with RUNMAT_DESIGN_COMMANDS_OUT={} cargo test -p runmat-design-commands --test design_catalog",
            sibling.display(),
            sibling.display()
        );
    }
}

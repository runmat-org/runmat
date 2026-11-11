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
    let sibling = std::env::var_os("RUNMAT_PRIVATE_CAD_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../runmat-private-cad")
        })
        .join("desktop/src/generated/design-commands.json");
    if let Ok(checked_in) = std::fs::read_to_string(&sibling) {
        assert!(
            checked_in == json,
            "{} is stale; regenerate with RUNMAT_DESIGN_COMMANDS_OUT={} cargo test -p runmat-design-commands --test design_catalog",
            sibling.display(),
            sibling.display()
        );
    }
}

/// `docs/design/commands.md`, the agent's list of commands with the fields each needs (the
/// design reference embeds it as `design/commands.md`). Set `RUNMAT_DESIGN_COMMANDS_DOC_OUT=1`
/// to rewrite it.
#[test]
fn design_commands_reference_is_current() {
    let doc = design_catalog().commands_reference();
    for line in [
        "- `part.sketch`: Start a sketch on a plane or planar face.",
        "  - Required: `plane` (pick: plane or face)",
        "## `board.` commands (`.circuit`, `.circuit.layout`)",
    ] {
        assert!(doc.contains(line), "{line}");
    }
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design/commands.md");
    if std::env::var_os("RUNMAT_DESIGN_COMMANDS_DOC_OUT").is_some() {
        std::fs::write(&path, &doc).expect("write commands.md");
        return;
    }
    let checked_in = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        checked_in == doc,
        "{} is stale; regenerate with RUNMAT_DESIGN_COMMANDS_DOC_OUT=1 cargo test -p runmat-design-commands --test design_catalog",
        path.display()
    );
}

/// Ticking Section view's Second plane cuts a corner at once: the second plane starts on a
/// plane other than the first plane's default.
#[test]
fn section_second_plane_starts_off_the_first_plane() {
    use runmat_design_commands::FieldKind;
    let c = design_catalog();
    let section = c.command("view.section").expect("view.section");
    let default_of = |id: &str| match &section.inputs.iter().find(|f| f.id == id).expect(id).kind {
        FieldKind::Choice { default, .. } => default.clone(),
        other => panic!("{id} is {other:?}"),
    };
    assert_eq!(default_of("plane").as_deref(), Some("front"));
    assert_eq!(default_of("second_plane").as_deref(), Some("top"));
}

/// The thread form offers only what the engine builds: cosmetic threads. A modelled (helical)
/// thread waits on a kernel helical cut, so no field offers one.
#[test]
fn thread_offers_no_modelled_form() {
    let c = design_catalog();
    let thread = c.command("part.thread").expect("part.thread");
    assert!(
        thread.inputs.iter().all(|f| f.id != "form"),
        "{:?}",
        thread.inputs.iter().map(|f| &f.id).collect::<Vec<_>>()
    );
    let json = c.to_json();
    assert!(
        !json.contains("\"modeled\""),
        "no command offers a modeled choice"
    );
}

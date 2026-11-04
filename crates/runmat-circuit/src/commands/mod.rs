//! Circuit editor commands (`.circuit`, `.circuit.layout`): schematic, board and 3D, per the UI
//! target §9, described with the design-command types of `runmat_cad::commands` so the Desktop
//! loads CAD and PCB commands from one file.
//!
//! [`design_catalog`] is the whole catalog the Desktop checks in as `design-commands.json`.

mod board;
mod items;
mod schematic;

use runmat_cad::commands::*;

/// Circuit commands, rows and item schemas.
pub fn catalog() -> Catalog {
    let mut c = Catalog::empty();
    schematic::add(&mut c);
    board::add(&mut c);
    items::add(&mut c);
    rows(&mut c);
    c.finish();
    c
}

/// CAD plus circuit: the file the Desktop loads.
pub fn design_catalog() -> Catalog {
    let mut c = runmat_cad::commands::catalog();
    c.merge(catalog());
    c
}

pub(crate) const CIRCUIT: &[EditorKind] = &[EditorKind::Circuit, EditorKind::CircuitLayout];

fn rows(c: &mut Catalog) {
    let tabs = vec![schematic::tab(), board::tab(), board::tab_3d()];
    let cluster = vec![
        c_item("circuit.variant"),
        variants(
            "circuit.select.menu",
            "Select",
            "select-filter",
            "circuit.select-filter",
            "filter",
            &[
                ("any", "Anything"),
                ("component", "Components"),
                ("wire", "Wires"),
                ("net", "Nets"),
                ("track", "Tracks"),
                ("via", "Vias"),
                ("pad", "Pads"),
                ("zone", "Pours"),
            ],
        ),
        variants(
            "circuit.display.menu",
            "Display",
            "display",
            "circuit.display",
            "layers",
            &[
                ("all", "All layers"),
                ("active", "Active layer only"),
                ("dim", "Dim other layers"),
            ],
        ),
        menu(
            "circuit.view.menu",
            "View",
            "fit",
            vec![
                e("view.fit"),
                e("view.zoom-selection"),
                e("circuit.flip-board"),
            ],
        ),
        c_item("view.measure"),
    ];
    for (editor, default_tab) in [
        (EditorKind::Circuit, "schematic"),
        (EditorKind::CircuitLayout, "board"),
    ] {
        c.rows.push(EditorRows {
            editor,
            mode: EditMode::Model,
            tabs: tabs.clone(),
            default_tab: default_tab.into(),
            view_cluster: cluster.clone(),
            shortcut_box: [
                "schematic.place-part",
                "schematic.place-wire",
                "schematic.net-label",
                "board.route",
                "board.place-via",
                "board.drc",
                "view.measure",
            ]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
            mode_actions: Vec::new(),
        });
    }
}

fn c_item(id: &str) -> RowItem {
    runmat_cad::commands::c(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_catalog_is_sound() {
        let c = design_catalog();
        let problems = c.lint();
        assert!(problems.is_empty(), "{problems:#?}");
        let rows = c
            .rows
            .iter()
            .find(|r| r.editor == EditorKind::Circuit)
            .unwrap();
        let labels: Vec<&str> = rows.tabs.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, ["Schematic", "Board", "3D"]);
        for id in [
            "schematic.place-wire",
            "schematic.place-part",
            "board.route",
            "board.place-via",
            "board.drc",
        ] {
            assert!(c.command(id).is_some(), "keymaps bind {id}");
        }
    }

    /// The Desktop checks the catalog in at `desktop/src/generated/design-commands.json`. Set
    /// `RUNMAT_DESIGN_COMMANDS_OUT` to that path to write it; with it unset, the test compares
    /// against the sibling Desktop checkout when there is one.
    #[test]
    fn design_commands_json_is_current() {
        let json = design_catalog().to_json();
        if let Ok(out) = std::env::var("RUNMAT_DESIGN_COMMANDS_OUT") {
            std::fs::write(&out, &json).expect("write design-commands.json");
            return;
        }
        let sibling = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../runmat-private-cad/desktop/src/generated/design-commands.json");
        if let Ok(checked_in) = std::fs::read_to_string(&sibling) {
            assert!(
                checked_in == json,
                "{} is stale; regenerate with RUNMAT_DESIGN_COMMANDS_OUT={} cargo test -p runmat-circuit design_commands_json",
                sibling.display(),
                sibling.display()
            );
        }
    }
}

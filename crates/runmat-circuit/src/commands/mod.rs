//! Circuit editor commands (`.circuit`, `.circuit.layout`): schematic, board and 3D, per the UI
//! target §9, described with the shared descriptor schema of `runmat_design_commands` so the
//! Desktop loads CAD and PCB commands from one file. That crate's tests merge this catalog with
//! the CAD one into the Desktop's `design-commands.json`.

mod board;
mod items;
mod schematic;

use runmat_design_commands::*;

/// Circuit commands, rows and item schemas, with the shared commands merged in.
pub fn catalog() -> Catalog {
    let mut c = runmat_design_commands::common();
    schematic::add(&mut c);
    board::add(&mut c);
    items::add(&mut c);
    rows(&mut c);
    c.finish();
    c
}

pub(crate) const CIRCUIT: &[EditorKind] = &[EditorKind::Circuit, EditorKind::CircuitLayout];

fn rows(c: &mut Catalog) {
    let tabs = vec![schematic::tab(), board::tab(), board::tab_3d()];
    let document_bar = vec![c_item("circuit.variant")];
    let cluster = vec![
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
            document_bar: document_bar.clone(),
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
    runmat_design_commands::c(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circuit_catalog_is_sound() {
        let c = catalog();
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
}

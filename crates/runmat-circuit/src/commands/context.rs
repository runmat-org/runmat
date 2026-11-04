//! Right-click menus for circuit components and nets (UI target decision 12), in the schematic
//! and on the board alike, and the commands only those menus reach.

use super::CIRCUIT;
use runmat_design_commands::*;
use SelectKind as K;

/// A submenu entry that runs `command` as is under its own label.
fn labeled(command: &str, label: &str) -> MenuEntry {
    MenuEntry {
        command: command.to_owned(),
        label: Some(label.to_owned()),
        values: Default::default(),
    }
}

pub(super) fn add(catalog: &mut Catalog) {
    let one = &[K::Component];
    catalog.add_commands(vec![
        cmd("circuit.change-part", "Change part", "replace", CIRCUIT)
            .desc("Replace the component's part with another from the pool; pins map by name.")
            .inputs(vec![text("part", "Part", Some("TPS62130RGTR"))])
            .when(selected(one))
            .edit(edit(EditOp::Set, Section::Components).item("part")),
        cmd("circuit.footprint", "Footprint", "footprint", CIRCUIT)
            .desc("Use another of the part's footprints.")
            .inputs(vec![text("footprint", "Footprint", Some("QFN-16"))])
            .when(selected(one))
            .edit(edit(EditOp::Set, Section::Components).item("footprint")),
        cmd("circuit.edit-value", "Edit value", "edit", CIRCUIT)
            .desc("Set the component's value with its unit and tolerance.")
            .kw(&["resistance", "capacitance", "tolerance"])
            .inputs(vec![text("value", "Value", Some("10 kΩ ± 1%"))])
            .when(selected_one(one))
            .edit(edit(EditOp::Set, Section::Components).item("value")),
        cmd("circuit.swap-pins", "Swap pins/gates", "swap", CIRCUIT)
            .desc("Swap equivalent pins or gates to shorten routes; the schematic follows.")
            .inputs(vec![radio("swap", "Swap", &[("pins", "Pins"), ("gates", "Gates")])])
            .when(selected_one(one))
            .edit(edit(EditOp::Set, Section::Components).item("swap")),
        cmd("circuit.fitted", "Fitted", "variant", CIRCUIT)
            .desc("Fit, leave out or substitute the component in the current variant.")
            .inputs(vec![choice(
                "fitting",
                "In this variant",
                &[("fitted", "Fitted"), ("not_fitted", "Not fitted"), ("alternate", "Alternate part")],
            )])
            .when(selected(one))
            .edit(edit(EditOp::Set, Section::Components).item("fitting")),
        cmd("circuit.locate", "Locate on board/schematic", "cross-probe", CIRCUIT)
            .desc("Jump to the item in the other view and frame it.")
            .when(selected_one(&[K::Component, K::Net]))
            .edit(view_action("locate")),
        cmd("circuit.net-class", "Net class", "net-class", CIRCUIT)
            .desc("Put the nets in a net class; its rules apply on the board.")
            .inputs(vec![text("class", "Net class", Some("power"))])
            .when(selected(&[K::Net]))
            .edit(edit(EditOp::Set, Section::Nets).item("class")),
        cmd("circuit.rename-net", "Rename", "rename", CIRCUIT)
            .desc("Rename the net; labels and ports follow.")
            .inputs(vec![text("name", "Name", Some("VBUS"))])
            .when(selected_one(&[K::Net]))
            .edit(edit(EditOp::Rename, Section::Nets)),
        cmd("circuit.net-length", "Length", "measure", CIRCUIT)
            .desc("Report the net's routed and unrouted length.")
            .when(selected(&[K::Net]))
            .edit(analyze("net_length")),
        cmd("board.route-net", "Route net", "route", CIRCUIT)
            .desc("Route the net's open connections with the interactive router.")
            .when(selected(&[K::Net]))
            .edit(edit(EditOp::Add, Section::Copper).item("track")),
    ]);

    let change = menu(
        "change-part",
        "Change part",
        "replace",
        vec![
            labeled("circuit.change-part", "From the pool…"),
            labeled("library.manufacturer-search", "Find alternates…"),
        ],
    );
    let footprint = menu(
        "footprint",
        "Footprint",
        "footprint",
        vec![labeled("circuit.footprint", "Choose footprint…"), labeled("library.open-part", "Edit in the part…")],
    );
    let fitted = choice_submenu(catalog, "fitted", "Fitted", "variant", "circuit.fitted", "fitting");
    let class = menu(
        "net-class",
        "Net class",
        "net-class",
        vec![
            preset("circuit.net-class", "Default", "class", "default"),
            preset("circuit.net-class", "Power", "class", "power"),
            labeled("circuit.net-class", "Choose…"),
        ],
    );

    catalog.context_menus.extend([
        context_menu(
            K::Component,
            CIRCUIT,
            &["circuit.cross-probe", "library.open-part", "circuit.edit-value", "board.lock", "circuit.locate"],
            vec![
                vec![c("circuit.cross-probe"), c("circuit.locate")],
                vec![c("library.open-part"), change, footprint, c("circuit.edit-value")],
                vec![c("circuit.swap-pins"), c("board.lock"), fitted],
                vec![c("design.show-source"), c("design.copy-reference")],
                vec![c("circuit.delete")],
            ],
        ),
        context_menu(
            K::Net,
            CIRCUIT,
            &["board.highlight-net", "circuit.cross-probe", "circuit.rename-net", "board.route-net"],
            vec![
                vec![c("board.highlight-net"), c("circuit.cross-probe")],
                vec![class, c("circuit.rename-net")],
                vec![c("circuit.net-length"), c("board.route-net")],
                vec![c("design.copy-reference")],
            ],
        ),
    ]);
}

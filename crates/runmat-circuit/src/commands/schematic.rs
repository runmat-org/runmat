//! Schematic commands (§9.1) and the circuit view cluster.

use super::CIRCUIT;
use runmat_design_commands::*;
use SelectKind as K;

fn sch(id: &str, label: &str, icon: &str, desc: &str) -> CommandDescriptor {
    cmd(id, label, icon, CIRCUIT).desc(desc).kw(&["schematic"])
}

/// Update board's inputs: the change list, the comparator's ignore setting per difference type
/// (Altium's *Ignore Differences*; saved with the board when the update is applied) and orphan
/// copper removal.
pub(super) fn eco_inputs() -> Vec<Field> {
    let mut inputs = vec![checklist("changes", "Changes to apply", &[K::EcoChange])];
    inputs.extend(
        crate::eco::Difference::ALL
            .into_iter()
            .map(|d| flag(&d.input_id(), d.label(), false).help(d.help())),
    );
    // An empty list is a valid setting: every parameter is compared.
    let mut ignored = list(
        "ignored_parameters",
        "Ignored parameters",
        text("parameter", "Parameter", Some("datasheet")),
    )
    .optional()
    .help("Parameters never compared, by name. Saved with the board.");
    if let FieldKind::List { min, .. } = &mut ignored.kind {
        *min = None;
    }
    inputs.push(ignored);
    inputs.push(flag(
        "remove_orphaned_copper",
        "Remove orphaned copper",
        false,
    ));
    inputs
}

fn place(item: &str, section: Section) -> ModelEdit {
    edit(EditOp::Add, section).item(item)
}

pub(super) fn add(c: &mut Catalog) {
    let point = |id: &str, label: &str| {
        pick(id, label, &[K::Pin, K::Wire, K::Port]).drag(Manipulator::Point)
    };
    c.add_commands(vec![
        sch(
            "schematic.select",
            "Select",
            "select",
            "Select items in the schematic.",
        )
        .edit(view_action("select")),
        sch(
            "schematic.move",
            "Move",
            "move",
            "Move selected items; wires stay connected.",
        )
        .inputs(vec![
            picks(
                "items",
                "Items",
                &[K::Component, K::Wire, K::Port, K::SheetSymbol],
            ),
            flag("drag_wires", "Keep wires attached", true),
        ])
        .when(selected(&[K::Component, K::Wire, K::Port, K::SheetSymbol]))
        .preview()
        .edit(edit(EditOp::Set, Section::Components).item("position")),
        sch(
            "schematic.place-part",
            "Part",
            "component",
            "Place a part from the pool.",
        )
        .kw(&["component", "symbol", "place"])
        .inputs(vec![
            text("part", "Part", Some("RC0603FR-0710KL")),
            text("designator", "Designator", Some("R?")),
            text("value", "Value", Some("10 kOhm 1%")).optional(),
            angle("rotation", "Rotation", 0.0),
            flag("mirror", "Mirror", false),
        ])
        .preview()
        .edit(place("component", Section::Components)),
        sch(
            "schematic.place-wire",
            "Wire",
            "wire",
            "Draw a wire between pins.",
        )
        .kw(&["connect", "net"])
        .inputs(vec![
            point("from", "From"),
            point("to", "To"),
            text("net", "Net", None).optional(),
        ])
        .preview()
        .edit(place("wire", Section::Wires)),
        sch(
            "schematic.place-bus",
            "Bus",
            "bus",
            "Draw a bus carrying a range of signals.",
        )
        .inputs(vec![
            text("name", "Bus", Some("D[0..7]")),
            point("from", "From"),
            point("to", "To"),
        ])
        .preview()
        .edit(place("bus", Section::Wires)),
        sch(
            "schematic.place-harness",
            "Harness",
            "harness",
            "Draw a harness that carries an interface (USB, SPI).",
        )
        .inputs(vec![
            text("name", "Harness", Some("USB")),
            text("interface", "Interface", Some("usb2")),
            point("from", "From"),
            point("to", "To"),
        ])
        .preview()
        .edit(place("harness", Section::Wires)),
        sch(
            "schematic.net-label",
            "Net label",
            "net-label",
            "Name a net with a label.",
        )
        .inputs(vec![
            text("name", "Net", Some("SENSE_A")),
            pick("on", "On wire", &[K::Wire, K::Pin]),
        ])
        .edit(place("label", Section::Nets)),
        sch(
            "schematic.power-port",
            "Power port",
            "power",
            "Place a power or ground symbol.",
        )
        .inputs(vec![
            text("net", "Net", Some("3V3")),
            choice(
                "style",
                "Style",
                &[
                    ("bar", "Bar"),
                    ("arrow", "Arrow"),
                    ("ground", "Ground"),
                    ("earth", "Earth"),
                    ("signal_ground", "Signal ground"),
                ],
            ),
        ])
        .edit(place("power", Section::Nets)),
        sch(
            "schematic.port",
            "Port",
            "port",
            "Place a port that connects this sheet to its parent.",
        )
        .inputs(vec![
            text("name", "Port", Some("VIN")),
            choice(
                "direction",
                "Direction",
                &[
                    ("input", "Input"),
                    ("output", "Output"),
                    ("bidirectional", "Bidirectional"),
                    ("power", "Power"),
                    ("passive", "Passive"),
                ],
            ),
        ])
        .edit(place("port", Section::Nets)),
        sch(
            "schematic.sheet-symbol",
            "Sheet symbol",
            "sheet-symbol",
            "Place a child sheet in the hierarchy.",
        )
        .inputs(vec![
            text("name", "Instance", Some("pwr")),
            text("sheet", "Sheet", Some("power")),
            count("channels", "Channels", 1, 1),
        ])
        .edit(place("instance", Section::Components)),
        sch(
            "schematic.sheet-entry",
            "Sheet entry",
            "sheet-entry",
            "Add an entry to a sheet symbol for a port of its sheet.",
        )
        .inputs(vec![
            pick("symbol", "Sheet symbol", &[K::SheetSymbol]),
            text("port", "Port", Some("VIN")),
        ])
        .when(selected_one(&[K::SheetSymbol]))
        .edit(place("binding", Section::Components)),
        sch(
            "schematic.no-erc",
            "No-ERC",
            "no-erc",
            "Mark a pin so ERC doesn't report it.",
        )
        .inputs(vec![pick("pin", "Pin", &[K::Pin])])
        .on(&[K::Pin])
        .edit(place("no_erc", Section::Directives)),
        sch(
            "schematic.directive",
            "Directive",
            "directive",
            "Attach a directive: differential pair, net class, parameter set or blanket.",
        )
        .inputs(vec![
            choice(
                "type",
                "Directive",
                &[
                    ("diff_pair", "Differential pair"),
                    ("net_class", "Net class"),
                    ("parameters", "Parameter set"),
                    ("blanket", "Blanket"),
                ],
            ),
            picks("nets", "Nets", &[K::Net, K::Wire]),
            text("value", "Class or parameters", Some("power")).optional(),
        ])
        .edit(place("directive", Section::Directives)),
        sch("schematic.text", "Text", "text", "Place text on the sheet.")
            .inputs(vec![text("text", "Text", None)])
            .edit(place("text", Section::Components)),
        sch(
            "schematic.repeat-channel",
            "Repeat channel",
            "repeat",
            "Repeat a sheet as channels with their own designators.",
        )
        .kw(&["multi-channel", "array"])
        .inputs(vec![
            pick("symbol", "Sheet symbol", &[K::SheetSymbol]),
            count("channels", "Channels", 4, 2),
            text(
                "designators",
                "Designators",
                Some("$Component_$ChannelIndex"),
            ),
        ])
        .when(selected_one(&[K::SheetSymbol]))
        .edit(edit(EditOp::Set, Section::Components).item("channels")),
        sch(
            "schematic.compile",
            "Compile",
            "compile",
            "Compile the design: flatten the hierarchy and extract nets.",
        )
        .edit(check("compile")),
        sch(
            "schematic.erc",
            "ERC",
            "erc",
            "Run the electrical rules check.",
        )
        .kw(&["electrical rules check"])
        .edit(check("erc")),
        sch(
            "schematic.annotate",
            "Annotate",
            "annotate",
            "Assign designators to parts that have none.",
        )
        .kw(&["designators", "renumber"])
        .inputs(vec![
            choice(
                "order",
                "Order",
                &[("across", "Across then down"), ("down", "Down then across")],
            ),
            flag("reset", "Renumber all", false),
        ])
        .edit(edit(EditOp::Set, Section::Components).item("designators")),
        sch(
            "schematic.update-board",
            "Update board (ECO)",
            "sync",
            "Send schematic changes to the board; review each line.",
        )
        .kw(&["eco", "engineering change", "sync", "update pcb"])
        .inputs(eco_inputs())
        .when(has(Facet::PendingEco))
        .on(&[K::EcoChange])
        .edit(edit(EditOp::Sync, Section::Placement).item("eco")),
        sch(
            "library.place",
            "Place from pool",
            "library",
            "Search the part pool and place a part.",
        )
        .inputs(vec![
            text("query", "Search", Some("10k 0603 1%")),
            flag("in_stock", "In stock only", true),
        ])
        .edit(place("component", Section::Components)),
        sch(
            "library.open-part",
            "Open part",
            "open",
            "Open the selected part's pool entry.",
        )
        .when(selected_one(&[K::Component]))
        .on(&[K::Component])
        .edit(view_action("open_part")),
        sch(
            "library.create-part",
            "Create part",
            "new-part",
            "Create a symbol, footprint or part in the pool.",
        )
        .inputs(vec![
            radio(
                "kind",
                "Create",
                &[
                    ("part", "Part"),
                    ("symbol", "Symbol"),
                    ("footprint", "Footprint"),
                ],
            ),
            text("name", "Name", Some("TMP117")),
        ])
        .edit(edit(EditOp::CreateFile, Section::Components).item("pool_entry")),
        sch(
            "library.manufacturer-search",
            "Manufacturer search",
            "search",
            "Search manufacturer parts with stock and pricing.",
        )
        .inputs(vec![
            text("query", "Search", Some("TPS62130")),
            text("manufacturer", "Manufacturer", None).optional(),
        ])
        .edit(analyze("manufacturer_search")),
        sch(
            "simulate.op-point",
            "Operating point",
            "simulate",
            "Solve the DC operating point and show voltages and power.",
        )
        .kw(&["spice", "ngspice", "dc"])
        .edit(analyze("op")),
        sch(
            "simulate.dc-sweep",
            "DC sweep",
            "simulate",
            "Sweep a source and plot the response.",
        )
        .inputs(vec![
            text("source", "Source", Some("V1")),
            qty("start", "Start", "voltage", Some(0.0)),
            qty("stop", "Stop", "voltage", Some(5.0)),
            qty("step", "Step", "voltage", Some(0.1)),
        ])
        .edit(analyze("dc")),
        sch(
            "simulate.transient",
            "Transient",
            "simulate",
            "Simulate over time.",
        )
        .inputs(vec![
            qty("stop", "Stop time", "time", Some(0.001)),
            qty("step", "Step", "time", Some(0.000001)),
        ])
        .edit(analyze("tran")),
        sch(
            "simulate.ac",
            "AC",
            "simulate",
            "Sweep frequency for gain and phase.",
        )
        .inputs(vec![
            qty("start", "Start", "frequency", Some(10.0)),
            qty("stop", "Stop", "frequency", Some(1.0e6)),
            count("points", "Points per decade", 20, 1),
        ])
        .edit(analyze("ac")),
        sch(
            "simulate.probe",
            "Probe",
            "probe",
            "Add a voltage or current probe.",
        )
        .inputs(vec![
            radio(
                "kind",
                "Probe",
                &[("voltage", "Voltage"), ("current", "Current")],
            ),
            pick("at", "At", &[K::Net, K::Pin, K::Component]),
        ])
        .edit(place("probe", Section::Directives)),
        // View cluster.
        cmd("circuit.variant", "Variant", "variant", CIRCUIT)
            .desc("Pick the assembly variant (fitted and not-fitted parts).")
            .inputs(vec![text("variant", "Variant", Some("default"))])
            .edit(view_action("variant")),
        cmd(
            "circuit.select-filter",
            "Selection filter",
            "select-filter",
            CIRCUIT,
        )
        .desc("Choose what clicks select.")
        .inputs(vec![choice(
            "filter",
            "Select",
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
        )])
        .edit(view_action("select_filter")),
        cmd("circuit.display", "Display", "display", CIRCUIT)
            .desc("Choose how layers draw.")
            .inputs(vec![
                choice(
                    "layers",
                    "Layers",
                    &[
                        ("all", "All layers"),
                        ("active", "Active layer only"),
                        ("dim", "Dim other layers"),
                    ],
                ),
                flag("ratsnest", "Ratsnest", true),
                flag("net_colors", "Net colors", false),
            ])
            .edit(view_action("display")),
        cmd("circuit.flip-board", "Flip board", "flip", CIRCUIT)
            .desc("View the board from the bottom.")
            .edit(view_action("flip")),
        // Tree actions.
        cmd("circuit.cross-probe", "Cross-probe", "cross-probe", CIRCUIT)
            .desc("Show the selection in the other view (schematic or board).")
            .when(selected(&[K::Component, K::Net, K::Pin]))
            .on(&[K::Component, K::Net, K::Pin])
            .edit(view_action("cross_probe")),
        cmd("circuit.delete", "Delete", "delete", CIRCUIT)
            .desc("Delete the selected items.")
            .when(selected(&[
                K::Component,
                K::Wire,
                K::Track,
                K::Via,
                K::Zone,
                K::Keepout,
                K::Room,
            ]))
            .on(&[
                K::Component,
                K::Wire,
                K::Track,
                K::Via,
                K::Zone,
                K::Keepout,
                K::Room,
            ])
            .edit(edit(EditOp::Remove, Section::Components)),
    ]);
}

pub(super) fn tab() -> DomainTab {
    tab_of(vec![
        group(
            "select",
            "Select",
            vec![c("schematic.select"), c("schematic.move")],
        ),
        group(
            "place",
            "Place",
            vec![
                c("schematic.place-part"),
                c("schematic.place-wire"),
                c("schematic.place-bus"),
                c("schematic.place-harness"),
                c("schematic.net-label"),
                c("schematic.power-port"),
                c("schematic.port"),
                c("schematic.sheet-symbol"),
                c("schematic.sheet-entry"),
                c("schematic.no-erc"),
                variants(
                    "schematic.directive.menu",
                    "Directive",
                    "directive",
                    "schematic.directive",
                    "type",
                    &[
                        ("diff_pair", "Differential pair"),
                        ("net_class", "Net class"),
                        ("parameters", "Parameter set"),
                        ("blanket", "Blanket"),
                    ],
                ),
                c("schematic.text"),
            ],
        ),
        group(
            "structure",
            "Structure",
            vec![c("schematic.repeat-channel")],
        ),
        group(
            "verify",
            "Verify",
            vec![
                c("schematic.compile"),
                c("schematic.erc"),
                c("schematic.annotate"),
            ],
        ),
        group("sync", "Sync", vec![c("schematic.update-board")]),
        group(
            "library",
            "Library",
            vec![menu(
                "library.menu",
                "Library",
                "library",
                vec![
                    e("library.place"),
                    e("library.open-part"),
                    e("library.create-part"),
                    e("library.manufacturer-search"),
                ],
            )],
        ),
        group(
            "simulate",
            "Simulate",
            vec![menu(
                "simulate.menu",
                "Simulate",
                "simulate",
                vec![
                    e("simulate.op-point"),
                    e("simulate.dc-sweep"),
                    e("simulate.transient"),
                    e("simulate.ac"),
                    e("simulate.probe"),
                ],
            )],
        ),
    ])
}

fn tab_of(groups: Vec<RowGroup>) -> DomainTab {
    runmat_design_commands::tab("schematic", "Schematic", groups)
}

//! Board commands (§9.2) and the 3D tab (§9.3).

use super::CIRCUIT;
use runmat_design_commands::*;
use SelectKind as K;

fn brd(id: &str, label: &str, icon: &str, desc: &str) -> CommandDescriptor {
    cmd(id, label, icon, CIRCUIT)
        .desc(desc)
        .kw(&["board", "pcb"])
        .when(has(Facet::Board))
}

fn add_to(item: &str, section: Section) -> ModelEdit {
    edit(EditOp::Add, section).item(item)
}

const RULES: &[(&str, &str)] = &[
    ("clearance", "Clearance"),
    ("width", "Width"),
    ("via", "Via"),
    ("diff_pair", "Differential pair"),
    ("length", "Length"),
    ("impedance", "Impedance"),
    ("plane", "Plane"),
    ("mask", "Solder mask"),
    ("silk", "Silkscreen"),
];

/// The board-house capability sets (`crate::capability::Capability`), word and label.
const CAPABILITIES: &[(&str, &str)] = &[
    ("standard", "Standard (0.15 mm track and space, 0.3 mm drill)"),
    ("advanced", "Advanced (0.1 mm track and space, 0.2 mm drill)"),
    ("hdi", "HDI (0.075 mm track and space, laser microvias)"),
];

/// Points clicked in the board view (`point:x,y` picks), in click order.
fn points(label: &str, help: &str) -> Field {
    picks("points", label, &[K::Point])
        .drag(Manipulator::Point)
        .help(help)
}

pub(super) fn add(c: &mut Catalog) {
    let layer = || text("layer", "Layer", Some("Top"));
    c.add_commands(vec![
        brd("board.select", "Select", "select", "Select items on the board.").edit(view_action("select")),
        brd("board.move", "Move", "move", "Move components or copper; tracks follow where they can.")
            .inputs(vec![picks("items", "Items", &[K::Component, K::Track, K::Via, K::Zone]), offset("dx", "Delta X", 0.0).drag(Manipulator::Triad), offset("dy", "Delta Y", 0.0), angle("rotation", "Rotation", 0.0)])
            .when(selected(&[K::Component, K::Track, K::Via, K::Zone]))
            .preview()
            .edit(edit(EditOp::Set, Section::Placement).item("position")),
        brd("board.flip", "Flip", "flip", "Move components to the other side of the board, mirrored where they stand.")
            .kw(&["bottom side", "top side", "mirror"])
            .inputs(vec![picks("components", "Components", &[K::Component])])
            .when(selected(&[K::Component]))
            .preview()
            .edit(edit(EditOp::Set, Section::Placement).item("side")),
        brd("board.place", "Place", "component", "Place components from the schematic, all at once or one at a time.")
            .inputs(vec![radio("mode", "Place", &[("from_schematic", "From schematic"), ("single", "Single component")]), picks("components", "Components", &[K::Component]).shown("mode", &["single"]), choice("side", "Side", &[("top", "Top"), ("bottom", "Bottom")])])
            .preview()
            .edit(edit(EditOp::Set, Section::Placement).item("place")),
        brd("board.align", "Align", "align", "Align or distribute components.")
            .inputs(vec![choice("align", "Align", &[("left", "Left edges"), ("right", "Right edges"), ("top", "Top edges"), ("bottom", "Bottom edges"), ("center_x", "Horizontal centers"), ("center_y", "Vertical centers"), ("distribute_x", "Distribute horizontally"), ("distribute_y", "Distribute vertically")]), picks("components", "Components", &[K::Component])])
            .when(selected_at_least(&[K::Component], 2))
            .preview()
            .edit(edit(EditOp::Set, Section::Placement).item("align")),
        brd("board.room", "Room", "room", "Define a room that keeps a group of components together.")
            .inputs(vec![text("name", "Room", Some("power")), picks("components", "Components", &[K::Component]), points("Outline", "Click the room's corners, or leave empty to fit the components.").optional(), flag("keep_inside", "Keep components inside", true)])
            .edit(add_to("room", Section::Placement)),
        brd("board.route", "Route", "route", "Route a track interactively; walkaround now, push and shove later behind the same button.")
            .kw(&["track", "trace", "interactive route"])
            .inputs(vec![
                pick("net", "Net", &[K::Net, K::Pad, K::Track, K::Via]),
                points("Points", "Click the start, then each corner."),
                qty("width", "Width", "length", Some(0.15)).min(0.0),
                text("via", "Via", Some("0.3/0.6")),
                layer(),
                choice("mode", "Mode", &[("walkaround", "Walkaround"), ("ignore", "Ignore obstacles"), ("stop", "Stop at obstacle")]),
                choice("corner", "Corners", &[("45", "45 degrees"), ("90", "90 degrees"), ("arc45", "45 degrees with arcs"), ("any", "Any angle")]),
            ])
            .preview()
            .edit(add_to("track", Section::Copper)),
        brd("board.diff-pair", "Diff pair", "diff-pair", "Route a differential pair together.")
            .inputs(vec![pick("pair", "Pair", &[K::Net]), qty("width", "Width", "length", Some(0.12)), qty("gap", "Gap", "length", Some(0.15)), layer()])
            .preview()
            .edit(add_to("track_pair", Section::Copper)),
        brd("board.multi-route", "Multi-route", "multi-route", "Route several nets as a bus.")
            .inputs(vec![picks("nets", "Nets", &[K::Net, K::Pad]), qty("width", "Width", "length", Some(0.15)), qty("spacing", "Spacing", "length", Some(0.15)), layer()])
            .preview()
            .edit(add_to("track_bus", Section::Copper)),
        brd("board.tune-length", "Tune length", "tune-length", "Add meanders until a net or pair meets its target length.")
            .inputs(vec![pick("net", "Net or pair", &[K::Net, K::Track]), qty("target", "Target length", "length", Some(50.0)), qty("tolerance", "Tolerance", "length", Some(0.1)), choice("style", "Style", &[("trombone", "Trombone"), ("sawtooth", "Sawtooth"), ("arc", "Rounded")])])
            .preview()
            .edit(edit(EditOp::Edit, Section::Copper).item("tuning")),
        brd("board.place-via", "Via", "via", "Place a via.")
            .inputs(vec![points("Location", "Click where the via goes."), text("size", "Size", Some("0.3/0.6")), pick("net", "Net", &[K::Net]).optional(), choice("span", "Span", &[("through", "Through"), ("blind", "Blind"), ("buried", "Buried"), ("micro", "Microvia")])])
            .preview()
            .edit(add_to("via", Section::Copper)),
        brd("board.stitch", "Stitch/shield", "stitch", "Stitch a pour with vias, or shield a net with a row of them.")
            .inputs(vec![radio("mode", "Mode", &[("stitch", "Stitch a pour"), ("shield", "Shield a net")]), pick("target", "Pour or net", &[K::Zone, K::Net]), text("shield_net", "Shield net", Some("GND")).shown("mode", &["shield"]), qty("pitch", "Pitch", "length", Some(1.5)), text("via", "Via", Some("0.3/0.6"))])
            .preview()
            .edit(add_to("via_array", Section::Copper)),
        brd("board.fanout", "BGA fanout", "fanout", "Fan out a component's pads to vias: BGA balls to the diagonals, other pads past their outer end.")
            .inputs(vec![picks("component", "Components", &[K::Component]), picks("nets", "Nets", &[K::Net]).optional().help("Empty: the nets poured on an inner layer."), flag("signals", "Fan out signal pads too", false), text("via", "Via", Some("0.2/0.45"))])
            .preview()
            .edit(add_to("fanout", Section::Copper)),
        brd("board.autoroute", "Autoroute", "route", "Route the unrouted connections of the picked nets, or of every net not on a plane.")
            .kw(&["auto route", "route all", "maze router"])
            .inputs(vec![
                picks("nets", "Nets", &[K::Net]).optional().help("Empty: every net with airwires that is not poured on an inner layer."),
                text("layers", "Layers", Some("F.Cu, B.Cu")).optional(),
                qty("width", "Width", "length", Some(0.2)).min(0.0).help("Nets whose class sets a track width use that."),
                text("via", "Via", Some("0.2/0.45")),
            ])
            .edit(add_to("track", Section::Copper)),
        brd("board.teardrops", "Teardrops", "teardrop", "Add or remove teardrops at pads and vias.")
            .inputs(vec![radio("action", "Action", &[("add", "Add"), ("remove", "Remove")]), flag("all", "Whole board", true)])
            .edit(edit(EditOp::Edit, Section::Copper).item("teardrops")),
        brd("board.pour", "Pour", "pour", "Draw a copper pour, repour, shelve or split a plane.")
            .inputs(vec![
                radio("action", "Action", &[("draw", "Draw pour"), ("repour", "Repour all"), ("shelve", "Shelve"), ("split_plane", "Split plane")]),
                pick("net", "Net", &[K::Net]).shown("action", &["draw", "split_plane"]),
                points("Outline", "Click the pour's corners.").shown("action", &["draw"]),
                layer().shown("action", &["draw", "split_plane"]),
                qty("clearance", "Clearance", "length", Some(0.25)).shown("action", &["draw"]),
                choice("connect", "Pad connection", &[("relief", "Thermal relief"), ("relief-tht", "Thermal relief on through-hole pads, SMD solid"), ("solid", "Solid"), ("none", "None")]).shown("action", &["draw"]),
                count("priority", "Priority", 0, 0).shown("action", &["draw"]).help("Where pours overlap, the higher priority fills first."),
            ])
            .preview()
            .edit(add_to("zone", Section::Copper)),
        brd("board.fill", "Fill", "fill", "Draw a solid copper fill.")
            .inputs(vec![points("Outline", "Click the fill's corners."), pick("net", "Net", &[K::Net]).optional(), layer()])
            .edit(add_to("fill", Section::Copper)),
        brd("board.keepout", "Keepout", "keepout", "Draw an area where copper, vias or components may not go.")
            .inputs(vec![points("Outline", "Click the keepout's corners."), choice("applies", "Keep out", &[("all", "Everything"), ("copper", "Copper"), ("tracks", "Tracks"), ("vias", "Vias"), ("components", "Components")]), layer()])
            .edit(add_to("keepout", Section::Copper)),
        brd("board.dimension", "Dimension", "dimension", "Add a board dimension annotation.")
            .inputs(vec![points("Points", "Click the two ends, then where the dimension line goes.").optional(), picks("items", "Between", &[K::Pad, K::Track, K::Via, K::Component]).optional(), text("layer", "Layer", Some("Mech1"))])
            .edit(add_to("dimension", Section::Outline)),
        brd(
            "board.update-schematic",
            "Update schematic (back-annotate)",
            "sync",
            "Send board changes (renumbering, pin and gate swaps, footprints) back to the schematic; review each line.",
        )
        .kw(&["eco", "back annotate", "back-annotation", "swap", "renumber"])
        .inputs(vec![checklist("changes", "Changes to apply", &[K::EcoBackChange])])
        .when(has(Facet::PendingBackAnnotation))
        .on(&[K::EcoBackChange])
        .edit(edit(EditOp::Sync, Section::Components).item("back_annotation")),
        brd("board.drc", "DRC", "drc", "Run the design rule check (also runs live while editing).")
            .kw(&["design rule check", "clearance"])
            .inputs(vec![radio("scope", "Check", &[("all", "Whole board"), ("selection", "Selection")]), flag("live", "Live checking", true)])
            .edit(check("drc")),
        brd("board.rules", "Rules", "rules", "Add or edit a design rule: clearance, width, via, pair, length, impedance, plane, mask or silk.")
            .inputs(vec![choice("kind", "Rule", RULES), text("applies", "Applies to", Some("InNetClass('power')")), text("value", "Value", Some("0.2 mm")), count("priority", "Priority", 1, 1)])
            .edit(add_to("rule", Section::Rules)),
        brd("board.capability", "Board house", "rules", "Load a board-house capability set: the fabrication limits every design rule falls back to.")
            .kw(&["capability", "fab house", "fabrication limits", "manufacturing limits", "hdi"])
            .inputs(vec![choice("capability", "Capability", CAPABILITIES)])
            .edit(edit(EditOp::Set, Section::Rules).item("limits")),
        brd("board.layer-stack", "Layer stack", "layers", "Edit the stackup, with the impedance calculator and thermal properties.")
            .kw(&["stackup", "impedance", "dielectric"])
            .inputs(vec![count("layers", "Copper layers", 4, 1), text("material", "Dielectric", Some("FR-4")), qty("thickness", "Board thickness", "length", Some(1.6)), qty("target_z", "Target impedance", "resistance", Some(50.0)).optional()])
            .edit(edit(EditOp::Set, Section::Stackup).item("stackup")),
        cmd("board.outline", "Outline", "outline", CIRCUIT)
            .desc("Draw the board outline, or take it from an enclosure part.")
            .kw(&["board shape", "edge cuts"])
            .inputs(vec![radio("source", "From", &[("draw", "Draw"), ("enclosure", "Enclosure part")]), points("Outline", "Click the board's corners.").shown("source", &["draw"]), text("part", "Part", Some("enclosure.part")).shown("source", &["enclosure"]), pick("face", "Face", &[K::Face]).shown("source", &["enclosure"])])
            .edit(add_to("outline", Section::Outline)),
        brd("board.panelize", "Panelize", "panel", "Step the board into a manufacturing panel.")
            .inputs(vec![count("columns", "Columns", 2, 1), count("rows", "Rows", 2, 1), qty("spacing", "Spacing", "length", Some(2.0)), choice("break", "Breakaway", &[("tabs", "Tabs with mouse bites"), ("v_score", "V-score")])])
            .when(has(Facet::BoardOutline))
            .edit(edit(EditOp::CreateFile, Section::Outline).item("panel")),
        brd("board.test-points", "Test points", "test-point", "Add test points on nets.")
            .inputs(vec![picks("nets", "Nets", &[K::Net]), choice("side", "Side", &[("bottom", "Bottom"), ("top", "Top")]), qty("diameter", "Pad diameter", "length", Some(1.0))])
            .edit(add_to("test_point", Section::Components)),
        brd("board.fab-outputs", "Fabrication outputs", "export", "Make the fabrication and assembly files for the board.")
            .kw(&["gerber", "odb++", "ipc-2581", "drill", "pick and place", "bom", "assembly drawing", "manufacturing", "fab house"])
            // The Desktop's panel holds the rest (outputs, layers, variants, folder, zip) from the
            // board's saved settings; the preset is the command's one input.
            .inputs(vec![choice(
                "preset",
                "Package",
                &[("fab_house", "Fab house"), ("assembly", "Assembly"), ("full", "Full package"), ("custom", "Custom")],
            )])
            .edit(export("fabrication")),
        brd("board.to-enclosure", "To enclosure", "to-enclosure", "Place the board in an assembly and check its clearance.")
            .inputs(vec![text("assembly", "Assembly", Some("enclosure.assembly")), qty("clearance", "Minimum clearance", "length", Some(1.0))])
            .edit(edit(EditOp::CreateFile, Section::None).item("assembly_occurrence")),
        brd("board.clearance-3d", "Clearance", "clearance", "Check clearance between the board's parts and the enclosure.")
            .inputs(vec![qty("minimum", "Minimum clearance", "length", Some(1.0))])
            .edit(analyze("clearance")),
        // Tree actions.
        brd("board.highlight-net", "Highlight net", "net", "Highlight the net on the board and dim the rest.")
            .when(selected(&[K::Net, K::Track, K::Pad, K::Via]))
            .on(&[K::Net, K::Track, K::Pad, K::Via])
            .edit(view_action("highlight_net")),
        brd(
            "board.copy-room-formats",
            "Copy room formats",
            "room",
            "Copy one channel room's placement and routing into the other channels' rooms.",
        )
        .kw(&["multi-channel", "channel", "replicate", "room", "copy placement", "copy routing"])
        .inputs(vec![
            pick("source", "From room", &[K::Room]),
            picks("targets", "To rooms", &[K::Room])
                .optional()
                .help("Empty: every other room of the same repeated sheet."),
            flag("placement", "Copy placement", true),
            flag("routing", "Copy routing", true)
                .help("Replaces the routing on the matching nets inside each target room."),
            flag("pours", "Copy pours", true)
                .help("Replaces the pours on the matching nets inside each target room."),
            flag("room_shape", "Copy room shape", true),
            flag("rotate", "Turn about the board center", false)
                .help("For channels set around the board's center: each copy turns by the quarter turns between the rooms."),
        ])
        .when(selected(&[K::Room]))
        .on(&[K::Room])
        .edit(edit(EditOp::Set, Section::Placement).item("room_formats")),
        brd("board.lock", "Lock", "lock", "Lock components or copper so they don't move.")
            .when(selected(&[K::Component, K::Track, K::Via, K::Zone]))
            .on(&[K::Component, K::Track, K::Via, K::Zone])
            .edit(edit(EditOp::Set, Section::Placement).item("locked")),
    ]);
}

pub(super) fn tab() -> DomainTab {
    runmat_design_commands::tab(
        "board",
        "Board",
        vec![
            group("select", "Select", vec![c("board.select"), c("board.move"), c("board.flip")]),
            group(
                "place",
                "Place",
                vec![
                    variants(
                        "board.place.menu",
                        "Place",
                        "component",
                        "board.place",
                        "mode",
                        &[
                            ("from_schematic", "Place from schematic"),
                            ("single", "Place a component"),
                        ],
                    ),
                    variants(
                        "board.align.menu",
                        "Align",
                        "align",
                        "board.align",
                        "align",
                        &[
                            ("left", "Align left"),
                            ("right", "Align right"),
                            ("top", "Align top"),
                            ("bottom", "Align bottom"),
                            ("center_x", "Align horizontal centers"),
                            ("center_y", "Align vertical centers"),
                            ("distribute_x", "Distribute horizontally"),
                            ("distribute_y", "Distribute vertically"),
                        ],
                    ),
                    c("board.room"),
                    c("board.copy-room-formats"),
                ],
            ),
            group(
                "route",
                "Route",
                vec![
                    c("board.route"),
                    c("board.diff-pair"),
                    c("board.multi-route"),
                    c("board.tune-length"),
                ],
            ),
            group(
                "vias",
                "Vias",
                vec![
                    c("board.place-via"),
                    c("board.stitch"),
                    c("board.fanout"),
                    c("board.teardrops"),
                ],
            ),
            group(
                "copper",
                "Copper",
                vec![
                    variants(
                        "board.pour.menu",
                        "Pour",
                        "pour",
                        "board.pour",
                        "action",
                        &[
                            ("draw", "Draw pour"),
                            ("repour", "Repour all"),
                            ("shelve", "Shelve pours"),
                            ("split_plane", "Split plane"),
                        ],
                    ),
                    c("board.fill"),
                    variants(
                        "board.keepout.menu",
                        "Keepout",
                        "keepout",
                        "board.keepout",
                        "applies",
                        &[
                            ("all", "Keepout"),
                            ("copper", "Copper keepout"),
                            ("tracks", "Track keepout"),
                            ("vias", "Via keepout"),
                            ("components", "Component keepout"),
                        ],
                    ),
                ],
            ),
            group("annotate", "Annotate", vec![c("board.dimension")]),
            group(
                "sync",
                "Sync",
                vec![c("schematic.update-board"), c("board.update-schematic")],
            ),
            group(
                "rules",
                "Rules",
                vec![
                    c("board.drc"),
                    variants(
                        "board.rules.menu",
                        "Rules",
                        "rules",
                        "board.rules",
                        "kind",
                        RULES,
                    ),
                    c("board.capability"),
                    c("board.layer-stack"),
                ],
            ),
            group(
                "board",
                "Board",
                vec![
                    variants(
                        "board.outline.menu",
                        "Outline",
                        "outline",
                        "board.outline",
                        "source",
                        &[
                            ("draw", "Draw outline"),
                            ("enclosure", "Outline from enclosure"),
                        ],
                    ),
                    c("board.panelize"),
                    c("board.test-points"),
                ],
            ),
            group("outputs", "Outputs", vec![c("board.fab-outputs")]),
            group("mechanical", "Mechanical", vec![c("board.to-enclosure")]),
            group(
                "analysis",
                "Analysis",
                vec![variants(
                    "study.new.menu",
                    "New study",
                    "study",
                    "study.new",
                    "physics",
                    &[("thermal", "Thermal"), ("ir_drop", "IR drop")],
                )],
            ),
        ],
    )
}

pub(super) fn tab_3d() -> DomainTab {
    runmat_design_commands::tab(
        "3d",
        "3D",
        vec![group("check", "Check", vec![c("board.clearance-3d")])],
    )
}

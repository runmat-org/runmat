//! The Sheet Metal tab.

use super::*;

pub(super) fn sheet_metal_commands() -> Vec<CommandDescriptor> {
    let sm = |id: &str, label: &str, icon: &str, variant: &str, desc: &str, inputs: Vec<Field>| {
        cmd(id, label, icon, P)
            .desc(desc)
            .kw(&["sheet metal"])
            .inputs(inputs)
            .when(sheet_metal())
            .preview()
            .edit(add_feature_variant("sheet_metal", "sheet_metal", variant))
    };
    let bend_radius = || length("radius", "Bend radius", 1.0);
    vec![
        cmd("sheet-metal.base-flange", "Base flange", "base-flange", P)
            .desc("Start a sheet metal part from a sketch.")
            .kw(&["sheet metal", "base", "tab"])
            .inputs(vec![
                pick("profile", "Sketch", &[K::Sketch]),
                length("thickness", "Thickness", 1.5),
                bend_radius(),
                length("depth", "Depth", 20.0).drag(Manipulator::Arrow),
                choice(
                    "bend_rule",
                    "Bend allowance",
                    &[
                        ("k_factor", "K-factor"),
                        ("table", "Bend table"),
                        ("allowance", "Bend allowance"),
                        ("deduction", "Bend deduction"),
                    ],
                ),
                qty("k", "K-factor", "ratio", Some(0.44)).shown("bend_rule", &["k_factor"]),
                text("gauge", "Gauge", Some("16 ga")).optional(),
            ])
            .when(has(Facet::Sketches))
            .preview()
            .edit(add_feature_variant(
                "sheet_metal",
                "sheet_metal",
                "base_flange",
            )),
        sm(
            "sheet-metal.edge-flange",
            "Edge flange",
            "edge-flange",
            "edge_flange",
            "Add a flange along edges.",
            vec![
                picks("edges", "Edges", &[K::Edge]),
                length("length", "Length", 10.0).drag(Manipulator::Arrow),
                angle("angle", "Angle", 90.0).drag(Manipulator::Angle),
                choice(
                    "position",
                    "Position",
                    &[
                        ("inside", "Material inside"),
                        ("outside", "Material outside"),
                        ("bend_outside", "Bend outside"),
                    ],
                ),
                bend_radius(),
                choice(
                    "relief",
                    "Relief",
                    &[
                        ("rectangular", "Rectangular"),
                        ("obround", "Obround"),
                        ("tear", "Tear"),
                    ],
                ),
            ],
        ),
        sm(
            "sheet-metal.miter-flange",
            "Miter flange",
            "miter-flange",
            "miter_flange",
            "Add flanges along a chain of edges with mitered corners.",
            vec![
                pick("profile", "Profile sketch", &[K::Sketch]),
                picks("edges", "Edges", &[K::Edge]),
                length("gap", "Miter gap", 0.25),
            ],
        ),
        sm(
            "sheet-metal.hem",
            "Hem",
            "hem",
            "hem",
            "Fold an edge back on itself.",
            vec![
                picks("edges", "Edges", &[K::Edge]),
                choice(
                    "hem",
                    "Type",
                    &[
                        ("closed", "Closed"),
                        ("open", "Open"),
                        ("teardrop", "Teardrop"),
                        ("rolled", "Rolled"),
                    ],
                ),
                length("length", "Length", 5.0),
                length("gap", "Gap", 0.5).shown("hem", &["open"]),
            ],
        ),
        sm(
            "sheet-metal.jog",
            "Jog",
            "jog",
            "jog",
            "Add two bends that offset a flat.",
            vec![
                pick("line", "Sketch line", &[K::SketchEntity, K::Sketch]),
                offset("offset", "Offset", 5.0),
                angle("angle", "Angle", 90.0),
            ],
        ),
        sm(
            "sheet-metal.sketched-bend",
            "Sketched bend",
            "sketched-bend",
            "sketched_bend",
            "Bend along a sketched line.",
            vec![
                pick("line", "Bend line", &[K::SketchEntity, K::Sketch]),
                angle("angle", "Angle", 90.0).drag(Manipulator::Angle),
                bend_radius(),
                pick("fixed", "Fixed face", &[K::Face]),
            ],
        ),
        sm(
            "sheet-metal.lofted-bend",
            "Lofted bend",
            "lofted-bend",
            "lofted_bend",
            "Bend between two open profiles (transitions, hoppers).",
            vec![
                pick("profile1", "First profile", &[K::Sketch]),
                pick("profile2", "Second profile", &[K::Sketch]),
                length("thickness", "Thickness", 1.5),
                count("bends", "Bends per segment", 8, 1),
            ],
        ),
        cmd(
            "sheet-metal.convert",
            "Convert to sheet metal",
            "convert-sheet",
            P,
        )
        .desc("Make a solid sheet metal from a fixed face and bend edges.")
        .kw(&["sheet metal", "rip"])
        .inputs(vec![
            pick("fixed", "Fixed face", &[K::Face]),
            picks("bends", "Bend edges", &[K::Edge]),
            length("thickness", "Thickness", 1.5),
            bend_radius(),
        ])
        .when(bodies_exist())
        .preview()
        .edit(add_feature_variant(
            "sheet_metal",
            "sheet_metal",
            "convert_to_sheet_metal",
        )),
        sm(
            "sheet-metal.closed-corner",
            "Closed corner",
            "closed-corner",
            "closed_corner",
            "Close the gap between flanges at a corner.",
            vec![
                picks("faces", "Faces to extend", &[K::Face]),
                choice(
                    "type",
                    "Corner",
                    &[
                        ("butt", "Butt"),
                        ("overlap", "Overlap"),
                        ("underlap", "Underlap"),
                    ],
                ),
                length("gap", "Gap", 0.1),
            ],
        ),
        sm(
            "sheet-metal.break-corner",
            "Break corner",
            "break-corner",
            "break_corner",
            "Chamfer or round a flat corner.",
            vec![
                picks("edges", "Corner edges", &[K::Edge, K::Vertex]),
                choice(
                    "break",
                    "Type",
                    &[("chamfer", "Chamfer"), ("fillet", "Fillet")],
                ),
                length("distance", "Distance", 2.0),
            ],
        ),
        sm(
            "sheet-metal.relief",
            "Corner relief",
            "relief",
            "corner_relief",
            "Cut relief at bend corners.",
            vec![
                picks("corners", "Corners", &[K::Vertex, K::Edge]),
                choice(
                    "relief",
                    "Relief",
                    &[
                        ("rectangular", "Rectangular"),
                        ("round", "Round"),
                        ("tear", "Tear"),
                    ],
                ),
                length("size", "Size", 1.5),
            ],
        ),
        sm(
            "sheet-metal.tab-and-slot",
            "Tab and slot",
            "tab-slot",
            "tab_and_slot",
            "Add interlocking tabs and slots between flanges.",
            vec![
                pick("edge", "Tab edge", &[K::Edge]),
                pick("face", "Slot face", &[K::Face]),
                length("length", "Tab length", 10.0),
                count("count", "Tabs", 2, 1),
                length("clearance", "Slot clearance", 0.1),
            ],
        ),
        sm(
            "sheet-metal.form-tool",
            "Form tool",
            "form-tool",
            "form_tool",
            "Stamp a form (louver, dimple, lance) into a face.",
            vec![
                text("tool", "Form tool", Some("louver")),
                pick("face", "Face", &[K::Face]),
                pick(
                    "position",
                    "Position",
                    &[K::SketchPoint, K::Point, K::Vertex],
                )
                .drag(Manipulator::Point),
                angle("rotation", "Rotation", 0.0),
            ],
        ),
        cmd("sheet-metal.unfold", "Unfold", "unfold", P)
            .desc("Unfold chosen bends to work on the flat.")
            .inputs(vec![
                pick("fixed", "Fixed face", &[K::Face]),
                picks("bends", "Bends", &[K::Face, K::Feature]),
            ])
            .when(sheet_metal())
            .edit(add_feature_variant("sheet_metal", "sheet_metal", "unfold")),
        cmd("sheet-metal.fold", "Fold", "fold", P)
            .desc("Fold unfolded bends back.")
            .inputs(vec![picks("bends", "Bends", &[K::Face, K::Feature])])
            .when(sheet_metal())
            .edit(add_feature_variant("sheet_metal", "sheet_metal", "fold")),
        cmd(
            "sheet-metal.flat-pattern",
            "Flat pattern",
            "flat-pattern",
            P,
        )
        .desc("Show the flat pattern with bend lines, for drawings and DXF.")
        .kw(&["flatten", "blank", "dxf"])
        .when(sheet_metal())
        .edit(add_feature_variant(
            "sheet_metal",
            "sheet_metal",
            "flat_pattern",
        )),
        cmd("sheet-metal.bend-table", "Bend table", "bend-table", P)
            .desc("Choose the bend table or rule for this part.")
            .inputs(vec![
                text("table", "Bend table", Some("company/aluminum.csv")),
                qty("k", "Default K-factor", "ratio", Some(0.44)),
            ])
            .edit(edit(EditOp::Set, Section::Settings).item("bend_table")),
        cmd("sheet-metal.gauge", "Gauge", "gauge", P)
            .desc("Pick the material gauge, which sets thickness and bend radius.")
            .inputs(vec![
                text("gauge", "Gauge", Some("16 ga")),
                text("table", "Gauge table", Some("company/steel-gauges.csv")).optional(),
            ])
            .edit(edit(EditOp::Set, Section::Settings).item("gauge")),
    ]
}

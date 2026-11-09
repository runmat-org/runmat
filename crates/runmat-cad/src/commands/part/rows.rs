//! The part editor's ribbon rows: each tab's groups, menus and presets.

use super::*;

pub(super) fn rows(catalog: &Catalog) -> EditorRows {
    let ext = catalog.command("part.extrude").expect("extrude");
    let extrude_menu = menu(
        "part.extrude.menu",
        "Extrude",
        "extrude",
        choice_pairs(ext, "op")
            .iter()
            .map(|(value, _)| {
                let name = match value.as_str() {
                    "add" => "Extrude",
                    "remove" => "Extruded cut",
                    "intersect" => "Extrude to intersect",
                    _ => "Extrude as a new body",
                };
                preset("part.extrude", name, "op", value)
            })
            .collect(),
    );
    EditorRows {
        editor: Part,
        mode: EditMode::Model,
        default_tab: "features".into(),
        view_cluster: cluster_3d(vec![]),
        document_bar: vec![c("view.configuration")],
        shortcut_box: [
            "part.sketch",
            "part.extrude",
            "part.revolve",
            "part.hole",
            "part.fillet",
            "part.chamfer",
            "part.pattern",
            "part.mirror",
            "part.plane",
            "view.measure",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect(),
        mode_actions: Vec::new(),
        tabs: vec![
            tab(
                "features",
                "Features",
                vec![
                    group("sketch", "Sketch", vec![c("part.sketch")]),
                    group(
                        "create",
                        "Create",
                        vec![
                            extrude_menu,
                            c("part.revolve"),
                            c("part.sweep"),
                            c("part.loft"),
                            c("part.boundary"),
                        ],
                    ),
                    group(
                        "modify",
                        "Modify",
                        vec![
                            variants(
                                "part.hole.menu",
                                "Hole",
                                "hole",
                                "part.hole",
                                "type",
                                &[
                                    ("simple", "Simple hole"),
                                    ("counterbore", "Counterbore"),
                                    ("countersink", "Countersink"),
                                    ("tapped", "Tapped hole"),
                                    ("pipe_tap", "Pipe tap"),
                                    ("slot", "Slot"),
                                ],
                            ),
                            variants(
                                "part.fillet.menu",
                                "Fillet",
                                "fillet",
                                "part.fillet",
                                "type",
                                &[
                                    ("constant", "Fillet"),
                                    ("variable", "Variable radius fillet"),
                                    ("face", "Face fillet"),
                                    ("full_round", "Full round fillet"),
                                    ("setback", "Setback fillet"),
                                ],
                            ),
                            c("part.chamfer"),
                            c("part.shell"),
                            c("part.draft"),
                            c("part.rib"),
                        ],
                    ),
                    group(
                        "repeat",
                        "Repeat",
                        vec![
                            variants(
                                "part.pattern.menu",
                                "Pattern",
                                "pattern",
                                "part.pattern",
                                "layout",
                                &[
                                    ("linear", "Linear pattern"),
                                    ("circular", "Circular pattern"),
                                    ("curve", "Curve-driven pattern"),
                                    ("sketch", "Sketch-driven pattern"),
                                    ("table", "Table-driven pattern"),
                                    ("fill", "Fill pattern"),
                                    ("variable", "Variable pattern"),
                                ],
                            ),
                            c("part.mirror"),
                        ],
                    ),
                    group(
                        "bodies",
                        "Bodies",
                        vec![
                            variants(
                                "part.boolean.menu",
                                "Boolean",
                                "boolean",
                                "part.boolean",
                                "kind",
                                &[
                                    ("union", "Union"),
                                    ("subtract", "Subtract"),
                                    ("intersect", "Intersect"),
                                ],
                            ),
                            c("part.split"),
                            menu(
                                "part.body.menu",
                                "Body",
                                "body",
                                vec![
                                    e("part.body-move"),
                                    e("part.body-delete"),
                                    e("part.body-scale"),
                                    e("part.body-promote"),
                                    e("part.body-combine"),
                                    e("part.body-indent"),
                                ],
                            ),
                        ],
                    ),
                    group(
                        "reference",
                        "Reference",
                        vec![
                            variants(
                                "part.plane.menu2",
                                "Plane",
                                "plane",
                                "part.plane",
                                "type",
                                &[
                                    ("offset", "Offset plane"),
                                    ("angle", "Plane at angle"),
                                    ("three_point", "Plane through 3 points"),
                                    ("mid", "Mid-plane"),
                                    ("normal_to_curve", "Plane normal to curve"),
                                    ("tangent_to_face", "Plane tangent to surface"),
                                ],
                            ),
                            c("part.axis"),
                            c("part.point"),
                            c("part.coordinate-system"),
                            c("part.mate-connector"),
                        ],
                    ),
                    group(
                        "curves",
                        "Curves",
                        vec![menu(
                            "part.curves.menu",
                            "Curves",
                            "helix",
                            vec![
                                e("part.curve-helix"),
                                e("part.curve-projected"),
                                e("part.curve-composite"),
                                e("part.curve-split-line"),
                                e("part.curve-intersection"),
                                e("part.curve-through-points"),
                                e("part.curve-equation"),
                            ],
                        )],
                    ),
                    group(
                        "detail",
                        "Detail",
                        vec![
                            c("part.thread"),
                            menu(
                                "part.emboss.menu",
                                "Emboss",
                                "emboss",
                                vec![e("part.emboss"), e("part.wrap")],
                            ),
                        ],
                    ),
                    group(
                        "reuse",
                        "Reuse",
                        vec![menu(
                            "part.library.menu",
                            "Library",
                            "library",
                            vec![e("part.library")],
                        )],
                    ),
                ],
            ),
            tab(
                "sketch",
                "Sketch",
                vec![
                    group(
                        "sketch",
                        "Sketch",
                        vec![c("part.sketch"), c("part.sketch-3d")],
                    ),
                    group(
                        "reference",
                        "Reference",
                        vec![
                            variants(
                                "part.plane.menu",
                                "Plane",
                                "plane",
                                "part.plane",
                                "type",
                                &[
                                    ("offset", "Offset plane"),
                                    ("angle", "Plane at angle"),
                                    ("three_point", "Plane through 3 points"),
                                    ("mid", "Mid-plane"),
                                    ("normal_to_curve", "Plane normal to curve"),
                                    ("tangent_to_face", "Plane tangent to surface"),
                                ],
                            ),
                            c("part.axis"),
                            c("part.point"),
                            c("part.coordinate-system"),
                        ],
                    ),
                ],
            ),
            tab(
                "direct",
                "Direct",
                vec![
                    group(
                        "faces",
                        "Faces",
                        vec![
                            c("direct.move-face"),
                            c("direct.offset-face"),
                            c("direct.replace-face"),
                            c("direct.delete-face"),
                        ],
                    ),
                    group("bodies", "Bodies", vec![c("direct.move-body")]),
                    group(
                        "drag",
                        "Drag",
                        vec![c("direct.instant-3d"), c("direct.pull")],
                    ),
                    group("relations", "Relations", vec![c("direct.live-relations")]),
                ],
            ),
            tab(
                "surfaces",
                "Surfaces",
                vec![
                    group(
                        "create",
                        "Create",
                        vec![
                            c("surface.extrude"),
                            c("surface.revolve"),
                            c("surface.sweep"),
                            c("surface.loft"),
                            c("surface.boundary"),
                            variants(
                                "surface.fill.menu",
                                "Fill",
                                "surface-fill",
                                "surface.fill",
                                "continuity",
                                &[
                                    ("g0", "Fill (contact)"),
                                    ("g1", "Fill (tangent)"),
                                    ("g2", "Fill (curvature)"),
                                ],
                            ),
                            c("surface.planar"),
                            c("surface.ruled"),
                            c("surface.offset"),
                        ],
                    ),
                    group(
                        "edit",
                        "Edit",
                        vec![
                            c("surface.extend"),
                            c("surface.trim"),
                            c("surface.untrim"),
                            c("surface.knit"),
                            c("surface.thicken"),
                            c("surface.delete-hole"),
                            c("surface.replace-face"),
                            c("surface.cut"),
                        ],
                    ),
                    group("mid", "Mid-surface", vec![c("surface.mid-surface")]),
                ],
            ),
            tab(
                "sheet-metal",
                "Sheet Metal",
                vec![
                    group(
                        "flanges",
                        "Flanges",
                        vec![
                            c("sheet-metal.base-flange"),
                            c("sheet-metal.edge-flange"),
                            c("sheet-metal.miter-flange"),
                            c("sheet-metal.hem"),
                            c("sheet-metal.jog"),
                        ],
                    ),
                    group(
                        "bends",
                        "Bends",
                        vec![
                            c("sheet-metal.sketched-bend"),
                            c("sheet-metal.lofted-bend"),
                            c("sheet-metal.convert"),
                        ],
                    ),
                    group(
                        "corners",
                        "Corners",
                        vec![
                            c("sheet-metal.closed-corner"),
                            c("sheet-metal.break-corner"),
                            c("sheet-metal.relief"),
                        ],
                    ),
                    group(
                        "features",
                        "Features",
                        vec![c("sheet-metal.tab-and-slot"), c("sheet-metal.form-tool")],
                    ),
                    group(
                        "flat",
                        "Flat",
                        vec![
                            c("sheet-metal.unfold"),
                            c("sheet-metal.fold"),
                            c("sheet-metal.flat-pattern"),
                        ],
                    ),
                    group(
                        "settings",
                        "Settings",
                        vec![
                            menu(
                                "sheet-metal.bend-table.menu",
                                "Bend table",
                                "bend-table",
                                vec![e("sheet-metal.bend-table")],
                            ),
                            menu(
                                "sheet-metal.gauge.menu",
                                "Gauge",
                                "gauge",
                                vec![e("sheet-metal.gauge")],
                            ),
                        ],
                    ),
                ],
            ),
            tab(
                "weldments",
                "Weldments",
                vec![
                    group(
                        "members",
                        "Members",
                        vec![c("weldment.structural-member"), c("weldment.trim-extend")],
                    ),
                    group(
                        "details",
                        "Details",
                        vec![
                            c("weldment.end-cap"),
                            c("weldment.gusset"),
                            c("weldment.weld-bead"),
                        ],
                    ),
                    group(
                        "lists",
                        "Lists",
                        vec![
                            c("weldment.cut-list"),
                            menu(
                                "weldment.profile-library.menu",
                                "Profile library",
                                "library",
                                vec![e("weldment.profile-library")],
                            ),
                        ],
                    ),
                ],
            ),
            tab(
                "mold",
                "Mold",
                vec![
                    group(
                        "parting",
                        "Parting",
                        vec![
                            c("mold.parting-line"),
                            c("mold.shut-off"),
                            c("mold.parting-surface"),
                            c("mold.core-cavity"),
                        ],
                    ),
                    group(
                        "check",
                        "Check",
                        vec![c("mold.draft-analysis"), c("mold.undercut")],
                    ),
                    group("shrink", "Shrink", vec![c("mold.scale")]),
                ],
            ),
            tab(
                "evaluate",
                "Evaluate",
                vec![
                    group(
                        "properties",
                        "Properties",
                        vec![
                            c("evaluate.mass-properties"),
                            c("evaluate.section-properties"),
                        ],
                    ),
                    group(
                        "check",
                        "Check",
                        vec![
                            c("evaluate.check-geometry"),
                            c("evaluate.interference"),
                            c("evaluate.clearance"),
                            c("evaluate.thickness"),
                        ],
                    ),
                    group(
                        "surface",
                        "Surface analysis",
                        vec![
                            c("evaluate.draft"),
                            c("evaluate.curvature"),
                            c("evaluate.zebra"),
                            c("evaluate.deviation"),
                            c("evaluate.undercut"),
                        ],
                    ),
                    group(
                        "manufacturing",
                        "Manufacturing",
                        vec![
                            variants(
                                "evaluate.dfm.menu",
                                "DFM check",
                                "dfm",
                                "evaluate.dfm",
                                "process",
                                &[
                                    ("machining", "Machining"),
                                    ("sheet_metal", "Sheet metal"),
                                    ("molding", "Molding"),
                                    ("additive", "Additive"),
                                ],
                            ),
                            c("evaluate.cost"),
                        ],
                    ),
                ],
            ),
            tab(
                "annotate",
                "Annotate",
                vec![
                    group(
                        "tolerancing",
                        "Tolerancing",
                        vec![
                            c("annotate.datum"),
                            c("annotate.gdt"),
                            c("annotate.dimension"),
                            c("annotate.auto-dimension"),
                        ],
                    ),
                    group(
                        "notes",
                        "Notes",
                        vec![
                            c("annotate.surface-finish"),
                            c("annotate.coating"),
                            c("annotate.weld-symbol"),
                            c("annotate.note"),
                        ],
                    ),
                    group("views", "Views", vec![c("annotate.annotation-view")]),
                ],
            ),
            tab(
                "simulate",
                "Simulate",
                vec![
                    group(
                        "regions",
                        "Regions",
                        vec![c("simulate.region"), c("simulate.publish")],
                    ),
                    group(
                        "representation",
                        "Representation",
                        vec![
                            variants(
                                "simulate.representation.menu",
                                "Representation",
                                "representation",
                                "simulate.representation",
                                "kind",
                                &[
                                    ("full", "Full"),
                                    ("envelope", "Envelope"),
                                    ("defeatured", "Defeatured"),
                                    ("mid_surface", "Mid-surface"),
                                    ("beam", "Beam"),
                                ],
                            ),
                            c("simulate.defeature"),
                        ],
                    ),
                    group(
                        "studies",
                        "Studies",
                        vec![
                            c("simulate.check"),
                            c("study.new"),
                            menu(
                                "simulate.studies.menu",
                                "Studies",
                                "study",
                                vec![e("simulate.studies")],
                            ),
                        ],
                    ),
                ],
            ),
            tab(
                "manage",
                "Manage",
                vec![
                    group(
                        "parameters",
                        "Parameters",
                        vec![
                            c("manage.equations"),
                            c("manage.configurations"),
                            c("manage.document-settings"),
                        ],
                    ),
                    group(
                        "references",
                        "References",
                        vec![
                            c("manage.find-references"),
                            c("manage.replace-reference"),
                            c("manage.freeze-bar"),
                        ],
                    ),
                    group(
                        "look",
                        "Look",
                        vec![
                            menu(
                                "manage.material.menu",
                                "Material",
                                "material",
                                vec![e("manage.material")],
                            ),
                            menu(
                                "manage.appearance.menu",
                                "Appearance",
                                "appearance",
                                vec![e("manage.appearance")],
                            ),
                            c("manage.decal"),
                        ],
                    ),
                    group(
                        "files",
                        "Files",
                        vec![
                            c("manage.import"),
                            variants(
                                "manage.export.menu",
                                "Export",
                                "export",
                                "manage.export",
                                "format",
                                &[
                                    ("step_ap242", "STEP AP242"),
                                    ("iges", "IGES"),
                                    ("stl", "STL"),
                                    ("3mf", "3MF"),
                                    ("gltf", "glTF"),
                                    ("dxf_face", "DXF of a face"),
                                ],
                            ),
                            c("manage.heal"),
                            c("manage.recognize-features"),
                            c("manage.split-to-files"),
                            c("manage.pack-and-share"),
                        ],
                    ),
                ],
            ),
        ],
    }
}

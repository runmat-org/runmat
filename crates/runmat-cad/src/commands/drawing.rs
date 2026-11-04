//! Drawing editor commands (`.drawing`), per the UI target §8.

use super::*;
use EditorKind::Drawing;
use SelectKind as K;

const D: &[EditorKind] = &[Drawing];

fn drw(id: &str, label: &str, icon: &str, desc: &str) -> CommandDescriptor {
    cmd(id, label, icon, D).desc(desc)
}

fn scale() -> Field {
    text("scale", "Scale", Some("1:2"))
}

fn views() -> Predicate {
    has(Facet::Views)
}

fn view_edit(kind: &str) -> ModelEdit {
    edit(EditOp::Add, Section::Views).item(kind)
}

fn annotation(kind: &str) -> ModelEdit {
    edit(EditOp::Add, Section::Annotations).item(kind)
}

fn table(kind: &str) -> ModelEdit {
    edit(EditOp::Add, Section::Tables).item(kind)
}

pub(super) fn add(catalog: &mut Catalog) {
    let parent = || pick("parent", "From view", &[K::View]);
    let commands = vec![
        // Views.
        drw(
            "drawing.standard-3",
            "Standard 3 view",
            "standard-views",
            "Place front, top and right views of a model.",
        )
        .inputs(vec![
            text("model", "Model", Some("bracket.part")),
            text("configuration", "Configuration", Some("default")).optional(),
            scale(),
            radio(
                "projection",
                "Projection",
                &[("first", "First angle"), ("third", "Third angle")],
            ),
        ])
        .edit(view_edit("standard")),
        drw(
            "drawing.model-view",
            "Model view",
            "model-view",
            "Place one view of a model in any orientation.",
        )
        .inputs(vec![
            text("model", "Model", Some("bracket.part")),
            choice(
                "orientation",
                "Orientation",
                &[
                    ("front", "Front"),
                    ("back", "Back"),
                    ("top", "Top"),
                    ("bottom", "Bottom"),
                    ("left", "Left"),
                    ("right", "Right"),
                    ("isometric", "Isometric"),
                    ("annotation", "Annotation view"),
                ],
            ),
            scale(),
            choice(
                "display",
                "Display",
                &[
                    ("hidden_removed", "Hidden lines removed"),
                    ("hidden_visible", "Hidden lines visible"),
                    ("wireframe", "Wireframe"),
                    ("shaded", "Shaded"),
                    ("shaded_edges", "Shaded with edges"),
                ],
            ),
        ])
        .preview()
        .edit(view_edit("model")),
        drw(
            "drawing.projected",
            "Projected view",
            "projected",
            "Project a view from an existing one.",
        )
        .inputs(vec![
            parent(),
            choice(
                "direction",
                "Direction",
                &[
                    ("up", "Up"),
                    ("down", "Down"),
                    ("left", "Left"),
                    ("right", "Right"),
                    ("iso", "Isometric corner"),
                ],
            ),
        ])
        .when(views())
        .preview()
        .edit(view_edit("projected")),
        drw(
            "drawing.section",
            "Section view",
            "section",
            "Cut a full, half, offset or aligned section through a view.",
        )
        .inputs(vec![
            parent(),
            choice(
                "type",
                "Type",
                &[
                    ("full", "Full"),
                    ("half", "Half"),
                    ("offset", "Offset"),
                    ("aligned", "Aligned"),
                ],
            ),
            text("label", "Label", Some("A")),
            flag("flip", "Flip direction", false),
            flag("hatch", "Hatch from material", true),
        ])
        .when(views())
        .preview()
        .edit(view_edit("section")),
        drw(
            "drawing.detail",
            "Detail view",
            "detail",
            "Enlarge a circled area of a view.",
        )
        .inputs(vec![
            parent(),
            text("label", "Label", Some("B")),
            text("scale", "Scale", Some("2:1")),
            choice(
                "profile",
                "Profile",
                &[("circle", "Circle"), ("profile", "Sketched profile")],
            ),
        ])
        .when(views())
        .preview()
        .edit(view_edit("detail")),
        drw(
            "drawing.auxiliary",
            "Auxiliary view",
            "auxiliary",
            "Project a view normal to an edge.",
        )
        .inputs(vec![
            parent(),
            pick("edge", "Edge", &[K::Edge]),
            text("label", "Label", Some("C")),
        ])
        .when(views())
        .preview()
        .edit(view_edit("auxiliary")),
        drw(
            "drawing.broken-out",
            "Broken-out section",
            "broken-out",
            "Cut away part of a view to show the inside.",
        )
        .inputs(vec![parent(), length("depth", "Depth", 10.0)])
        .when(views())
        .edit(view_edit("broken_out")),
        drw(
            "drawing.break",
            "Break view",
            "break",
            "Shorten a long view with break lines.",
        )
        .inputs(vec![
            parent(),
            radio(
                "direction",
                "Break",
                &[("horizontal", "Horizontal"), ("vertical", "Vertical")],
            ),
            length("gap", "Gap", 10.0),
            choice(
                "style",
                "Style",
                &[
                    ("straight", "Straight"),
                    ("zigzag", "Zig zag"),
                    ("curved", "Curved"),
                ],
            ),
        ])
        .when(views())
        .edit(view_edit("break")),
        drw(
            "drawing.crop",
            "Crop view",
            "crop",
            "Crop a view to a sketched boundary.",
        )
        .inputs(vec![parent(), pick("profile", "Boundary", &[K::Sketch])])
        .when(views())
        .edit(view_edit("crop")),
        drw(
            "drawing.isometric",
            "Isometric view",
            "view-iso",
            "Place an isometric view.",
        )
        .inputs(vec![
            text("model", "Model", Some("bracket.part")),
            scale(),
            choice(
                "display",
                "Display",
                &[
                    ("shaded_edges", "Shaded with edges"),
                    ("hidden_removed", "Hidden lines removed"),
                ],
            ),
        ])
        .edit(view_edit("isometric")),
        drw(
            "drawing.alternate-position",
            "Alternate position",
            "alternate",
            "Overlay another position of an assembly in a view.",
        )
        .inputs(vec![
            parent(),
            text("configuration", "Position", Some("open")),
        ])
        .when(views())
        .edit(view_edit("alternate")),
        drw(
            "drawing.auto",
            "Auto-drawing",
            "auto-drawing",
            "Lay out views, dimensions and tables for a model; fasteners are skipped.",
        )
        .kw(&["automatic", "generate"])
        .inputs(vec![
            text("model", "Model", Some("bracket.part")),
            choice(
                "sheet",
                "Sheet size",
                &[
                    ("a3", "A3"),
                    ("a2", "A2"),
                    ("a4", "A4"),
                    ("b", "ANSI B"),
                    ("d", "ANSI D"),
                ],
            ),
            flag("dimensions", "Add dimensions", true),
            flag("tables", "Add tables", true),
        ])
        .edit(view_edit("auto")),
        // Annotate: dimensions.
        drw(
            "drawing.smart-dimension",
            "Smart dimension",
            "dimension",
            "Dimension edges, points or circles.",
        )
        .inputs(vec![
            picks("items", "Items", &[K::Edge, K::Vertex, K::View]),
            choice(
                "tolerance",
                "Tolerance",
                &[
                    ("none", "None"),
                    ("symmetric", "Symmetric"),
                    ("bilateral", "Bilateral"),
                    ("limit", "Limit"),
                    ("fit", "Fit"),
                ],
            ),
        ])
        .when(views())
        .edit(annotation("dimension")),
        drw(
            "drawing.ordinate",
            "Ordinate dimension",
            "ordinate",
            "Dimension features from a zero line.",
        )
        .inputs(vec![
            pick("origin", "Zero", &[K::Edge, K::Vertex]),
            picks("items", "Items", &[K::Edge, K::Vertex]),
        ])
        .when(views())
        .edit(annotation("ordinate")),
        drw(
            "drawing.baseline",
            "Baseline dimension",
            "baseline",
            "Dimension features from one baseline.",
        )
        .inputs(vec![
            pick("base", "Baseline", &[K::Edge, K::Vertex]),
            picks("items", "Items", &[K::Edge, K::Vertex]),
        ])
        .when(views())
        .edit(annotation("baseline")),
        drw(
            "drawing.chain",
            "Chain dimension",
            "chain",
            "Dimension features end to end.",
        )
        .inputs(vec![picks("items", "Items", &[K::Edge, K::Vertex])])
        .when(views())
        .edit(annotation("chain")),
        drw(
            "drawing.hole-callout",
            "Hole callout",
            "hole",
            "Label a hole with its size, depth and thread from the model.",
        )
        .inputs(vec![pick("hole", "Hole", &[K::Edge])])
        .when(views())
        .edit(annotation("hole_callout")),
        drw(
            "drawing.model-items",
            "Model items",
            "model-items",
            "Bring dimensions and GD&T from the model into views.",
        )
        .inputs(vec![
            picks("views", "Views", &[K::View]).optional(),
            flag("dimensions", "Dimensions", true),
            flag("gdt", "GD&T", true),
            flag("datums", "Datums", true),
            flag("finishes", "Surface finishes", true),
        ])
        .when(views())
        .edit(annotation("model_items")),
        // Annotate: tolerancing.
        drw(
            "drawing.gdt",
            "GD&T frame",
            "gdt",
            "Add a feature control frame.",
        )
        .inputs(vec![
            text("frame", "Frame", Some("position|0.1|A|B|C")),
            pick(
                "attach",
                "Attached to",
                &[K::Edge, K::DrawingDimension, K::Annotation],
            ),
        ])
        .when(views())
        .edit(annotation("gdt")),
        drw(
            "drawing.datum",
            "Datum",
            "datum",
            "Add a datum feature symbol.",
        )
        .inputs(vec![
            text("label", "Label", Some("A")),
            pick("attach", "Attached to", &[K::Edge, K::DrawingDimension]),
        ])
        .when(views())
        .edit(annotation("datum")),
        drw(
            "drawing.datum-target",
            "Datum target",
            "datum-target",
            "Add a datum target point, line or area.",
        )
        .inputs(vec![
            text("label", "Label", Some("A1")),
            choice(
                "shape",
                "Shape",
                &[
                    ("point", "Point"),
                    ("line", "Line"),
                    ("circle", "Circle"),
                    ("rectangle", "Rectangle"),
                ],
            ),
            pick("attach", "Attached to", &[K::Edge, K::Vertex]),
        ])
        .when(views())
        .edit(annotation("datum_target")),
        drw(
            "drawing.surface-finish",
            "Surface finish",
            "surface-finish",
            "Add a surface finish symbol.",
        )
        .inputs(vec![
            text("ra", "Ra", Some("1.6")),
            pick("attach", "Attached to", &[K::Edge]),
        ])
        .when(views())
        .edit(annotation("surface_finish")),
        drw(
            "drawing.weld-symbol",
            "Weld symbol",
            "weld-symbol",
            "Add a weld symbol.",
        )
        .inputs(vec![
            choice(
                "type",
                "Weld",
                &[
                    ("fillet", "Fillet"),
                    ("groove", "Groove"),
                    ("plug", "Plug"),
                    ("spot", "Spot"),
                ],
            ),
            length("size", "Size", 4.0),
            pick("attach", "Attached to", &[K::Edge]),
        ])
        .when(views())
        .edit(annotation("weld")),
        // Annotate: notes.
        drw(
            "drawing.note",
            "Note",
            "note",
            "Add a note, with an optional leader.",
        )
        .inputs(vec![
            text("text", "Text", None),
            pick("attach", "Leader to", &[K::Edge, K::Vertex, K::View]).optional(),
        ])
        .edit(annotation("note")),
        drw(
            "drawing.balloon",
            "Balloon",
            "balloon",
            "Label a component with its BOM item number.",
        )
        .inputs(vec![
            pick("attach", "Component", &[K::Edge, K::Face]),
            choice(
                "style",
                "Style",
                &[
                    ("circle", "Circle"),
                    ("split", "Split circle"),
                    ("triangle", "Triangle"),
                    ("hexagon", "Hexagon"),
                ],
            ),
        ])
        .when(views())
        .edit(annotation("balloon")),
        drw(
            "drawing.auto-balloon",
            "Auto-balloon",
            "balloon",
            "Balloon every component in a view.",
        )
        .inputs(vec![
            picks("views", "Views", &[K::View]),
            choice(
                "layout",
                "Layout",
                &[
                    ("square", "Square"),
                    ("circular", "Circular"),
                    ("top", "Top"),
                    ("left", "Left"),
                ],
            ),
            flag("fasteners", "Include fasteners", false),
        ])
        .when(views())
        .edit(annotation("auto_balloon")),
        drw(
            "drawing.stacked-balloon",
            "Stacked balloon",
            "balloon",
            "Stack balloons for a fastener set.",
        )
        .inputs(vec![
            picks("attach", "Components", &[K::Edge, K::Face]),
            radio(
                "direction",
                "Stack",
                &[
                    ("up", "Up"),
                    ("down", "Down"),
                    ("left", "Left"),
                    ("right", "Right"),
                ],
            ),
        ])
        .when(views())
        .edit(annotation("stacked_balloon")),
        // Annotate: marks.
        drw(
            "drawing.center-mark",
            "Center mark",
            "center-mark",
            "Mark the center of circles and arcs.",
        )
        .inputs(vec![
            picks("edges", "Circles", &[K::Edge]),
            flag("auto", "All holes in the view", false),
        ])
        .when(views())
        .edit(annotation("center_mark")),
        drw(
            "drawing.centerline",
            "Centerline",
            "centerline",
            "Draw a centerline between edges or on a cylinder.",
        )
        .inputs(vec![picks("edges", "Edges", &[K::Edge])])
        .when(views())
        .edit(annotation("centerline")),
        drw(
            "drawing.area-hatch",
            "Area hatch",
            "hatch",
            "Fill a closed area with a hatch.",
        )
        .inputs(vec![
            pick("area", "Area", &[K::Face, K::Sketch]),
            text("pattern", "Pattern", Some("ANSI31")),
            qty("hatch_scale", "Scale", "scale", Some(1.0)),
        ])
        .when(views())
        .edit(annotation("hatch")),
        drw(
            "drawing.revision-cloud",
            "Revision cloud",
            "revision-cloud",
            "Circle changed areas for a revision.",
        )
        .inputs(vec![
            text("revision", "Revision", Some("B")),
            radio(
                "shape",
                "Shape",
                &[
                    ("rectangle", "Rectangle"),
                    ("ellipse", "Ellipse"),
                    ("freehand", "Freehand"),
                ],
            ),
        ])
        .edit(annotation("revision_cloud")),
        drw(
            "drawing.sketch",
            "Sketch on sheet",
            "sketch",
            "Draw lines and shapes on the sheet.",
        )
        .inputs(vec![pick("on", "On", &[K::Sheet, K::View])])
        .edit(edit(EditOp::EnterMode, Section::Annotations).item("sketch")),
        // Tables.
        drw("table.bom", "BOM", "bom", "Add a bill of materials table.")
            .kw(&["bill of materials", "parts list"])
            .inputs(vec![
                pick("view", "From view", &[K::View]),
                choice(
                    "type",
                    "Type",
                    &[
                        ("top", "Top level"),
                        ("parts", "Parts only"),
                        ("indented", "Indented"),
                    ],
                ),
                text("template", "Template", Some("company/bom")).optional(),
            ])
            .when(views())
            .edit(table("bom")),
        drw(
            "table.hole",
            "Hole table",
            "hole-table",
            "List holes with positions from an origin.",
        )
        .inputs(vec![
            pick("origin", "Origin", &[K::Vertex, K::Edge]),
            picks("faces", "Faces", &[K::Face, K::Edge]),
            flag("combine", "Combine same sizes", true),
        ])
        .when(views())
        .edit(table("hole")),
        drw(
            "table.revision",
            "Revision table",
            "revision-table",
            "Add the revision history table.",
        )
        .edit(table("revision")),
        drw(
            "table.bend",
            "Bend table",
            "bend-table",
            "List bends of a flat pattern.",
        )
        .inputs(vec![pick("view", "Flat pattern view", &[K::View])])
        .when(views())
        .edit(table("bend")),
        drw("table.weld", "Weld table", "weld-table", "List weld beads.")
            .inputs(vec![pick("view", "From view", &[K::View])])
            .when(views())
            .edit(table("weld")),
        drw(
            "table.cut-list",
            "Cut list",
            "cut-list",
            "List weldment members by profile and length.",
        )
        .inputs(vec![pick("view", "From view", &[K::View])])
        .when(views())
        .edit(table("cut_list")),
        drw(
            "table.general",
            "General table",
            "table",
            "Add an empty table.",
        )
        .inputs(vec![
            count("rows", "Rows", 3, 1),
            count("columns", "Columns", 3, 1),
        ])
        .edit(table("general")),
        // Sheet.
        drw("sheet.add", "Add sheet", "add-sheet", "Add a sheet.")
            .inputs(vec![
                text("name", "Name", Some("Sheet 2")),
                choice(
                    "size",
                    "Size",
                    &[
                        ("a3", "A3"),
                        ("a4", "A4"),
                        ("a2", "A2"),
                        ("a1", "A1"),
                        ("a0", "A0"),
                        ("a", "ANSI A"),
                        ("b", "ANSI B"),
                        ("c", "ANSI C"),
                        ("d", "ANSI D"),
                    ],
                ),
            ])
            .edit(edit(EditOp::Add, Section::Sheets).item("sheet")),
        drw(
            "sheet.format",
            "Sheet format",
            "title-block",
            "Choose the sheet format and title block; fields fill from the model's properties.",
        )
        .kw(&["title block", "border"])
        .inputs(vec![
            text("format", "Format", Some("company/a3-landscape")),
            radio(
                "projection",
                "Projection",
                &[("first", "First angle"), ("third", "Third angle")],
            ),
            scale(),
        ])
        .edit(edit(EditOp::Set, Section::Sheets).item("format")),
        drw(
            "sheet.layers",
            "Layers",
            "layers",
            "Add layers and set their color and line style.",
        )
        .inputs(vec![
            text("name", "Layer", Some("Notes")),
            text("color", "Color", Some("#1f2937")),
            flag("print", "Print", true),
        ])
        .edit(edit(EditOp::Add, Section::Sheets).item("layer")),
        drw(
            "sheet.line-styles",
            "Line styles",
            "line-styles",
            "Set line weights and styles by line type.",
        )
        .inputs(vec![
            choice(
                "line",
                "Line",
                &[
                    ("visible", "Visible"),
                    ("hidden", "Hidden"),
                    ("center", "Center"),
                    ("section", "Section"),
                    ("dimension", "Dimension"),
                ],
            ),
            qty("weight", "Weight", "length", Some(0.35)),
        ])
        .edit(edit(EditOp::Set, Section::Sheets).item("line_style")),
        // Manage.
        drw(
            "drawing.check",
            "Check drawing",
            "check",
            "Check the drawing against a standards pack.",
        )
        .inputs(vec![text("pack", "Standard", Some("ISO 128"))])
        .edit(check("drawing")),
        drw(
            "drawing.document-settings",
            "Document settings",
            "settings",
            "Standard, fonts, arrows and units.",
        )
        .inputs(vec![
            choice(
                "standard",
                "Standard",
                &[
                    ("iso", "ISO"),
                    ("ansi", "ANSI"),
                    ("din", "DIN"),
                    ("jis", "JIS"),
                ],
            ),
            choice("units", "Units", &[("mm", "Millimeters"), ("in", "Inches")]),
            count("decimals", "Decimals", 1, 0),
        ])
        .edit(edit(EditOp::Set, Section::Settings).item("document")),
        drw(
            "drawing.export",
            "Export",
            "export",
            "Export as PDF, DXF, DWG or SVG.",
        )
        .inputs(vec![
            choice(
                "format",
                "Format",
                &[
                    ("pdf", "PDF"),
                    ("dxf", "DXF"),
                    ("dwg", "DWG"),
                    ("svg", "SVG"),
                ],
            ),
            radio(
                "sheets",
                "Sheets",
                &[("all", "All sheets"), ("current", "Current sheet")],
            ),
        ])
        .edit(export("selected")),
        drw("drawing.print", "Print", "print", "Print the drawing.")
            .inputs(vec![
                radio(
                    "sheets",
                    "Sheets",
                    &[("all", "All sheets"), ("current", "Current sheet")],
                ),
                flag("scale_to_fit", "Scale to fit", false),
            ])
            .edit(export("print")),
        // View cluster.
        drw(
            "drawing.sheet",
            "Sheet",
            "sheet",
            "Switch to another sheet.",
        )
        .inputs(vec![pick("sheet", "Sheet", &[K::Sheet])])
        .edit(view_action("sheet")),
        drw(
            "drawing.standard",
            "Standard",
            "standard",
            "Choose the drafting standard.",
        )
        .inputs(vec![choice(
            "standard",
            "Standard",
            &[
                ("iso", "ISO"),
                ("ansi", "ANSI"),
                ("din", "DIN"),
                ("jis", "JIS"),
            ],
        )])
        .edit(edit(EditOp::Set, Section::Settings).item("standard")),
        drw(
            "drawing.layer",
            "Layer",
            "layers",
            "Choose the layer new annotations go on.",
        )
        .inputs(vec![text("layer", "Layer", Some("Notes"))])
        .edit(view_action("layer")),
        drw(
            "drawing.zoom",
            "Zoom",
            "zoom",
            "Fit the sheet, zoom to actual size or to the selection.",
        )
        .inputs(vec![choice(
            "zoom",
            "Zoom",
            &[
                ("sheet", "Fit sheet"),
                ("actual", "Actual size"),
                ("selection", "Selection"),
            ],
        )])
        .edit(view_action("zoom")),
        // Tree actions.
        drw(
            "drawing.edit-view",
            "Edit view",
            "edit",
            "Open the view's properties.",
        )
        .when(selected_one(&[K::View]))
        .on(&[K::View])
        .edit(edit(EditOp::Edit, Section::Views)),
        drw(
            "drawing.delete",
            "Delete",
            "delete",
            "Delete views, annotations or tables.",
        )
        .when(selected(&[
            K::View,
            K::Annotation,
            K::Table,
            K::DrawingDimension,
        ]))
        .on(&[
            K::View,
            K::Annotation,
            K::Table,
            K::DrawingDimension,
            K::Sheet,
        ])
        .edit(edit(EditOp::Remove, Section::Views)),
        drw(
            "drawing.open-model",
            "Open model",
            "open",
            "Open the model a view shows.",
        )
        .when(selected_one(&[K::View]))
        .on(&[K::View])
        .edit(view_action("open")),
    ];
    catalog.add_commands(commands);

    catalog.rows.push(EditorRows {
        editor: Drawing,
        mode: EditMode::Model,
        default_tab: "views".into(),
        document_bar: vec![
            c("drawing.sheet"),
            variants(
                "drawing.standard.menu",
                "Standard",
                "standard",
                "drawing.standard",
                "standard",
                &[
                    ("iso", "ISO"),
                    ("ansi", "ANSI"),
                    ("din", "DIN"),
                    ("jis", "JIS"),
                ],
            ),
            c("drawing.layer"),
        ],
        view_cluster: vec![
            variants(
                "drawing.zoom.menu",
                "Zoom",
                "zoom",
                "drawing.zoom",
                "zoom",
                &[
                    ("sheet", "Fit sheet"),
                    ("actual", "Actual size"),
                    ("selection", "Zoom to selection"),
                ],
            ),
            c("view.measure"),
        ],
        shortcut_box: [
            "drawing.model-view",
            "drawing.projected",
            "drawing.section",
            "drawing.detail",
            "drawing.smart-dimension",
            "drawing.note",
            "drawing.balloon",
            "table.bom",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect(),
        mode_actions: Vec::new(),
        tabs: vec![
            tab(
                "views",
                "Views",
                vec![
                    group(
                        "views",
                        "Views",
                        vec![
                            c("drawing.standard-3"),
                            c("drawing.model-view"),
                            c("drawing.projected"),
                            variants(
                                "drawing.section.menu",
                                "Section",
                                "section",
                                "drawing.section",
                                "type",
                                &[
                                    ("full", "Full section"),
                                    ("half", "Half section"),
                                    ("offset", "Offset section"),
                                    ("aligned", "Aligned section"),
                                ],
                            ),
                            c("drawing.detail"),
                            c("drawing.auxiliary"),
                            c("drawing.broken-out"),
                            c("drawing.break"),
                            c("drawing.crop"),
                            c("drawing.isometric"),
                            c("drawing.alternate-position"),
                        ],
                    ),
                    group("auto", "Auto", vec![c("drawing.auto")]),
                ],
            ),
            tab(
                "annotate",
                "Annotate",
                vec![
                    group(
                        "dimensions",
                        "Dimensions",
                        vec![
                            c("drawing.smart-dimension"),
                            c("drawing.ordinate"),
                            c("drawing.baseline"),
                            c("drawing.chain"),
                            c("drawing.hole-callout"),
                            c("drawing.model-items"),
                        ],
                    ),
                    group(
                        "tolerancing",
                        "Tolerancing",
                        vec![
                            c("drawing.gdt"),
                            c("drawing.datum"),
                            c("drawing.datum-target"),
                            c("drawing.surface-finish"),
                            c("drawing.weld-symbol"),
                        ],
                    ),
                    group(
                        "notes",
                        "Notes",
                        vec![
                            c("drawing.note"),
                            c("drawing.balloon"),
                            c("drawing.auto-balloon"),
                            c("drawing.stacked-balloon"),
                        ],
                    ),
                    group(
                        "marks",
                        "Marks",
                        vec![
                            c("drawing.center-mark"),
                            c("drawing.centerline"),
                            c("drawing.area-hatch"),
                            c("drawing.revision-cloud"),
                            c("drawing.sketch"),
                        ],
                    ),
                ],
            ),
            tab(
                "tables",
                "Tables",
                vec![group(
                    "tables",
                    "Tables",
                    vec![
                        c("table.bom"),
                        c("table.hole"),
                        c("table.revision"),
                        c("table.bend"),
                        c("table.weld"),
                        c("table.cut-list"),
                        c("table.general"),
                    ],
                )],
            ),
            tab(
                "sheet",
                "Sheet",
                vec![group(
                    "sheet",
                    "Sheet",
                    vec![
                        c("sheet.add"),
                        c("sheet.format"),
                        c("sheet.layers"),
                        c("sheet.line-styles"),
                    ],
                )],
            ),
            tab(
                "manage",
                "Manage",
                vec![group(
                    "manage",
                    "Manage",
                    vec![
                        c("drawing.check"),
                        c("drawing.document-settings"),
                        variants(
                            "drawing.export.menu",
                            "Export",
                            "export",
                            "drawing.export",
                            "format",
                            &[
                                ("pdf", "PDF"),
                                ("dxf", "DXF"),
                                ("dwg", "DWG"),
                                ("svg", "SVG"),
                            ],
                        ),
                        c("drawing.print"),
                    ],
                )],
            ),
        ],
    });
}

//! Sketch mode (§6.3): replaces both rows of the part editor while a sketch is open.

use super::*;
use EditorKind::{Assembly, Part};
use SelectKind as K;

const S: &[EditorKind] = &[Part, Assembly];

fn sketching() -> Predicate {
    in_mode(
        &[EditMode::Sketch, EditMode::Sketch3d],
        "Open a sketch first",
    )
}

fn entity(
    id: &str,
    label: &str,
    icon: &str,
    item: &str,
    desc: &str,
    inputs: Vec<Field>,
) -> CommandDescriptor {
    cmd(id, label, icon, S)
        .desc(desc)
        .kw(&["sketch", "draw"])
        .inputs(inputs)
        .when(sketching())
        .preview()
        .edit(edit(EditOp::Add, Section::Sketch).item(item))
}

fn tool(
    id: &str,
    label: &str,
    icon: &str,
    item: &str,
    desc: &str,
    inputs: Vec<Field>,
) -> CommandDescriptor {
    cmd(id, label, icon, S)
        .desc(desc)
        .kw(&["sketch"])
        .inputs(inputs)
        .when(sketching())
        .preview()
        .edit(edit(EditOp::Edit, Section::Sketch).item(item))
}

const CONSTRAINTS: &[(&str, &str)] = &[
    ("coincident", "Coincident"),
    ("horizontal", "Horizontal"),
    ("vertical", "Vertical"),
    ("parallel", "Parallel"),
    ("perpendicular", "Perpendicular"),
    ("tangent", "Tangent"),
    ("equal", "Equal"),
    ("concentric", "Concentric"),
    ("midpoint", "Midpoint"),
    ("symmetric", "Symmetric"),
    ("collinear", "Collinear"),
    ("coradial", "Coradial"),
    ("fix", "Fix"),
    ("pierce", "Pierce"),
    ("curvature", "Curvature (G2)"),
];

const DIMENSIONS: &[(&str, &str)] = &[
    ("smart", "Smart dimension"),
    ("horizontal", "Horizontal dimension"),
    ("vertical", "Vertical dimension"),
    ("ordinate", "Ordinate dimension"),
    ("path_length", "Path length"),
];

pub(super) fn add(catalog: &mut Catalog) {
    let pt = |id: &str, label: &str| {
        pick(id, label, &[K::SketchPoint, K::Vertex, K::Point]).drag(Manipulator::Point)
    };
    let commands = vec![
        entity("sketch.line", "Line", "line", "line", "Draw lines, centerlines or a line from its midpoint.", vec![
            radio("type", "Type", &[("line", "Line"), ("centerline", "Centerline"), ("midpoint", "Midpoint line")]),
            pt("start", "Start"),
            pt("end", "End"),
            length("length", "Length", 0.0).optional(),
            angle("angle", "Angle", 0.0).optional(),
            flag("chain", "Keep drawing", true),
        ]),
        entity("sketch.rectangle", "Rectangle", "rectangle", "rectangle", "Draw a rectangle by corners, center, three points or as a parallelogram.", vec![
            radio("type", "Type", &[("corner", "Corner"), ("center", "Center"), ("three_point", "3-point"), ("parallelogram", "Parallelogram")]),
            pt("first", "First point"),
            pt("second", "Second point"),
            pt("third", "Third point").shown("type", &["three_point", "parallelogram"]),
            flag("construction_diagonals", "Construction diagonals", false),
        ]),
        entity("sketch.circle", "Circle", "circle", "circle", "Draw a circle by center or through three points.", vec![
            radio("type", "Type", &[("center", "Center and radius"), ("three_point", "3-point")]),
            pt("center", "Center").shown("type", &["center"]),
            length("radius", "Radius", 10.0).shown("type", &["center"]).drag(Manipulator::Radius),
            picks("points", "Points", &[K::SketchPoint, K::Vertex]).shown("type", &["three_point"]),
        ]),
        entity("sketch.arc", "Arc", "arc", "arc", "Draw an arc by center, tangent to an end or through three points.", vec![
            radio("type", "Type", &[("center", "Center point"), ("tangent", "Tangent"), ("three_point", "3-point")]),
            pt("start", "Start"),
            pt("end", "End"),
            pt("center", "Center").shown("type", &["center"]),
            length("radius", "Radius", 10.0).optional().drag(Manipulator::Radius),
        ]),
        entity("sketch.slot", "Slot", "slot", "slot", "Draw a straight, center-point or arc slot.", vec![
            radio("type", "Type", &[("straight", "Straight"), ("center", "Center point"), ("arc", "Arc")]),
            pt("start", "Start"),
            pt("end", "End"),
            length("width", "Width", 6.0),
            flag("dimension", "Add dimensions", true),
        ]),
        entity("sketch.polygon", "Polygon", "polygon", "polygon", "Draw a regular polygon.", vec![
            count("sides", "Sides", 6, 3),
            pt("center", "Center"),
            length("radius", "Radius", 10.0).drag(Manipulator::Radius),
            radio("fit", "Size by", &[("inscribed", "Inscribed circle"), ("circumscribed", "Circumscribed circle")]),
            angle("rotation", "Rotation", 0.0),
        ]),
        entity("sketch.spline", "Spline", "spline", "spline", "Draw a spline through points, by control points, from an equation or as a style spline.", vec![
            radio("type", "Type", &[("through_points", "Through points"), ("control_points", "Control points"), ("equation", "Equation"), ("style", "Style spline")]),
            picks("points", "Points", &[K::SketchPoint, K::Vertex]).shown("type", &["through_points", "control_points", "style"]),
            count("degree", "Degree", 3, 1).shown("type", &["control_points", "style"]),
            text("x", "x(t)", Some("t")).shown("type", &["equation"]),
            text("y", "y(t)", Some("sin(t)")).shown("type", &["equation"]),
            flag("closed", "Closed", false),
        ]),
        entity("sketch.ellipse", "Ellipse", "ellipse", "ellipse", "Draw an ellipse, a partial ellipse, a parabola or a conic.", vec![
            radio("type", "Type", &[("full", "Ellipse"), ("partial", "Partial ellipse"), ("parabola", "Parabola"), ("conic", "Conic")]),
            pt("center", "Center"),
            length("major", "Major radius", 20.0),
            length("minor", "Minor radius", 10.0),
            qty("rho", "Rho", "ratio", Some(0.5)).shown("type", &["conic"]),
        ]),
        entity("sketch.point", "Point", "sketch-point", "point", "Place a sketch point.", vec![pt("at", "At")]),
        entity("sketch.text", "Text", "text", "text", "Place text that features can extrude or emboss.", vec![
            text("text", "Text", None),
            text("font", "Font", Some("Inter")),
            length("height", "Height", 5.0),
            pick("along", "Along curve", &[K::SketchEntity]).optional(),
        ]),
        cmd("sketch.construction", "Construction", "construction", S)
            .desc("Draw new entities as construction geometry, or toggle selected ones.")
            .kw(&["reference geometry", "for construction"])
            .when(sketching())
            .on(&[K::SketchEntity])
            .edit(edit(EditOp::Set, Section::Sketch).item("construction")),
        tool("sketch.fillet", "Sketch fillet", "fillet", "fillet", "Round sketch corners.", vec![
            picks("corners", "Corners", &[K::SketchPoint, K::SketchEntity]),
            length("radius", "Radius", 2.0).drag(Manipulator::Radius),
            flag("keep_corner", "Keep constrained corners", true),
        ]),
        tool("sketch.chamfer", "Sketch chamfer", "chamfer", "chamfer", "Bevel sketch corners.", vec![
            picks("corners", "Corners", &[K::SketchPoint, K::SketchEntity]),
            length("distance", "Distance", 1.0),
            angle("angle", "Angle", 45.0),
        ]),
        tool("sketch.trim", "Trim", "trim", "trim", "Trim entities by power trim, to a corner, inside or outside.", vec![
            radio("type", "Type", &[("power", "Power trim"), ("corner", "Corner"), ("inside", "Trim away inside"), ("outside", "Trim away outside")]),
            picks("entities", "Entities", &[K::SketchEntity]),
            picks("boundaries", "Boundaries", &[K::SketchEntity]).shown("type", &["inside", "outside"]),
        ]),
        tool("sketch.extend", "Extend", "extend", "extend", "Extend an entity to the next one it meets.", vec![picks("entities", "Entities", &[K::SketchEntity])]),
        tool("sketch.offset", "Offset", "offset", "offset", "Offset entities by a distance.", vec![
            picks("entities", "Entities", &[K::SketchEntity, K::Edge]),
            offset("distance", "Distance", 2.0).drag(Manipulator::Arrow),
            flag("both", "Both directions", false),
            flag("cap", "Cap ends", false),
            flag("chain", "Select chain", true),
        ]),
        tool("sketch.split", "Split", "split", "split", "Split an entity at a point.", vec![pick("entity", "Entity", &[K::SketchEntity]), pt("at", "At")]),
        tool("sketch.stretch", "Stretch", "stretch", "stretch", "Stretch entities from a base point.", vec![
            picks("entities", "Entities", &[K::SketchEntity, K::SketchPoint]),
            pt("base", "Base point"),
            offset("dx", "Delta X", 0.0),
            offset("dy", "Delta Y", 0.0),
        ]),
        tool("sketch.mirror", "Mirror", "mirror", "mirror", "Mirror entities about a line, with an optional live mirror.", vec![
            picks("entities", "Entities", &[K::SketchEntity]),
            pick("about", "Mirror about", &[K::SketchEntity]),
            flag("copy", "Keep originals", true),
            flag("dynamic", "Dynamic mirror", false),
        ]),
        tool("sketch.pattern", "Sketch pattern", "pattern", "pattern", "Repeat entities in a line or a circle.", vec![
            radio("layout", "Layout", &[("linear", "Linear"), ("circular", "Circular")]),
            picks("entities", "Entities", &[K::SketchEntity]),
            length("spacing", "Spacing", 10.0).shown("layout", &["linear"]),
            angle("direction", "Direction", 0.0).shown("layout", &["linear"]),
            pt("center", "Center").shown("layout", &["circular"]),
            angle("angle", "Angle", 360.0).shown("layout", &["circular"]),
            count("instances", "Instances", 4, 2),
        ]),
        tool("sketch.transform", "Transform", "move", "transform", "Move, copy, rotate or scale entities.", vec![
            radio("type", "Type", &[("move", "Move"), ("copy", "Copy"), ("rotate", "Rotate"), ("scale", "Scale")]),
            picks("entities", "Entities", &[K::SketchEntity, K::SketchPoint]),
            pt("base", "Base point"),
            offset("dx", "Delta X", 0.0).shown("type", &["move", "copy"]),
            offset("dy", "Delta Y", 0.0).shown("type", &["move", "copy"]),
            angle("angle", "Angle", 0.0).shown("type", &["rotate"]),
            qty("factor", "Factor", "scale", Some(1.0)).shown("type", &["scale"]),
        ]),
        entity("sketch.use-edges", "Use edges", "use-edges", "reference", "Bring model edges into the sketch by projection, intersection, silhouette or offset.", vec![
            radio("type", "Type", &[("project", "Project"), ("intersection", "Intersection"), ("silhouette", "Silhouette"), ("offset_on_surface", "Offset on surface")]),
            picks("items", "Edges, faces or loops", &[K::Edge, K::Face, K::Body]),
            length("distance", "Offset", 1.0).shown("type", &["offset_on_surface"]),
        ]),
        cmd("sketch.dimension", "Dimension", "dimension", S)
            .desc("Add a driving or driven dimension.")
            .kw(&["smart dimension", "distance", "radius", "angle"])
            .inputs(vec![
                choice("type", "Type", DIMENSIONS),
                picks("entities", "Entities", &[K::SketchEntity, K::SketchPoint]),
                qty("value", "Value", "length", None).optional().help("Leave empty to keep the current size"),
                flag("driven", "Driven (reference) dimension", false),
            ])
            .when(sketching())
            .preview()
            .edit(edit(EditOp::Add, Section::Sketch).item("dimension")),
        cmd("sketch.constraint", "Add relation", "constraint", S)
            .desc("Add a geometric relation between sketch entities.")
            .kw(&["constraint", "relation", "coincident", "tangent", "parallel"])
            .inputs(vec![choice("relation", "Relation", CONSTRAINTS), picks("entities", "Entities", &[K::SketchEntity, K::SketchPoint, K::Edge])])
            .when(sketching())
            .on(&[K::SketchEntity, K::SketchPoint])
            .preview()
            .edit(edit(EditOp::Add, Section::Sketch).item("constraint")),
        cmd("sketch.auto-constrain", "Auto-constrain", "auto-constrain", S)
            .desc("Suggest relations and dimensions; accept them one at a time.")
            .kw(&["infer", "suggest"])
            .inputs(vec![picks("entities", "Entities", &[K::SketchEntity]).optional(), length("tolerance", "Tolerance", 0.01)])
            .when(sketching())
            .preview()
            .edit(edit(EditOp::Add, Section::Sketch).item("constraint").variant("auto", "true")),
        cmd("sketch.fully-define", "Fully define", "fully-define", S)
            .desc("Add relations and dimensions until the sketch has no degrees of freedom.")
            .inputs(vec![
                pick("datum_x", "Horizontal datum", &[K::SketchEntity, K::SketchPoint, K::Plane]).optional(),
                pick("datum_y", "Vertical datum", &[K::SketchEntity, K::SketchPoint, K::Plane]).optional(),
                radio("scheme", "Dimensions", &[("baseline", "Baseline"), ("chain", "Chain"), ("ordinate", "Ordinate")]),
            ])
            .when(sketching())
            .preview()
            .edit(edit(EditOp::Add, Section::Sketch).item("dimension").variant("auto", "true")),
        cmd("sketch.block", "Block", "block", S)
            .desc("Make a block from entities, insert one, or explode it.")
            .inputs(vec![
                radio("action", "Action", &[("make", "Make"), ("insert", "Insert"), ("explode", "Explode")]),
                picks("entities", "Entities", &[K::SketchEntity]).shown("action", &["make", "explode"]),
                text("name", "Block", Some("hinge-link")).shown("action", &["make", "insert"]),
                pt("at", "At").shown("action", &["insert"]),
            ])
            .when(sketching())
            .edit(edit(EditOp::Edit, Section::Sketch).item("block")),
        cmd("sketch.image", "Image", "image", S)
            .desc("Place a picture to trace over.")
            .kw(&["trace", "picture"])
            .inputs(vec![text("file", "Image", Some("sketch.png")), length("width", "Width", 100.0), qty("opacity", "Opacity", "ratio", Some(0.5))])
            .when(sketching())
            .edit(edit(EditOp::Add, Section::Sketch).item("image")),
        cmd("sketch.dxf-import", "DXF import", "import", S)
            .desc("Bring DXF or DWG geometry into the sketch.")
            .inputs(vec![text("file", "File", Some("profile.dxf")), choice("units", "Units", &[("mm", "Millimeters"), ("in", "Inches")])])
            .when(sketching())
            .edit(edit(EditOp::Import, Section::Sketch).format("dxf")),
        cmd("sketch.show", "Show", "visibility", S)
            .desc("Show or hide relation glyphs, dimensions and DOF arrows.")
            .inputs(vec![flag("glyphs", "Relation glyphs", true), flag("dimensions", "Dimensions", true), flag("dof", "DOF arrows", true)])
            .when(sketching())
            .edit(view_action("sketch_display")),
        cmd("sketch.check", "Check sketch for feature", "check", S)
            .desc("Check the sketch for a feature: open contours, overlaps, self-intersections.")
            .inputs(vec![choice("for", "Use", &[("extrude", "Extrude"), ("revolve", "Revolve"), ("sweep_path", "Sweep path"), ("loft_profile", "Loft profile")])])
            .when(sketching())
            .edit(check("sketch")),
        cmd("sketch.repair", "Repair sketch", "repair", S)
            .desc("Close small gaps and remove overlapping entities.")
            .inputs(vec![length("gap", "Close gaps up to", 0.05)])
            .when(sketching())
            .edit(edit(EditOp::Edit, Section::Sketch).item("repair")),
        cmd("sketch.exit", "Exit sketch", "confirm", S)
            .desc("Keep the sketch changes and leave the sketch.")
            .when(sketching())
            .edit(edit(EditOp::ExitMode, Section::Sketch)),
        cmd("sketch.discard", "Discard", "cancel", S)
            .desc("Leave the sketch without keeping the changes.")
            .when(sketching())
            .edit(edit(EditOp::DiscardMode, Section::Sketch)),
    ];
    catalog.add_commands(commands);

    let constraint_menu = variants(
        "sketch.constraint.menu",
        "Relations",
        "constraint",
        "sketch.constraint",
        "relation",
        CONSTRAINTS,
    );
    let mut dimension_entries: Vec<MenuEntry> = DIMENSIONS
        .iter()
        .map(|(v, l)| preset("sketch.dimension", l, "type", v))
        .collect();
    dimension_entries.push({
        let mut entry = preset("sketch.dimension", "Driven dimension", "type", "smart");
        entry
            .values
            .insert("driven".into(), serde_json::Value::Bool(true));
        entry
    });
    for editor in [Part, Assembly] {
        catalog.rows.push(EditorRows {
            editor,
            mode: EditMode::Sketch,
            default_tab: "sketch".into(),
            document_bar: Vec::new(),
            view_cluster: vec![
                menu(
                    "sketch.view.menu",
                    "View",
                    "view-iso",
                    vec![
                        e("view.normal-to"),
                        e("view.isometric"),
                        e("view.front"),
                        e("view.top"),
                        e("view.right"),
                        e("view.fit"),
                    ],
                ),
                c("view.normal-to"),
                c("view.measure"),
            ],
            shortcut_box: [
                "sketch.line",
                "sketch.rectangle",
                "sketch.circle",
                "sketch.arc",
                "sketch.dimension",
                "sketch.constraint",
                "sketch.trim",
                "sketch.offset",
                "sketch.construction",
            ]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
            mode_actions: vec![
                "sketch.exit".into(),
                "sketch.discard".into(),
                "design.exit-nested".into(),
            ],
            tabs: vec![tab(
                "sketch",
                "Sketch",
                vec![
                    group(
                        "draw",
                        "Draw",
                        vec![
                            variants(
                                "sketch.line.menu",
                                "Line",
                                "line",
                                "sketch.line",
                                "type",
                                &[
                                    ("line", "Line"),
                                    ("centerline", "Centerline"),
                                    ("midpoint", "Midpoint line"),
                                ],
                            ),
                            variants(
                                "sketch.rectangle.menu",
                                "Rectangle",
                                "rectangle",
                                "sketch.rectangle",
                                "type",
                                &[
                                    ("corner", "Corner rectangle"),
                                    ("center", "Center rectangle"),
                                    ("three_point", "3-point rectangle"),
                                    ("parallelogram", "Parallelogram"),
                                ],
                            ),
                            variants(
                                "sketch.circle.menu",
                                "Circle",
                                "circle",
                                "sketch.circle",
                                "type",
                                &[("center", "Circle"), ("three_point", "3-point circle")],
                            ),
                            variants(
                                "sketch.arc.menu",
                                "Arc",
                                "arc",
                                "sketch.arc",
                                "type",
                                &[
                                    ("center", "Center point arc"),
                                    ("tangent", "Tangent arc"),
                                    ("three_point", "3-point arc"),
                                ],
                            ),
                            variants(
                                "sketch.slot.menu",
                                "Slot",
                                "slot",
                                "sketch.slot",
                                "type",
                                &[
                                    ("straight", "Straight slot"),
                                    ("center", "Center point slot"),
                                    ("arc", "Arc slot"),
                                ],
                            ),
                            c("sketch.polygon"),
                            variants(
                                "sketch.spline.menu",
                                "Spline",
                                "spline",
                                "sketch.spline",
                                "type",
                                &[
                                    ("through_points", "Spline"),
                                    ("control_points", "Control point spline"),
                                    ("equation", "Equation spline"),
                                    ("style", "Style spline"),
                                ],
                            ),
                            variants(
                                "sketch.ellipse.menu",
                                "Ellipse",
                                "ellipse",
                                "sketch.ellipse",
                                "type",
                                &[
                                    ("full", "Ellipse"),
                                    ("partial", "Partial ellipse"),
                                    ("parabola", "Parabola"),
                                    ("conic", "Conic"),
                                ],
                            ),
                            c("sketch.point"),
                            c("sketch.text"),
                        ],
                    ),
                    group("mode", "Mode", vec![c("sketch.construction")]),
                    group(
                        "corners",
                        "Corners",
                        vec![c("sketch.fillet"), c("sketch.chamfer")],
                    ),
                    group(
                        "edit",
                        "Edit",
                        vec![
                            variants(
                                "sketch.trim.menu",
                                "Trim",
                                "trim",
                                "sketch.trim",
                                "type",
                                &[
                                    ("power", "Power trim"),
                                    ("corner", "Corner trim"),
                                    ("inside", "Trim away inside"),
                                    ("outside", "Trim away outside"),
                                ],
                            ),
                            c("sketch.extend"),
                            c("sketch.offset"),
                            c("sketch.split"),
                            c("sketch.stretch"),
                        ],
                    ),
                    group(
                        "repeat",
                        "Repeat",
                        vec![
                            c("sketch.mirror"),
                            variants(
                                "sketch.pattern.menu",
                                "Pattern",
                                "pattern",
                                "sketch.pattern",
                                "layout",
                                &[
                                    ("linear", "Linear pattern"),
                                    ("circular", "Circular pattern"),
                                ],
                            ),
                            variants(
                                "sketch.transform.menu",
                                "Transform",
                                "move",
                                "sketch.transform",
                                "type",
                                &[
                                    ("move", "Move"),
                                    ("copy", "Copy"),
                                    ("rotate", "Rotate"),
                                    ("scale", "Scale"),
                                ],
                            ),
                        ],
                    ),
                    group(
                        "reference",
                        "Reference",
                        vec![variants(
                            "sketch.use-edges.menu",
                            "Use edges",
                            "use-edges",
                            "sketch.use-edges",
                            "type",
                            &[
                                ("project", "Project edges"),
                                ("intersection", "Intersection curve"),
                                ("silhouette", "Silhouette edges"),
                                ("offset_on_surface", "Offset on surface"),
                            ],
                        )],
                    ),
                    group(
                        "constrain",
                        "Constrain",
                        vec![
                            menu(
                                "sketch.dimension.menu",
                                "Dimension",
                                "dimension",
                                dimension_entries.clone(),
                            ),
                            constraint_menu.clone(),
                            c("sketch.auto-constrain"),
                            c("sketch.fully-define"),
                        ],
                    ),
                    group(
                        "reuse",
                        "Reuse",
                        vec![
                            variants(
                                "sketch.block.menu",
                                "Block",
                                "block",
                                "sketch.block",
                                "action",
                                &[
                                    ("make", "Make block"),
                                    ("insert", "Insert block"),
                                    ("explode", "Explode block"),
                                ],
                            ),
                            c("sketch.image"),
                            c("sketch.dxf-import"),
                        ],
                    ),
                    group(
                        "show",
                        "Show",
                        vec![menu(
                            "sketch.show.menu",
                            "Show",
                            "visibility",
                            vec![e("sketch.show"), e("sketch.check"), e("sketch.repair")],
                        )],
                    ),
                ],
            )],
        });
    }
}

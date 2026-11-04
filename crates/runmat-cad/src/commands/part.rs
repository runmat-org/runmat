//! Part editor commands (`.part`): Sketch, Features, Direct, Surfaces, Sheet Metal, Weldments,
//! Mold, Evaluate, Annotate, Simulate and Manage, per the UI target §6.

use super::view::cluster_3d;
use super::*;
use EditorKind::Part;
use SelectKind as K;

const P: &[EditorKind] = &[Part];

const END_CONDITIONS: &[(&str, &str)] = &[
    ("blind", "Blind"),
    ("through_all", "Through all"),
    ("up_to_next", "Up to next"),
    ("up_to_vertex", "Up to vertex"),
    ("up_to_face", "Up to face"),
    ("offset_from_face", "Offset from face"),
    ("up_to_body", "Up to body"),
    ("mid_plane", "Mid-plane"),
];

const BODY_OPS: &[(&str, &str)] = &[
    ("add", "Add"),
    ("remove", "Remove"),
    ("intersect", "Intersect"),
    ("new", "New body"),
];

fn end_fields(prefix: &str, label: &str) -> Vec<Field> {
    let id = |s: &str| format!("{prefix}{s}");
    vec![
        choice(&id("end"), label, END_CONDITIONS),
        length(&id("distance"), "Depth", 10.0)
            .shown(&id("end"), &["blind", "mid_plane", "offset_from_face"])
            .drag(Manipulator::Arrow),
        pick(
            &id("up_to"),
            "Up to",
            &[K::Face, K::Vertex, K::Body, K::Plane],
        )
        .shown(
            &id("end"),
            &[
                "up_to_vertex",
                "up_to_face",
                "offset_from_face",
                "up_to_body",
            ],
        ),
        flag(&id("flip"), "Flip direction", false),
    ]
}

fn op_field() -> Field {
    radio("op", "Op", BODY_OPS)
}

fn merge_field() -> Field {
    pick("merge", "Merge into", &[K::Body])
        .optional()
        .shown("op", &["add", "remove", "intersect"])
}

fn profile() -> Field {
    picks("profile", "Profile", &[K::SketchRegion, K::Sketch, K::Face])
}

fn bodies_exist() -> Predicate {
    has(Facet::Bodies)
}

fn sheet_metal() -> Predicate {
    has(Facet::SheetMetal)
}

pub(super) fn add(catalog: &mut Catalog) {
    let mut commands = Vec::new();
    commands.extend(features());
    commands.extend(direct());
    commands.extend(surfaces());
    commands.extend(sheet_metal_commands());
    commands.extend(weldments());
    commands.extend(mold());
    commands.extend(evaluate());
    commands.extend(annotate());
    commands.extend(simulate());
    commands.extend(manage());
    commands.extend(tree_actions());
    catalog.add_commands(commands);
    catalog.rows.push(rows(catalog));
}

fn features() -> Vec<CommandDescriptor> {
    let mut extrude_inputs = vec![
        profile(),
        op_field(),
        choice(
            "from",
            "From",
            &[
                ("sketch_plane", "Sketch plane"),
                ("surface", "Surface or face"),
                ("vertex", "Vertex"),
                ("offset", "Offset"),
            ],
        ),
    ];
    extrude_inputs.extend(end_fields("", "End"));
    extrude_inputs.push(flag("second_direction", "Second direction", false));
    extrude_inputs.extend(
        end_fields("second_", "Second end")
            .into_iter()
            .map(|f| f.if_checked("second_direction")),
    );
    extrude_inputs.extend([
        flag("draft", "Draft", false),
        angle("draft_angle", "Draft angle", 0.0)
            .if_checked("draft")
            .drag(Manipulator::Angle),
        flag("draft_outward", "Draft outward", false).if_checked("draft"),
        flag("thin", "Thin", false),
        length("thin_thickness", "Thickness", 1.0).if_checked("thin"),
        choice(
            "thin_side",
            "Thin side",
            &[
                ("one", "One side"),
                ("mid", "Mid-plane"),
                ("two", "Two sides"),
            ],
        )
        .if_checked("thin"),
        merge_field(),
    ]);
    vec![
        cmd("part.sketch", "Sketch", "sketch", P)
            .desc("Start a sketch on a plane or planar face.")
            .kw(&["2d", "draw", "profile"])
            .inputs(vec![pick("plane", "On", &[K::Plane, K::Face])])
            .edit(edit(EditOp::EnterMode, Section::Features).item("sketch")),
        cmd("part.sketch-3d", "3D Sketch", "sketch-3d", P)
            .desc("Start a sketch in space for paths and routes.")
            .kw(&["3d", "path", "route"])
            .edit(edit(EditOp::EnterMode, Section::Features).item("sketch").variant("space", "3d")),
        cmd("part.extrude", "Extrude", "extrude", P)
            .desc("Extend a profile along a direction to add, cut or intersect material.")
            .kw(&["boss", "pad", "cut", "extruded cut"])
            .inputs(extrude_inputs)
            .when(any(vec![selected(&[K::Sketch, K::SketchRegion, K::Face]), has(Facet::Sketches)]))
            .preview()
            .edit(add_feature("extrude")),
        cmd("part.revolve", "Revolve", "revolve", P)
            .desc("Turn a profile around an axis.")
            .kw(&["revolved boss", "revolved cut", "lathe"])
            .inputs(vec![
                profile(),
                pick("axis", "Axis", &[K::Axis, K::Edge, K::SketchEntity]),
                op_field(),
                choice("type", "Type", &[("one", "One direction"), ("mid", "Mid-plane"), ("two", "Two directions")]),
                angle("angle", "Angle", 360.0).min(0.0).max(360.0).drag(Manipulator::Angle),
                angle("angle2", "Second angle", 0.0).shown("type", &["two"]),
                flag("thin", "Thin", false),
                length("thin_thickness", "Thickness", 1.0).if_checked("thin"),
                merge_field(),
            ])
            .when(has(Facet::Sketches))
            .preview()
            .edit(add_feature("revolve")),
        cmd("part.sweep", "Sweep", "sweep", P)
            .desc("Move a profile along a path, with guide curves, twist and tangency.")
            .kw(&["swept", "path"])
            .inputs(vec![
                profile(),
                pick("path", "Path", &[K::Sketch, K::Curve, K::Edge]),
                picks("guides", "Guide curves", &[K::Sketch, K::Curve, K::Edge]).optional(),
                op_field(),
                choice("orientation", "Orientation", &[("follow_path", "Follow path"), ("keep_normal", "Keep normal constant"), ("follow_path_first_guide", "Follow path and first guide")]),
                choice("twist", "Twist", &[("none", "None"), ("turns", "Turns"), ("angle", "Angle")]),
                angle("twist_angle", "Twist", 0.0).shown("twist", &["turns", "angle"]),
                choice("start_tangency", "Start tangency", &[("none", "None"), ("normal", "Normal to profile")]),
                choice("end_tangency", "End tangency", &[("none", "None"), ("normal", "Normal to profile")]),
                flag("thin", "Thin", false),
                length("thin_thickness", "Thickness", 1.0).if_checked("thin"),
                merge_field(),
            ])
            .when(has(Facet::Sketches))
            .preview()
            .edit(add_feature("sweep")),
        cmd("part.loft", "Loft", "loft", P)
            .desc("Blend between two or more profiles, with guides and end tangency.")
            .kw(&["blend", "lofted"])
            .inputs(vec![
                picks("profiles", "Profiles", &[K::Sketch, K::SketchRegion, K::Face, K::Point]),
                picks("guides", "Guide curves", &[K::Sketch, K::Curve, K::Edge]).optional(),
                pick("centerline", "Centerline", &[K::Sketch, K::Curve, K::Edge]).optional(),
                op_field(),
                choice("start_tangency", "Start tangency", &[("none", "None"), ("normal", "Normal to profile"), ("direction", "Direction vector"), ("faces", "Tangent to faces")]),
                choice("end_tangency", "End tangency", &[("none", "None"), ("normal", "Normal to profile"), ("direction", "Direction vector"), ("faces", "Tangent to faces")]),
                flag("closed", "Close the loft", false),
                flag("thin", "Thin", false),
                length("thin_thickness", "Thickness", 1.0).if_checked("thin"),
                merge_field(),
            ])
            .when(has(Facet::Sketches))
            .preview()
            .edit(add_feature("loft")),
        cmd("part.boundary", "Boundary", "boundary", P)
            .desc("Make a solid from curves in two directions.")
            .inputs(vec![
                picks("direction1", "Direction 1", &[K::Sketch, K::Curve, K::Edge]),
                picks("direction2", "Direction 2", &[K::Sketch, K::Curve, K::Edge]).optional(),
                op_field(),
                merge_field(),
            ])
            .when(has(Facet::Sketches))
            .preview()
            .edit(add_feature("loft").variant("boundary", "true")),
        cmd("part.hole", "Hole", "hole", P)
            .desc("Add standard holes: simple, counterbore, countersink, tapped, pipe tap or slot.")
            .kw(&["hole wizard", "tap", "counterbore", "countersink", "thread"])
            .inputs(vec![
                choice("type", "Type", &[("simple", "Simple"), ("counterbore", "Counterbore"), ("countersink", "Countersink"), ("tapped", "Tapped"), ("pipe_tap", "Pipe tap"), ("slot", "Slot")]),
                choice("standard", "Standard", &[("iso", "ISO"), ("ansi_metric", "ANSI metric"), ("ansi_inch", "ANSI inch")]),
                text("size", "Size", Some("M6")),
                choice("fit", "Fit", &[("normal", "Normal"), ("close", "Close"), ("loose", "Loose")]).shown("type", &["simple", "counterbore", "countersink", "slot"]),
                length("diameter", "Diameter", 6.0).shown("type", &["simple"]).drag(Manipulator::Radius),
                length("cbore_diameter", "Counterbore diameter", 11.0).shown("type", &["counterbore"]),
                length("cbore_depth", "Counterbore depth", 6.0).shown("type", &["counterbore"]),
                length("csink_diameter", "Countersink diameter", 12.0).shown("type", &["countersink"]),
                angle("csink_angle", "Countersink angle", 90.0).shown("type", &["countersink"]),
                length("slot_length", "Slot length", 10.0).shown("type", &["slot"]),
                choice("end", "End", &[("blind", "Blind"), ("through_all", "Through all"), ("up_to_next", "Up to next"), ("up_to_face", "Up to face")]),
                length("depth", "Depth", 12.0).shown("end", &["blind"]).drag(Manipulator::Arrow),
                length("thread_depth", "Thread depth", 10.0).shown("type", &["tapped", "pipe_tap"]),
                flag("cosmetic_thread", "Cosmetic thread", true),
                pick("face", "Face", &[K::Face]),
                list("positions", "Positions", pick("position", "Position", &[K::SketchPoint, K::Vertex, K::Point]).drag(Manipulator::Point)),
                flag("series", "Hole series across parts", false),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("hole")),
        cmd("part.fillet", "Fillet", "fillet", P)
            .desc("Round edges or blend faces: constant, variable, face, full round or setback.")
            .kw(&["round", "blend", "radius"])
            .inputs(vec![
                choice("type", "Type", &[("constant", "Constant radius"), ("variable", "Variable radius"), ("face", "Face fillet"), ("full_round", "Full round"), ("setback", "Setback")]),
                picks("edges", "Edges and faces", &[K::Edge, K::Face, K::Feature]),
                length("radius", "Radius", 2.0).shown("type", &["constant", "face", "setback"]).drag(Manipulator::Radius),
                list("radii", "Radii at vertices", length("radius", "Radius", 2.0)).shown("type", &["variable"]),
                choice("profile", "Profile", &[("circular", "Circular"), ("conic", "Conic"), ("curvature", "Curvature continuous")]),
                length("setback", "Setback distance", 3.0).shown("type", &["setback"]),
                flag("tangent_propagation", "Tangent propagation", true),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("fillet")),
        cmd("part.chamfer", "Chamfer", "chamfer", P)
            .desc("Bevel edges or vertices by distance and angle or two distances.")
            .kw(&["bevel"])
            .inputs(vec![
                choice("type", "Type", &[("distance_angle", "Distance and angle"), ("distance_distance", "Two distances"), ("equal", "Equal distance"), ("vertex", "Vertex")]),
                picks("edges", "Edges", &[K::Edge, K::Vertex, K::Face]),
                length("distance", "Distance", 1.0).drag(Manipulator::Arrow),
                length("distance2", "Second distance", 1.0).shown("type", &["distance_distance", "vertex"]),
                angle("angle", "Angle", 45.0).shown("type", &["distance_angle"]),
                flag("flip", "Flip direction", false),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("chamfer")),
        cmd("part.shell", "Shell", "shell", P)
            .desc("Hollow a body, with different thicknesses on chosen faces.")
            .kw(&["hollow", "wall"])
            .inputs(vec![
                pick("body", "Body", &[K::Body]),
                picks("open_faces", "Faces to remove", &[K::Face]).optional(),
                length("thickness", "Thickness", 2.0).drag(Manipulator::Arrow),
                flag("outward", "Shell outward", false),
                list("face_thickness", "Other thicknesses", length("thickness", "Thickness", 3.0)).optional(),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("shell")),
        cmd("part.draft", "Draft", "draft", P)
            .desc("Taper faces for molding by neutral plane, parting line or step.")
            .kw(&["taper", "mold"])
            .inputs(vec![
                choice("type", "Type", &[("neutral_plane", "Neutral plane"), ("parting_line", "Parting line"), ("step", "Step draft")]),
                picks("faces", "Faces to draft", &[K::Face]),
                pick("neutral", "Neutral plane", &[K::Plane, K::Face]).shown("type", &["neutral_plane"]),
                picks("parting", "Parting line", &[K::Edge, K::Curve]).shown("type", &["parting_line", "step"]),
                angle("angle", "Angle", 1.0).min(0.0).max(89.0).drag(Manipulator::Angle),
                flag("flip", "Flip direction", false),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("draft")),
        cmd("part.rib", "Rib", "rib", P)
            .desc("Add a thin wall from an open sketch.")
            .kw(&["web", "stiffener"])
            .inputs(vec![
                pick("profile", "Profile", &[K::Sketch]),
                length("thickness", "Thickness", 2.0),
                choice("side", "Thickness side", &[("mid", "Both sides"), ("one", "One side"), ("other", "Other side")]),
                choice("direction", "Extrude", &[("parallel", "Parallel to sketch"), ("normal", "Normal to sketch")]),
                flag("reverse", "Flip material side", false),
                flag("draft", "Draft", false),
                angle("draft_angle", "Draft angle", 1.0).if_checked("draft"),
            ])
            .when(all(vec![bodies_exist(), has(Facet::Sketches)]))
            .preview()
            .edit(add_feature("rib")),
        cmd("part.pattern", "Pattern", "pattern", P)
            .desc("Repeat features, faces or bodies: linear, circular, curve, sketch, table, fill or variable.")
            .kw(&["array", "repeat", "linear pattern", "circular pattern"])
            .inputs(vec![
                choice("layout", "Layout", &[("linear", "Linear"), ("circular", "Circular"), ("curve", "Curve-driven"), ("sketch", "Sketch-driven"), ("table", "Table-driven"), ("fill", "Fill"), ("variable", "Variable")]),
                picks("seeds", "Features, faces or bodies", &[K::Feature, K::Face, K::Body]),
                pick("direction", "Direction", &[K::Edge, K::Axis, K::Plane, K::Face]).shown("layout", &["linear", "fill"]),
                length("spacing", "Spacing", 10.0).shown("layout", &["linear", "fill"]).drag(Manipulator::Arrow),
                count("instances", "Instances", 3, 1).shown("layout", &["linear", "circular", "curve"]),
                flag("second_direction", "Second direction", false).shown("layout", &["linear"]),
                pick("direction2", "Second direction", &[K::Edge, K::Axis, K::Plane]).if_checked("second_direction"),
                length("spacing2", "Second spacing", 10.0).if_checked("second_direction"),
                count("instances2", "Second instances", 2, 1).if_checked("second_direction"),
                pick("axis", "Axis", &[K::Axis, K::Edge, K::Face]).shown("layout", &["circular"]),
                angle("angle", "Angle", 360.0).shown("layout", &["circular"]),
                flag("equal_spacing", "Equal spacing", true).shown("layout", &["circular"]),
                pick("curve", "Curve", &[K::Curve, K::Edge, K::Sketch]).shown("layout", &["curve"]),
                pick("sketch", "Sketch points", &[K::Sketch]).shown("layout", &["sketch"]),
                text("table", "Table", Some("coordinates.csv")).shown("layout", &["table"]),
                pick("boundary", "Fill boundary", &[K::Face, K::Sketch]).shown("layout", &["fill"]),
                list("skip", "Instances to skip", count("instance", "Instance", 1, 1)).optional(),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("pattern")),
        cmd("part.mirror", "Mirror", "mirror", P)
            .desc("Mirror features, faces or bodies about a plane.")
            .kw(&["symmetry", "reflect"])
            .inputs(vec![
                pick("plane", "Mirror plane", &[K::Plane, K::Face]),
                picks("seeds", "Features, faces or bodies", &[K::Feature, K::Face, K::Body]),
                flag("merge", "Merge result", true),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("mirror")),
        cmd("part.boolean", "Boolean", "boolean", P)
            .desc("Combine bodies by union, subtract or intersect.")
            .kw(&["combine", "union", "subtract", "intersect"])
            .inputs(vec![
                radio("kind", "Operation", &[("union", "Union"), ("subtract", "Subtract"), ("intersect", "Intersect")]),
                pick("target", "Target body", &[K::Body]),
                picks("tools", "Tool bodies", &[K::Body]),
                flag("keep_tools", "Keep tool bodies", false),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("boolean")),
        cmd("part.split", "Split", "split", P)
            .desc("Split a body with a plane, face or surface.")
            .inputs(vec![
                pick("body", "Body", &[K::Body]),
                pick("tool", "Split with", &[K::Plane, K::Face]),
                flag("keep_all", "Keep all pieces", true),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("split")),
        cmd("part.body-move", "Move/copy body", "move", P)
            .desc("Move, rotate or copy bodies.")
            .kw(&["translate", "rotate", "copy"])
            .inputs(vec![
                picks("bodies", "Bodies", &[K::Body]),
                offset("dx", "Delta X", 0.0).drag(Manipulator::Triad),
                offset("dy", "Delta Y", 0.0),
                offset("dz", "Delta Z", 0.0),
                pick("axis", "Rotate about", &[K::Axis, K::Edge]).optional(),
                angle("angle", "Angle", 0.0),
                flag("copy", "Copy", false),
                count("copies", "Copies", 1, 1).if_checked("copy"),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature_variant("body_op", "op", "translate")),
        cmd("part.body-delete", "Delete/keep body", "delete-body", P)
            .desc("Delete bodies, or keep only the chosen ones.")
            .inputs(vec![
                picks("bodies", "Bodies", &[K::Body]),
                radio("keep", "Mode", &[("delete", "Delete"), ("keep", "Keep")]),
            ])
            .when(bodies_exist())
            .edit(add_feature_variant("body_op", "op", "delete")),
        cmd("part.body-scale", "Scale body", "scale", P)
            .desc("Scale bodies about a point.")
            .inputs(vec![
                picks("bodies", "Bodies", &[K::Body]),
                choice("about", "About", &[("centroid", "Centroid"), ("origin", "Origin"), ("point", "Point")]),
                pick("point", "Point", &[K::Point, K::Vertex]).shown("about", &["point"]),
                qty("factor", "Factor", "scale", Some(1.0)).min(0.0),
                flag("uniform", "Uniform", true),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature_variant("mold", "mold", "shrink")),
        cmd("part.body-promote", "Promote to part", "promote", P)
            .desc("Make a body its own part in this file, with its own number and material.")
            .kw(&["multibody", "part"])
            .inputs(vec![pick("body", "Body", &[K::Body]), text("name", "Part name", Some("bracket"))])
            .when(selected(&[K::Body]))
            .edit(edit(EditOp::Add, Section::Properties).item("part")),
        cmd("part.body-combine", "Combine bodies", "combine", P)
            .desc("Merge touching bodies into one.")
            .inputs(vec![picks("bodies", "Bodies", &[K::Body])])
            .when(bodies_exist())
            .edit(add_feature("boolean").variant("kind", "union")),
        cmd("part.body-indent", "Indent", "indent", P)
            .desc("Press one body into another, leaving a clearance.")
            .inputs(vec![
                pick("target", "Target body", &[K::Body]),
                pick("tool", "Tool body", &[K::Body]),
                length("clearance", "Clearance", 0.5),
                length("thickness", "Thickness", 1.0),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("boolean").variant("kind", "subtract")),
        cmd("part.plane", "Plane", "plane", P)
            .desc("Add a reference plane: offset, angle, 3-point, mid, normal to curve or tangent to surface.")
            .kw(&["datum plane", "reference plane", "workplane"])
            .inputs(vec![
                choice("type", "Type", &[("offset", "Offset"), ("angle", "At angle"), ("three_point", "Three points"), ("mid", "Mid-plane"), ("normal_to_curve", "Normal to curve"), ("tangent_to_face", "Tangent to surface")]),
                pick("base", "Reference", &[K::Plane, K::Face]).shown("type", &["offset", "angle", "mid"]),
                offset("distance", "Distance", 10.0).shown("type", &["offset"]).drag(Manipulator::Arrow),
                count("count", "Number of planes", 1, 1).shown("type", &["offset"]),
                pick("axis", "Axis", &[K::Axis, K::Edge]).shown("type", &["angle"]),
                angle("angle", "Angle", 45.0).shown("type", &["angle"]).drag(Manipulator::Angle),
                picks("points", "Points", &[K::Vertex, K::Point, K::SketchPoint]).shown("type", &["three_point"]),
                pick("other", "Second reference", &[K::Plane, K::Face]).shown("type", &["mid"]),
                pick("curve", "Curve", &[K::Curve, K::Edge]).shown("type", &["normal_to_curve"]),
                pick("at", "At point", &[K::Vertex, K::Point]).shown("type", &["normal_to_curve"]),
                pick("face", "Surface", &[K::Face]).shown("type", &["tangent_to_face"]),
                flag("flip", "Flip", false),
            ])
            .preview()
            .edit(add_feature_variant("plane", "plane", "offset")),
        cmd("part.axis", "Axis", "axis", P)
            .desc("Add a reference axis.")
            .kw(&["datum axis"])
            .inputs(vec![
                choice("type", "Type", &[("edge", "Line or edge"), ("two_planes", "Two planes"), ("two_points", "Two points"), ("cylinder", "Cylindrical face"), ("point_face", "Point and face")]),
                picks("references", "References", &[K::Edge, K::Plane, K::Face, K::Vertex, K::Point]),
            ])
            .preview()
            .edit(add_feature("axis")),
        cmd("part.point", "Point", "point", P)
            .desc("Add a reference point.")
            .kw(&["datum point"])
            .inputs(vec![
                choice("type", "Type", &[("coordinates", "Coordinates"), ("center", "Arc center"), ("face_center", "Face center"), ("intersection", "Intersection"), ("along_curve", "Along curve")]),
                picks("references", "References", &[K::Edge, K::Face, K::Vertex, K::Curve]).shown("type", &["center", "face_center", "intersection", "along_curve"]),
                offset("x", "X", 0.0).shown("type", &["coordinates"]),
                offset("y", "Y", 0.0).shown("type", &["coordinates"]),
                offset("z", "Z", 0.0).shown("type", &["coordinates"]),
                length("distance", "Distance along", 0.0).shown("type", &["along_curve"]),
            ])
            .preview()
            .edit(add_feature("point")),
        cmd("part.coordinate-system", "Coordinate system", "csys", P)
            .desc("Add a coordinate system from an origin and axes.")
            .kw(&["frame", "csys"])
            .inputs(vec![
                pick("origin", "Origin", &[K::Vertex, K::Point]),
                pick("x", "X axis", &[K::Edge, K::Axis]).optional(),
                pick("y", "Y axis", &[K::Edge, K::Axis]).optional(),
            ])
            .preview()
            .edit(add_feature("frame")),
        cmd("part.mate-connector", "Mate connector", "mate-connector", P)
            .desc("Add a named frame other files mate to.")
            .kw(&["connector", "interface"])
            .inputs(vec![
                text("name", "Name", Some("pivot")),
                pick("on", "On", &[K::Face, K::Edge, K::Vertex, K::Point]),
                choice("origin", "Origin", &[("centroid", "Centroid"), ("center", "Circle center"), ("vertex", "Vertex")]),
                flag("flip", "Flip primary axis", false),
                angle("rotate", "Rotate", 0.0),
            ])
            .preview()
            .edit(add_feature("mate_connector")),
        curve("part.curve-helix", "Helix/spiral", "helix", "helix", "Add a helix or spiral by pitch, turns and height.", vec![
            choice("type", "Type", &[("helix", "Helix"), ("spiral", "Spiral")]),
            pick("on", "Circle", &[K::Sketch, K::Edge]),
            length("pitch", "Pitch", 2.0),
            count("revolutions", "Revolutions", 5, 1),
            angle("start_angle", "Start angle", 0.0),
            flag("clockwise", "Clockwise", false),
            flag("taper", "Taper", false),
            angle("taper_angle", "Taper angle", 0.0).if_checked("taper"),
        ]),
        curve("part.curve-projected", "Projected curve", "projected-curve", "projected", "Project a sketch onto faces.", vec![
            pick("sketch", "Sketch", &[K::Sketch]),
            picks("faces", "Faces", &[K::Face]),
            flag("both", "Both directions", false),
        ]),
        curve("part.curve-composite", "Composite curve", "composite-curve", "composite", "Join edges and curves into one.", vec![picks("curves", "Curves", &[K::Edge, K::Curve, K::SketchEntity])]),
        curve("part.curve-split-line", "Split line", "split-line", "split_line", "Split faces along a projected sketch, silhouette or intersection.", vec![
            choice("type", "Type", &[("projection", "Projection"), ("silhouette", "Silhouette"), ("intersection", "Intersection")]),
            pick("tool", "Sketch or plane", &[K::Sketch, K::Plane, K::Face]),
            picks("faces", "Faces to split", &[K::Face]),
        ]),
        curve("part.curve-intersection", "Intersection curve", "intersection-curve", "intersection", "Make curves where faces and planes meet.", vec![picks("items", "Faces and planes", &[K::Face, K::Plane])]),
        curve("part.curve-through-points", "Curve through points", "through-points", "through_points", "Fit a curve through points.", vec![
            picks("points", "Points", &[K::Vertex, K::Point, K::SketchPoint]),
            flag("closed", "Closed", false),
        ]),
        curve("part.curve-equation", "Equation curve", "equation-curve", "equation", "Draw a curve from x(t), y(t), z(t).", vec![
            text("x", "x(t)", Some("10*cos(t)")),
            text("y", "y(t)", Some("10*sin(t)")),
            text("z", "z(t)", Some("t")),
            qty("t0", "Start t", "ratio", Some(0.0)),
            qty("t1", "End t", "ratio", Some(std::f64::consts::TAU)),
        ]),
        cmd("part.thread", "Thread", "thread", P)
            .desc("Add a cosmetic or modeled thread on a cylindrical face.")
            .kw(&["tap", "screw"])
            .inputs(vec![
                pick("face", "Cylindrical face", &[K::Face]),
                choice("standard", "Standard", &[("iso", "ISO metric"), ("unified", "Unified inch")]),
                text("designation", "Size", Some("M6x1")),
                length("length", "Length", 10.0).optional(),
                radio("form", "Form", &[("cosmetic", "Cosmetic"), ("modeled", "Modeled")]),
                flag("left_hand", "Left hand", false),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("thread")),
        cmd("part.emboss", "Emboss", "emboss", P)
            .desc("Raise or engrave a sketch or text on a face.")
            .kw(&["engrave", "text", "logo"])
            .inputs(vec![
                pick("profile", "Sketch", &[K::Sketch]),
                pick("face", "Face", &[K::Face]),
                length("depth", "Depth", 0.5),
                radio("mode", "Mode", &[("emboss", "Emboss"), ("deboss", "Deboss")]),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("emboss")),
        cmd("part.wrap", "Wrap", "wrap", P)
            .desc("Wrap a sketch around a cylindrical or conical face.")
            .inputs(vec![
                pick("profile", "Sketch", &[K::Sketch]),
                pick("face", "Face", &[K::Face]),
                length("depth", "Depth", 0.5),
                radio("mode", "Mode", &[("emboss", "Emboss"), ("deboss", "Deboss"), ("scribe", "Scribe")]),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature("emboss").variant("wrap", "true")),
        cmd("part.library", "Library feature", "library", P)
            .desc("Insert a feature definition from your library or a company pack.")
            .kw(&["feature def", "pack", "reuse"])
            .inputs(vec![
                text("definition", "Definition", Some("pack/standard_slot")),
                pick("placement", "Placement", &[K::Face, K::Plane, K::Point, K::MateConnector]),
            ])
            .preview()
            .edit(add_feature("library")),
    ]
}

fn curve(
    id: &str,
    label: &str,
    icon: &str,
    variant: &str,
    desc: &str,
    inputs: Vec<Field>,
) -> CommandDescriptor {
    cmd(id, label, icon, P)
        .desc(desc)
        .kw(&["curve"])
        .inputs(inputs)
        .preview()
        .edit(add_feature_variant("curve", "curve", variant))
}

fn direct() -> Vec<CommandDescriptor> {
    let d = |id: &str, label: &str, icon: &str, variant: &str| {
        cmd(id, label, icon, P)
            .when(bodies_exist())
            .preview()
            .edit(add_feature_variant("direct", "direct", variant))
    };
    vec![
        d("direct.move-face", "Move face", "move-face", "move_faces")
            .desc("Move or rotate faces; neighboring faces follow. Recorded as a feature.")
            .kw(&["push", "translate face"])
            .inputs(vec![
                picks("faces", "Faces", &[K::Face]),
                radio(
                    "motion",
                    "Motion",
                    &[
                        ("translate", "Translate"),
                        ("rotate", "Rotate"),
                        ("normal", "Along normal"),
                    ],
                ),
                offset("distance", "Distance", 1.0).drag(Manipulator::Arrow),
                pick("direction", "Direction", &[K::Edge, K::Axis, K::Plane])
                    .shown("motion", &["translate"]),
                pick("axis", "Axis", &[K::Axis, K::Edge]).shown("motion", &["rotate"]),
                angle("angle", "Angle", 0.0).shown("motion", &["rotate"]),
            ]),
        d(
            "direct.offset-face",
            "Offset face",
            "offset-face",
            "move_faces",
        )
        .desc("Offset faces along their normals.")
        .inputs(vec![
            picks("faces", "Faces", &[K::Face]),
            offset("distance", "Distance", 1.0).drag(Manipulator::Arrow),
        ]),
        d(
            "direct.replace-face",
            "Replace face",
            "replace-face",
            "replace_faces",
        )
        .desc("Replace faces with another face or surface.")
        .inputs(vec![
            picks("faces", "Faces to replace", &[K::Face]),
            pick("with", "Replacement", &[K::Face]),
        ]),
        d(
            "direct.delete-face",
            "Delete face",
            "delete-face",
            "delete_faces",
        )
        .desc("Delete faces and heal the gap from the neighbors.")
        .kw(&["heal", "defeature"])
        .inputs(vec![
            picks("faces", "Faces", &[K::Face]),
            radio(
                "heal",
                "Then",
                &[
                    ("heal", "Heal"),
                    ("patch", "Patch"),
                    ("leave", "Leave open"),
                ],
            ),
        ]),
        d(
            "direct.move-body",
            "Move/rotate body",
            "move-body",
            "move_bodies",
        )
        .desc("Move or rotate bodies directly.")
        .inputs(vec![
            picks("bodies", "Bodies", &[K::Body]),
            offset("dx", "Delta X", 0.0).drag(Manipulator::Triad),
            offset("dy", "Delta Y", 0.0),
            offset("dz", "Delta Z", 0.0),
            angle("angle", "Rotate", 0.0),
            pick("axis", "About", &[K::Axis, K::Edge]).optional(),
        ]),
        d(
            "direct.instant-3d",
            "Instant 3D",
            "instant-3d",
            "move_faces",
        )
        .desc("Drag any face or dimension in the view to change it.")
        .kw(&["drag", "dimension"])
        .inputs(vec![
            pick(
                "target",
                "Face or dimension",
                &[K::Face, K::DrawingDimension, K::Feature],
            ),
            offset("value", "Value", 0.0).drag(Manipulator::Arrow),
        ]),
        d("direct.pull", "Pull", "pull", "move_faces")
            .desc("Push or pull faces and edges; adds or removes material.")
            .kw(&["push pull", "press pull"])
            .inputs(vec![
                picks("faces", "Faces or edges", &[K::Face, K::Edge]),
                offset("distance", "Distance", 1.0).drag(Manipulator::Arrow),
            ]),
        cmd("direct.live-relations", "Live relations", "relations", P)
            .desc("Keep faces parallel, concentric or tangent while editing directly.")
            .kw(&["keep", "parallel", "concentric", "tangent"])
            .inputs(vec![
                choice(
                    "keep",
                    "Keep",
                    &[
                        ("parallel", "Parallel"),
                        ("concentric", "Concentric"),
                        ("tangent", "Tangent"),
                        ("coplanar", "Coplanar"),
                        ("symmetric", "Symmetric"),
                    ],
                ),
                picks("faces", "Faces", &[K::Face]),
            ])
            .when(bodies_exist())
            .edit(
                edit(EditOp::Set, Section::Features)
                    .item("direct")
                    .variant("keep", "parallel"),
            ),
    ]
}

fn surfaces() -> Vec<CommandDescriptor> {
    let s = |id: &str, label: &str, icon: &str, variant: &str, desc: &str, inputs: Vec<Field>| {
        cmd(id, label, icon, P)
            .desc(desc)
            .kw(&["surface"])
            .inputs(inputs)
            .preview()
            .edit(add_feature_variant("surface", "kind", variant))
    };
    vec![
        s(
            "surface.extrude",
            "Extruded surface",
            "surface-extrude",
            "extrude",
            "Extrude open or closed curves into a surface.",
            {
                let mut f = vec![picks(
                    "profile",
                    "Curves",
                    &[K::Sketch, K::SketchEntity, K::Edge],
                )];
                f.extend(end_fields("", "End"));
                f
            },
        ),
        s(
            "surface.revolve",
            "Revolved surface",
            "surface-revolve",
            "revolve",
            "Revolve curves into a surface.",
            vec![
                picks("profile", "Curves", &[K::Sketch, K::SketchEntity, K::Edge]),
                pick("axis", "Axis", &[K::Axis, K::Edge, K::SketchEntity]),
                angle("angle", "Angle", 360.0).drag(Manipulator::Angle),
            ],
        ),
        s(
            "surface.sweep",
            "Swept surface",
            "surface-sweep",
            "sweep",
            "Sweep a curve along a path.",
            vec![
                pick("profile", "Profile", &[K::Sketch, K::Curve, K::Edge]),
                pick("path", "Path", &[K::Sketch, K::Curve, K::Edge]),
                picks("guides", "Guide curves", &[K::Curve, K::Edge]).optional(),
            ],
        ),
        s(
            "surface.loft",
            "Lofted surface",
            "surface-loft",
            "loft",
            "Loft a surface through curves.",
            vec![
                picks("profiles", "Profiles", &[K::Sketch, K::Curve, K::Edge]),
                picks("guides", "Guide curves", &[K::Curve, K::Edge]).optional(),
            ],
        ),
        s(
            "surface.boundary",
            "Boundary surface",
            "surface-boundary",
            "boundary",
            "Make a surface from curves in two directions.",
            vec![
                picks("direction1", "Direction 1", &[K::Curve, K::Edge, K::Sketch]),
                picks("direction2", "Direction 2", &[K::Curve, K::Edge, K::Sketch]).optional(),
            ],
        ),
        s(
            "surface.fill",
            "Fill",
            "surface-fill",
            "fill",
            "Patch a closed boundary with G0, G1 or G2 continuity.",
            vec![
                picks("boundary", "Boundary edges", &[K::Edge, K::Curve]),
                choice(
                    "continuity",
                    "Continuity",
                    &[
                        ("g0", "Contact (G0)"),
                        ("g1", "Tangent (G1)"),
                        ("g2", "Curvature (G2)"),
                    ],
                ),
                picks(
                    "constraints",
                    "Constraint curves",
                    &[K::Curve, K::SketchEntity],
                )
                .optional(),
                flag("merge", "Merge result", true),
            ],
        ),
        s(
            "surface.planar",
            "Planar surface",
            "surface-planar",
            "planar",
            "Fill a planar boundary.",
            vec![picks("boundary", "Boundary", &[K::Edge, K::Sketch])],
        ),
        s(
            "surface.ruled",
            "Ruled surface",
            "surface-ruled",
            "ruled",
            "Extend a ruled surface from edges.",
            vec![
                picks("edges", "Edges", &[K::Edge]),
                choice(
                    "type",
                    "Type",
                    &[
                        ("tangent", "Tangent to surface"),
                        ("normal", "Normal to surface"),
                        ("tapered", "Tapered to vector"),
                        ("sweep", "Sweep"),
                    ],
                ),
                length("distance", "Distance", 10.0).drag(Manipulator::Arrow),
            ],
        ),
        s(
            "surface.offset",
            "Offset surface",
            "surface-offset",
            "offset",
            "Offset faces into a new surface.",
            vec![
                picks("faces", "Faces", &[K::Face]),
                offset("distance", "Distance", 1.0).drag(Manipulator::Arrow),
            ],
        ),
        s(
            "surface.extend",
            "Extend",
            "surface-extend",
            "extend",
            "Extend a surface past its edges.",
            vec![
                picks("edges", "Edges", &[K::Edge]),
                length("distance", "Distance", 5.0).drag(Manipulator::Arrow),
                choice(
                    "type",
                    "Type",
                    &[("same", "Same surface"), ("linear", "Linear")],
                ),
            ],
        ),
        s(
            "surface.trim",
            "Trim",
            "surface-trim",
            "trim",
            "Trim surfaces with other surfaces, planes or sketches.",
            vec![
                picks("tools", "Trim tools", &[K::Face, K::Plane, K::Sketch]),
                picks("keep", "Pieces to keep", &[K::Face]),
                radio(
                    "mode",
                    "Mode",
                    &[("standard", "Standard"), ("mutual", "Mutual")],
                ),
            ],
        ),
        s(
            "surface.untrim",
            "Untrim",
            "surface-untrim",
            "untrim",
            "Restore a trimmed surface's natural boundary.",
            vec![picks("faces", "Faces", &[K::Face])],
        ),
        s(
            "surface.knit",
            "Knit",
            "surface-knit",
            "knit",
            "Knit surfaces together, and make a solid when they close.",
            vec![
                picks("faces", "Surfaces", &[K::Face, K::Body]),
                flag("solid", "Make a solid", true),
                length("tolerance", "Gap tolerance", 0.01),
            ],
        ),
        s(
            "surface.thicken",
            "Thicken",
            "surface-thicken",
            "thicken",
            "Give a surface thickness to make a solid.",
            vec![
                pick("surface", "Surface", &[K::Face, K::Body]),
                length("thickness", "Thickness", 1.0).drag(Manipulator::Arrow),
                radio(
                    "side",
                    "Side",
                    &[("one", "Side 1"), ("both", "Both"), ("other", "Side 2")],
                ),
            ],
        ),
        s(
            "surface.delete-hole",
            "Delete hole",
            "delete-hole",
            "delete_hole",
            "Remove a hole's boundary from a surface.",
            vec![picks("edges", "Hole edges", &[K::Edge])],
        ),
        s(
            "surface.replace-face",
            "Replace face",
            "replace-face",
            "fill",
            "Replace solid faces with a surface.",
            vec![
                picks("faces", "Faces", &[K::Face]),
                pick("with", "Surface", &[K::Face]),
            ],
        ),
        s(
            "surface.cut",
            "Cut with surface",
            "surface-cut",
            "cut_with_surface",
            "Cut a solid with a surface.",
            vec![
                pick("surface", "Surface", &[K::Face]),
                picks("bodies", "Bodies", &[K::Body]),
                flag("flip", "Flip side", false),
            ],
        ),
        s(
            "surface.mid-surface",
            "Mid-surface",
            "mid-surface",
            "mid_surface",
            "Make mid-surfaces between face pairs for shell studies.",
            vec![
                picks("pairs", "Face pairs", &[K::Face]),
                flag("auto", "Find pairs", true),
            ],
        ),
    ]
}

fn sheet_metal_commands() -> Vec<CommandDescriptor> {
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
                        ("circular", "Circular"),
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

fn weldments() -> Vec<CommandDescriptor> {
    let w = |id: &str, label: &str, icon: &str, variant: &str, desc: &str, inputs: Vec<Field>| {
        cmd(id, label, icon, P)
            .desc(desc)
            .kw(&["weldment", "frame"])
            .inputs(inputs)
            .preview()
            .edit(add_feature_variant("weldment", "weldment", variant))
    };
    vec![
        w(
            "weldment.structural-member",
            "Structural member",
            "structural-member",
            "structural_member",
            "Place profiles along sketch segments.",
            vec![
                choice(
                    "standard",
                    "Standard",
                    &[("iso", "ISO"), ("ansi", "ANSI"), ("company", "Company")],
                ),
                choice(
                    "profile",
                    "Type",
                    &[
                        ("rect_tube", "Rectangular tube"),
                        ("square_tube", "Square tube"),
                        ("pipe", "Pipe"),
                        ("angle", "Angle"),
                        ("channel", "Channel"),
                        ("i_beam", "I-beam"),
                        ("flat_bar", "Flat bar"),
                    ],
                ),
                text("size", "Size", Some("40x40x3")),
                picks("segments", "Path segments", &[K::SketchEntity, K::Sketch]),
                choice(
                    "corner",
                    "Corner treatment",
                    &[
                        ("miter", "Miter"),
                        ("butt1", "Butt (first)"),
                        ("butt2", "Butt (second)"),
                        ("none", "None"),
                    ],
                ),
                angle("rotation", "Rotation", 0.0),
                flag("mirror_profile", "Mirror profile", false),
            ],
        )
        .when(has(Facet::Sketches)),
        w(
            "weldment.trim-extend",
            "Trim/extend",
            "trim-extend",
            "trim_extend",
            "Trim or extend members to each other or to faces.",
            vec![
                picks("members", "Members to trim", &[K::Body]),
                picks(
                    "boundary",
                    "Trimming boundary",
                    &[K::Body, K::Face, K::Plane],
                ),
                choice(
                    "type",
                    "Corner",
                    &[
                        ("end_trim", "End trim"),
                        ("miter", "Miter"),
                        ("butt", "Butt"),
                    ],
                ),
            ],
        )
        .when(has(Facet::Weldments)),
        w(
            "weldment.end-cap",
            "End cap",
            "end-cap",
            "end_cap",
            "Close a tube end with a plate.",
            vec![
                picks("faces", "End faces", &[K::Face]),
                length("thickness", "Thickness", 3.0),
                length("inset", "Inset", 0.0),
                flag("chamfer", "Chamfer corners", true),
            ],
        )
        .when(has(Facet::Weldments)),
        w(
            "weldment.gusset",
            "Gusset",
            "gusset",
            "gusset",
            "Add a triangular or polygonal plate between members.",
            vec![
                picks("faces", "Supporting faces", &[K::Face]),
                choice(
                    "shape",
                    "Profile",
                    &[("triangle", "Triangular"), ("polygon", "Polygonal")],
                ),
                length("d1", "Leg 1", 50.0),
                length("d2", "Leg 2", 50.0),
                length("thickness", "Thickness", 5.0),
            ],
        )
        .when(has(Facet::Weldments)),
        w(
            "weldment.weld-bead",
            "Weld bead",
            "weld-bead",
            "weld_bead",
            "Add a fillet or groove weld between faces.",
            vec![
                picks("faces", "Faces", &[K::Face, K::Edge]),
                choice(
                    "weld",
                    "Type",
                    &[("fillet", "Fillet"), ("groove", "Groove")],
                ),
                length("leg", "Leg size", 4.0),
                flag("intermittent", "Intermittent", false),
                length("pitch", "Pitch", 50.0).if_checked("intermittent"),
            ],
        )
        .when(has(Facet::Weldments)),
        cmd("weldment.cut-list", "Cut list", "cut-list", P)
            .desc("Show the cut list: members grouped by profile and length.")
            .when(has(Facet::Weldments))
            .edit(analyze("cut_list")),
        cmd("weldment.profile-library", "Profile library", "library", P)
            .desc("Browse and add structural profiles.")
            .inputs(vec![text("library", "Library", Some("company/profiles"))])
            .edit(edit(EditOp::Set, Section::Settings).item("profile_library")),
    ]
}

fn mold() -> Vec<CommandDescriptor> {
    let m = |id: &str, label: &str, icon: &str, variant: &str, desc: &str, inputs: Vec<Field>| {
        cmd(id, label, icon, P)
            .desc(desc)
            .kw(&["mold", "tooling"])
            .inputs(inputs)
            .when(bodies_exist())
            .preview()
            .edit(add_feature_variant("mold", "mold", variant))
    };
    vec![
        m(
            "mold.parting-line",
            "Parting line",
            "parting-line",
            "parting_line",
            "Find the parting line for a pull direction.",
            vec![
                pick(
                    "direction",
                    "Pull direction",
                    &[K::Plane, K::Face, K::Axis, K::Edge],
                ),
                angle("draft", "Minimum draft", 1.0),
            ],
        ),
        m(
            "mold.shut-off",
            "Shut-off surfaces",
            "shut-off",
            "shut_off",
            "Close through-holes so core and cavity separate.",
            vec![
                picks("loops", "Hole edges", &[K::Edge]),
                choice(
                    "type",
                    "Patch",
                    &[
                        ("contact", "Contact"),
                        ("tangent", "Tangent"),
                        ("none", "No fill"),
                    ],
                ),
            ],
        ),
        m(
            "mold.parting-surface",
            "Parting surface",
            "parting-surface",
            "parting_surface",
            "Extend the parting line outward into a surface.",
            vec![
                pick("parting_line", "Parting line", &[K::Feature, K::Edge]),
                length("distance", "Distance", 50.0),
                choice(
                    "type",
                    "Type",
                    &[
                        ("normal", "Normal to pull"),
                        ("tangent", "Tangent to surface"),
                        ("perpendicular", "Perpendicular to surface"),
                    ],
                ),
            ],
        ),
        m(
            "mold.core-cavity",
            "Core/cavity split",
            "core-cavity",
            "core_cavity",
            "Split a mold block into core and cavity.",
            vec![
                pick("block", "Block sketch", &[K::Sketch]),
                length("core_depth", "Core depth", 30.0),
                length("cavity_depth", "Cavity depth", 30.0),
                flag("interlock", "Interlock surface", false),
            ],
        ),
        cmd("mold.draft-analysis", "Draft analysis", "draft-analysis", P)
            .desc("Color faces by draft angle for a pull direction.")
            .kw(&["mold", "draft"])
            .inputs(vec![
                pick("direction", "Pull direction", &[K::Plane, K::Face, K::Axis]),
                angle("angle", "Required draft", 1.0),
            ])
            .when(bodies_exist())
            .preview()
            .edit(analyze("draft")),
        m(
            "mold.undercut",
            "Undercut detection",
            "undercut",
            "undercuts",
            "Find faces that can't release in the pull direction.",
            vec![pick(
                "direction",
                "Pull direction",
                &[K::Plane, K::Face, K::Axis],
            )],
        ),
        m(
            "mold.scale",
            "Scale (shrink)",
            "scale",
            "shrink",
            "Scale the part for the material's shrinkage.",
            vec![
                qty("factor", "Shrink factor", "ratio", Some(1.005)).min(0.0),
                flag("uniform", "Uniform", true),
                qty("fx", "X factor", "ratio", Some(1.005)).shown("uniform", &["false"]),
                qty("fy", "Y factor", "ratio", Some(1.005)).shown("uniform", &["false"]),
                qty("fz", "Z factor", "ratio", Some(1.005)).shown("uniform", &["false"]),
            ],
        ),
    ]
}

fn evaluate() -> Vec<CommandDescriptor> {
    let a = |id: &str, label: &str, icon: &str, query: &str, desc: &str, inputs: Vec<Field>| {
        cmd(id, label, icon, P)
            .desc(desc)
            .inputs(inputs)
            .when(bodies_exist())
            .edit(analyze(query))
    };
    vec![
        a(
            "evaluate.mass-properties",
            "Mass properties",
            "mass",
            "mass_properties",
            "Mass, volume, area, center of mass and inertia.",
            vec![
                picks("bodies", "Bodies", &[K::Body, K::Part]).optional(),
                pick("about", "Report about", &[K::CoordinateSystem]).optional(),
                flag("override_mass", "Override mass", false),
                qty("mass", "Mass", "mass", Some(1.0)).if_checked("override_mass"),
            ],
        )
        .kw(&["weight", "volume", "inertia", "center of gravity"]),
        a(
            "evaluate.section-properties",
            "Section properties",
            "section-properties",
            "section_properties",
            "Area, centroid and second moments of a planar face or sketch.",
            vec![pick("section", "Face or sketch", &[K::Face, K::Sketch])],
        ),
        a(
            "evaluate.check-geometry",
            "Check geometry",
            "check",
            "check_geometry",
            "Find invalid faces, short edges, gaps and self-intersections.",
            vec![flag("strict", "Strict", false)],
        )
        .kw(&["validate", "repair"]),
        a(
            "evaluate.interference",
            "Interference",
            "interference",
            "interference",
            "Find bodies that overlap.",
            vec![picks("bodies", "Bodies", &[K::Body]).optional()],
        ),
        a(
            "evaluate.clearance",
            "Clearance",
            "clearance",
            "clearance",
            "Find places closer than a clearance.",
            vec![
                picks("items", "Bodies or faces", &[K::Body, K::Face]),
                length("minimum", "Minimum clearance", 1.0),
            ],
        ),
        a(
            "evaluate.thickness",
            "Thickness",
            "thickness",
            "thickness",
            "Color thin and thick regions.",
            vec![
                length("target", "Target thickness", 2.0),
                choice(
                    "show",
                    "Show",
                    &[("thin", "Thin regions"), ("thick", "Thick regions")],
                ),
            ],
        ),
        a(
            "evaluate.draft",
            "Draft analysis",
            "draft-analysis",
            "draft",
            "Color faces by draft angle.",
            vec![
                pick("direction", "Pull direction", &[K::Plane, K::Face, K::Axis]),
                angle("angle", "Draft angle", 1.0),
            ],
        ),
        a(
            "evaluate.curvature",
            "Curvature",
            "curvature",
            "curvature",
            "Color faces by curvature.",
            vec![],
        ),
        a(
            "evaluate.zebra",
            "Zebra stripes",
            "zebra",
            "zebra",
            "Show stripes to judge surface continuity.",
            vec![count("stripes", "Stripes", 20, 2)],
        )
        .kw(&["continuity", "class a"]),
        a(
            "evaluate.deviation",
            "Deviation",
            "deviation",
            "deviation",
            "Measure the angle between faces along shared edges.",
            vec![picks("edges", "Edges", &[K::Edge])],
        ),
        a(
            "evaluate.undercut",
            "Undercut",
            "undercut",
            "undercut",
            "Find undercut faces for a pull direction.",
            vec![pick(
                "direction",
                "Pull direction",
                &[K::Plane, K::Face, K::Axis],
            )],
        ),
        a(
            "evaluate.dfm",
            "DFM check",
            "dfm",
            "dfm",
            "Check manufacturability for a process.",
            vec![
                choice(
                    "process",
                    "Process",
                    &[
                        ("machining", "Machining"),
                        ("sheet_metal", "Sheet metal"),
                        ("molding", "Molding"),
                        ("additive", "Additive"),
                    ],
                ),
                text("rules", "Rule pack", Some("company/dfm")).optional(),
            ],
        )
        .kw(&["manufacturability", "design for manufacturing"]),
        a(
            "evaluate.cost",
            "Cost estimate",
            "cost",
            "cost",
            "Estimate the part's cost for a process and quantity.",
            vec![
                choice(
                    "process",
                    "Process",
                    &[
                        ("machining", "Machining"),
                        ("sheet_metal", "Sheet metal"),
                        ("molding", "Molding"),
                        ("additive", "Additive"),
                    ],
                ),
                count("quantity", "Quantity", 100, 1),
            ],
        ),
    ]
}

fn annotate() -> Vec<CommandDescriptor> {
    let pmi = |id: &str, label: &str, icon: &str, item: &str, desc: &str, inputs: Vec<Field>| {
        cmd(id, label, icon, P)
            .desc(desc)
            .kw(&["pmi", "mbd"])
            .inputs(inputs)
            .when(bodies_exist())
            .edit(edit(EditOp::Add, Section::Pmi).item(item))
    };
    vec![
        pmi(
            "annotate.datum",
            "Datum",
            "datum",
            "datum",
            "Label a face or feature as a datum.",
            vec![
                text("label", "Label", Some("A")),
                pick("target", "Face or feature", &[K::Face, K::Feature, K::Axis]),
            ],
        ),
        pmi(
            "annotate.gdt",
            "GD&T frame",
            "gdt",
            "tolerance",
            "Add a feature control frame.",
            vec![
                choice(
                    "characteristic",
                    "Characteristic",
                    &[
                        ("position", "Position"),
                        ("flatness", "Flatness"),
                        ("perpendicularity", "Perpendicularity"),
                        ("parallelism", "Parallelism"),
                        ("profile_surface", "Profile of a surface"),
                        ("runout", "Circular runout"),
                        ("concentricity", "Concentricity"),
                        ("cylindricity", "Cylindricity"),
                    ],
                ),
                length("tolerance", "Tolerance", 0.1),
                choice(
                    "modifier",
                    "Material condition",
                    &[
                        ("none", "None"),
                        ("mmc", "Maximum material"),
                        ("lmc", "Least material"),
                    ],
                ),
                text("datums", "Datums", Some("A|B|C")).optional(),
                picks("targets", "Applies to", &[K::Face, K::Feature]),
            ],
        ),
        pmi(
            "annotate.dimension",
            "Dimension",
            "dimension",
            "size",
            "Add a 3D dimension with tolerance.",
            vec![
                picks("targets", "Between", &[K::Face, K::Edge, K::Vertex]),
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
                length("plus", "Plus", 0.1).shown("tolerance", &["symmetric", "bilateral"]),
                length("minus", "Minus", 0.1).shown("tolerance", &["bilateral"]),
                text("fit", "Fit", Some("H7")).shown("tolerance", &["fit"]),
            ],
        ),
        pmi(
            "annotate.auto-dimension",
            "Auto-dimension scheme",
            "auto-dimension",
            "scheme",
            "Dimension and tolerance a part from its datums.",
            vec![
                choice(
                    "scheme",
                    "Scheme",
                    &[("plus_minus", "Plus and minus"), ("geometric", "Geometric")],
                ),
                picks("datums", "Datums", &[K::Face, K::Feature]).optional(),
            ],
        ),
        pmi(
            "annotate.surface-finish",
            "Surface finish",
            "surface-finish",
            "finish",
            "Mark a face's required roughness.",
            vec![
                picks("faces", "Faces", &[K::Face]),
                qty("ra", "Ra", "length", Some(0.0016)),
                choice(
                    "process",
                    "Process",
                    &[
                        ("any", "Any"),
                        ("removal_required", "Material removal required"),
                        ("removal_prohibited", "No material removal"),
                    ],
                ),
            ],
        ),
        pmi(
            "annotate.coating",
            "Coating",
            "coating",
            "coating",
            "Specify a coating on faces or the part.",
            vec![
                picks("faces", "Faces", &[K::Face, K::Body]).optional(),
                text("spec", "Specification", Some("anodize type II, black")),
            ],
        ),
        pmi(
            "annotate.weld-symbol",
            "Weld symbol",
            "weld-symbol",
            "weld",
            "Add a weld symbol on edges.",
            vec![
                picks("edges", "Edges", &[K::Edge]),
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
                flag("all_around", "All around", false),
            ],
        ),
        pmi(
            "annotate.note",
            "Note",
            "note",
            "note",
            "Add a note attached to the model.",
            vec![
                text("text", "Text", None),
                pick(
                    "anchor",
                    "Attached to",
                    &[K::Face, K::Edge, K::Vertex, K::Feature],
                )
                .optional(),
            ],
        ),
        pmi(
            "annotate.annotation-view",
            "Annotation view",
            "annotation-view",
            "view",
            "Save a view direction that groups annotations.",
            vec![
                text("name", "Name", Some("Front annotations")),
                choice(
                    "orientation",
                    "Orientation",
                    &[
                        ("current", "Current view"),
                        ("front", "Front"),
                        ("top", "Top"),
                        ("right", "Right"),
                    ],
                ),
            ],
        ),
    ]
}

fn simulate() -> Vec<CommandDescriptor> {
    vec![
        cmd("simulate.region", "Region", "region", P)
            .desc("Name faces, edges or bodies so studies can load and fix them.")
            .kw(&["named selection", "boundary", "load face"])
            .inputs(vec![
                text("name", "Name", Some("mount_face")),
                radio("kind", "Kind", &[("face", "Faces"), ("edge", "Edges"), ("body", "Bodies"), ("vertex", "Vertices")]),
                picks("select", "Items", &[K::Face, K::Edge, K::Body, K::Vertex]),
                choice("expect", "Expect", &[("at_least_one", "At least one"), ("exactly_one", "Exactly one"), ("any", "Any number")]),
            ])
            .when(bodies_exist())
            .edit(edit(EditOp::Add, Section::Regions).item("region")),
        cmd("simulate.publish", "Publish", "publish", P)
            .desc("Publish a region, frame or value for assemblies and studies to use.")
            .kw(&["interface", "export name"])
            .inputs(vec![
                text("name", "Name", Some("mount")),
                radio("target", "Publishes", &[("region", "Region"), ("frame", "Mate connector"), ("value", "Value")]),
                pick("item", "Item", &[K::Region, K::MateConnector, K::Input]),
            ])
            .edit(edit(EditOp::Add, Section::Publishes).item("publish")),
        cmd("simulate.representation", "Representation", "representation", P)
            .desc("Define how the part appears in assemblies and studies: full, envelope, defeatured, mid-surface or beam.")
            .inputs(vec![
                choice("kind", "Representation", &[("full", "Full"), ("envelope", "Envelope"), ("defeatured", "Defeatured"), ("mid_surface", "Mid-surface"), ("beam", "Beam")]),
                text("name", "Name", Some("defeatured")),
            ])
            .when(bodies_exist())
            .edit(edit(EditOp::Add, Section::Configurations).item("representation")),
        cmd("simulate.defeature", "Defeature", "defeature", P)
            .desc("Remove small fillets, holes and chamfers for simulation.")
            .kw(&["simplify"])
            .inputs(vec![
                length("max_fillet", "Fillets up to", 2.0),
                length("max_hole", "Holes up to", 3.0),
                length("max_chamfer", "Chamfers up to", 1.0),
                flag("keep_regions", "Keep named regions", true),
            ])
            .when(bodies_exist())
            .preview()
            .edit(add_feature_variant("direct", "direct", "delete_faces")),
        cmd("simulate.studies", "Studies", "study", P)
            .desc("Open a study that uses this part.")
            .inputs(vec![text("study", "Study", None)])
            .edit(view_action("open_study")),
    ]
}

fn manage() -> Vec<CommandDescriptor> {
    vec![
        cmd("manage.equations", "Equations and inputs", "equations", P)
            .desc("Edit the part's inputs, equations and global values.")
            .kw(&["parameters", "variables", "global"])
            .inputs(vec![
                text("name", "Input", Some("width")),
                text("value", "Value", Some("80 mm")),
            ])
            .edit(edit(EditOp::Set, Section::Inputs).item("input")),
        cmd(
            "manage.configurations",
            "Configurations",
            "configuration",
            P,
        )
        .desc("Add configurations and edit the design table.")
        .kw(&["design table", "variants", "family"])
        .inputs(vec![
            text("name", "New configuration", Some("long")),
            text("based_on", "Based on", Some("default")).optional(),
        ])
        .edit(edit(EditOp::Add, Section::Configurations).item("configuration")),
        cmd(
            "manage.document-settings",
            "Document settings",
            "settings",
            P,
        )
        .desc("Units, drafting standard and default tolerances.")
        .kw(&["units", "standard", "tolerance"])
        .inputs(vec![
            choice(
                "units",
                "Units",
                &[
                    ("mmgs", "Millimeter, gram, second"),
                    ("mks", "Meter, kilogram, second"),
                    ("ips", "Inch, pound, second"),
                ],
            ),
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
            choice(
                "tolerance_class",
                "General tolerance",
                &[
                    ("m", "ISO 2768-m"),
                    ("f", "ISO 2768-f"),
                    ("c", "ISO 2768-c"),
                ],
            ),
            count("decimals", "Decimals", 2, 0),
        ])
        .edit(edit(EditOp::Set, Section::Settings).item("document")),
        cmd("manage.find-references", "Find references", "references", P)
            .desc("List the assemblies, drawings and studies that use this part.")
            .kw(&["where used", "where-used"])
            .edit(analyze("where_used")),
        cmd(
            "manage.replace-reference",
            "Replace reference",
            "replace",
            P,
        )
        .desc("Point a reference at another file or entity.")
        .inputs(vec![
            pick("reference", "Reference", &[K::Feature, K::Input]),
            text("with", "Replace with", Some("other.part")),
        ])
        .edit(edit(EditOp::Edit, Section::Features).item("reference")),
        cmd("manage.freeze-bar", "Freeze bar", "freeze", P)
            .desc("Freeze features above the bar so they don't rebuild.")
            .inputs(vec![pick("after", "Freeze through", &[K::Feature])])
            .edit(edit(EditOp::Set, Section::Features).item("freeze")),
        cmd("manage.material", "Material", "material", P)
            .desc("Assign a material to the part or a body.")
            .kw(&["alloy", "density"])
            .inputs(vec![
                text("material", "Material", Some("Al 2024-T3")),
                picks("bodies", "Bodies", &[K::Body, K::Part])
                    .optional()
                    .help("Leave empty for the whole part"),
            ])
            .on(&[K::Body, K::Part])
            .edit(edit(EditOp::Set, Section::Materials).item("material")),
        cmd("manage.appearance", "Appearance", "appearance", P)
            .desc("Set the color and finish of the part, a body or faces.")
            .kw(&["color", "colour", "texture", "render"])
            .inputs(vec![
                text("color", "Color", Some("#9aa4b1")),
                choice(
                    "finish",
                    "Finish",
                    &[
                        ("matte", "Matte"),
                        ("satin", "Satin"),
                        ("gloss", "Gloss"),
                        ("metal", "Metal"),
                    ],
                ),
                picks(
                    "targets",
                    "Applies to",
                    &[K::Part, K::Body, K::Face, K::Feature],
                )
                .optional(),
            ])
            .on(&[K::Body, K::Face, K::Feature, K::Part])
            .edit(edit(EditOp::Set, Section::Appearances).item("appearance")),
        cmd("manage.decal", "Decal", "decal", P)
            .desc("Place an image on faces.")
            .inputs(vec![
                text("image", "Image", Some("logo.png")),
                picks("faces", "Faces", &[K::Face]),
                length("width", "Width", 20.0),
            ])
            .edit(edit(EditOp::Set, Section::Appearances).item("decal")),
        cmd("manage.import", "Import", "import", P)
            .desc("Import STEP or IGES bodies, or DXF and images into a sketch.")
            .kw(&["step", "iges", "dxf", "image"])
            .inputs(vec![
                text("file", "File", Some("bracket.step")),
                choice(
                    "as",
                    "Import as",
                    &[
                        ("bodies", "Bodies"),
                        ("sketch", "Sketch (DXF)"),
                        ("image", "Sketch picture"),
                    ],
                ),
                pick("plane", "On plane", &[K::Plane, K::Face]).shown("as", &["sketch", "image"]),
            ])
            .edit(add_feature("import")),
        cmd("manage.export", "Export", "export", P)
            .desc("Export the part as STEP AP242, IGES, STL, 3MF, glTF or a face as DXF.")
            .kw(&["step", "stl", "3mf", "gltf", "iges", "dxf"])
            .inputs(vec![
                choice(
                    "format",
                    "Format",
                    &[
                        ("step_ap242", "STEP AP242"),
                        ("iges", "IGES"),
                        ("stl", "STL"),
                        ("3mf", "3MF"),
                        ("gltf", "glTF"),
                        ("dxf_face", "DXF of a face"),
                    ],
                ),
                pick("face", "Face", &[K::Face]).shown("format", &["dxf_face"]),
                choice(
                    "resolution",
                    "Mesh resolution",
                    &[
                        ("fine", "Fine"),
                        ("standard", "Standard"),
                        ("coarse", "Coarse"),
                    ],
                )
                .shown("format", &["stl", "3mf", "gltf"]),
                flag("pmi", "Include PMI", true).shown("format", &["step_ap242"]),
            ])
            .edit(export("selected")),
        cmd("manage.heal", "Heal import", "heal", P)
            .desc("Close gaps and fix faces on imported bodies.")
            .inputs(vec![
                picks("bodies", "Bodies", &[K::Body]),
                length("tolerance", "Tolerance", 0.01),
            ])
            .when(bodies_exist())
            .edit(add_feature("import").variant("heal", "true")),
        cmd(
            "manage.recognize-features",
            "Recognize features",
            "recognize",
            P,
        )
        .desc("Turn an imported body into editable features.")
        .kw(&["feature recognition", "convert"])
        .inputs(vec![
            pick("body", "Body", &[K::Body]),
            flag("interactive", "Pick features one at a time", false),
        ])
        .when(bodies_exist())
        .edit(edit(EditOp::Edit, Section::Features).item("recognize")),
        cmd("manage.split-to-files", "Split to files", "split-files", P)
            .desc("Save bodies as separate parts and an assembly.")
            .inputs(vec![
                picks("bodies", "Bodies", &[K::Body]),
                flag("assembly", "Create an assembly", true),
            ])
            .when(bodies_exist())
            .edit(edit(EditOp::CreateFile, Section::None).item("parts")),
        cmd("manage.pack-and-share", "Pack and share", "pack", P)
            .desc("Collect the part and every file it references into one folder.")
            .kw(&["pack and go", "zip"])
            .inputs(vec![text("folder", "Folder", Some("bracket-pack"))])
            .edit(export("pack")),
    ]
}

/// Right-click actions on tree nodes (§6.4): edit, edit sketch, suppress, roll back, rename,
/// reorder, show references, delete.
fn tree_actions() -> Vec<CommandDescriptor> {
    let f = &[K::Feature, K::Sketch];
    vec![
        cmd("part.edit-feature", "Edit feature", "edit", P)
            .desc("Open the feature's inputs in Properties.")
            .when(selected_one(f))
            .on(f)
            .edit(edit(EditOp::Edit, Section::Features)),
        cmd("part.edit-sketch", "Edit sketch", "sketch", P)
            .desc("Open the sketch for editing.")
            .when(selected_one(&[K::Sketch, K::Feature]))
            .on(f)
            .edit(edit(EditOp::EnterMode, Section::Sketch).item("sketch")),
        cmd("part.suppress", "Suppress", "suppress", P)
            .desc("Turn the feature off without deleting it.")
            .when(selected(f))
            .on(f)
            .edit(edit(EditOp::Suppress, Section::Features)),
        cmd("part.unsuppress", "Unsuppress", "unsuppress", P)
            .desc("Turn a suppressed feature back on.")
            .when(selected(f))
            .on(f)
            .edit(edit(EditOp::Unsuppress, Section::Features)),
        cmd("part.rollback-here", "Roll back to here", "rollback", P)
            .desc("Move the rollback bar to just after this feature.")
            .kw(&["rollback"])
            .when(selected_one(f))
            .on(f)
            .edit(edit(EditOp::Rollback, Section::Features)),
        cmd("part.rollback-end", "Roll to end", "rollback", P)
            .desc("Move the rollback bar to the end of the tree.")
            .edit(edit(EditOp::Rollback, Section::Features).item("end")),
        cmd("part.rename", "Rename", "rename", P)
            .desc("Rename the item; references follow.")
            .inputs(vec![text("name", "Name", None)])
            .when(selected_one(&[
                K::Feature,
                K::Sketch,
                K::Body,
                K::Region,
                K::Input,
                K::Plane,
                K::MateConnector,
            ]))
            .on(&[
                K::Feature,
                K::Sketch,
                K::Body,
                K::Region,
                K::Input,
                K::Plane,
                K::MateConnector,
            ])
            .edit(edit(EditOp::Rename, Section::Features)),
        cmd("part.reorder", "Reorder", "reorder", P)
            .desc("Move the feature before or after another one.")
            .inputs(vec![
                radio(
                    "position",
                    "Place",
                    &[("before", "Before"), ("after", "After")],
                ),
                pick("anchor", "Feature", &[K::Feature, K::Sketch]),
            ])
            .when(selected_one(f))
            .on(f)
            .edit(edit(EditOp::Reorder, Section::Features)),
        cmd("part.show-references", "Show references", "references", P)
            .desc("Show what this item uses and what uses it.")
            .when(selected_one(&[K::Feature, K::Sketch, K::Region, K::Input]))
            .on(&[K::Feature, K::Sketch, K::Region, K::Input])
            .edit(analyze("references")),
        cmd("part.delete", "Delete", "delete", P)
            .desc("Delete the item, after showing what depends on it.")
            .when(selected(&[
                K::Feature,
                K::Sketch,
                K::Region,
                K::Input,
                K::Plane,
                K::MateConnector,
                K::Annotation,
            ]))
            .on(&[
                K::Feature,
                K::Sketch,
                K::Region,
                K::Input,
                K::Plane,
                K::MateConnector,
                K::Annotation,
            ])
            .edit(edit(EditOp::Remove, Section::Features)),
    ]
}

fn rows(catalog: &Catalog) -> EditorRows {
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

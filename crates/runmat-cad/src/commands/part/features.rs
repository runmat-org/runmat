//! The Features tab: sketch-based features, patterns, reference geometry and curves.

use super::*;

pub(super) fn features() -> Vec<CommandDescriptor> {
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
                length("depth", "Depth", 10.0),
                flag("reverse", "Flip material side", false),
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
                qty("parameter", "Fraction along", "ratio", Some(0.5)).shown("type", &["along_curve"]),
            ])
            .preview()
            .edit(add_feature("point")),
        cmd("part.coordinate-system", "Coordinate system", "csys", P)
            .desc("Add a coordinate system from an origin and axes, or on a planar face.")
            .kw(&["frame", "csys"])
            .inputs(vec![
                radio("type", "Type", &[("axes", "Origin and axes"), ("face", "On a face")]),
                pick("origin", "Origin", &[K::Vertex, K::Point]).shown("type", &["axes"]),
                pick("z", "Z axis", &[K::Edge, K::Axis]).shown("type", &["axes"]),
                pick("x", "X axis", &[K::Edge, K::Axis]).optional().shown("type", &["axes"]),
                pick("face", "Face", &[K::Face]).shown("type", &["face"]),
            ])
            .preview()
            .edit(add_feature("frame")),
        cmd("part.mate-connector", "Mate connector", "mate-connector", P)
            .desc("Add a named frame other files mate to.")
            .kw(&["connector", "interface"])
            .inputs(vec![
                text("name", "Name", Some("pivot")),
                pick("on", "On", &[K::Face, K::Edge, K::Vertex]),
                choice("origin", "Origin", &[("centroid", "Middle of the face or edge"), ("center", "Axis of a round face")]),
                offset("offset_x", "Offset X", 0.0).drag(Manipulator::Triad),
                offset("offset_y", "Offset Y", 0.0),
                offset("offset_z", "Offset Z", 0.0),
                flag("flip", "Flip primary axis", false),
                angle("rotate", "Rotate", 0.0),
            ])
            .preview()
            .edit(add_feature("mate_connector")),
        curve("part.curve-helix", "Helix/spiral", "helix", "helix", "Add a helix or spiral by pitch, turns and height.", vec![
            radio("type", "Type", &[("helix", "Helix"), ("spiral", "Spiral")]),
            pick("axis", "Axis", &[K::Axis, K::Edge]).shown("type", &["helix"]),
            pick("start", "Start point", &[K::Vertex, K::Point, K::SketchPoint]).shown("type", &["helix"]),
            pick("plane", "Plane", &[K::Plane, K::Face]).shown("type", &["spiral"]),
            length("start_radius", "Start radius", 5.0).shown("type", &["spiral"]),
            length("pitch", "Pitch", 2.0),
            count("revolutions", "Revolutions", 5, 1),
            flag("clockwise", "Clockwise", false).shown("type", &["helix"]),
            flag("taper", "Taper", false).shown("type", &["helix"]),
            length("taper_step", "Radius change per turn", 0.5).if_checked("taper"),
        ]),
        curve("part.curve-projected", "Projected curve", "projected-curve", "projected", "Project a sketch onto faces.", vec![
            pick("sketch", "Sketch", &[K::Sketch]),
            picks("faces", "Faces", &[K::Face]),
            flag("both", "Both directions", false),
        ]),
        curve("part.curve-composite", "Composite curve", "composite-curve", "composite", "Join edges end to end into one curve.", vec![picks("curves", "Edges", &[K::Edge])]),
        curve("part.curve-split-line", "Split line", "split-line", "split_line", "Split faces along a projected sketch, silhouette or intersection.", vec![
            choice("type", "Type", &[("projection", "Projection"), ("silhouette", "Silhouette"), ("intersection", "Intersection")]),
            pick("tool", "Sketch or plane", &[K::Sketch, K::Plane, K::Face]),
            picks("faces", "Faces to split", &[K::Face]),
        ]),
        curve("part.curve-intersection", "Intersection curve", "intersection-curve", "intersection", "Make the curve where a planar face meets a plane.", vec![
            pick("face", "Face", &[K::Face]),
            pick("plane", "Plane", &[K::Plane, K::Face]),
        ]),
        curve("part.curve-through-points", "Curve through points", "through-points", "through_points", "Fit a curve through points.", vec![
            picks("points", "Points", &[K::Vertex, K::Point, K::SketchPoint]),
            flag("closed", "Closed", false),
        ]),
        curve("part.curve-equation", "Equation curve", "equation-curve", "equation", "Draw a curve from x(t), y(t), z(t).", vec![
            text("x", "x(t)", Some("cos(t) * 10 mm")),
            text("y", "y(t)", Some("sin(t) * 10 mm")),
            text("z", "z(t)", Some("t * 2 mm")),
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

pub(super) fn curve(
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

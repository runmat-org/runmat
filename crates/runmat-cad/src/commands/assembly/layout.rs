//! Assembly commands on the Layout, Fasteners and Motion tabs.

use super::*;
use SelectKind as K;

pub(super) fn layout() -> Vec<CommandDescriptor> {
    vec![
        // Layout.
        asm(
            "assembly.skeleton",
            "Skeleton part",
            "skeleton",
            "Add a skeleton part that drives the layout.",
        )
        .inputs(vec![text("name", "Name", Some("skeleton.part"))])
        .edit(edit(EditOp::CreateFile, Section::Occurrences).item("skeleton")),
        asm(
            "assembly.publish",
            "Publish",
            "publish",
            "Publish an interface, frame or value for parent assemblies.",
        )
        .inputs(vec![
            text("name", "Name", Some("mount"))
                .help("The name parent assemblies read it by; an existing publish's name changes that publish."),
            pick("item", "Item", &[K::MateConnector, K::Occurrence])
                .help("A mate connector (a frame a parent mates to) or a whole component."),
        ])
        .on(&[K::Publish])
        .edit(edit(EditOp::Add, Section::Publishes).item("publish")),
        asm(
            "assembly.envelope",
            "Envelope",
            "envelope",
            "Reserve space with an envelope body others must stay out of.",
        )
        .inputs(vec![
            text("name", "Name", Some("service_space")),
            pick("body", "Body", &[K::Occurrence, K::Body]),
        ])
        .edit(edit(EditOp::Add, Section::Occurrences).item("envelope")),
        asm(
            "assembly.interface-check",
            "Interface check",
            "check",
            "Check that mated interfaces still match their published definitions.",
        )
        .inputs(vec![length("tolerance", "Tolerance", 0.01).help(
            "A pinned interface that moved further than this in its file's working copy is listed.",
        )])
        .when(has(Facet::Mates))
        .edit(analyze("interfaces")),
        asm(
            "assembly.feature",
            "Assembly feature",
            "assembly-feature",
            "Cut, drill across parts or add a weld bead at the assembly level.",
        )
        .inputs(vec![
            text("name", "Name", Some("cut1"))
                .optional()
                .help("Leave empty for the next free name; an existing feature's name changes that feature."),
            choice(
                "type",
                "Feature",
                &[
                    ("cut", "Cut"),
                    ("hole", "Hole across parts"),
                    ("weld", "Weld bead"),
                ],
            ),
            pick("profile", "Sketch", &[K::Sketch, K::SketchEntity, K::SketchPoint])
                .help("A cut uses the sketch's closed profile; a hole goes at each point and circle center; a weld bead follows its lines and arcs."),
            choice(
                "end",
                "End",
                &[
                    ("through_all", "Through all"),
                    ("through_all_both", "Through all, both directions"),
                    ("blind", "Blind"),
                    ("symmetric", "Mid-plane"),
                ],
            )
            .shown("type", &["cut", "hole"]),
            length("distance", "Depth", 10.0).shown("end", &["blind", "symmetric"]),
            flag("reverse", "Flip direction", false).shown("type", &["cut", "hole"]),
            choice(
                "hole_type",
                "Hole type",
                &[
                    ("simple", "Simple"),
                    ("counterbore", "Counterbore"),
                    ("countersink", "Countersink"),
                ],
            )
            .shown("type", &["hole"]),
            text("size", "Fastener size", Some("M6")).optional().shown("type", &["hole"]),
            choice(
                "fit",
                "Fit",
                &[("normal", "Normal"), ("close", "Close"), ("loose", "Loose")],
            )
            .shown("type", &["hole"]),
            qty("diameter", "Diameter", "length", None)
                .optional()
                .help("Leave empty for the fastener size's clearance diameter.")
                .shown("type", &["hole"]),
            qty("cbore_diameter", "Counterbore diameter", "length", None)
                .optional()
                .shown("hole_type", &["counterbore"]),
            qty("cbore_depth", "Counterbore depth", "length", None)
                .optional()
                .shown("hole_type", &["counterbore"]),
            qty("csk_diameter", "Countersink diameter", "length", None)
                .optional()
                .shown("hole_type", &["countersink"]),
            qty("csk_angle", "Countersink angle", "angle", Some(90.0))
                .optional()
                .shown("hole_type", &["countersink"]),
            length("weld_size", "Bead size", 5.0).shown("type", &["weld"]),
            flag("flip", "Other side of the path", false).shown("type", &["weld"]),
            picks("components", "Affects", &[K::Occurrence])
                .optional()
                .help("Leave empty to affect every component the feature meets."),
        ])
        .when(occurrences())
        .preview()
        .edit(edit(EditOp::Add, Section::Occurrences).item("assembly_feature")),
        asm(
            "assembly.hole-series",
            "Hole series",
            "hole",
            "Add aligned holes through several parts with matching sizes.",
        )
        .inputs(vec![
            picks("components", "Through", &[K::Occurrence]),
            text("size", "Fastener size", Some("M6")),
            pick("start", "Start face", &[K::Face]),
        ])
        .when(occurrences())
        .preview()
        .edit(edit(EditOp::Add, Section::Occurrences).item("hole_series")),
        // Fasteners.
        asm(
            "assembly.smart-fasteners",
            "Smart fasteners",
            "fastener",
            "Fill holes with fasteners sized to the hole.",
        )
        .kw(&["bolt", "screw", "nut", "washer"])
        .inputs(vec![
            picks("holes", "Holes", &[K::Hole, K::Feature, K::Face]).optional(),
            text("library", "Library", Some("iso/hex-bolts")),
            flag("washers", "Add washers and nuts", true),
        ])
        .when(occurrences())
        .preview()
        .edit(edit(EditOp::Add, Section::Fasteners).item("set")),
        asm(
            "assembly.fastener-rule",
            "Fastener rule",
            "fastener-rule",
            "Place fasteners along an edge by pitch, type and drill-through.",
        )
        .kw(&["rivet", "hi-lok", "pitch"])
        .inputs(vec![
            picks("path", "Along", &[K::Edge, K::Curve]),
            length("pitch", "Pitch", 25.0),
            length("edge_distance", "Edge distance", 12.0),
            text("fastener", "Fastener", Some("NAS1097")),
            flag("drill", "Drill through", true),
        ])
        .when(occurrences())
        .preview()
        .edit(edit(EditOp::Add, Section::Fasteners).item("rule")),
        asm(
            "assembly.fastener-library",
            "Fastener library",
            "library",
            "Browse fasteners and add one.",
        )
        .inputs(vec![text("library", "Library", Some("iso"))])
        .edit(edit(EditOp::Set, Section::Settings).item("fastener_library")),
        asm(
            "assembly.hole-check",
            "Hole/fastener check",
            "check",
            "Check hole alignment, fastener length and grip.",
        )
        .inputs(vec![
            length("tolerance", "Tolerance", 0.1)
                .help("Holes and fasteners further off their axes than this are listed."),
            picks("components", "Components", &[K::Occurrence])
                .optional()
                .help("Leave empty to check every part."),
        ])
        .when(occurrences())
        .edit(analyze("fasteners")),
        // Motion.
        asm(
            "motion.drag",
            "Drag through DOF",
            "drag",
            "Drag a component through its remaining degrees of freedom.",
        )
        .inputs(vec![
            pick("component", "Component", &[K::Occurrence])
                .help("A floating component; the handle sits at its centre."),
            offset("dx", "Pull X", 0.0)
                .drag(Manipulator::Triad)
                .help("How far the handle is pulled; the mates decide where the component goes."),
            offset("dy", "Pull Y", 0.0),
            offset("dz", "Pull Z", 0.0),
            flag("collisions", "Stop at collisions", true)
                .help("Stop at the last pose clear of the components it starts apart from."),
        ])
        .when(has(Facet::Mates))
        .preview()
        .edit(edit(EditOp::Set, Section::Occurrences).item("transform")),
        asm(
            "motion.collision",
            "Collision detection",
            "collision",
            "Check a motion study for components that meet as they move.",
        )
        .inputs(vec![
            pick("study", "Study", &[K::MotionStudy])
                .optional()
                .help("Leave empty when the assembly has one motion study."),
            radio(
                "mode",
                "On contact",
                &[("stop", "Stop"), ("highlight", "Highlight")],
            )
            .help("Stop reports the first contact between components that start apart; Highlight lists every contact over the whole study."),
            picks("components", "Check", &[K::Occurrence])
                .optional()
                .help("Leave empty to check every moving component against the rest."),
        ])
        .when(has(Facet::Mates))
        .on(&[K::MotionStudy])
        .edit(edit(EditOp::Analyze, Section::Motion).item("collision")),
        asm(
            "motion.mate-controller",
            "Mate controller",
            "controller",
            "Save named positions of mates and move the mechanism between them.",
        )
        .inputs(vec![
            text("name", "Position", Some("open"))
                .help("An existing position's name saves over that position."),
            radio(
                "action",
                "Action",
                &[("save", "Save values"), ("move", "Move to position")],
            ),
            picks("mates", "Mates", &[K::Mate])
                .optional()
                .shown("action", &["save"])
                .help("The mates whose current values the position keeps. Leave empty to save a position's mates again."),
        ])
        .when(has(Facet::Mates))
        .on(&[K::MatePosition])
        .edit(edit(EditOp::Add, Section::Motion).item("position")),
        asm(
            "motion.animation",
            "Animation study",
            "animation",
            "Drive a mate through a range over time and play the solved motion.",
        )
        .inputs(vec![
            text("name", "Name", Some("open_close"))
                .help("An existing study's name changes that study."),
            pick("mate", "Mate", &[K::Mate])
                .help("A revolute, slider, cylindrical, distance or angle mate."),
            choice("quantity", "Drive", &[("angle", "Angle"), ("distance", "Distance")]),
            angle("from_angle", "From", 0.0).shown("quantity", &["angle"]),
            angle("to_angle", "To", 90.0).shown("quantity", &["angle"]),
            offset("from_distance", "From", 0.0).shown("quantity", &["distance"]),
            offset("to_distance", "To", 50.0).shown("quantity", &["distance"]),
            qty("duration", "Duration", "time", Some(4.0)),
            choice(
                "profile",
                "Motion",
                &[
                    ("linear", "Constant speed"),
                    ("cycloidal", "Smooth start and stop"),
                ],
            ),
            pick("track", "Plot", &[K::Occurrence])
                .optional()
                .help("The component whose speed and acceleration are plotted; the first moving one when empty."),
        ])
        .when(has(Facet::Mates))
        .on(&[K::MotionStudy])
        .edit(edit(EditOp::Add, Section::Motion).item("animation")),
        asm(
            "motion.basic",
            "Basic motion",
            "motion",
            "Simulate gravity, springs and motors on the mechanism.",
        )
        .inputs(vec![
            flag("gravity", "Gravity", true),
            picks("motors", "Motor mates", &[K::Mate]).optional(),
            qty("speed", "Motor speed", "angular_speed", Some(30.0)),
            qty("duration", "Duration", "time", Some(5.0)),
        ])
        .when(has(Facet::Mates))
        .edit(edit(EditOp::Add, Section::Motion).item("basic")),
        asm(
            "motion.to-fea",
            "Results to .fea",
            "study",
            "Send motion loads at an instant to a new study.",
        )
        .inputs(vec![
            qty("time", "At time", "time", Some(0.0)),
            text("name", "Study", Some("loads.fea")),
        ])
        .when(has(Facet::Mates))
        .edit(edit(EditOp::CreateFile, Section::Studies).item("fea")),
    ]
}

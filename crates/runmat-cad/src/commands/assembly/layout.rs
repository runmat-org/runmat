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
            text("name", "Name", Some("mount")),
            pick(
                "item",
                "Item",
                &[K::MateConnector, K::Region, K::Occurrence],
            ),
        ])
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
        .when(has(Facet::Mates))
        .edit(check("interfaces")),
        asm(
            "assembly.feature",
            "Assembly feature",
            "assembly-feature",
            "Cut, drill across parts or add a weld bead at the assembly level.",
        )
        .inputs(vec![
            choice(
                "type",
                "Feature",
                &[
                    ("cut", "Cut"),
                    ("hole", "Hole across parts"),
                    ("weld", "Weld bead"),
                ],
            ),
            pick("profile", "Sketch", &[K::Sketch]).shown("type", &["cut"]),
            picks("components", "Affects", &[K::Occurrence]),
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
        .when(occurrences())
        .edit(check("fasteners")),
        // Motion.
        asm(
            "motion.drag",
            "Drag through DOF",
            "drag",
            "Drag a component through its remaining degrees of freedom.",
        )
        .inputs(vec![
            pick("component", "Component", &[K::Occurrence]),
            flag("collisions", "Stop at collisions", true),
        ])
        .when(has(Facet::Mates))
        .preview()
        .edit(edit(EditOp::Analyze, Section::Motion).item("drag")),
        asm(
            "motion.collision",
            "Collision detection",
            "collision",
            "Stop or highlight when moving components touch.",
        )
        .inputs(vec![
            radio(
                "mode",
                "On contact",
                &[("stop", "Stop"), ("highlight", "Highlight")],
            ),
            picks("components", "Check", &[K::Occurrence]).optional(),
        ])
        .edit(edit(EditOp::Analyze, Section::Motion).item("collision")),
        asm(
            "motion.mate-controller",
            "Mate controller",
            "controller",
            "Save named positions of mates and step between them.",
        )
        .inputs(vec![
            text("name", "Position", Some("open")),
            picks("mates", "Mates", &[K::Mate]),
        ])
        .when(has(Facet::Mates))
        .edit(edit(EditOp::Add, Section::Motion).item("position")),
        asm(
            "motion.animation",
            "Animation study",
            "animation",
            "Animate mates over time.",
        )
        .inputs(vec![
            text("name", "Name", Some("open-close")),
            qty("duration", "Duration", "time", Some(5.0)),
        ])
        .when(has(Facet::Mates))
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

//! Assembly commands on the Assemble tab: components, mates, motion, patterns and structure.

use super::*;
use SelectKind as K;

pub(super) fn assemble(conn: &[SelectKind]) -> Vec<CommandDescriptor> {
    vec![
        // Assemble: components.
        asm(
            "assembly.insert",
            "Insert",
            "insert",
            "Insert a part, assembly, circuit board, STEP file or library part.",
        )
        .kw(&["add component", "place", "instance"])
        .inputs(vec![
            choice(
                "source",
                "Source",
                &[
                    ("part", "Part"),
                    ("assembly", "Assembly"),
                    ("circuit", "Circuit board"),
                    ("step", "STEP file"),
                    ("library", "Library part"),
                ],
            ),
            text("file", "File", Some("bracket.part")),
            text("configuration", "Configuration", Some("default")).optional(),
            count("quantity", "Instances", 1, 1),
            flag("fix", "Fix in place", false),
        ])
        .preview()
        .edit(edit(EditOp::Add, Section::Occurrences).item("occurrence")),
        asm(
            "assembly.new-part",
            "New part in context",
            "new-part",
            "Create a part inside the assembly, referencing its neighbors.",
        )
        .kw(&["top-down", "in context"])
        .inputs(vec![
            text("name", "Part name", Some("cover.part")),
            pick("plane", "Start on", &[K::Plane, K::Face]).optional(),
        ])
        .edit(edit(EditOp::CreateFile, Section::Occurrences).item("part")),
        asm(
            "assembly.replace",
            "Replace",
            "replace",
            "Replace components with another file; mates follow where they can.",
        )
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]),
            text("with", "Replace with", Some("bracket-v2.part")),
            flag("remate", "Re-attach mates", true),
        ])
        .when(selected(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(edit(EditOp::Edit, Section::Occurrences).item("definition")),
        // Mates.
        asm(
            "assembly.mate",
            "Mate",
            "mate",
            "Constrain two components: standard, advanced and mechanical mates, with limits.",
        )
        .kw(&["constraint", "joint", "fasten", "revolute", "slider"])
        .inputs(vec![
            choice("type", "Type", MATE_TYPES),
            pick("first", "First", conn),
            pick("second", "Second", conn),
            offset("offset", "Offset", 0.0)
                .shown(
                    "type",
                    &["fasten", "planar", "distance", "slider", "cylindrical"],
                )
                .drag(Manipulator::Arrow),
            angle("angle", "Angle", 0.0).shown("type", &["angle", "revolute", "fasten"]),
            flag("flip", "Flip", false),
            flag("limits", "Limits", false),
            qty("min", "Min", "angle", Some(-30.0)).if_checked("limits"),
            qty("max", "Max", "angle", Some(45.0)).if_checked("limits"),
            qty("ratio", "Ratio", "ratio", Some(1.0))
                .shown("type", &["gear", "linear_coupler", "rack_pinion", "screw"]),
            length("pitch", "Pitch", 1.0).shown("type", &["screw", "rack_pinion"]),
            pick("path", "Path", &[K::Edge, K::Curve]).shown("type", &["path"]),
        ])
        .when(occurrences())
        .preview()
        .edit(edit(EditOp::Add, Section::Mates).item("mate")),
        asm(
            "assembly.mate-connector",
            "Mate connector",
            "mate-connector",
            "Add a frame on a component to mate to.",
        )
        .inputs(vec![
            text("name", "Name", Some("pin")),
            pick("on", "On", &[K::Face, K::Edge, K::Vertex]),
            flag("flip", "Flip", false),
        ])
        .when(occurrences())
        .edit(edit(EditOp::Add, Section::Mates).item("connector")),
        asm(
            "assembly.smart-mate",
            "Smart mate",
            "smart-mate",
            "Drag a component onto another and accept the suggested mate.",
        )
        .kw(&["drag", "snap"])
        .inputs(vec![
            pick("component", "Component", &[K::Occurrence]),
            pick("target", "Onto", conn),
        ])
        .when(occurrences())
        .preview()
        .edit(
            edit(EditOp::Add, Section::Mates)
                .item("mate")
                .variant("smart", "true"),
        ),
        // Motion.
        asm(
            "assembly.move",
            "Move component",
            "move",
            "Move components by dragging, with a triad or by aligning.",
        )
        .inputs(vec![
            radio(
                "mode",
                "Move",
                &[
                    ("drag", "Free drag"),
                    ("triad", "Triad"),
                    ("align", "Along an entity"),
                ],
            ),
            picks("components", "Components", &[K::Occurrence]),
            offset("dx", "Delta X", 0.0)
                .shown("mode", &["triad"])
                .drag(Manipulator::Triad),
            offset("dy", "Delta Y", 0.0).shown("mode", &["triad"]),
            offset("dz", "Delta Z", 0.0).shown("mode", &["triad"]),
            angle("rx", "Rotate X", 0.0).shown("mode", &["triad"]),
            angle("ry", "Rotate Y", 0.0).shown("mode", &["triad"]),
            angle("rz", "Rotate Z", 0.0).shown("mode", &["triad"]),
            pick("along", "Along", &[K::Edge, K::Axis, K::Face]).shown("mode", &["align"]),
            flag("collisions", "Stop at collisions", false),
        ])
        .when(selected(&[K::Occurrence]))
        .preview()
        .edit(edit(EditOp::Set, Section::Occurrences).item("transform")),
        asm(
            "assembly.fix",
            "Fix/Float",
            "fix",
            "Fix a component in place, or let mates move it.",
        )
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]),
            radio("state", "State", &[("fixed", "Fixed"), ("float", "Float")]),
        ])
        .when(selected(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(edit(EditOp::Set, Section::Occurrences).item("fixed")),
        // Repeat.
        asm(
            "assembly.pattern",
            "Component pattern",
            "pattern",
            "Repeat components in a line, a circle, or by a part's feature pattern.",
        )
        .inputs(vec![
            choice(
                "layout",
                "Layout",
                &[
                    ("linear", "Linear"),
                    ("circular", "Circular"),
                    ("feature", "Follow feature pattern"),
                ],
            ),
            picks("components", "Components", &[K::Occurrence]),
            pick("direction", "Direction", &[K::Edge, K::Axis, K::Face])
                .shown("layout", &["linear", "circular"]),
            length("spacing", "Spacing", 20.0).shown("layout", &["linear"]),
            angle("angle", "Angle", 360.0).shown("layout", &["circular"]),
            count("instances", "Instances", 3, 1).shown("layout", &["linear", "circular"]),
            pick("feature", "Feature pattern", &[K::Feature]).shown("layout", &["feature"]),
        ])
        .when(occurrences())
        .preview()
        .edit(edit(EditOp::Add, Section::Occurrences).item("pattern")),
        asm(
            "assembly.mirror",
            "Mirror components",
            "mirror",
            "Mirror components about a plane, as copies or opposite-hand parts.",
        )
        .inputs(vec![
            pick("plane", "Mirror plane", &[K::Plane, K::Face]),
            picks("components", "Components", &[K::Occurrence]),
            radio(
                "result",
                "Result",
                &[
                    ("copy", "Mirrored copies"),
                    ("opposite", "Opposite-hand parts"),
                ],
            ),
        ])
        .when(occurrences())
        .preview()
        .edit(edit(EditOp::Add, Section::Occurrences).item("mirror")),
        asm(
            "assembly.copy-with-mates",
            "Copy with mates",
            "copy",
            "Copy components with their mates, picking new references.",
        )
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]),
            picks("references", "New references", conn),
        ])
        .when(selected(&[K::Occurrence]))
        .edit(edit(EditOp::Add, Section::Occurrences).item("copy")),
        // Structure.
        asm(
            "assembly.subassembly",
            "Subassembly",
            "subassembly",
            "Form a subassembly from components, dissolve one, or make it flexible or rigid.",
        )
        .inputs(vec![
            radio(
                "action",
                "Action",
                &[
                    ("form", "Form new"),
                    ("dissolve", "Dissolve"),
                    ("flexible", "Make flexible"),
                    ("rigid", "Make rigid"),
                ],
            ),
            picks("components", "Components", &[K::Occurrence]),
            text("name", "File name", Some("hinge.assembly")).shown("action", &["form"]),
        ])
        .when(selected(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(edit(EditOp::Edit, Section::Occurrences).item("structure")),
        asm(
            "assembly.edit-in-context",
            "Edit in context",
            "edit",
            "Edit a component's part inside the assembly.",
        )
        .when(selected_one(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(edit(EditOp::EnterMode, Section::Occurrences).item("in_context")),
    ]
}

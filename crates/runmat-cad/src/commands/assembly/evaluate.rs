//! Assembly commands on the Evaluate, Explode and Simulate tabs.

use super::*;
use SelectKind as K;

pub(super) fn evaluate() -> Vec<CommandDescriptor> {
    vec![
        // Evaluate.
        asm(
            "assembly.interference",
            "Interference",
            "interference",
            "Find components that overlap.",
        )
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]).optional(),
            flag("fasteners", "Include fasteners", false),
        ])
        .when(occurrences())
        .edit(analyze("interference")),
        asm(
            "assembly.clearance",
            "Clearance",
            "clearance",
            "Find components closer than a clearance.",
        )
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]).optional(),
            length("minimum", "Minimum clearance", 2.0),
        ])
        .when(occurrences())
        .edit(analyze("clearance")),
        asm(
            "assembly.hole-alignment",
            "Hole alignment",
            "hole",
            "Find holes across parts that don't line up.",
        )
        .inputs(vec![length("tolerance", "Tolerance", 0.1)])
        .when(occurrences())
        .edit(analyze("hole_alignment")),
        asm(
            "assembly.mass-properties",
            "Mass properties",
            "mass",
            "Mass, center of mass and inertia of the assembly.",
        )
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]).optional()
        ])
        .when(occurrences())
        .edit(analyze("mass_properties")),
        asm(
            "assembly.bom",
            "BOM",
            "bom",
            "Show the bill of materials: top level, indented or parts only.",
        )
        .kw(&["bill of materials", "parts list"])
        .inputs(vec![choice(
            "type",
            "Type",
            &[
                ("top", "Top level"),
                ("indented", "Indented"),
                ("parts", "Parts only"),
            ],
        )])
        .when(occurrences())
        .edit(analyze("bom")),
        asm(
            "assembly.where-used",
            "Where-used",
            "references",
            "List where a component's file is used.",
        )
        .when(selected_one(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(analyze("where_used")),
        asm(
            "assembly.visualization",
            "Assembly visualization",
            "visualization",
            "Sort and color components by mass, cost or another property.",
        )
        .inputs(vec![choice(
            "by",
            "Sort by",
            &[
                ("mass", "Mass"),
                ("cost", "Cost"),
                ("rebuild", "Rebuild time"),
                ("count", "Count"),
            ],
        )])
        .when(occurrences())
        .edit(analyze("visualization")),
        // Explode.
        asm(
            "explode.new",
            "New exploded view",
            "explode",
            "Create an exploded view.",
        )
        .inputs(vec![text("name", "Name", Some("Exploded 1"))])
        .when(occurrences())
        .edit(edit(EditOp::Add, Section::Explode).item("view")),
        asm(
            "explode.step",
            "Explode step",
            "explode-step",
            "Move components along a direction in one step.",
        )
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]),
            pick("direction", "Direction", &[K::Edge, K::Axis, K::Face]),
            offset("distance", "Distance", 50.0).drag(Manipulator::Arrow),
        ])
        .when(has(Facet::ExplodedViews))
        .preview()
        .edit(edit(EditOp::Add, Section::Explode).item("step")),
        asm(
            "explode.radial",
            "Radial step",
            "explode-radial",
            "Spread components out radially about an axis.",
        )
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]),
            pick("axis", "Axis", &[K::Axis, K::Edge]),
            length("distance", "Distance", 50.0),
        ])
        .when(has(Facet::ExplodedViews))
        .preview()
        .edit(edit(EditOp::Add, Section::Explode).item("radial_step")),
        asm(
            "explode.auto",
            "Auto-explode",
            "explode-auto",
            "Explode the assembly automatically from its mates.",
        )
        .inputs(vec![
            length("spacing", "Spacing", 20.0),
            flag("fasteners", "Fasteners with their holes", true),
        ])
        .when(occurrences())
        .preview()
        .edit(edit(EditOp::Add, Section::Explode).item("auto")),
        asm(
            "explode.lines",
            "Explode lines",
            "explode-lines",
            "Draw lines showing how components move.",
        )
        .inputs(vec![picks("steps", "Steps", &[K::ExplodeStep]).optional()])
        .when(has(Facet::ExplodedViews))
        .edit(edit(EditOp::Add, Section::Explode).item("lines")),
        // Simulate.
        asm(
            "assembly.contacts",
            "Contact regions",
            "contact",
            "Find touching faces between components and name them as contact regions.",
        )
        .inputs(vec![
            length("tolerance", "Gap tolerance", 0.05),
            flag("bonded", "Treat as bonded", true),
        ])
        .when(occurrences())
        .edit(edit(EditOp::Add, Section::Regions).item("contact")),
        asm(
            "assembly.representation",
            "Representation per component",
            "representation",
            "Choose each component's representation for studies.",
        )
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]),
            choice(
                "representation",
                "Representation",
                &[
                    ("full", "Full"),
                    ("envelope", "Envelope"),
                    ("defeatured", "Defeatured"),
                    ("mid_surface", "Mid-surface"),
                    ("beam", "Beam"),
                ],
            ),
        ])
        .when(selected(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(edit(EditOp::Set, Section::Occurrences).item("representation")),
    ]
}

//! Assembly editor commands (`.assembly`), per the UI target §7.

use super::view::cluster_3d;
use super::*;
use EditorKind::Assembly;
use SelectKind as K;

const A: &[EditorKind] = &[Assembly];

pub(super) const MATE_TYPES: &[(&str, &str)] = &[
    ("fasten", "Fasten"),
    ("revolute", "Revolute"),
    ("slider", "Slider"),
    ("cylindrical", "Cylindrical"),
    ("planar", "Planar"),
    ("ball", "Ball"),
    ("pin_slot", "Pin slot"),
    ("parallel", "Parallel"),
    ("tangent", "Tangent"),
    ("distance", "Distance"),
    ("angle", "Angle"),
    ("width", "Width"),
    ("symmetric", "Symmetric"),
    ("path", "Path"),
    ("linear_coupler", "Linear coupler"),
    ("gear", "Gear"),
    ("rack_pinion", "Rack and pinion"),
    ("screw", "Screw"),
    ("cam", "Cam"),
    ("hinge", "Hinge"),
    ("universal_joint", "Universal joint"),
];

fn occurrences() -> Predicate {
    has(Facet::Occurrences)
}

fn asm(id: &str, label: &str, icon: &str, desc: &str) -> CommandDescriptor {
    cmd(id, label, icon, A).desc(desc)
}

pub(super) fn add(catalog: &mut Catalog) {
    let conn = &[K::MateConnector, K::Face, K::Edge, K::Vertex, K::Occurrence];
    let commands = vec![
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
        // Manage.
        asm(
            "assembly.options-effectivity",
            "Options and effectivity",
            "options",
            "Define product options and the serial ranges they apply to.",
        )
        .inputs(vec![
            text("option", "Option", Some("engine")),
            text("values", "Values", Some("a, b")),
            text("effectivity", "Effectivity", Some("MSN 1-99")).optional(),
        ])
        .edit(edit(EditOp::Add, Section::Options).item("option")),
        asm(
            "assembly.configurations",
            "Configurations and display states",
            "configuration",
            "Add configurations and display states.",
        )
        .inputs(vec![
            radio(
                "kind",
                "Add",
                &[
                    ("configuration", "Configuration"),
                    ("display_state", "Display state"),
                ],
            ),
            text("name", "Name", Some("open")),
        ])
        .edit(edit(EditOp::Add, Section::Configurations).item("configuration")),
        asm(
            "assembly.document-settings",
            "Document settings",
            "settings",
            "Units, standard and default tolerances.",
        )
        .inputs(vec![choice(
            "units",
            "Units",
            &[
                ("mmgs", "Millimeter, gram, second"),
                ("mks", "Meter, kilogram, second"),
                ("ips", "Inch, pound, second"),
            ],
        )])
        .edit(edit(EditOp::Set, Section::Settings).item("document")),
        asm(
            "assembly.update-pins",
            "References and pins",
            "pin",
            "Review pinned revisions of components and update them.",
        )
        .kw(&["revision", "update", "version"])
        .inputs(vec![
            picks("components", "Components", &[K::Occurrence]).optional(),
            radio(
                "to",
                "Update to",
                &[("latest", "Latest"), ("released", "Latest released")],
            ),
        ])
        .when(occurrences())
        .edit(edit(EditOp::Set, Section::Occurrences).item("revision")),
        asm(
            "assembly.isolate",
            "Isolate",
            "isolate",
            "Show only the selected components.",
        )
        .when(selected(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(view_action("isolate")),
        asm(
            "assembly.hide",
            "Hide/show",
            "visibility",
            "Hide or show components.",
        )
        .when(selected(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(view_action("hide")),
        asm(
            "assembly.transparency",
            "Transparency",
            "transparency",
            "Draw components see-through.",
        )
        .when(selected(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(view_action("transparency")),
        asm(
            "assembly.suppress",
            "Suppress",
            "suppress",
            "Leave components out of this configuration.",
        )
        .when(selected(&[K::Occurrence, K::Mate]))
        .on(&[K::Occurrence, K::Mate])
        .edit(edit(EditOp::Suppress, Section::Occurrences)),
        asm(
            "assembly.pack-and-share",
            "Pack and share",
            "pack",
            "Collect the assembly and every file it references.",
        )
        .inputs(vec![text("folder", "Folder", Some("rig-pack"))])
        .edit(export("pack")),
        asm(
            "assembly.export",
            "Export",
            "export",
            "Export as a STEP assembly, glTF or USD.",
        )
        .inputs(vec![
            choice(
                "format",
                "Format",
                &[("step", "STEP assembly"), ("gltf", "glTF"), ("usd", "USD")],
            ),
            flag("hidden", "Include hidden components", false),
        ])
        .edit(export("selected")),
        asm(
            "assembly.delete",
            "Delete",
            "delete",
            "Delete components or mates.",
        )
        .when(selected(&[K::Occurrence, K::Mate]))
        .on(&[K::Occurrence, K::Mate])
        .edit(edit(EditOp::Remove, Section::Occurrences)),
        asm(
            "assembly.edit-mate",
            "Edit mate",
            "edit",
            "Open the mate's inputs in Properties.",
        )
        .when(selected_one(&[K::Mate]))
        .on(&[K::Mate])
        .edit(edit(EditOp::Edit, Section::Mates)),
        asm(
            "assembly.open",
            "Open part",
            "open",
            "Open the component's file in its own tab.",
        )
        .when(selected_one(&[K::Occurrence]))
        .on(&[K::Occurrence])
        .edit(view_action("open")),
    ];
    catalog.add_commands(commands);

    let mate_menu = variants(
        "assembly.mate.menu",
        "Mate",
        "mate",
        "assembly.mate",
        "type",
        MATE_TYPES,
    );
    catalog.rows.push(EditorRows {
        editor: Assembly,
        mode: EditMode::Model,
        default_tab: "assemble".into(),
        document_bar: vec![
            c("view.configuration"),
            c("assembly.options"),
            c("view.representation"),
            c("view.working-set"),
        ],
        view_cluster: cluster_3d(vec![]),
        shortcut_box: [
            "assembly.insert",
            "assembly.mate",
            "assembly.move",
            "assembly.pattern",
            "assembly.interference",
            "view.measure",
            "view.section",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect(),
        mode_actions: Vec::new(),
        tabs: vec![
            tab(
                "assemble",
                "Assemble",
                vec![
                    group(
                        "components",
                        "Components",
                        vec![
                            variants(
                                "assembly.insert.menu",
                                "Insert",
                                "insert",
                                "assembly.insert",
                                "source",
                                &[
                                    ("part", "Part"),
                                    ("assembly", "Assembly"),
                                    ("circuit", "Circuit board"),
                                    ("step", "STEP file"),
                                    ("library", "Library part"),
                                ],
                            ),
                            c("assembly.new-part"),
                            c("assembly.replace"),
                        ],
                    ),
                    group(
                        "mates",
                        "Mates",
                        vec![
                            mate_menu,
                            c("assembly.mate-connector"),
                            c("assembly.smart-mate"),
                        ],
                    ),
                    group(
                        "motion",
                        "Motion",
                        vec![
                            variants(
                                "assembly.move.menu",
                                "Move",
                                "move",
                                "assembly.move",
                                "mode",
                                &[
                                    ("drag", "Free drag"),
                                    ("triad", "Move with triad"),
                                    ("align", "Move along an entity"),
                                ],
                            ),
                            c("assembly.fix"),
                        ],
                    ),
                    group(
                        "repeat",
                        "Repeat",
                        vec![
                            variants(
                                "assembly.pattern.menu",
                                "Pattern",
                                "pattern",
                                "assembly.pattern",
                                "layout",
                                &[
                                    ("linear", "Linear pattern"),
                                    ("circular", "Circular pattern"),
                                    ("feature", "Follow feature pattern"),
                                ],
                            ),
                            c("assembly.mirror"),
                            c("assembly.copy-with-mates"),
                        ],
                    ),
                    group(
                        "structure",
                        "Structure",
                        vec![
                            variants(
                                "assembly.subassembly.menu",
                                "Subassembly",
                                "subassembly",
                                "assembly.subassembly",
                                "action",
                                &[
                                    ("form", "Form new subassembly"),
                                    ("dissolve", "Dissolve"),
                                    ("flexible", "Make flexible"),
                                    ("rigid", "Make rigid"),
                                ],
                            ),
                            c("assembly.edit-in-context"),
                        ],
                    ),
                ],
            ),
            tab(
                "layout",
                "Layout",
                vec![
                    group(
                        "skeleton",
                        "Skeleton",
                        vec![
                            c("assembly.skeleton"),
                            c("assembly.publish"),
                            c("assembly.envelope"),
                            c("assembly.interface-check"),
                        ],
                    ),
                    group(
                        "features",
                        "Features",
                        vec![c("assembly.feature"), c("assembly.hole-series")],
                    ),
                ],
            ),
            tab(
                "fasteners",
                "Fasteners",
                vec![group(
                    "fasteners",
                    "Fasteners",
                    vec![
                        c("assembly.smart-fasteners"),
                        c("assembly.fastener-rule"),
                        menu(
                            "assembly.fastener-library.menu",
                            "Library",
                            "library",
                            vec![e("assembly.fastener-library")],
                        ),
                        c("assembly.hole-check"),
                    ],
                )],
            ),
            tab(
                "motion",
                "Motion",
                vec![group(
                    "motion",
                    "Motion",
                    vec![
                        c("motion.drag"),
                        c("motion.collision"),
                        c("motion.mate-controller"),
                        c("motion.animation"),
                        c("motion.basic"),
                        c("motion.to-fea"),
                    ],
                )],
            ),
            tab(
                "evaluate",
                "Evaluate",
                vec![
                    group(
                        "check",
                        "Check",
                        vec![
                            c("assembly.interference"),
                            c("assembly.clearance"),
                            c("assembly.hole-alignment"),
                        ],
                    ),
                    group(
                        "properties",
                        "Properties",
                        vec![
                            c("assembly.mass-properties"),
                            c("assembly.bom"),
                            c("assembly.where-used"),
                            c("assembly.visualization"),
                        ],
                    ),
                ],
            ),
            tab(
                "explode",
                "Explode",
                vec![group(
                    "explode",
                    "Explode",
                    vec![
                        c("explode.new"),
                        c("explode.step"),
                        c("explode.radial"),
                        c("explode.auto"),
                        c("explode.lines"),
                    ],
                )],
            ),
            tab(
                "simulate",
                "Simulate",
                vec![group(
                    "simulate",
                    "Simulate",
                    vec![
                        c("assembly.contacts"),
                        c("assembly.representation"),
                        c("study.new"),
                    ],
                )],
            ),
            tab(
                "manage",
                "Manage",
                vec![
                    group(
                        "parameters",
                        "Parameters",
                        vec![
                            c("assembly.options-effectivity"),
                            c("assembly.configurations"),
                            c("assembly.document-settings"),
                        ],
                    ),
                    group("references", "References", vec![c("assembly.update-pins")]),
                    group(
                        "display",
                        "Display",
                        vec![
                            c("assembly.isolate"),
                            c("assembly.hide"),
                            c("assembly.transparency"),
                        ],
                    ),
                    group(
                        "files",
                        "Files",
                        vec![
                            c("assembly.pack-and-share"),
                            variants(
                                "assembly.export.menu",
                                "Export",
                                "export",
                                "assembly.export",
                                "format",
                                &[("step", "STEP assembly"), ("gltf", "glTF"), ("usd", "USD")],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    });
}

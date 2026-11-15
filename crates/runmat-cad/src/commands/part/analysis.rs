//! The Evaluate, Annotate and Simulate tabs.

use super::*;

pub(super) fn evaluate() -> Vec<CommandDescriptor> {
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
            "Area, centroid, second and polar moments and principal moments of planar faces or a sketch.",
            vec![picks("section", "Faces or sketch", &[K::Face, K::Sketch])],
        ),
        a(
            "evaluate.check-geometry",
            "Check geometry",
            "check",
            "check_geometry",
            "Find invalid faces, short edges, gaps and self-intersections.",
            vec![
                picks("bodies", "Bodies", &[K::Body]).optional(),
                flag("strict", "Strict (also self-intersections; slower)", false),
                flag("short_edges", "Short edges", true),
                length("short_edge", "Shorter than", 0.01).if_checked("short_edges"),
                flag("gaps", "Edge and vertex gaps", true),
                length("gap", "Gaps over", 0.01).if_checked("gaps"),
                flag("min_radius", "Smallest radius", true),
            ],
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
            "Color the model by wall thickness and list each face's thinnest and thickest point.",
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
            "Color faces by curvature: convex red, concave blue, flat neutral.",
            vec![choice(
                "measure",
                "Show",
                &[
                    ("largest", "Largest curvature"),
                    ("mean", "Mean curvature"),
                    ("gaussian", "Gaussian curvature"),
                ],
            )],
        ),
        a(
            "evaluate.zebra",
            "Zebra stripes",
            "zebra",
            "zebra",
            "Show stripes to judge surface continuity, and each edge's continuity (G0, G1, G2).",
            vec![
                count("stripes", "Stripes", 20, 2),
                pick("direction", "Light direction", &[K::Plane, K::Face, K::Axis]).optional(),
            ],
        )
        .kw(&["continuity", "class a"]),
        a(
            "evaluate.deviation",
            "Deviation",
            "deviation",
            "deviation",
            "Measure the angle between faces along shared edges.",
            vec![
                picks("edges", "Edges", &[K::Edge]).optional(),
                angle("tolerance", "Tangent within", 0.5),
            ],
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
                text("rules", "Company rule pack", Some("company/dfm")).optional(),
                pick("direction", "Pull or build direction", &[K::Plane, K::Face, K::Axis])
                    .optional()
                    .shown("process", &["molding", "additive"]),
            ],
        )
        .kw(&["manufacturability", "design for manufacturing"]),
        a(
            "evaluate.cost",
            "Cost estimate",
            "cost",
            "cost",
            "Estimate the part's cost for a process and quantity from your shop's rates.",
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
                text("currency", "Currency", Some("USD")).optional(),
                qty("price", "Material price per kg", "ratio", None),
                qty("rate", "Machine rate per hour", "ratio", None),
                qty("setup", "Setup per batch", "ratio", Some(0.0)).optional(),
                qty("removal_rate", "Removal rate (cm³/min)", "ratio", None)
                    .shown("process", &["machining"]),
                length("allowance", "Stock allowance per side", 0.0)
                    .shown("process", &["machining"]),
                qty("build_rate", "Build rate (cm³/h)", "ratio", None)
                    .shown("process", &["additive"]),
                qty("cycle", "Cycle time per part", "time", None)
                    .shown("process", &["molding", "sheet_metal"]),
                qty("tooling", "Tooling (mold or die)", "ratio", Some(0.0))
                    .optional()
                    .shown("process", &["molding", "sheet_metal"]),
            ],
        ),
    ]
}

pub(super) fn annotate() -> Vec<CommandDescriptor> {
    let pmi = |id: &str, label: &str, icon: &str, item: &str, desc: &str, inputs: Vec<Field>| {
        cmd(id, label, icon, P)
            .desc(desc)
            .kw(&["pmi", "mbd"])
            .inputs(inputs)
            .when(bodies_exist())
            .edit(edit(EditOp::Add, Section::Pmi).item(item))
    };
    // Characteristics that take a diametral zone (a feature of size's axis) and a material
    // condition (ASME Y14.5, ISO 1101).
    let diametral = &["position", "straightness", "perpendicularity", "parallelism", "angularity", "concentricity"];
    let sized = &["position", "straightness", "perpendicularity", "parallelism", "angularity"];
    let related = &[
        "position",
        "perpendicularity",
        "parallelism",
        "angularity",
        "profile_of_line",
        "profile_of_surface",
        "concentricity",
        "symmetry",
        "circular_runout",
        "total_runout",
    ];
    vec![
        pmi(
            "annotate.datum",
            "Datum",
            "datum",
            "datum",
            "Label a face or axis as a datum feature.",
            vec![
                text("label", "Label", Some("A")).optional(),
                pick("target", "Face or axis", &[K::Face, K::Axis]),
            ],
        ),
        pmi(
            "annotate.gdt",
            "GD&T frame",
            "gdt",
            "tolerance",
            "Add a feature control frame to faces.",
            vec![
                picks("targets", "Applies to", &[K::Face]),
                choice(
                    "characteristic",
                    "Characteristic",
                    &[
                        ("position", "Position"),
                        ("flatness", "Flatness"),
                        ("straightness", "Straightness"),
                        ("circularity", "Circularity"),
                        ("cylindricity", "Cylindricity"),
                        ("profile_of_line", "Profile of a line"),
                        ("profile_of_surface", "Profile of a surface"),
                        ("perpendicularity", "Perpendicularity"),
                        ("parallelism", "Parallelism"),
                        ("angularity", "Angularity"),
                        ("concentricity", "Concentricity"),
                        ("symmetry", "Symmetry"),
                        ("circular_runout", "Circular runout"),
                        ("total_runout", "Total runout"),
                    ],
                ),
                flag("diametral", "Diametral zone (⌀)", false).shown("characteristic", diametral),
                length("tolerance", "Tolerance", 0.1),
                choice(
                    "modifier",
                    "Material condition",
                    &[
                        ("none", "None (regardless of size)"),
                        ("mmc", "Maximum material (Ⓜ)"),
                        ("lmc", "Least material (Ⓛ)"),
                    ],
                )
                .shown("characteristic", sized),
                text("datums", "Datums", Some("A|B|C"))
                    .optional()
                    .help("Primary, secondary and tertiary, with (M) or (L) after a label for its material condition.")
                    .shown("characteristic", related),
                flag("projected", "Projected zone (Ⓟ)", false).shown("characteristic", &["position", "perpendicularity"]),
                length("projected_height", "Projected height", 10.0).if_checked("projected"),
            ],
        ),
        pmi(
            "annotate.dimension",
            "Dimension",
            "dimension",
            "size",
            "Add a 3D dimension with tolerance: a hole's diameter, or the distance between two faces.",
            vec![
                picks("targets", "Faces", &[K::Face]),
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
                length("plus", "Plus", 0.1).shown("tolerance", &["symmetric", "bilateral", "limit"]),
                length("minus", "Minus", 0.1).shown("tolerance", &["bilateral", "limit"]),
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
                picks("datums", "Datums (A, B, C)", &[K::Face])
                    .optional()
                    .help("Planar faces in order. Empty: the part's datums, else its largest faces."),
                length("tolerance", "Tolerance", 0.1),
            ],
        ),
        pmi(
            "annotate.surface-finish",
            "Surface finish",
            "surface-finish",
            "finish",
            "Mark the roughness faces must have.",
            vec![
                picks("faces", "Faces", &[K::Face]),
                qty("ra", "Ra", "roughness", Some(1.6)).min(0.0),
                choice(
                    "removal",
                    "Process",
                    &[
                        ("any", "Any"),
                        ("removal_required", "Material removal required"),
                        ("removal_prohibited", "No material removal"),
                    ],
                ),
                text("method", "Method", Some("ground")).optional(),
                choice(
                    "lay",
                    "Lay",
                    &[
                        ("none", "Any"),
                        ("=", "Parallel (=)"),
                        ("⊥", "Perpendicular (⊥)"),
                        ("X", "Crossed (X)"),
                        ("M", "Multidirectional (M)"),
                        ("C", "Circular (C)"),
                        ("R", "Radial (R)"),
                        ("P", "Particulate (P)"),
                    ],
                ),
            ],
        ),
        pmi(
            "annotate.coating",
            "Coating",
            "coating",
            "coating",
            "Specify a coating on faces or the whole part.",
            vec![
                text("spec", "Specification", Some("anodize type II, black")),
                picks("faces", "Faces", &[K::Face])
                    .optional()
                    .help("Empty: the whole part."),
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
                radio(
                    "side",
                    "Side",
                    &[("arrow", "Arrow side"), ("other", "Other side"), ("both", "Both sides")],
                ),
                length("size", "Size", 4.0),
                flag("all_around", "All around", false),
                flag("field", "Field weld", false),
                flag("intermittent", "Intermittent", false).shown("type", &["fillet", "groove"]),
                length("length", "Length", 50.0).if_checked("intermittent"),
                length("pitch", "Pitch", 100.0).if_checked("intermittent"),
                text("tail", "Process (tail)", Some("GMAW")).optional(),
            ],
        ),
        pmi(
            "annotate.note",
            "Note",
            "note",
            "note",
            "Add a note in the model, with a leader to a face, edge or vertex.",
            vec![
                text("text", "Text", None),
                pick("anchor", "Attached to", &[K::Face, K::Edge, K::Vertex]).optional(),
            ],
        ),
        pmi(
            "annotate.annotation-view",
            "Annotation view",
            "annotation-view",
            "view",
            "Save a view direction that groups annotations.",
            vec![
                text("name", "Name", Some("front_pmi")),
                choice(
                    "orientation",
                    "Orientation",
                    &[
                        ("current", "Current view"),
                        ("front", "Front"),
                        ("back", "Back"),
                        ("top", "Top"),
                        ("bottom", "Bottom"),
                        ("right", "Right"),
                        ("left", "Left"),
                        ("isometric", "Isometric"),
                    ],
                ),
                picks("annotations", "Annotations", &[K::Annotation])
                    .optional()
                    .help("Empty: every annotation not in a view yet."),
            ],
        ),
    ]
}

pub(super) fn simulate() -> Vec<CommandDescriptor> {
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
            .desc("Publish a region, frame, value or path for assemblies and studies to use.")
            .kw(&["interface", "export name", "cam profile", "path"])
            .inputs(vec![
                text("name", "Name", Some("mount")),
                radio("target", "Publishes", &[("region", "Region"), ("frame", "Mate connector"), ("value", "Value"), ("path", "Path")]),
                pick("item", "Item", &[K::Region, K::MateConnector, K::Input]).shown("target", &["region", "frame", "value"]),
                // A cam's closed profile or a path mate's track: edges joined end to end.
                picks("edges", "Edges", &[K::Edge]).shown("target", &["path"]),
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
        cmd("simulate.check", "Study checks", "check", &[EditorKind::Part, EditorKind::Assembly])
            .desc("Check fit and mass before a study meshes the design: overlapping parts, open contacts and parts without a material.")
            .kw(&["preflight", "fit check", "mass check", "before simulation"])
            .inputs(vec![length("gap", "Contact gap", 0.1)])
            .edit(analyze("study_check")),
        cmd("simulate.studies", "Studies", "study", P)
            .desc("Open a study that uses this part.")
            .inputs(vec![text("study", "Study", None)])
            .edit(view_action("open_study")),
    ]
}

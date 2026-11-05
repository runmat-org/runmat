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

pub(super) fn annotate() -> Vec<CommandDescriptor> {
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

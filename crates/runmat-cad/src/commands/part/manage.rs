//! The Manage tab and the right-click actions on tree nodes.

use super::*;

pub(super) fn manage() -> Vec<CommandDescriptor> {
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
pub(super) fn tree_actions() -> Vec<CommandDescriptor> {
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

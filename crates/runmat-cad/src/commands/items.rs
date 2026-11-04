//! Editable properties of selected CAD items (Properties panel, selection mode): features,
//! bodies, parts, occurrences, mates, views. Item metadata lives here only (§5a): Annotate and
//! Manage don't repeat it.

use super::*;
use EditorKind::*;
use SelectKind as K;

fn set(section: Section, property: &str) -> ModelEdit {
    edit(EditOp::Set, section).item(property)
}

fn prop(field: Field, section: Section, property: &str) -> ItemField {
    ItemField {
        field,
        edit: set(section, property),
        read_only: false,
    }
}

fn shown(field: Field, property: &str) -> ItemField {
    ItemField {
        field: field.optional(),
        edit: set(Section::None, property),
        read_only: true,
    }
}

fn item(
    kind: SelectKind,
    editors: &[EditorKind],
    title: &str,
    sections: Vec<ItemSection>,
) -> ItemSchema {
    ItemSchema {
        kind,
        editors: editors.to_vec(),
        title: title.to_owned(),
        sections,
    }
}

fn section(title: &str, fields: Vec<ItemField>) -> ItemSection {
    ItemSection {
        title: title.to_owned(),
        fields,
    }
}

/// Item metadata shared by parts and occurrences.
pub(super) fn metadata(section_of: Section) -> ItemSection {
    section(
        "Item",
        vec![
            prop(
                text("part_number", "Part number", Some("BRK-0042")).optional(),
                section_of,
                "part_number",
            ),
            prop(
                text("revision", "Revision", Some("B")).optional(),
                section_of,
                "revision",
            ),
            prop(
                text("description", "Description", None).optional(),
                section_of,
                "description",
            ),
            prop(
                text("supplier", "Supplier", None).optional(),
                section_of,
                "supplier",
            ),
            prop(
                text("supplier_part", "Supplier part", None).optional(),
                section_of,
                "supplier_part",
            ),
            prop(
                choice(
                    "make_buy",
                    "Make or buy",
                    &[
                        ("make", "Make"),
                        ("buy", "Buy"),
                        ("reference", "Reference only"),
                    ],
                )
                .optional(),
                section_of,
                "make_buy",
            ),
        ],
    )
}

fn look(section_of: Section) -> ItemSection {
    section(
        "Look",
        vec![
            prop(
                text("material", "Material", Some("Al 2024-T3")).optional(),
                Section::Materials,
                "material",
            ),
            prop(
                text("color", "Color", Some("#9aa4b1")).optional(),
                Section::Appearances,
                "color",
            ),
            prop(
                choice(
                    "finish",
                    "Finish",
                    &[
                        ("matte", "Matte"),
                        ("satin", "Satin"),
                        ("gloss", "Gloss"),
                        ("metal", "Metal"),
                    ],
                )
                .optional(),
                Section::Appearances,
                "finish",
            ),
            prop(
                qty("transparency", "Transparency", "ratio", Some(0.0))
                    .optional()
                    .min(0.0)
                    .max(1.0),
                section_of,
                "transparency",
            ),
        ],
    )
}

pub(super) fn add(catalog: &mut Catalog) {
    let mass = || {
        section(
            "Mass",
            vec![
                shown(qty("mass", "Mass", "mass", None), "mass"),
                shown(qty("volume", "Volume", "length", None), "volume"),
                shown(qty("area", "Surface area", "area", None), "area"),
            ],
        )
    };
    catalog.items.extend([
        item(
            K::Feature,
            &[Part],
            "Feature",
            vec![
                section(
                    "Feature",
                    vec![
                        prop(text("name", "Name", None), Section::Features, "name"),
                        prop(
                            flag("suppressed", "Suppressed", false),
                            Section::Features,
                            "suppressed",
                        ),
                        shown(text("kind", "Type", None), "kind"),
                        shown(text("status", "Status", None), "status"),
                        shown(
                            qty("rebuild_ms", "Rebuild time", "time", None),
                            "rebuild_ms",
                        ),
                    ],
                ),
                section(
                    "Look",
                    vec![prop(
                        text("color", "Color", None).optional(),
                        Section::Appearances,
                        "color",
                    )],
                ),
            ],
        ),
        item(
            K::Sketch,
            &[Part],
            "Sketch",
            vec![section(
                "Sketch",
                vec![
                    prop(text("name", "Name", None), Section::Features, "name"),
                    shown(text("plane", "On", None), "plane"),
                    shown(text("definition", "Definition", None), "definition"),
                    shown(count("dof", "Degrees of freedom", 0, 0), "dof"),
                ],
            )],
        ),
        item(
            K::SketchEntity,
            &[Part, Assembly],
            "Sketch entity",
            vec![section(
                "Entity",
                vec![
                    prop(
                        flag("construction", "Construction", false),
                        Section::Sketch,
                        "construction",
                    ),
                    shown(qty("length", "Length", "length", None), "length"),
                    shown(text("relations", "Relations", None), "relations"),
                    shown(count("dof", "Entity DOF", 0, 0), "dof"),
                ],
            )],
        ),
        item(
            K::Body,
            &[Part],
            "Body",
            vec![
                section(
                    "Body",
                    vec![
                        prop(text("name", "Name", None), Section::Features, "body_name"),
                        shown(count("faces", "Faces", 0, 0), "faces"),
                        shown(count("edges", "Edges", 0, 0), "edges"),
                    ],
                ),
                look(Section::Appearances),
                mass(),
            ],
        ),
        item(
            K::Part,
            &[Part],
            "Part",
            vec![
                metadata(Section::Properties),
                look(Section::Appearances),
                mass(),
            ],
        ),
        item(
            K::Input,
            &[Part, Assembly],
            "Input",
            vec![section(
                "Input",
                vec![
                    prop(
                        text("value", "Value", Some("80 mm")),
                        Section::Inputs,
                        "value",
                    ),
                    shown(text("kind", "Kind", None), "kind"),
                    prop(
                        text("description", "Description", None).optional(),
                        Section::Inputs,
                        "description",
                    ),
                ],
            )],
        ),
        item(
            K::Region,
            &[Part, Assembly],
            "Region",
            vec![section(
                "Region",
                vec![
                    prop(text("name", "Name", None), Section::Regions, "name"),
                    shown(text("kind", "Kind", None), "kind"),
                    shown(count("entities", "Entities", 0, 0), "entities"),
                    prop(
                        choice(
                            "expect",
                            "Expect",
                            &[
                                ("at_least_one", "At least one"),
                                ("exactly_one", "Exactly one"),
                                ("any", "Any number"),
                            ],
                        ),
                        Section::Regions,
                        "expect",
                    ),
                ],
            )],
        ),
        item(
            K::Occurrence,
            &[Assembly],
            "Component",
            vec![
                section(
                    "Component",
                    vec![
                        shown(text("definition", "File", None), "definition"),
                        prop(
                            text("configuration", "Configuration", Some("default")).optional(),
                            Section::Occurrences,
                            "configuration",
                        ),
                        prop(
                            choice(
                                "representation",
                                "Representation",
                                &[
                                    ("full", "Full"),
                                    ("display", "Display only"),
                                    ("envelope", "Envelope"),
                                    ("defeatured", "Defeatured"),
                                ],
                            ),
                            Section::Occurrences,
                            "representation",
                        ),
                        prop(flag("fixed", "Fixed", false), Section::Occurrences, "fixed"),
                        prop(
                            flag("flexible", "Flexible", false),
                            Section::Occurrences,
                            "flexible",
                        ),
                        shown(text("pin", "Pinned revision", None), "revision"),
                        shown(count("dof", "Degrees of freedom", 0, 0), "dof"),
                    ],
                ),
                metadata(Section::Occurrences),
                look(Section::Occurrences),
            ],
        ),
        item(
            K::Mate,
            &[Assembly],
            "Mate",
            vec![section(
                "Mate",
                vec![
                    prop(
                        choice("type", "Type", super::assembly::MATE_TYPES),
                        Section::Mates,
                        "type",
                    ),
                    prop(
                        offset("offset", "Offset", 0.0).optional(),
                        Section::Mates,
                        "offset",
                    ),
                    prop(flag("flip", "Flip", false), Section::Mates, "flip"),
                    prop(flag("limits", "Limits", false), Section::Mates, "limits"),
                    prop(
                        qty("min", "Min", "angle", None)
                            .optional()
                            .if_checked("limits"),
                        Section::Mates,
                        "min",
                    ),
                    prop(
                        qty("max", "Max", "angle", None)
                            .optional()
                            .if_checked("limits"),
                        Section::Mates,
                        "max",
                    ),
                    prop(
                        flag("suppressed", "Suppressed", false),
                        Section::Mates,
                        "suppressed",
                    ),
                    shown(text("status", "Status", None), "status"),
                    shown(count("dof", "DOF remaining", 0, 0), "dof"),
                ],
            )],
        ),
        item(
            K::View,
            &[Drawing],
            "View",
            vec![section(
                "View",
                vec![
                    prop(text("scale", "Scale", Some("1:2")), Section::Views, "scale"),
                    prop(
                        choice(
                            "display",
                            "Display",
                            &[
                                ("hidden_removed", "Hidden lines removed"),
                                ("hidden_visible", "Hidden lines visible"),
                                ("wireframe", "Wireframe"),
                                ("shaded", "Shaded"),
                                ("shaded_edges", "Shaded with edges"),
                            ],
                        ),
                        Section::Views,
                        "display",
                    ),
                    prop(
                        choice(
                            "tangent_edges",
                            "Tangent edges",
                            &[
                                ("visible", "Visible"),
                                ("font", "With font"),
                                ("removed", "Removed"),
                            ],
                        ),
                        Section::Views,
                        "tangent_edges",
                    ),
                    prop(
                        text("configuration", "Configuration", Some("default")).optional(),
                        Section::Views,
                        "configuration",
                    ),
                    shown(text("model", "Model", None), "model"),
                ],
            )],
        ),
        item(
            K::Sheet,
            &[Drawing],
            "Sheet",
            vec![section(
                "Sheet",
                vec![
                    prop(text("name", "Name", None), Section::Sheets, "name"),
                    prop(text("size", "Size", Some("A3")), Section::Sheets, "size"),
                    prop(
                        text("scale", "Scale", Some("1:2")),
                        Section::Sheets,
                        "scale",
                    ),
                    prop(
                        text("format", "Format", None).optional(),
                        Section::Sheets,
                        "format",
                    ),
                ],
            )],
        ),
        item(
            K::Annotation,
            &[Drawing, Part],
            "Annotation",
            vec![section(
                "Annotation",
                vec![
                    shown(text("kind", "Kind", None), "kind"),
                    prop(
                        text("text", "Text", None).optional(),
                        Section::Annotations,
                        "text",
                    ),
                    prop(
                        text("layer", "Layer", None).optional(),
                        Section::Annotations,
                        "layer",
                    ),
                    shown(text("attached", "Attached to", None), "attached"),
                ],
            )],
        ),
    ]);
}

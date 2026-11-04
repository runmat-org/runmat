//! Editable properties of selected circuit items: components, nets, net classes, tracks, vias,
//! pads and pours. Values set by rules carry the rule's explain line in the projection.

use super::CIRCUIT;
use runmat_design_commands::*;
use SelectKind as K;

fn prop(field: Field, section: Section, property: &str) -> ItemField {
    ItemField {
        field: field.optional(),
        edit: edit(EditOp::Set, section).item(property),
        read_only: false,
    }
}

fn shown(field: Field, property: &str) -> ItemField {
    ItemField {
        field: field.optional(),
        edit: edit(EditOp::Set, Section::None).item(property),
        read_only: true,
    }
}

fn schema(kind: SelectKind, title: &str, sections: Vec<ItemSection>) -> ItemSchema {
    ItemSchema {
        kind,
        editors: CIRCUIT.to_vec(),
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

pub(super) fn add(c: &mut Catalog) {
    c.items.extend([
        schema(
            K::Component,
            "Component",
            vec![
                section(
                    "Component",
                    vec![
                        prop(
                            text("designator", "Designator", Some("R1")),
                            Section::Components,
                            "designator",
                        ),
                        prop(
                            text("value", "Value", Some("10 kOhm 1%")),
                            Section::Components,
                            "value",
                        ),
                        prop(text("part", "Part", None), Section::Components, "part"),
                        prop(
                            text("footprint", "Footprint", None),
                            Section::Components,
                            "footprint",
                        ),
                        prop(
                            choice(
                                "fitting",
                                "Variant",
                                &[
                                    ("fitted", "Fitted"),
                                    ("not_fitted", "Not fitted"),
                                    ("alternate", "Alternate"),
                                ],
                            ),
                            Section::Components,
                            "fitting",
                        ),
                        shown(text("stock", "Stock", None), "stock"),
                        shown(qty("power", "Power", "power", None), "power"),
                        shown(text("sheet", "Sheet", None), "sheet"),
                    ],
                ),
                section(
                    "Placement",
                    vec![
                        prop(qty("x", "X", "length", None), Section::Placement, "x"),
                        prop(qty("y", "Y", "length", None), Section::Placement, "y"),
                        prop(
                            qty("rotation", "Rotation", "angle", None),
                            Section::Placement,
                            "rotation",
                        ),
                        prop(
                            choice("side", "Side", &[("top", "Top"), ("bottom", "Bottom")]),
                            Section::Placement,
                            "side",
                        ),
                        prop(
                            flag("locked", "Locked", false),
                            Section::Placement,
                            "locked",
                        ),
                    ],
                ),
            ],
        ),
        schema(
            K::Net,
            "Net",
            vec![section(
                "Net",
                vec![
                    prop(text("name", "Name", None), Section::Nets, "name"),
                    prop(text("class", "Net class", None), Section::Nets, "class"),
                    shown(count("pins", "Pins", 0, 0), "pins"),
                    shown(qty("clearance", "Clearance", "length", None), "clearance"),
                    shown(qty("width", "Track width", "length", None), "width"),
                    shown(qty("length", "Routed length", "length", None), "length"),
                ],
            )],
        ),
        schema(
            K::NetClass,
            "Net class",
            vec![section(
                "Net class",
                vec![
                    prop(text("name", "Name", None), Section::Nets, "class_name"),
                    prop(
                        qty("clearance", "Clearance", "length", None),
                        Section::Rules,
                        "clearance",
                    ),
                    prop(
                        qty("width", "Track width", "length", None),
                        Section::Rules,
                        "width",
                    ),
                    prop(text("via", "Via", None), Section::Rules, "via"),
                    shown(count("nets", "Nets", 0, 0), "nets"),
                ],
            )],
        ),
        schema(
            K::Track,
            "Track",
            vec![section(
                "Track",
                vec![
                    shown(text("net", "Net", None), "net"),
                    prop(
                        qty("width", "Width", "length", None),
                        Section::Copper,
                        "width",
                    ),
                    prop(text("layer", "Layer", None), Section::Copper, "layer"),
                    shown(qty("length", "Length", "length", None), "length"),
                    shown(qty("clearance", "Clearance", "length", None), "clearance"),
                    prop(flag("locked", "Locked", false), Section::Copper, "locked"),
                ],
            )],
        ),
        schema(
            K::Via,
            "Via",
            vec![section(
                "Via",
                vec![
                    shown(text("net", "Net", None), "net"),
                    prop(
                        qty("drill", "Drill", "length", None),
                        Section::Copper,
                        "drill",
                    ),
                    prop(
                        qty("diameter", "Diameter", "length", None),
                        Section::Copper,
                        "diameter",
                    ),
                    prop(
                        choice(
                            "span",
                            "Span",
                            &[
                                ("through", "Through"),
                                ("blind", "Blind"),
                                ("buried", "Buried"),
                                ("micro", "Microvia"),
                            ],
                        ),
                        Section::Copper,
                        "span",
                    ),
                    prop(flag("tented", "Tented", true), Section::Copper, "tented"),
                ],
            )],
        ),
        schema(
            K::Pad,
            "Pad",
            vec![section(
                "Pad",
                vec![
                    shown(text("component", "Component", None), "component"),
                    shown(text("net", "Net", None), "net"),
                    shown(text("shape", "Shape", None), "shape"),
                    shown(text("layer", "Layers", None), "layer"),
                    shown(qty("clearance", "Clearance", "length", None), "clearance"),
                ],
            )],
        ),
        schema(
            K::Rule,
            "Design rule",
            vec![section(
                "Rule",
                vec![
                    shown(text("selector", "Applies to", None), "selector"),
                    shown(text("other", "Against", None), "other"),
                    shown(count("priority", "Priority", 0, 0), "priority"),
                    shown(text("source", "Written at", None), "source"),
                ],
            )],
        ),
        schema(
            K::Zone,
            "Pour",
            vec![section(
                "Pour",
                vec![
                    prop(text("net", "Net", None), Section::Copper, "net"),
                    prop(text("layer", "Layer", None), Section::Copper, "layer"),
                    prop(
                        qty("clearance", "Clearance", "length", None),
                        Section::Copper,
                        "clearance",
                    ),
                    prop(
                        choice(
                            "connect",
                            "Pad connection",
                            &[
                                ("relief", "Thermal relief"),
                                ("relief_tht", "Thermal relief, through-hole only"),
                                ("solid", "Solid"),
                                ("none", "None"),
                            ],
                        ),
                        Section::Copper,
                        "connect",
                    ),
                    prop(
                        count("priority", "Priority", 0, 0),
                        Section::Copper,
                        "priority",
                    ),
                    prop(
                        qty("min_width", "Minimum width", "length", None),
                        Section::Copper,
                        "min_width",
                    ),
                    prop(
                        qty("thermal_gap", "Relief gap", "length", None),
                        Section::Copper,
                        "thermal_gap",
                    ),
                    prop(
                        qty("spoke_width", "Spoke width", "length", None),
                        Section::Copper,
                        "spoke_width",
                    ),
                ],
            )],
        ),
    ]);
}

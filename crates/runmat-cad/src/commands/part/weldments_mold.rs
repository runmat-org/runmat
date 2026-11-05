//! The Weldments and Mold tabs.

use super::*;

pub(super) fn weldments() -> Vec<CommandDescriptor> {
    let w = |id: &str, label: &str, icon: &str, variant: &str, desc: &str, inputs: Vec<Field>| {
        cmd(id, label, icon, P)
            .desc(desc)
            .kw(&["weldment", "frame"])
            .inputs(inputs)
            .preview()
            .edit(add_feature_variant("weldment", "weldment", variant))
    };
    vec![
        w(
            "weldment.structural-member",
            "Structural member",
            "structural-member",
            "structural_member",
            "Place profiles along sketch segments.",
            vec![
                choice(
                    "standard",
                    "Standard",
                    &[("iso", "ISO"), ("ansi", "ANSI"), ("company", "Company")],
                ),
                choice(
                    "profile",
                    "Type",
                    &[
                        ("rect_tube", "Rectangular tube"),
                        ("square_tube", "Square tube"),
                        ("pipe", "Pipe"),
                        ("angle", "Angle"),
                        ("channel", "Channel"),
                        ("i_beam", "I-beam"),
                        ("flat_bar", "Flat bar"),
                    ],
                ),
                text("size", "Size", Some("40x40x3")),
                picks("segments", "Path segments", &[K::SketchEntity, K::Sketch]),
                choice(
                    "corner",
                    "Corner treatment",
                    &[
                        ("miter", "Miter"),
                        ("butt1", "Butt (first)"),
                        ("butt2", "Butt (second)"),
                        ("none", "None"),
                    ],
                ),
                angle("rotation", "Rotation", 0.0),
                flag("mirror_profile", "Mirror profile", false),
            ],
        )
        .when(has(Facet::Sketches)),
        w(
            "weldment.trim-extend",
            "Trim/extend",
            "trim-extend",
            "trim_extend",
            "Trim or extend members to each other or to faces.",
            vec![
                picks("members", "Members to trim", &[K::Body]),
                picks(
                    "boundary",
                    "Trimming boundary",
                    &[K::Body, K::Face, K::Plane],
                ),
                choice(
                    "type",
                    "Corner",
                    &[
                        ("end_trim", "End trim"),
                        ("miter", "Miter"),
                        ("butt", "Butt"),
                    ],
                ),
            ],
        )
        .when(has(Facet::Weldments)),
        w(
            "weldment.end-cap",
            "End cap",
            "end-cap",
            "end_cap",
            "Close a tube end with a plate.",
            vec![
                picks("faces", "End faces", &[K::Face]),
                length("thickness", "Thickness", 3.0),
                length("inset", "Inset", 0.0),
                flag("chamfer", "Chamfer corners", true),
            ],
        )
        .when(has(Facet::Weldments)),
        w(
            "weldment.gusset",
            "Gusset",
            "gusset",
            "gusset",
            "Add a triangular or polygonal plate between members.",
            vec![
                picks("faces", "Supporting faces", &[K::Face]),
                choice(
                    "shape",
                    "Profile",
                    &[("triangle", "Triangular"), ("polygon", "Polygonal")],
                ),
                length("d1", "Leg 1", 50.0),
                length("d2", "Leg 2", 50.0),
                length("thickness", "Thickness", 5.0),
            ],
        )
        .when(has(Facet::Weldments)),
        w(
            "weldment.weld-bead",
            "Weld bead",
            "weld-bead",
            "weld_bead",
            "Add a fillet or groove weld between faces.",
            vec![
                picks("faces", "Faces", &[K::Face, K::Edge]),
                choice(
                    "weld",
                    "Type",
                    &[("fillet", "Fillet"), ("groove", "Groove")],
                ),
                length("leg", "Leg size", 4.0),
                flag("intermittent", "Intermittent", false),
                length("pitch", "Pitch", 50.0).if_checked("intermittent"),
            ],
        )
        .when(has(Facet::Weldments)),
        cmd("weldment.cut-list", "Cut list", "cut-list", P)
            .desc("Show the cut list: members grouped by profile and length.")
            .when(has(Facet::Weldments))
            .edit(analyze("cut_list")),
        cmd("weldment.profile-library", "Profile library", "library", P)
            .desc("Browse and add structural profiles.")
            .inputs(vec![text("library", "Library", Some("company/profiles"))])
            .edit(edit(EditOp::Set, Section::Settings).item("profile_library")),
    ]
}

pub(super) fn mold() -> Vec<CommandDescriptor> {
    let m = |id: &str, label: &str, icon: &str, variant: &str, desc: &str, inputs: Vec<Field>| {
        cmd(id, label, icon, P)
            .desc(desc)
            .kw(&["mold", "tooling"])
            .inputs(inputs)
            .when(bodies_exist())
            .preview()
            .edit(add_feature_variant("mold", "mold", variant))
    };
    vec![
        m(
            "mold.parting-line",
            "Parting line",
            "parting-line",
            "parting_line",
            "Find the parting line for a pull direction.",
            vec![
                pick(
                    "direction",
                    "Pull direction",
                    &[K::Plane, K::Face, K::Axis, K::Edge],
                ),
                angle("draft", "Minimum draft", 1.0),
            ],
        ),
        m(
            "mold.shut-off",
            "Shut-off surfaces",
            "shut-off",
            "shut_off",
            "Close through-holes so core and cavity separate.",
            vec![
                picks("loops", "Hole edges", &[K::Edge]),
                choice(
                    "type",
                    "Patch",
                    &[
                        ("contact", "Contact"),
                        ("tangent", "Tangent"),
                        ("none", "No fill"),
                    ],
                ),
            ],
        ),
        m(
            "mold.parting-surface",
            "Parting surface",
            "parting-surface",
            "parting_surface",
            "Extend the parting line outward into a surface.",
            vec![
                pick("parting_line", "Parting line", &[K::Feature, K::Edge]),
                length("distance", "Distance", 50.0),
                choice(
                    "type",
                    "Type",
                    &[
                        ("normal", "Normal to pull"),
                        ("tangent", "Tangent to surface"),
                        ("perpendicular", "Perpendicular to surface"),
                    ],
                ),
            ],
        ),
        m(
            "mold.core-cavity",
            "Core/cavity split",
            "core-cavity",
            "core_cavity",
            "Split a mold block into core and cavity.",
            vec![
                pick("block", "Block sketch", &[K::Sketch]),
                length("core_depth", "Core depth", 30.0),
                length("cavity_depth", "Cavity depth", 30.0),
                flag("interlock", "Interlock surface", false),
            ],
        ),
        cmd("mold.draft-analysis", "Draft analysis", "draft-analysis", P)
            .desc("Color faces by draft angle for a pull direction.")
            .kw(&["mold", "draft"])
            .inputs(vec![
                pick("direction", "Pull direction", &[K::Plane, K::Face, K::Axis]),
                angle("angle", "Required draft", 1.0),
            ])
            .when(bodies_exist())
            .preview()
            .edit(analyze("draft")),
        m(
            "mold.undercut",
            "Undercut detection",
            "undercut",
            "undercuts",
            "Find faces that can't release in the pull direction.",
            vec![pick(
                "direction",
                "Pull direction",
                &[K::Plane, K::Face, K::Axis],
            )],
        ),
        m(
            "mold.scale",
            "Scale (shrink)",
            "scale",
            "shrink",
            "Scale the part for the material's shrinkage.",
            vec![
                qty("factor", "Shrink factor", "ratio", Some(1.005)).min(0.0),
                flag("uniform", "Uniform", true),
                qty("fx", "X factor", "ratio", Some(1.005)).shown("uniform", &["false"]),
                qty("fy", "Y factor", "ratio", Some(1.005)).shown("uniform", &["false"]),
                qty("fz", "Z factor", "ratio", Some(1.005)).shown("uniform", &["false"]),
            ],
        ),
    ]
}

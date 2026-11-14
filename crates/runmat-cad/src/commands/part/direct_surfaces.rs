//! The Direct and Surfaces tabs.

use super::*;

/// What Move face does, for its tooltip and panel: how it differs from Offset face.
const MOVE_FACE_HELP: &str = "Move or turn the picked faces as one rigid set; the neighboring faces extend or trim to meet them. Recorded as a feature. Offset moves each face along its own normal, the same edit as Offset face.";

/// What Offset face does, for its tooltip and panel.
const OFFSET_FACE_HELP: &str = "Move each face along its own normal by the same distance; the neighboring faces extend or trim to follow, so a hole's radius changes and an angled face moves square to itself. Move face moves the faces rigidly instead.";

const MOTION_HELP: &str = "Offset moves each face along its own normal. Translate moves the faces together along one direction, and Rotate turns them about an axis; the neighboring faces extend to meet them.";

pub(super) fn direct() -> Vec<CommandDescriptor> {
    let d = |id: &str, label: &str, icon: &str, variant: &str| {
        cmd(id, label, icon, P)
            .when(bodies_exist())
            .preview()
            .edit(add_feature_variant("direct", "direct", variant))
    };
    vec![
        d("direct.move-face", "Move face", "move-face", "move_faces")
            .desc(MOVE_FACE_HELP)
            .kw(&["push", "translate face", "rotate face", "offset face"])
            .inputs(vec![
                picks("faces", "Faces", &[K::Face]),
                radio(
                    "motion",
                    "Motion",
                    &[
                        ("normal", "Offset (each face along its own normal)"),
                        ("translate", "Translate along a direction"),
                        ("rotate", "Rotate about an axis"),
                    ],
                )
                .help(MOTION_HELP),
                pick(
                    "direction",
                    "Direction (edge, axis, or a plane's normal)",
                    &[K::Edge, K::Axis, K::Plane, K::Face],
                )
                .help("A plane or planar face moves the faces along its normal; an edge or axis along itself")
                .shown("motion", &["translate"]),
                pick("axis", "Axis", &[K::Axis, K::Edge]).shown("motion", &["rotate"]),
                offset("distance", "Distance", 1.0)
                    .drag(Manipulator::Arrow)
                    .shown("motion", &["normal", "translate"]),
                angle("angle", "Angle", 0.0)
                    .drag(Manipulator::Angle)
                    .shown("motion", &["rotate"]),
                flag("reverse", "Reverse direction", false).shown("motion", &["translate", "rotate"]),
                choice(
                    "keep",
                    "Keep (live relation)",
                    &[
                        ("none", "Nothing"),
                        ("fixed", "Fixed"),
                        ("parallel", "Parallel"),
                        ("perpendicular", "Perpendicular"),
                        ("coplanar", "Coplanar"),
                        ("concentric", "Concentric"),
                        ("tangent", "Tangent"),
                    ],
                )
                .help("The edit fails, naming the relation, rather than break it"),
                picks("keep_faces", "Faces to keep", &[K::Face])
                    .help("Fixed: faces that must not move. Otherwise one face kept to the first moved face, or two faces kept to each other")
                    .shown("keep", &["fixed", "parallel", "perpendicular", "coplanar", "concentric", "tangent"]),
            ]),
        d(
            "direct.offset-face",
            "Offset face",
            "offset-face",
            "move_faces",
        )
        .desc(OFFSET_FACE_HELP)
        .inputs(vec![
            picks("faces", "Faces", &[K::Face]),
            offset("distance", "Distance", 1.0)
                .drag(Manipulator::Arrow)
                .help("Each face moves this far along its own normal; negative moves it inward"),
        ]),
        d(
            "direct.replace-face",
            "Replace face",
            "replace-face",
            "replace_faces",
        )
        .desc("Replace faces with another face or surface.")
        .inputs(vec![
            picks("faces", "Faces to replace", &[K::Face]),
            pick("with", "Replacement", &[K::Face]),
        ]),
        d(
            "direct.delete-face",
            "Delete face",
            "delete-face",
            "delete_faces",
        )
        .desc("Delete faces and heal the gap from the neighbors.")
        .kw(&["heal", "defeature"])
        .inputs(vec![
            picks("faces", "Faces", &[K::Face]),
            radio(
                "heal",
                "Then",
                &[
                    ("heal", "Heal"),
                    ("patch", "Patch"),
                    ("leave", "Leave open"),
                ],
            ),
        ]),
        d(
            "direct.move-body",
            "Move/rotate body",
            "move-body",
            "move_bodies",
        )
        .desc("Move or rotate bodies directly.")
        .inputs(vec![
            picks("bodies", "Bodies", &[K::Body]),
            offset("dx", "Delta X", 0.0).drag(Manipulator::Triad),
            offset("dy", "Delta Y", 0.0),
            offset("dz", "Delta Z", 0.0),
            angle("angle", "Rotate", 0.0),
            pick("axis", "About", &[K::Axis, K::Edge]).optional(),
        ]),
        d(
            "direct.instant-3d",
            "Instant 3D",
            "instant-3d",
            "move_faces",
        )
        .desc("Drag any face or dimension in the view to change it.")
        .kw(&["drag", "dimension"])
        .inputs(vec![
            pick(
                "target",
                "Face or dimension",
                &[K::Face, K::DrawingDimension, K::Feature],
            ),
            offset("value", "Value", 0.0).drag(Manipulator::Arrow),
        ]),
        d("direct.pull", "Pull", "pull", "move_faces")
            .desc("Offset face that adds or removes material: each face moves along its own normal, and the body grows or shrinks to follow.")
            .kw(&["push pull", "press pull"])
            .inputs(vec![
                picks("faces", "Faces", &[K::Face]),
                offset("distance", "Distance", 1.0).drag(Manipulator::Arrow),
            ]),
    ]
}

pub(super) fn surfaces() -> Vec<CommandDescriptor> {
    let s = |id: &str, label: &str, icon: &str, variant: &str, desc: &str, inputs: Vec<Field>| {
        cmd(id, label, icon, P)
            .desc(desc)
            .kw(&["surface"])
            .inputs(inputs)
            .preview()
            .edit(add_feature_variant("surface", "kind", variant))
    };
    vec![
        s(
            "surface.extrude",
            "Extruded surface",
            "surface-extrude",
            "extrude",
            "Extrude a sketch's curves, open or closed, into a surface.",
            vec![
                pick("profile", "Sketch", &[K::Sketch]),
                length("depth", "Depth", 10.0).drag(Manipulator::Arrow),
                flag("symmetric", "Both sides", false),
            ],
        ),
        s(
            "surface.revolve",
            "Revolved surface",
            "surface-revolve",
            "revolve",
            "Revolve a sketch's curves into a surface.",
            vec![
                pick("profile", "Sketch", &[K::Sketch]),
                pick("axis", "Axis", &[K::Axis, K::Edge]),
                angle("angle", "Angle", 360.0).drag(Manipulator::Angle),
            ],
        ),
        s(
            "surface.sweep",
            "Swept surface",
            "surface-sweep",
            "sweep",
            "Sweep a sketch's curves along another sketch's path.",
            vec![
                pick("profile", "Profile", &[K::Sketch]),
                pick("path", "Path", &[K::Sketch]),
            ],
        ),
        s(
            "surface.loft",
            "Lofted surface",
            "surface-loft",
            "loft",
            "Loft a surface through curves.",
            vec![
                picks("profiles", "Profiles", &[K::Sketch, K::Curve, K::Edge]),
                picks("guides", "Guide curves", &[K::Curve, K::Edge]).optional(),
            ],
        ),
        s(
            "surface.boundary",
            "Boundary surface",
            "surface-boundary",
            "boundary",
            "Make a surface from curves in two directions.",
            vec![
                picks("direction1", "Direction 1", &[K::Edge]),
                picks("direction2", "Direction 2", &[K::Edge]).optional(),
                choice(
                    "continuity",
                    "Continuity",
                    &[
                        ("g0", "Contact (G0)"),
                        ("g1", "Tangent (G1)"),
                        ("g2", "Curvature (G2)"),
                    ],
                ),
            ],
        ),
        s(
            "surface.fill",
            "Fill",
            "surface-fill",
            "fill",
            "Patch a closed boundary with G0, G1 or G2 continuity.",
            vec![
                picks("boundary", "Boundary edges", &[K::Edge]),
                choice(
                    "continuity",
                    "Continuity",
                    &[
                        ("g0", "Contact (G0)"),
                        ("g1", "Tangent (G1)"),
                        ("g2", "Curvature (G2)"),
                    ],
                ),
            ],
        ),
        s(
            "surface.planar",
            "Planar surface",
            "surface-planar",
            "planar",
            "Fill a closed sketch with a flat surface.",
            vec![pick("profile", "Sketch", &[K::Sketch])],
        ),
        s(
            "surface.ruled",
            "Ruled surface",
            "surface-ruled",
            "ruled",
            "Grow strips from edges along their faces' normal, tilted outward by an angle.",
            vec![
                picks("edges", "Edges", &[K::Edge]),
                length("distance", "Distance", 10.0).drag(Manipulator::Arrow),
                angle("angle", "Angle", 0.0),
            ],
        ),
        s(
            "surface.offset",
            "Offset surface",
            "surface-offset",
            "offset",
            "Offset faces into a new surface.",
            vec![
                picks("faces", "Faces", &[K::Face]),
                offset("distance", "Distance", 1.0).drag(Manipulator::Arrow),
            ],
        ),
        s(
            "surface.extend",
            "Extend",
            "surface-extend",
            "extend",
            "Extend a surface past its edges.",
            vec![
                picks("edges", "Edges", &[K::Edge]),
                length("distance", "Distance", 5.0).drag(Manipulator::Arrow),
            ],
        ),
        s(
            "surface.trim",
            "Trim",
            "surface-trim",
            "trim",
            "Trim surfaces with a plane, keeping one side.",
            vec![
                picks("bodies", "Surfaces", &[K::Face, K::Body]),
                pick("tool", "Trim plane", &[K::Plane, K::Face]),
                radio(
                    "keep",
                    "Keep",
                    &[
                        ("positive", "Above the plane"),
                        ("negative", "Below the plane"),
                    ],
                ),
            ],
        ),
        s(
            "surface.untrim",
            "Untrim",
            "surface-untrim",
            "untrim",
            "Restore a trimmed surface's natural boundary.",
            vec![picks("faces", "Faces", &[K::Face])],
        ),
        s(
            "surface.knit",
            "Knit",
            "surface-knit",
            "knit",
            "Knit surfaces together, and make a solid when they close.",
            vec![
                picks("bodies", "Surfaces", &[K::Face, K::Body]),
                flag("solid", "Make a solid", true),
            ],
        ),
        s(
            "surface.thicken",
            "Thicken",
            "surface-thicken",
            "thicken",
            "Give a surface thickness to make a solid.",
            vec![
                pick("surface", "Surface", &[K::Face, K::Body]),
                length("thickness", "Thickness", 1.0).drag(Manipulator::Arrow),
                radio(
                    "side",
                    "Side",
                    &[("one", "Side 1"), ("both", "Both"), ("other", "Side 2")],
                ),
            ],
        ),
        s(
            "surface.delete-hole",
            "Delete hole",
            "delete-hole",
            "delete_hole",
            "Remove a hole's boundary from a surface.",
            vec![picks("edges", "Hole edges", &[K::Edge])],
        ),
        s(
            "surface.replace-face",
            "Replace face",
            "replace-face",
            "fill",
            "Replace solid faces with a surface.",
            vec![
                picks("faces", "Faces", &[K::Face]),
                pick("with", "Surface", &[K::Face]),
            ],
        ),
        s(
            "surface.cut",
            "Cut with surface",
            "surface-cut",
            "cut_with_surface",
            "Cut a solid with a surface, keeping one side.",
            vec![
                pick("surface", "Surface", &[K::Face, K::Body]),
                pick("body", "Solid", &[K::Body, K::Face]),
                radio(
                    "keep",
                    "Keep",
                    &[
                        ("positive", "Front of the surface"),
                        ("negative", "Behind the surface"),
                    ],
                ),
            ],
        ),
        s(
            "surface.mid-surface",
            "Mid-surface",
            "mid-surface",
            "mid_surface",
            "Make mid-surfaces between pairs of opposite faces for shell studies.",
            vec![picks("pairs", "Face pairs", &[K::Face])],
        ),
    ]
}

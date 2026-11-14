//! Solid, hole, blend, pattern, body and datum operations (`docs/design/mcad/part/features.md`).

use super::Quantity::*;
use super::{call, choice, opt, req, Allowed, Argument, Group, Operation};

const fn feature(
    word: &'static str,
    summary: &'static str,
    arguments: &'static [Argument],
) -> Operation {
    Operation {
        word,
        group: Group::Feature,
        summary,
        arguments,
    }
}

const TARGETS: &[Argument] = &[opt(
    "targets",
    References,
    "The bodies to combine with; empty means every body at this point in the history.",
)];

/// How a body-producing feature combines with existing bodies.
const MODES: &[Allowed] = &[
    choice("new", "A new body."),
    call("add", "Union with the target bodies; the part's first body feature makes the first body.", TARGETS),
    call("cut", "Subtract from the target bodies.", TARGETS),
    call("intersect", "Keep only what the feature shares with the target bodies.", TARGETS),
];

const MODE: Argument = opt("mode", Choice, "How the result combines with existing bodies.")
    .choices(MODES)
    .default("new");

const PROFILE: Argument = req(
    "profile",
    Profile,
    "The closed sketch profile: `sketch.<name>`, or one region of it.",
);

const EXTENTS: &[Allowed] = &[
    call("blind", "A set distance from the start.", &[req("distance", Length, "How far the extrude goes.")]),
    call("symmetric", "Half the distance each way from the sketch plane.", &[req("distance", Length, "The total length.")]),
    choice("through_all", "Through every body in the way."),
    choice("up_to_next", "Up to the next face the extrude meets, which must cover the whole profile."),
    call("up_to", "Up to a face; the end cap takes the face's shape.", &[req("face", Reference, "The face the extrude ends on.")]),
    call("up_to_plane", "Up to a plane parallel to the profile plane.", &[req("plane", Plane, "The end plane.")]),
    call("up_to_vertex", "Up to the plane through a vertex, parallel to the profile plane.", &[req("vertex", Point, "The vertex or point.")]),
    call(
        "offset_from_face",
        "Up to a face, `offset` short of it unless `beyond`.",
        &[
            req("face", Reference, "The face measured from."),
            req("offset", Length, "The gap to the face.").sites(&["second.offset"]),
            opt("beyond", Flag, "Go past the face by `offset`.").default("false"),
        ],
    ),
    call(
        "up_to_body",
        "Up to where the extrude first meets a body.",
        &[req("body", Reference, "The body."), opt("offset", Length, "A gap short of the body.")],
    ),
];

const EXTRUDE: &[Argument] = &[
    PROFILE,
    opt("distance", Length, "How far the extrude goes (a blind extent).").sites(&["extent.distance", "second.distance"]),
    opt("through_all", Flag, "Extrude through every body in the way."),
    opt("up_to_next", Flag, "Extrude up to the next face it meets."),
    opt("up_to", Reference, "Extrude up to this face."),
    opt("symmetric", Flag, "Extrude `distance` in total, half each way."),
    opt("extent", Choice, "The end condition in full form, when no short argument fits.").choices(EXTENTS),
    opt("reverse", Flag, "Extrude against the sketch normal.").default("false"),
    opt("second", Choice, "A second direction, opposite the first, with its own end condition.").choices(EXTENTS),
    opt("from", Choice, "Where the extrude starts.")
        .choices(&[
            choice("sketch_plane", "On the sketch plane."),
            call("offset", "On a plane offset from the sketch plane.", &[req("distance", Length, "The offset along the sketch normal.").sites(&["from.distance"])]),
            call("surface", "On a face; blind depths are measured from it.", &[req("face", Reference, "The start face.")]),
            call("vertex", "On the plane through a vertex, parallel to the sketch plane.", &[req("vertex", Point, "The vertex or point.")]),
        ])
        .default("sketch_plane"),
    opt("draft", Record, "Taper the side faces: `{ angle, outward, second_angle, second_outward }`."),
    opt("draft.angle", Angle, "The draft angle; positive tapers the profile inward away from the start."),
    opt("draft.second_angle", Angle, "The second direction's draft angle (default: the same)."),
    opt("thin", Record, "A wall along the profile curves: `{ thickness, side, second_thickness, cap_ends }`."),
    opt("thin.thickness", Length, "The wall thickness."),
    opt("thin.second_thickness", Length, "The outside thickness when `side = two`."),
    opt("thin.cap_ends", Length, "Close both ends with caps this thick."),
    MODE,
];

const HOLE_FORMS: &[Allowed] = &[
    choice("simple", "A plain drilled hole."),
    call(
        "counterbore",
        "A wider, flat-bottomed recess at the top for a bolt head.",
        &[
            opt("diameter", Length, "The counterbore diameter (default: from `spec`)."),
            opt("depth", Length, "The counterbore depth (default: from `spec`)."),
        ],
    ),
    call(
        "countersink",
        "A cone at the top for a flat-head screw.",
        &[
            opt("diameter", Length, "The countersink diameter at the face (default: from `spec`)."),
            opt("angle", Angle, "The included cone angle (default: from `spec`)."),
        ],
    ),
    call(
        "tapped",
        "Drilled at the tap drill and threaded.",
        &[
            opt("thread_depth", Length, "The threaded depth (default: the hole less 1.5 pitches when blind)."),
            opt("near_countersink", Angle, "A countersink at the entry."),
        ],
    ),
    call("pipe_tap", "A tapered pipe thread; the drill tapers at the thread taper.", &[opt("thread_depth", Length, "The thread length.")]),
    call(
        "slot",
        "A slot of the hole's width, `length` center to center.",
        &[
            req("length", Length, "Center-to-center length."),
            opt("angle", Angle, "The slot's turn from the face frame's x axis.").default("0 deg"),
            opt("counterbore", Record, "A counterbore around the slot: `(diameter, depth)`."),
        ],
    ),
];

const HOLE: &[Argument] = &[
    req("face", Reference, "The planar face the holes start on."),
    req("at", Record, "Keyed positions: `{ a = point(15 mm, 25 mm, 10 mm), … }`.").sites(&["positions"]),
    opt("diameter", Length, "The drilled diameter (default: from `spec`)."),
    opt("depth", Length, "A blind hole's full-diameter depth.").sites(&["end.depth"]),
    opt("through_all", Flag, "Drill through every body in the way."),
    opt("up_to_next", Flag, "Drill to the next face the hole meets."),
    opt("end", Choice, "How deep the hole goes, in full form.").choices(&[
        call("blind", "To a drilled depth.", &[req("depth", Length, "The full-diameter depth.")]),
        choice("through_all", "Through every body in the way."),
        choice("up_to_next", "Through the material the hole starts in, to the next face."),
        call(
            "up_to_face",
            "To a face, with an offset past it.",
            &[req("face", Reference, "The face."), opt("offset", Length, "Past the face.").sites(&["end.offset"])],
        ),
    ]),
    opt("form", Choice, "The hole's form.").choices(HOLE_FORMS).default("simple"),
    opt("spec", Record, "A standard size and fit: `{ standard = iso, size = M6 }`; the size sets the diameters."),
    opt("form.counterbore.diameter", Length, "A slot's counterbore diameter."),
    opt("form.counterbore.depth", Length, "A slot's counterbore depth."),
    opt("drill_point", Angle, "A blind hole's drill-point angle (default: a flat bottom)."),
    opt("series", Record, "A hole through several bodies: `{ bodies, end, middle_diameter }`."),
    opt("series.middle_diameter", Length, "The middle bodies' diameter (default: the start clearance)."),
];

const PROFILES: &[Allowed] = &[
    choice("circular", "A circular cross-section."),
    call("conic", "A conic section.", &[req("rho", Number, "Shape: 0.5 a parabola, below an ellipse, above a hyperbola.").sites(&["profile.rho"])]),
    choice("curvature", "Curvature-continuous (G2) to both faces."),
];

const BLEND_PROFILE: Argument = opt("profile", Choice, "The blend's cross-section.")
    .choices(PROFILES)
    .default("circular");

const CHAMFER_FORMS: &[Allowed] = &[
    call("equal", "The same setback on both faces.", &[req("distance", Length, "The setback.")]),
    call(
        "distances",
        "A setback on each face.",
        &[req("distance", Length, "On the first face."), req("distance2", Length, "On the second face.")],
    ),
    call(
        "distance_angle",
        "A setback and an angle from the first face.",
        &[req("distance", Length, "On the first face."), req("angle", Angle, "From the first face.")],
    ),
    call("vertex", "A corner cut through points along each of a vertex's three edges.", &[req("distances", Length, "Three lengths, in the vertex's edge order.")]),
];

const BOOLEAN: &[Argument] = &[
    req("target", Reference, "The body changed."),
    req("tools", References, "The bodies combined with it."),
    opt("keep_tools", Flag, "Keep the tool bodies.").default("false"),
];

const BODIES: Argument = req("bodies", References, "The bodies.");
const DX: Argument = req("dx", Length, "Offset along x.");
const DY: Argument = req("dy", Length, "Offset along y.");
const DZ: Argument = req("dz", Length, "Offset along z.");

const SEEDS: &[Argument] = &[
    opt("features", Features, "Features whose tools repeat at each copy."),
    opt("bodies", References, "Bodies copied at each copy."),
    opt("faces", References, "Faces of one body whose material repeats (a boss or a pocket)."),
    opt("seeds", Record, "The seeds in full form: `{ seeds = features, features = […] }`."),
];

const LAYOUTS: &[Allowed] = &[
    call(
        "linear",
        "`count` copies in all, `spacing` apart along `direction`.",
        &[req("direction", Axis, "The direction."), req("spacing", Length, "The gap between copies."), req("count", Count, "Copies, the seed included.")],
    ),
    call(
        "grid",
        "A two-way array.",
        &[
            req("first", Axis, "The first direction."),
            req("first_spacing", Length, "The gap along the first direction."),
            req("first_count", Count, "Copies along the first direction."),
            req("second", Axis, "The second direction."),
            req("second_spacing", Length, "The gap along the second direction."),
            req("second_count", Count, "Copies along the second direction."),
        ],
    ),
    call(
        "circular",
        "`count` copies about `axis`, equally spaced over `angle`.",
        &[
            req("axis", Axis, "The axis."),
            req("count", Count, "Copies, the seed included."),
            req("angle", Angle, "The total angle, or the step without equal spacing."),
            opt("equal_spacing", Flag, "Spread the copies over `angle`.").default("true"),
        ],
    ),
    call("table", "One copy per row of offsets.", &[req("rows", Record, "`{ key, dx, dy, dz }` rows.")]),
    call(
        "curve",
        "Copies along an edge or curve from its start.",
        &[
            req("curve", Reference, "The edge or curve."),
            req("count", Count, "Copies, the seed included."),
            opt("spacing", Length, "The gap along the curve (default: equal over its length)."),
            opt("orientation", Choice, "How copies turn.").choices(&[
                choice("translate", "Translated only."),
                choice("align_to_curve", "Turned with the curve."),
            ]),
        ],
    ),
    call(
        "sketch",
        "One copy at each point of a sketch.",
        &[
            req("sketch", Profile, "The sketch."),
            opt("points", Record, "The points used (default: all its points)."),
            opt("reference", Point, "The point the seed sits at (default: the first point)."),
        ],
    ),
    call(
        "fill",
        "Copies filling a sketch region on a lattice.",
        &[
            req("boundary", Profile, "The region filled."),
            req("spacing", Length, "The lattice pitch."),
            opt("margin", Length, "Kept inside the boundary.").default("0 mm"),
            opt("direction", Axis, "The lattice's first direction."),
            opt("seed", Point, "Where the lattice starts (default: the seed's position)."),
            opt("lattice", Choice, "The lattice.").choices(&[
                choice("square", "A square grid."),
                choice("staggered", "Rows offset by half a pitch (a perforation)."),
                choice("circular", "Rings about the seed."),
            ]),
        ],
    ),
    call("variable", "Copies each with their own offset and input values.", &[req("instances", Record, "`{ key, dx, dy, dz, values }` rows.")]),
];

const PATTERN: &[Argument] = &[
    SEEDS[0],
    SEEDS[1],
    SEEDS[2],
    SEEDS[3],
    req("layout", Choice, "Where the copies go.").choices(LAYOUTS),
    opt("skip", Record, "Copies left out, by key."),
    opt("rows.dx", Length, "A table row's offset along x."),
    opt("rows.dy", Length, "A table row's offset along y."),
    opt("rows.dz", Length, "A table row's offset along z."),
    opt("instances.dx", Length, "A variable copy's offset along x."),
    opt("instances.dy", Length, "A variable copy's offset along y."),
    opt("instances.dz", Length, "A variable copy's offset along z."),
];

const MIRROR: &[Argument] = &[SEEDS[0], SEEDS[1], SEEDS[2], SEEDS[3], req("plane", Plane, "The mirror plane.")];

pub(super) const OPERATIONS: &[Operation] = &[
    feature("extrude", "A prism from a closed sketch profile.", EXTRUDE),
    feature(
        "revolve",
        "A solid of revolution: a profile turned about an axis.",
        &[PROFILE, req("axis", Axis, "The axis turned about."), req("angle", Angle, "How far it turns."), MODE],
    ),
    feature(
        "sweep",
        "A profile moved along a path.",
        &[
            PROFILE,
            req("path", Profile, "The path: a sketch's open chain."),
            opt("follow_path", Flag, "Keep the profile's angle to the path through bends.").default("false"),
            MODE,
        ],
    ),
    feature(
        "loft",
        "A solid through profiles in order.",
        &[req("profiles", Record, "The profiles, in order."), opt("guides", References, "Guide curves."), MODE],
    ),
    feature(
        "rib",
        "A rib from an open sketch.",
        &[
            PROFILE,
            req("thickness", Length, "The rib's thickness."),
            req("depth", Length, "How far the rib extends along the sketch normal."),
            opt("reverse", Flag, "Extend the other way.").default("false"),
        ],
    ),
    feature(
        "emboss",
        "Raised or recessed text or shapes on a face.",
        &[
            PROFILE,
            req("face", Reference, "The face."),
            req("height", Length, "Positive raises (emboss), negative sinks (deboss)."),
            opt("wrap", Flag, "Wrap onto a curved face.").default("false"),
        ],
    ),
    feature("hole", "Drilled holes at keyed positions on a face.", HOLE),
    feature(
        "thread",
        "A thread on a cylindrical face.",
        &[
            req("face", Reference, "The cylindrical face threaded."),
            req("spec", Record, "The thread standard and size."),
            opt("length", Length, "The threaded length (default: the whole face)."),
            opt("modelled", Flag, "Cut the helix into the geometry; otherwise the thread is cosmetic.").default("false"),
        ],
    ),
    feature(
        "fillet",
        "Rounds edges.",
        &[
            req("edges", References, "The edges rounded."),
            req("radius", Length, "The radius; with `variable`, the radius where no point gives one."),
            opt("variable", Record, "Radii at points along each edge: `{ at, radius }`, `at` from 0 to 1."),
            opt("variable.at", Number, "Where along the edge, from 0 at its start to 1 at its end.").sites(&["variable.at"]),
            opt("variable.radius", Length, "The radius there."),
            BLEND_PROFILE,
            opt("setbacks", Record, "Corner setbacks: `{ vertex, distance }`."),
            opt("setbacks.distance", Length, "How far short of the vertex the blends stop."),
            opt("propagate", Flag, "Extend along tangent-continuous edges.").default("true"),
        ],
    ),
    feature(
        "face_fillet",
        "A blend tangent to two face sets.",
        &[
            req("first", References, "The first face set."),
            req("second", References, "The second face set."),
            req("radius", Length, "The radius."),
            BLEND_PROFILE,
        ],
    ),
    feature(
        "full_round",
        "Replaces center faces by a blend tangent to two side face sets.",
        &[
            req("first", References, "One side."),
            req("center", References, "The faces replaced."),
            req("second", References, "The other side."),
        ],
    ),
    feature(
        "chamfer",
        "Bevels edges or vertices.",
        &[
            req("edges", References, "The edges (for a vertex chamfer, the vertices)."),
            opt("distance", Length, "The setback.").sites(&["form.distance"]),
            opt("distance2", Length, "The setback on the second face."),
            opt("angle", Angle, "The angle from the first face.").sites(&["form.angle"]),
            opt("distances", Length, "A vertex chamfer's three lengths."),
            opt("form", Choice, "The chamfer in full form.").choices(CHAMFER_FORMS),
            opt("flip", Flag, "Swap the faces the distances are measured on.").default("false"),
        ],
    ),
    feature(
        "shell",
        "Hollows a body.",
        &[
            req("body", Reference, "The body hollowed."),
            req("thickness", Length, "The wall thickness."),
            opt("open_faces", References, "Faces removed to open the shell."),
            opt("faces", Record, "Walls with their own thickness: `{ faces, thickness }`."),
            opt("faces.thickness", Length, "Those walls' thickness."),
            opt("outward", Flag, "Grow the walls outward; the body becomes the cavity.").default("false"),
        ],
    ),
    feature(
        "draft",
        "Tilts faces for molding.",
        &[
            req("faces", References, "The faces drafted."),
            req("neutral", Plane, "The neutral plane; its normal is the pull direction."),
            req("angle", Angle, "The draft angle."),
            opt("kind", Choice, "How the faces hinge.")
                .choices(&[
                    choice("neutral_plane", "About their line on the neutral plane."),
                    call("parting_line", "About the parting-line edges each face touches.", &[req("edges", References, "The parting-line edges.")]),
                    call("step", "As a parting-line draft, with step faces between hinge heights.", &[req("edges", References, "The parting-line edges.")]),
                ])
                .default("neutral_plane"),
            opt("flip", Flag, "Pull the other way.").default("false"),
        ],
    ),
    feature("pattern", "Repeats features, bodies or faces; each copy is keyed.", PATTERN),
    feature(
        "mirror",
        "Reflects features, bodies or faces about a plane.",
        MIRROR,
    ),
    feature("boolean.union", "Joins tool bodies to a target body.", BOOLEAN),
    feature("boolean.subtract", "Cuts tool bodies from a target body.", BOOLEAN),
    feature("boolean.intersect", "Keeps what a target body shares with the tool bodies.", BOOLEAN),
    feature(
        "split",
        "Splits a body with a plane or surface into `body(side = positive)` and `body(side = negative)`.",
        &[req("body", Reference, "The body split."), req("tool", Plane, "The plane or surface.")],
    ),
    feature("body_op.delete", "Removes bodies.", &[BODIES]),
    feature("body_op.translate", "Moves bodies by an offset.", &[BODIES, DX, DY, DZ]),
    feature(
        "body_op.rotate",
        "Turns bodies about an axis.",
        &[BODIES, req("axis", Axis, "The axis turned about."), req("angle", Angle, "The turn.")],
    ),
    feature(
        "body_op.copy",
        "Copies bodies, the originals kept: copy `k` moves `k` times by the offset, then turns `k` times by `angle` about `axis`.",
        &[
            BODIES,
            DX,
            DY,
            DZ,
            opt("axis", Axis, "The axis each copy turns about."),
            opt("angle", Angle, "The turn per copy."),
            opt("count", Count, "Copies.").default("1"),
        ],
    ),
    feature(
        "body_op.indent",
        "Presses a tool body into a target; the tool, grown by `clearance`, is cut from the target and kept.",
        &[
            req("target", Reference, "The body indented."),
            req("tool", Reference, "The body pressed in."),
            opt("clearance", Length, "The gap around the tool.").default("0 mm"),
        ],
    ),
    feature(
        "import",
        "A body from a STEP file, held by the file's digest; its named faces read as `feature.<name>.face(\"Top\")`.",
        &[req("source", Text, "The project-relative path of the file."), req("digest", Text, "The file's content digest.")],
    ),
    feature(
        "library",
        "An instance of a reusable feature definition (`feature_def`).",
        &[
            req("def", Text, "The definition."),
            opt("args", Record, "Its inputs, by name."),
            opt("bindings", Record, "Its references, by name."),
            MODE,
        ],
    ),
];

//! Sketch entities, constraints, dimensions and reference functions
//! (`docs/design/mcad/part/sketches.md`).

use super::Quantity::*;
use super::{opt, req, Argument, Group, Operation};

const fn entity(word: &'static str, summary: &'static str, arguments: &'static [Argument]) -> Operation {
    Operation {
        word,
        group: Group::SketchEntity,
        summary,
        arguments,
    }
}

const fn constraint(word: &'static str, summary: &'static str, arguments: &'static [Argument]) -> Operation {
    Operation {
        word,
        group: Group::SketchConstraint,
        summary,
        arguments,
    }
}

const CONSTRUCTION: Argument = opt(
    "construction",
    Flag,
    "A construction entity: it guides other entities and makes no profile.",
);

const A: Argument = req("a", Reference, "The first entity or point.");
const B: Argument = req("b", Reference, "The second entity or point.");
const VALUE_LENGTH: Argument = req("value", Length, "The dimension's value.");

pub(super) const ENTITIES: &[Operation] = &[
    entity(
        "text",
        "Text drawn as sketch curves, for an emboss or a cut.",
        &[
            req("text", Text, "The characters."),
            req("at", Point, "Where the text starts."),
            req("height", Length, "The character height."),
            opt("font", Choice, "The font.").default("single_line"),
            opt("stroke", Length, "The stroke width of a single-line font."),
        ],
    ),
    entity("point", "A sketch point.", &[req("at", Point, "Where it sits: `[x, y]`."), CONSTRUCTION]),
    entity(
        "line",
        "A straight sketch line; its points are `start` and `end`.",
        &[req("start", Point, "The start point: `[x, y]` or another entity's point."), req("end", Point, "The end point."), CONSTRUCTION],
    ),
    entity(
        "arc",
        "A circular arc counter-clockwise from `start` to `end` about `center`.",
        &[req("center", Point, "The center."), req("start", Point, "The start point."), req("end", Point, "The end point."), CONSTRUCTION],
    ),
    entity(
        "circle",
        "A circle; its point is `center`.",
        &[req("center", Point, "The center."), req("radius", Length, "The radius."), CONSTRUCTION],
    ),
    entity(
        "spline",
        "An interpolating cubic through points in order: `start`, `p1` … and `end`.",
        &[req("points", Record, "The points, in order."), opt("closed", Flag, "Run back to the first point."), CONSTRUCTION],
    ),
    entity(
        "ellipse",
        "An ellipse about `center`; with `start` and `end`, an elliptical arc.",
        &[
            req("center", Point, "The center."),
            req("major", Length, "The semi-major axis."),
            req("minor", Length, "The semi-minor axis."),
            opt("angle", Angle, "The major axis's turn from the sketch x axis.").default("0 deg"),
            opt("start", Point, "An arc's start point."),
            opt("end", Point, "An arc's end point."),
            CONSTRUCTION,
        ],
    ),
];

pub(super) const CONSTRAINTS: &[Operation] = &[
    constraint("coincident", "Two points meet, or a point lies on a curve.", &[A, B]),
    constraint("horizontal", "A line runs along the sketch x axis.", &[req("line", Reference, "The line.")]),
    constraint("vertical", "A line runs along the sketch y axis.", &[req("line", Reference, "The line.")]),
    constraint("parallel", "Two lines are parallel.", &[A, B]),
    constraint("perpendicular", "Two lines meet at a right angle.", &[A, B]),
    constraint("tangent", "Two curves touch without crossing.", &[A, B]),
    constraint("equal", "Two lines have the same length, or two circles the same radius.", &[A, B]),
    constraint("fix", "Holds an entity or point where it is.", &[req("target", Reference, "The entity or point.")]),
    constraint(
        "midpoint",
        "A point sits at the middle of a line.",
        &[req("point", Reference, "The point."), req("line", Reference, "The line.")],
    ),
    constraint("concentric", "Two circles or arcs share a center.", &[A, B]),
    constraint("coradial", "Two circles or arcs share a center and a radius.", &[A, B]),
    constraint("collinear", "Two lines lie on one infinite line.", &[A, B]),
    constraint(
        "symmetric",
        "Points `a` and `b` are mirror images in the line `axis`.",
        &[A, B, req("axis", Reference, "The mirror line.")],
    ),
    constraint("distance", "The distance between two entities or points.", &[A, B, VALUE_LENGTH]),
    constraint("length", "A line's length.", &[req("line", Reference, "The line."), VALUE_LENGTH]),
    constraint("radius", "A circle's or arc's radius.", &[req("curve", Reference, "The circle or arc."), VALUE_LENGTH]),
    constraint("diameter", "A circle's diameter.", &[req("curve", Reference, "The circle."), VALUE_LENGTH]),
    constraint("major_radius", "An ellipse's semi-major axis.", &[req("curve", Reference, "The ellipse."), VALUE_LENGTH]),
    constraint("minor_radius", "An ellipse's semi-minor axis.", &[req("curve", Reference, "The ellipse."), VALUE_LENGTH]),
    constraint("angle", "The angle between two lines.", &[A, B, req("value", Angle, "The angle.")]),
    constraint(
        "limit",
        "Keeps a dimension within a range and leaves it free inside it.",
        &[req("dimension", Record, "The dimension limited."), req("min", Length, "The lower bound."), req("max", Length, "The upper bound.")],
    ),
];

pub(super) const REFERENCES: &[Operation] = &[
    Operation {
        word: "point",
        group: Group::Reference,
        summary: "A point by its coordinates in the part: `point(x, y, z)`.",
        arguments: &[
            req("x", Length, "The x coordinate."),
            req("y", Length, "The y coordinate."),
            req("z", Length, "The z coordinate."),
        ],
    },
    Operation {
        word: "edges_between",
        group: Group::Reference,
        summary: "The edges where two faces meet.",
        arguments: &[A, B],
    },
    Operation {
        word: "vertex_of",
        group: Group::Reference,
        summary: "The vertex where edges or faces meet.",
        arguments: &[],
    },
    Operation {
        word: "piece",
        group: Group::Reference,
        summary: "One piece of a face a later feature split.",
        arguments: &[],
    },
];

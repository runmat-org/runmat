//! Datums, curves, connectors and the feature families written `<family>.<operation>(…)`:
//! surfaces, sheet metal, weldments, molds and direct edits.
//!
//! The datum and curve entries list their arguments. The family entries carry each operation's
//! summary; their arguments are documented with the family (`docs/design/mcad/part/features.md`).

use super::Quantity::*;
use super::{opt, req, Argument, Group, Operation};

const fn op(
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

const fn named(word: &'static str, summary: &'static str) -> Operation {
    op(word, summary, &[])
}

pub(super) const OPERATIONS: &[Operation] = &[
    op(
        "plane.offset",
        "A datum plane parallel to a base plane, `distance` along its normal; read as `plane.<name>`.",
        &[req("base", Plane, "The base plane or planar face."), req("distance", Length, "The offset along the normal.")],
    ),
    op(
        "plane.angle",
        "A datum plane turned about an axis lying in a base plane.",
        &[req("base", Plane, "The base plane."), req("axis", Axis, "The hinge axis."), req("angle", Angle, "The turn.")],
    ),
    op(
        "plane.three_point",
        "A datum plane through three points.",
        &[req("a", Point, "The first point."), req("b", Point, "The second point."), req("c", Point, "The third point.")],
    ),
    op(
        "plane.mid",
        "A datum plane midway between two planes.",
        &[req("a", Plane, "The first plane."), req("b", Plane, "The second plane.")],
    ),
    op(
        "plane.normal_to_curve",
        "A datum plane normal to a curve at a point on it.",
        &[req("curve", Reference, "The curve or edge."), req("parameter", Number, "Where along the curve, from 0 at its start to 1 at its end.")],
    ),
    op(
        "plane.tangent_to_face",
        "A datum plane tangent to a face at a point.",
        &[req("face", Reference, "The face."), req("at", Point, "The point.")],
    ),
    op(
        "axis.through",
        "A datum axis through two points; read as `axis.<name>`.",
        &[req("a", Point, "The first point."), req("b", Point, "The second point.")],
    ),
    op("axis.of_face", "The axis of a cylindrical or conical face.", &[req("face", Reference, "The face.")]),
    op("axis.of_edge", "The axis along a straight edge, or a circular edge's axis.", &[req("edge", Reference, "The edge.")]),
    op(
        "axis.two_planes",
        "The axis where two planes meet.",
        &[req("a", Plane, "The first plane."), req("b", Plane, "The second plane.")],
    ),
    op(
        "axis.normal",
        "The axis normal to a plane through a point.",
        &[req("plane", Plane, "The plane."), req("through", Point, "The point.")],
    ),
    op("point.at", "A datum point at another point.", &[req("at", Point, "The point.")]),
    op("point.coords", "A datum point at coordinates.", &[req("x", Length, "x."), req("y", Length, "y."), req("z", Length, "z.")]),
    op(
        "point.offset",
        "A datum point offset from another.",
        &[req("base", Point, "The base point."), req("dx", Length, "Offset along x."), req("dy", Length, "Offset along y."), req("dz", Length, "Offset along z.")],
    ),
    op("point.center_of", "A datum point at the center of a circular edge or face.", &[req("of", Reference, "The edge or face.")]),
    op(
        "point.on_curve",
        "A datum point at a fraction along a curve.",
        &[req("curve", Reference, "The curve or edge."), req("parameter", Number, "Where along it, from 0 to 1.")],
    ),
    op(
        "point.intersection",
        "A datum point where an axis meets a plane.",
        &[req("axis", Axis, "The axis."), req("plane", Plane, "The plane.")],
    ),
    named("frame", "A datum coordinate frame."),
    op(
        "curve.helix",
        "A helix about an axis.",
        &[
            req("axis", Axis, "The axis."),
            req("start", Point, "Where the helix starts."),
            req("pitch", Length, "The rise per turn."),
            req("turns", Number, "The number of turns."),
        ],
    ),
    named("curve.spiral", "A flat spiral."),
    named("curve.projected", "A sketch curve projected onto faces."),
    named("curve.composite", "One curve joined from edges."),
    named("curve.split_line", "A curve that splits faces."),
    named("curve.intersection", "The curve where a face meets a plane."),
    named("curve.through_points", "A curve through points in order."),
    named("curve.equation", "A curve from parametric equations."),
    op(
        "mate_connector",
        "A frame an assembly mates to.",
        &[
            req("placement", Record, "Where the frame sits: on a frame, a face or a point."),
            opt("offset", Length, "A shift from the placement: `[dx, dy, dz]`."),
            opt("rotation", Angle, "A turn about the frame's z axis."),
            opt("flip", Flag, "Point the frame's z axis the other way.").default("false"),
        ],
    ),
    named("surface.extrude", "A surface extruded from a sketch curve."),
    named("surface.revolve", "A surface revolved from a sketch curve."),
    named("surface.sweep", "A surface swept along a path."),
    named("surface.loft", "A surface through curves in order."),
    named("surface.boundary", "A surface bounded by curves."),
    named("surface.fill", "A surface filling a closed boundary of edges."),
    named("surface.planar", "A flat surface inside a closed sketch profile."),
    named("surface.ruled", "A ruled surface between curves."),
    named("surface.offset", "A surface offset from faces."),
    named("surface.extend", "A surface extended past its edges."),
    named("surface.trim", "A surface trimmed by another."),
    named("surface.untrim", "Removes a surface's trims."),
    named("surface.knit", "Joins surfaces into one body, a solid when they close."),
    named("surface.thicken", "Thickens a surface into a solid."),
    named("surface.delete_hole", "Fills holes bounded by edges."),
    named("surface.cut_with_surface", "Cuts a body with a surface."),
    named("surface.mid_surface", "The surface midway between a thin body's faces."),
    named("sheet_metal.base_flange", "The first sheet-metal plate; it carries the thickness, bend radius and K-factor."),
    named("sheet_metal.edge_flange", "A flange bent up from an edge."),
    named("sheet_metal.miter_flange", "Flanges along connected edges, mitered at the corners."),
    named("sheet_metal.hem", "A folded-back edge."),
    named("sheet_metal.jog", "Two bends that offset a plate."),
    named("sheet_metal.sketched_bend", "A bend along a sketch line."),
    named("sheet_metal.lofted_bend", "A plate formed between two profiles."),
    named("sheet_metal.convert_to_sheet_metal", "Turns a thin solid into sheet metal."),
    named("sheet_metal.closed_corner", "Closes the gap at a corner."),
    named("sheet_metal.break_corner", "Chamfers or rounds a plate corner."),
    named("sheet_metal.corner_relief", "Relief cut where bends meet."),
    named("sheet_metal.tab_and_slot", "Tabs on one plate and slots in another."),
    named("sheet_metal.form_tool", "A formed feature stamped into a plate."),
    named("sheet_metal.unfold", "Unfolds bends."),
    named("sheet_metal.fold", "Folds unfolded bends again."),
    named("sheet_metal.flat_pattern", "The flat blank, unfolded from a fixed face."),
    named("weldment.structural_member", "Profiles along the segments of a layout sketch."),
    named("weldment.trim_extend", "Trims or extends members to meet."),
    named("weldment.end_cap", "A plate closing a member's end."),
    named("weldment.gusset", "A plate bracing a joint."),
    named("weldment.weld_bead", "A weld bead along a joint."),
    named("mold.shrink", "Scales a part for mold shrinkage."),
    named("mold.parting_line", "The parting line along a pull direction."),
    named("mold.shut_off", "Surfaces that close openings through the part."),
    named("mold.parting_surface", "The surface the mold halves meet on."),
    named("mold.core_cavity", "Splits a mold block into core and cavity."),
    named("mold.undercuts", "Faces that cannot release along the pull."),
    named("direct.move_faces", "Moves faces of an imported or finished body."),
    named("direct.replace_faces", "Replaces faces with another face's surface."),
    named("direct.delete_faces", "Deletes faces and heals the body."),
    named("direct.move_bodies", "Moves bodies."),
];

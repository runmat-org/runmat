//! Geometric split hints with inherited acute-corner spacing. Exact materialized
//! clearance remains a separate admission step. See Si, Constrained Delaunay
//! Mesh Generation (2008), section 5.3.1, rules 1–3:
//! https://webdoc.sub.gwdg.de/ebook/dissts/Berlin_TU/Si2008.pdf
use super::*;

pub(super) struct SplitHints {
    pub(super) parameters: [f64; 3],
    pub(super) radial_origin: Option<[f64; 3]>,
}

pub(super) fn preferred(
    endpoints: [[f64; 3]; 2],
    witness: [f64; 3],
    context: SegmentContext,
    parameters: [f64; 2],
    acute: [bool; 2],
) -> SplitHints {
    let [left, right] = endpoints;
    let [lower, upper] = parameters;
    let length = distance(context.first_coordinates, context.last_coordinates);
    let radial = if acute[0] {
        distance(context.first_coordinates, witness) / length
    } else if acute[1] {
        1.0 - distance(context.last_coordinates, witness) / length
    } else if 2.0 * distance(left, witness) < distance(left, right) {
        lower + distance(left, witness) / length
    } else if 2.0 * distance(right, witness) < distance(left, right) {
        upper - distance(right, witness) / length
    } else {
        lower + 0.5 * (upper - lower)
    };
    let point = interpolate(context.first_coordinates, context.last_coordinates, radial);
    let spaced = if acute[0] || acute[1] {
        let (near, far, near_parameter, sign) = if acute[0] {
            (left, right, lower, 1.0)
        } else {
            (right, left, upper, -1.0)
        };
        let witness_gap = distance(point, witness);
        if distance(point, far) < witness_gap {
            // Rule 3 splits the near-to-rejected-point interval, preserving
            // the original acute reference. A midpoint of the whole fragment
            // is a different policy and does not retain this spacing rule.
            let span = distance(near, point);
            let advance = if witness_gap < 0.5 * span {
                span - witness_gap
            } else {
                0.5 * span
            };
            near_parameter + sign * advance / length
        } else {
            radial
        }
    } else {
        radial
    };
    let direction = subtract(context.last_coordinates, context.first_coordinates);
    let projection =
        dot(subtract(witness, context.first_coordinates), direction) / dot(direction, direction);
    SplitHints {
        parameters: [spaced, radial, projection],
        // Rule 2's protection radius must not shrink through parameter rounding.
        // Rule 3 intentionally chooses a smaller radius after rejecting rule 2.
        radial_origin: if spaced == radial && acute[0] {
            Some(context.first_coordinates)
        } else if spaced == radial && acute[1] {
            Some(context.last_coordinates)
        } else {
            None
        },
    }
}

#[cfg(test)]
mod tests;

//! Independent edge-trace discriminator. A Tet10 displacement restricted to a
//! straight affine element edge is quadratic. A yield front crossing that edge
//! changes the continuum displacement curvature, irrespective of integration.
use super::*;

#[test]
fn crossing_front_has_a_nonquadratic_trace_despite_polynomial_terminal_state() {
    for cells in [1, 2, 4] {
        for factor in [0.25, 0.375] {
            let r = front(factor);
            let left = (r * cells as f64).floor() / cells as f64;
            let right = left + 1.0 / cells as f64;
            assert!(left < r && r < right);
            // Exact quadratic interpolation through edge end/midpoint values.
            // This is a trace test, NOT the best-FE-approximation error bound.
            let mid = (left + right) / 2.0;
            let nodal = [left, mid, right].map(|x| reference(x, factor).displacement);
            let mut largest: f64 = 0.0;
            for i in 1..32 {
                let s = i as f64 / 32.0;
                let x = left + (right - left) * s;
                let interpolated = nodal[0] * (1.0 - s) * (1.0 - 2.0 * s)
                    + nodal[1] * 4.0 * s * (1.0 - s)
                    + nodal[2] * s * (2.0 * s - 1.0);
                largest = largest.max((interpolated - reference(x, factor).displacement).abs());
            }
            // The analytic second derivatives differ by this nonzero amount on
            // open intervals on either side of the front. No single quadratic,
            // not merely this nodal interpolant, has both curvatures.
            let curvature_jump =
                factor * tangent() * B * (1.0 / tangent() - 1.0 / elastic_tangent());
            assert!(curvature_jump > 0.0);
            assert!(
                largest > 1e-8,
                "cells={cells} factor={factor} trace_error={largest}"
            );
            eprintln!("independent trace cells={cells} factor={factor} front={r} nodal_error_m={largest} curvature_jump={curvature_jump}");
        }
    }
    // At terminal full yield the branch boundary disappears; representing this
    // final polynomial does not imply exact representation of its entire path.
    let nodal = [0.0, 0.5, 1.0].map(|x| reference(x, 1.0).displacement);
    for i in 0..=32 {
        let s = i as f64 / 32.0;
        let interpolated = nodal[0] * (1.0 - s) * (1.0 - 2.0 * s)
            + nodal[1] * 4.0 * s * (1.0 - s)
            + nodal[2] * s * (2.0 * s - 1.0);
        assert!((interpolated - reference(s, 1.0).displacement).abs() < 1e-14);
    }
}

//! A drawing view's lines from the kernel's exact hidden-line projection (OCCT: every model
//! edge split where it passes behind a face, and the outlines of curved faces from HLRBRep).
//!
//! The result keeps the display contract of [`super::view_lines`]: flat `[u, v, …]` polylines
//! in view millimetres, visible and hidden, with centerlines from the display. A segment that
//! two edges draw on the same projected line (a through cut's top and bottom edges) draws once,
//! and a hidden run under a visible line is left to the visible line.

use super::{Axes, ViewLines};
use runmat_cad::model::datum::Frame;
use runmat_cad::regen::kernel::{BodyHandle, ExactQueries, Placed, ProjectionRequest, Transform};
use serde_json::Value;
use std::collections::BTreeSet;

/// Kernel lengths are metres; views are millimetres.
const MM_PER_M: f64 = 1000.0;

/// The view's lines of `bodies` (each at its world transform, kernel metres) projected under
/// `axes`, or `None` when the kernel has no exact projection (the reference test double).
pub fn view_lines(
    queries: &dyn ExactQueries,
    bodies: &[(&BodyHandle, Transform)],
    display: &Value,
    axes: &Axes,
) -> Option<ViewLines> {
    let placed: Vec<Placed<'_>> = bodies
        .iter()
        .map(|(body, transform)| Placed {
            body,
            transform: *transform,
        })
        .collect();
    let request = ProjectionRequest {
        frame: Frame {
            origin: [0.0; 3],
            x: axes.u,
            y: axes.v,
            // Towards the viewer: u × v, so the frame is right-handed in every orientation.
            z: super::cross(axes.u, axes.v),
        },
        hidden: true,
        smooth: false,
        // 0.01 mm chord deviation: well under a 0.18 mm line at any sheet scale up to 10:1.
        tolerance: 1.0e-5,
    };
    let curves = queries.project(&placed, &request).ok()?;
    let mut out = ViewLines::default();
    let mut seen: BTreeSet<(bool, Vec<i64>)> = BTreeSet::new();
    let mut ordered: Vec<_> = curves.into_iter().collect();
    // Visible runs first, so a hidden copy of a visible line is recognised.
    ordered.sort_by_key(|c| !c.visible);
    for curve in ordered {
        let pts: Vec<f64> = curve
            .polyline
            .iter()
            .flat_map(|p| [p[0] * MM_PER_M, p[1] * MM_PER_M])
            .collect();
        if pts.len() < 4 {
            continue;
        }
        let key = line_key(&pts);
        if seen.contains(&(true, key.clone())) || seen.contains(&(curve.visible, key.clone())) {
            continue;
        }
        seen.insert((curve.visible, key));
        if curve.visible {
            out.visible.push(pts);
        } else {
            out.hidden.push(pts);
        }
    }
    super::merge_overlaps(&mut out);
    out.centers = super::centers(&super::model_of(display), axes);
    Some(out)
}

/// A polyline's points rounded to 1 µm, in a direction-independent order.
fn line_key(pts: &[f64]) -> Vec<i64> {
    let r: Vec<i64> = pts.iter().map(|x| (x * 1.0e3).round() as i64).collect();
    let pairs: Vec<[i64; 2]> = r.chunks_exact(2).map(|c| [c[0], c[1]]).collect();
    let mut reversed = pairs.clone();
    reversed.reverse();
    let pick = if reversed < pairs { reversed } else { pairs };
    pick.into_iter().flatten().collect()
}

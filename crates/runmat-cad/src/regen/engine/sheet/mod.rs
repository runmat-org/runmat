//! Sheet metal features beyond the base flange's own lowering: the miter flange, and the
//! flat pattern and unfold of a plate with sketched bends and jogs.
//!
//! # Miter flange
//!
//! The profile is an open chain of lines drawn on a plane normal to one of the chain's edges,
//! starting on that edge. Its coordinates are read in the edge's section plane: `x` out of the
//! sheet across the edge (the thickness face's outward normal) and `y` along the outward normal
//! of the sheet face the edges lie on. The sheet itself is the stretch `x ≤ 0`, `−t ≤ y ≤ 0`.
//!
//! The section is the sheet's own mold line (`x ≤ 0`, `y = 0`) followed by the profile, with
//! the material below the mold line (on the sheet's side), so a bend of the sheet's radius
//! joins the sheet to the profile's first segment and a bend joins each later pair. The
//! section then moves out by the first bend's setback so that bend starts at the sheet edge
//! (nothing of the sheet is trimmed, as an edge flange placed `bend_outside`).
//!
//! The same section runs along every edge of the chain. At a corner shared by two edges each
//! side is clipped by the plane through the corner vertex that bisects the two flanges (it
//! holds the sheet normal and the sum of the two edges' outward directions); a positive `gap`
//! moves each clip plane back by `gap / 2`. Open chain ends stay square.
//!
//! Faces are named by the edge they grow from (the edge between its sheet face and thickness
//! face) and an ordinal. The flat of profile curve `j` takes `16 j` plus 0 for the face on the
//! profile's side (the inner face of a flange turned up), 1 for the other face, 1 for its free
//! end, 2 and 3 for its square ends and 4 and 5 for the ends that meet bends. The bend before
//! curve `j` takes `16 j + 8` plus the bend's own numbers (0 inner and 1 outer face, 2 to 5
//! its ends). A mitre cut adds 6 (at the chain's earlier corner) or 7 (at the later one).

mod miter;
mod plate_bends;
mod strips;

use super::families::{bend_tool, frame_of, labeled, op, plane_normal, polygon};
use super::*;
use crate::model::sheet_metal::{
    section, FlatBendLine, FlatLayout, MaterialSide, Section, SheetMetalFeature, StripRegion,
};
use crate::refs::history::{FaceGeom, KernelTag};
use crate::refs::name::Role;
use crate::refs::naming::Restore;
use crate::regen::records::{FeatureRecord, SheetRules};

/// One edge of a miter flange's chain, oriented along the chain.
struct ChainEdge {
    index: u32,
    start: [f64; 3],
    end: [f64; 3],
    /// Outward normal of the thickness face along the edge (the section's x).
    out: [f64; 3],
}

/// A direction under a rigid motion's rotation.
fn rotate_by(t: &Transform, v: [f64; 3]) -> [f64; 3] {
    let r = &t.rotation;
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

/// A half-plane `(p − origin)·normal ≥ 0` of a plate's frame: `(origin, normal)`.
type HalfPlane = ([f64; 2], [f64; 2]);

/// Whether the half-planes `(p − o)·n ≥ 0` of `a` and `b` share an area inside the polygon
/// `outline` (its vertices; arcs count by their ends).
fn half_planes_overlap_in(outline: &[[f64; 2]], a: HalfPlane, b: HalfPlane) -> bool {
    let tol = 1e-9;
    let side =
        |h: ([f64; 2], [f64; 2]), p: [f64; 2]| (p[0] - h.0[0]) * h.1[0] + (p[1] - h.0[1]) * h.1[1];
    // The region's corners: outline vertices, where each line crosses the outline, and where
    // the two lines cross.
    let mut pts: Vec<[f64; 2]> = outline.to_vec();
    let n = outline.len();
    for h in [a, b] {
        for k in 0..n {
            let (p, q) = (outline[k], outline[(k + 1) % n]);
            let (sp, sq) = (side(h, p), side(h, q));
            if (sp < 0.0) != (sq < 0.0) && (sp - sq).abs() > 0.0 {
                let f = sp / (sp - sq);
                pts.push([p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])]);
            }
        }
    }
    let den = a.1[0] * b.1[1] - a.1[1] * b.1[0];
    if den.abs() > 1e-12 {
        // Lines n·p = n·o: solve the 2 × 2 system.
        let (ca, cb) = (
            a.1[0] * a.0[0] + a.1[1] * a.0[1],
            b.1[0] * b.0[0] + b.1[1] * b.0[1],
        );
        let x = [
            (ca * b.1[1] - cb * a.1[1]) / den,
            (a.1[0] * cb - b.1[0] * ca) / den,
        ];
        if inside(outline, x) {
            pts.push(x);
        }
    }
    let both: Vec<[f64; 2]> = pts
        .into_iter()
        .filter(|p| side(a, *p) >= -tol && side(b, *p) >= -tol)
        .collect();
    // A shared area has three corners that are not in a line.
    let scale = outline
        .iter()
        .map(|p| p[0].abs().max(p[1].abs()))
        .fold(1e-3, f64::max);
    let min_area = 1e-6 * scale * scale;
    for i in 0..both.len() {
        for j in i + 1..both.len() {
            for k in j + 1..both.len() {
                let (p, q, r) = (both[i], both[j], both[k]);
                let area =
                    0.5 * ((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])).abs();
                if area > min_area {
                    return true;
                }
            }
        }
    }
    false
}

/// Point in polygon (even-odd).
fn inside(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
    let mut c = false;
    let n = poly.len();
    for k in 0..n {
        let (a, b) = (poly[k], poly[(k + 1) % n]);
        if (a[1] > p[1]) != (b[1] > p[1])
            && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        {
            c = !c;
        }
    }
    c
}

fn near(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
    let d = sub(a, b);
    dot(d, d).sqrt() <= tol
}

/// Orders edges into one chain, end to end: an open chain starts at its free end; a closed
/// one at the first edge given. `None` when they do not form a single chain.
fn order_chain(mut loose: Vec<ChainEdge>, tol: f64) -> Option<Vec<ChainEdge>> {
    if loose.is_empty() {
        return None;
    }
    let touches = |p: [f64; 3], e: &ChainEdge| near(p, e.start, tol) || near(p, e.end, tol);
    // A free end: an endpoint no other edge shares.
    let mut first = 0;
    let mut flip = false;
    'find: for (k, e) in loose.iter().enumerate() {
        for (end, p) in [(false, e.start), (true, e.end)] {
            if !loose
                .iter()
                .enumerate()
                .any(|(j, o)| j != k && touches(p, o))
            {
                first = k;
                flip = end;
                break 'find;
            }
        }
    }
    let mut e = loose.remove(first);
    if flip {
        std::mem::swap(&mut e.start, &mut e.end);
    }
    let mut chain = vec![e];
    while !loose.is_empty() {
        let tip = chain.last().expect("non-empty").end;
        let found: Vec<usize> = (0..loose.len())
            .filter(|j| touches(tip, &loose[*j]))
            .collect();
        if found.len() != 1 {
            return None;
        }
        let mut e = loose.remove(found[0]);
        if !near(e.start, tip, tol) {
            std::mem::swap(&mut e.start, &mut e.end);
        }
        chain.push(e);
    }
    Some(chain)
}

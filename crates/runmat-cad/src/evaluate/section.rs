//! Section properties of a plane region: area, centroid, second moments of area about the
//! centroid in the region's plane axes, the polar moment, and the principal moments with the
//! angle of the first principal axis. A sketch's regions integrate here from their boundary
//! polygons; a planar face's come from the kernel and reduce here to the same form.

use crate::display::regions::{region_polygons, Sampling};
use crate::model::sketch::{Sketch, SolvedSketch};
use runmat_geometry_kernel::analysis::FaceArea;
use runmat_geometry_kernel::frame::Frame;
use serde::Serialize;

/// Sketch boundaries for integration: 4096 points round a circle, 1024 along an arc or curve,
/// so a circle's area is within 2·10⁻⁶ of π r².
pub const SECTION_SAMPLING: Sampling = Sampling {
    closed: 4096,
    open: 1024,
};

/// A plane region's properties, in metres. `ixx`, `iyy`, `ixy` are about the centroid along the
/// plane's `x` and `y` axes (`ixx` = ∫ y² dA).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PlaneSection {
    pub area: f64,
    /// World coordinates.
    pub centroid: [f64; 3],
    /// The plane's axes the moments are taken along (world unit vectors), and its normal.
    pub x: [f64; 3],
    pub y: [f64; 3],
    pub normal: [f64; 3],
    pub ixx: f64,
    pub iyy: f64,
    pub ixy: f64,
    /// `ixx + iyy`, about the normal through the centroid.
    pub polar: f64,
    /// Principal moments, larger first.
    pub i1: f64,
    pub i2: f64,
    /// Angle from the plane's `x` axis to the axis of `i1`, radians, in (−π/2, π/2].
    pub angle: f64,
}

/// Area moments of a counter-clockwise polygon (clockwise holes subtract) about the origin:
/// (A, ∫x, ∫y, ∫x², ∫y², ∫xy).
fn polygon_moments(p: &[[f64; 2]]) -> [f64; 6] {
    let mut m = [0.0; 6];
    for i in 0..p.len() {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        let c = a[0] * b[1] - b[0] * a[1];
        m[0] += c / 2.0;
        m[1] += (a[0] + b[0]) * c / 6.0;
        m[2] += (a[1] + b[1]) * c / 6.0;
        m[3] += (a[0] * a[0] + a[0] * b[0] + b[0] * b[0]) * c / 12.0;
        m[4] += (a[1] * a[1] + a[1] * b[1] + b[1] * b[1]) * c / 12.0;
        m[5] += (a[0] * b[1] + 2.0 * a[0] * a[1] + 2.0 * b[0] * b[1] + b[0] * a[1]) * c / 24.0;
    }
    m
}

/// The section from area, centroid (plane coordinates) and central moments.
pub fn plane_section(
    frame: &Frame,
    area: f64,
    c: [f64; 2],
    ixx: f64,
    iyy: f64,
    ixy: f64,
) -> PlaneSection {
    let mean = (ixx + iyy) / 2.0;
    let r = (((ixx - iyy) / 2.0).powi(2) + ixy * ixy).sqrt();
    // The axis of the larger moment about it: ∫ (distance to the axis)² is largest.
    let angle = if r < 1e-30 * mean.abs().max(1e-300) {
        0.0
    } else {
        let a = 0.5 * (-2.0 * ixy).atan2(ixx - iyy);
        // Keep it in (−π/2, π/2].
        if a <= -std::f64::consts::FRAC_PI_2 {
            a + std::f64::consts::PI
        } else {
            a
        }
    };
    PlaneSection {
        area,
        centroid: frame.to_world(c),
        x: frame.x,
        y: frame.y,
        normal: frame.z,
        ixx,
        iyy,
        ixy,
        polar: ixx + iyy,
        i1: mean + r,
        i2: mean - r,
        angle,
    }
}

/// A planar region from the kernel's [`FaceArea`], with its moments along axes in its plane:
/// `x` is world X projected into the plane (world Y when the plane is square to X), `y` is
/// `normal × x`.
pub fn face_section(face: &FaceArea) -> PlaneSection {
    let n = face.normal;
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let seed = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let d = dot(seed, n);
    let x = [seed[0] - d * n[0], seed[1] - d * n[1], seed[2] - d * n[2]];
    let l = dot(x, x).sqrt();
    let x = [x[0] / l, x[1] / l, x[2] / l];
    let y = [
        n[1] * x[2] - n[2] * x[1],
        n[2] * x[0] - n[0] * x[2],
        n[0] * x[1] - n[1] * x[0],
    ];
    // For a plane region about its centroid, ∫ r rᵀ = (tr M / 2) E − M.
    let m = face.inertia;
    let half_tr = (m[0][0] + m[1][1] + m[2][2]) / 2.0;
    let rr = |a: [f64; 3], b: [f64; 3]| {
        let mut s = half_tr * dot(a, b);
        for (ai, row) in a.iter().zip(&m) {
            for (mij, bj) in row.iter().zip(&b) {
                s -= ai * mij * bj;
            }
        }
        s
    };
    let frame = Frame {
        origin: face.centroid,
        x,
        y,
        z: n,
    };
    let mut s = plane_section(&frame, face.area, [0.0, 0.0], rr(y, y), rr(x, x), rr(x, y));
    s.centroid = face.centroid;
    s
}

/// Every closed region of a solved sketch, in its plane `frame`, as one section (the regions
/// together, holes subtracted), or `None` when the sketch has no closed region.
pub fn sketch_section(
    sketch: &Sketch,
    solved: &SolvedSketch,
    frame: &Frame,
) -> Option<PlaneSection> {
    regions_section(sketch, solved, frame, |_| true)
}

/// The region of a sketch bounded by `loops` (each loop's curve ids, outer first), as a region
/// pick names it.
pub fn region_section(
    sketch: &Sketch,
    solved: &SolvedSketch,
    frame: &Frame,
    loops: &[Vec<String>],
) -> Option<PlaneSection> {
    let key = |l: &[Vec<String>]| {
        let mut k: Vec<Vec<String>> = l
            .iter()
            .map(|c| {
                let mut c = c.clone();
                c.sort();
                c
            })
            .collect();
        k.sort();
        k
    };
    let want = key(loops);
    regions_section(sketch, solved, frame, |l| key(l) == want)
}

fn regions_section(
    sketch: &Sketch,
    solved: &SolvedSketch,
    frame: &Frame,
    keep: impl Fn(&[Vec<String>]) -> bool,
) -> Option<PlaneSection> {
    let mut m = [0.0; 6];
    for region in region_polygons(sketch, solved, SECTION_SAMPLING) {
        if !keep(&region.loops) {
            continue;
        }
        for p in &region.polygons {
            for (k, v) in polygon_moments(p).iter().enumerate() {
                m[k] += v;
            }
        }
    }
    let area = m[0];
    if area <= 0.0 {
        return None;
    }
    let c = [m[1] / area, m[2] / area];
    Some(plane_section(
        frame,
        area,
        c,
        m[4] - area * c[1] * c[1],
        m[3] - area * c[0] * c[0],
        m[5] - area * c[0] * c[1],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rectangle_has_bh3_over_12_about_its_centroid() {
        // 60 × 20 with its corner at (10, 5).
        let p = [[10.0, 5.0], [70.0, 5.0], [70.0, 25.0], [10.0, 25.0]];
        let m = polygon_moments(&p);
        let a = m[0];
        let c = [m[1] / a, m[2] / a];
        assert!((a - 1200.0).abs() < 1e-9);
        assert!((c[0] - 40.0).abs() < 1e-12 && (c[1] - 15.0).abs() < 1e-12);
        let s = plane_section(
            &Frame::WORLD,
            a,
            c,
            m[4] - a * c[1] * c[1],
            m[3] - a * c[0] * c[0],
            m[5] - a * c[0] * c[1],
        );
        assert!((s.ixx - 60.0 * 20f64.powi(3) / 12.0).abs() < 1e-6, "{s:?}");
        assert!((s.iyy - 20.0 * 60f64.powi(3) / 12.0).abs() < 1e-6, "{s:?}");
        assert!(s.ixy.abs() < 1e-6);
        // The larger principal axis is the long side's (x) direction's perpendicular: about y.
        assert!((s.i1 - s.iyy).abs() < 1e-6 && (s.i2 - s.ixx).abs() < 1e-6);
        assert!(
            (s.angle - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
            "{s:?}"
        );
        assert!((s.polar - s.ixx - s.iyy).abs() < 1e-9);
    }
}

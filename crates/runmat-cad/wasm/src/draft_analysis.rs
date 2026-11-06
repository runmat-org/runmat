//! Draft analysis (CAD-36): each face's draft angle against a pull direction, sorted into bands
//! a mould maker reads: positive (releases toward the pull), negative (releases from the other
//! half), and faces that need draft (steeper than the angle asked for either way, or both
//! signs on one face).
//!
//! The draft of a facet is 90° less its normal's angle to the pull: a face square to the pull
//! has 90°, a wall parallel to it 0°. A face's range comes from its tessellation, so a curved
//! face reports the least and greatest draft over it.

use runmat_cad::display::{FaceDisplay, PartDisplay};
use serde::Serialize;

/// A face's band.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DraftBand {
    /// Every facet at or above the angle toward the pull.
    Positive,
    /// Every facet at or below minus the angle: it releases from the other half.
    Negative,
    /// Some facet between minus the angle and the angle: the face needs draft.
    NeedsDraft,
    /// Facets on both sides beyond the angle: the face straddles the parting line.
    Straddle,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FaceDraft {
    /// The face's canonical name, or `face <index>` when it has none.
    pub face: String,
    /// Degrees.
    pub min: f64,
    pub max: f64,
    pub band: DraftBand,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(a: [f64; 3]) -> Option<[f64; 3]> {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    (l > 1e-15).then(|| [a[0] / l, a[1] / l, a[2] / l])
}

/// Each facet's normal, times twice its area.
fn facets(face: &FaceDisplay) -> Vec<[f64; 3]> {
    let p = |i: u32| {
        let k = i as usize * 3;
        [f64::from(face.positions[k]), f64::from(face.positions[k + 1]), f64::from(face.positions[k + 2])]
    };
    face.triangles
        .chunks_exact(3)
        .filter(|t| t.iter().all(|&i| (i as usize) * 3 + 2 < face.positions.len()))
        .map(|t| cross(sub(p(t[1]), p(t[0])), sub(p(t[2]), p(t[0]))))
        .collect()
}

/// The area-weighted normal of the face named `reference`.
pub fn face_normal(display: &PartDisplay, reference: &str) -> Option<[f64; 3]> {
    let face = display.bodies.iter().flat_map(|b| &b.faces).find(|f| f.reference.as_deref() == Some(reference))?;
    let sum = facets(face).into_iter().fold([0.0; 3], |s, n| [s[0] + n[0], s[1] + n[1], s[2] + n[2]]);
    unit(sum)
}

/// The pull direction a pick names: an origin plane's normal, an origin axis, or a planar
/// face's normal (away from the material).
pub fn pull_direction(display: &PartDisplay, pick: &str) -> Result<[f64; 3], String> {
    let named = match pick {
        "origin:top" | "origin.xy" | "origin:z" | "origin.z" => Some([0.0, 0.0, 1.0]),
        "origin:front" | "origin.zx" | "origin:y" | "origin.y" => Some([0.0, 1.0, 0.0]),
        "origin:right" | "origin.yz" | "origin:x" | "origin.x" => Some([1.0, 0.0, 0.0]),
        _ => None,
    };
    named
        .or_else(|| face_normal(display, pick))
        .ok_or_else(|| format!("`{pick}` isn't an origin plane, an origin axis or a face of the part"))
}

/// Every face's draft against `pull` (unit), banded at `angle` degrees.
pub fn draft_faces(display: &PartDisplay, pull: [f64; 3], angle: f64) -> Vec<FaceDraft> {
    let mut out = Vec::new();
    for face in display.bodies.iter().flat_map(|b| &b.faces) {
        let drafts: Vec<f64> = facets(face)
            .into_iter()
            .filter_map(unit)
            .map(|n| (n[0] * pull[0] + n[1] * pull[1] + n[2] * pull[2]).clamp(-1.0, 1.0).asin().to_degrees())
            .collect();
        if drafts.is_empty() {
            continue;
        }
        let min = drafts.iter().copied().fold(f64::INFINITY, f64::min);
        let max = drafts.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // A hair under the angle still passes: tessellated normals of a drafted face scatter.
        let eps = 1e-6;
        let band = if min >= angle - eps {
            DraftBand::Positive
        } else if max <= -angle + eps {
            DraftBand::Negative
        } else if drafts.iter().any(|d| d.abs() < angle - eps) {
            DraftBand::NeedsDraft
        } else {
            DraftBand::Straddle
        };
        out.push(FaceDraft {
            face: face.reference.clone().unwrap_or_else(|| format!("face {}", face.index)),
            min,
            max,
            band,
        });
    }
    out
}

//! Evaluate-tab hooks: typed analysis requests and results behind a kernel seam, and the cost
//! estimate seam.
//!
//! Mass and section properties, thickness, curvature, zebra and deviation need the kernel's
//! geometry, so they go through [`GeometryAnalyzer`] (K1 implements it beside `CadKernel`; the
//! fake kernel implements mass properties exactly on its grid). Draft and undercut need only
//! face normals, so [`draft_analysis`] runs on any kernel's topology summary. Results name
//! faces by their structured names, so they survive regeneration like references do.

use crate::refs::history::{SurfaceKind, Topology};
use crate::regen::kernel::{BodyHandle, KernelError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "analysis")]
pub enum AnalysisRequest {
    /// Volume, mass, centroid and inertia about the centroid (density in kg/m³).
    MassProperties { density: f64 },
    /// Area properties of the section cut by a plane (origin, normal).
    SectionProperties { origin: [f64; 3], normal: [f64; 3] },
    /// Wall thickness below `min` or above `max` (m), reported per face.
    Thickness { min: f64, max: Option<f64> },
    /// Draft against a pull direction (also done kernel-free by [`draft_analysis`]).
    Draft { pull: [f64; 3], min: f64 },
    /// Curvature range per face.
    Curvature,
    /// Zebra stripes: continuity across edges (G0, G1, G2) per edge.
    Zebra,
    /// Deviation between operand 0 and operand 1 (max and RMS distance).
    Deviation,
    /// Faces that cannot release along the pull.
    Undercut { pull: [f64; 3] },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MassProperties {
    pub volume: f64,
    pub mass: f64,
    pub centroid: [f64; 3],
    /// Inertia tensor about the centroid (kg·m²), rows.
    pub inertia: [[f64; 3]; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SectionProperties {
    pub area: f64,
    pub centroid: [f64; 3],
    /// Second moments about the centroid in the section plane's frame.
    pub ixx: f64,
    pub iyy: f64,
    pub ixy: f64,
}

/// A per-face measurement, by face index of the analysed body.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaceValue {
    pub face: u32,
    pub min: f64,
    pub max: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeContinuity {
    G0,
    G1,
    G2,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "result")]
pub enum AnalysisResult {
    MassProperties(MassProperties),
    SectionProperties(SectionProperties),
    Thickness { faces: Vec<FaceValue> },
    Draft(DraftReport),
    Curvature { faces: Vec<FaceValue> },
    Zebra { edges: Vec<(u32, EdgeContinuity)> },
    Deviation { max: f64, rms: f64 },
    Undercut { faces: Vec<u32> },
}

/// The kernel seam for geometry analyses. A kernel that cannot run one returns
/// `KernelError::NotSupported`.
pub trait GeometryAnalyzer {
    fn analyze(
        &self,
        request: &AnalysisRequest,
        bodies: &[&BodyHandle],
    ) -> Result<AnalysisResult, KernelError>;
}

/// Draft per face against a pull direction, from topology alone.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DraftReport {
    /// `(face, draft angle)`; positive releases along the pull.
    pub faces: Vec<(u32, f64)>,
    /// Faces whose draft magnitude is below the minimum (and are not perpendicular to the
    /// pull).
    pub insufficient: Vec<u32>,
    /// Faces whose draft is not known from the summary (free-form surfaces).
    pub unknown: Vec<u32>,
}

pub fn draft_analysis(topology: &Topology, pull: [f64; 3], min: f64) -> DraftReport {
    let n = (pull[0] * pull[0] + pull[1] * pull[1] + pull[2] * pull[2]).sqrt();
    let p = [pull[0] / n, pull[1] / n, pull[2] / n];
    let mut out = DraftReport::default();
    for (i, f) in topology.faces.iter().enumerate() {
        let draft = match f.geom.surface {
            SurfaceKind::Plane { normal } => {
                let l = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
                ((normal[0] * p[0] + normal[1] * p[1] + normal[2] * p[2]) / l)
                    .clamp(-1.0, 1.0)
                    .asin()
            }
            SurfaceKind::Cylinder {
                axis_direction: a, ..
            } => {
                let l = (a[0].powi(2) + a[1].powi(2) + a[2].powi(2)).sqrt();
                let c = ((a[0] * p[0] + a[1] * p[1] + a[2] * p[2]) / l).abs();
                if c > 1.0 - 1e-9 {
                    0.0
                } else {
                    out.unknown.push(i as u32);
                    continue;
                }
            }
            SurfaceKind::Other => {
                out.unknown.push(i as u32);
                continue;
            }
        };
        if draft.abs() < min && draft.abs() < std::f64::consts::FRAC_PI_2 - 1e-9 {
            out.insufficient.push(i as u32);
        }
        out.faces.push((i as u32, draft));
    }
    out
}

/// What a cost estimate is asked for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostInput {
    pub process: String,
    pub material: Option<String>,
    pub quantity: u32,
    pub mass: Option<f64>,
    pub volume: Option<f64>,
    /// Process facts (`bends`, `holes`, `setups`, `cut_length`), by name.
    #[serde(default)]
    pub facts: std::collections::BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CostEstimate {
    pub currency: String,
    pub unit_cost: f64,
    pub setup_cost: f64,
    /// Lines that make up the estimate (material, machine time, finishing).
    pub lines: Vec<(String, f64)>,
}

/// The cost-estimate seam: a costing service or a shop's rate tables plug in here.
pub trait CostEstimator {
    fn estimate(&self, input: &CostInput) -> Result<CostEstimate, String>;
}

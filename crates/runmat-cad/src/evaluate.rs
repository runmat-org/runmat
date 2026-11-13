//! Evaluate-tab hooks: typed analysis requests and results behind a kernel seam, and the cost
//! estimate seam.
//!
//! Mass and section properties, thickness, curvature, zebra and deviation need the kernel's
//! geometry, so they go through [`GeometryAnalyzer`] (K1 implements it beside `CadKernel`; the
//! fake kernel implements mass properties exactly on its grid). Draft and undercut need only
//! face normals, so [`draft_analysis`] runs on any kernel's topology summary. Results name
//! faces by their structured names, so they survive regeneration like references do.

use serde::{Deserialize, Serialize};

/// The analysis seam and its typed requests and results, defined with the kernel contract.
pub use runmat_geometry_kernel::analysis::*;

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

//! Provider-neutral planning and execution of typed exact-geometry feature graphs.

mod materialize;
mod plan;
mod provider;
mod regions;
mod resolve;
mod types;

pub use materialize::*;
pub use plan::{plan_feature_graph, GeometryFeaturePlanLimits};
pub use provider::{execute_feature_plan, GeometryFeatureProvider};
pub use regions::*;
pub use resolve::{resolve_geometry_parameters, ResolvedGeometryParameters};
pub use types::*;

#[cfg(test)]
mod tests;

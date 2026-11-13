//! Shared local relocation geometry; no canonical mesh construction.
use super::*;
pub(super) mod apex;
pub(super) mod block;
mod centroids;
mod kernel;
mod score;
mod smoothing;
pub(super) use centroids::{neighbor_centroid, volume_weighted_star_centroid};
#[cfg(test)]
pub(super) use kernel::relocated_star_minimum_jacobian;
pub(super) use kernel::{point_is_in_vertex_star_kernel, sliver_height_target};
pub(super) use score::SmoothingObjective;
#[cfg(test)]
pub(super) use smoothing::pattern_search_vertex_star;
pub(super) use smoothing::smooth_vertex_star_points_with_metrics;

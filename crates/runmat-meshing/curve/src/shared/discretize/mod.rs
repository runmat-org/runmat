mod arc_length;
mod degenerate;
mod endpoints;
mod error;
mod generate;
mod math;
mod pcurves;
mod sampling;
mod types;
mod vertex_uv;

pub use generate::discretize_shared_curves;
pub use types::*;

pub(super) use arc_length::world_arc_length;
pub(super) use error::{edge_error, geometry_error, validate_options};
pub(super) use generate::{discretize_edge, discretize_edge_with_parameters};
pub(super) use math::{
    average_metric, metric_length, normalize, point_segment_distance, sub, tangent_angle,
};
pub(super) use pcurves::face_uses_for_parameters;

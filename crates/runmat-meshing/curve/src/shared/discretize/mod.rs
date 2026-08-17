mod arc_length;
mod degenerate;
mod error;
mod generate;
mod math;
mod pcurves;
mod sampling;
mod types;

pub use generate::discretize_shared_curves;
pub use types::*;

pub(super) use arc_length::world_arc_length;
pub(super) use error::{edge_error, geometry_error, validate_options};
pub(super) use math::{
    average_metric, metric_length, normalize, point_segment_distance, sub, tangent_angle,
};

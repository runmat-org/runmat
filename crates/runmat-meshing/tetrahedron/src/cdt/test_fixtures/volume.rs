//! Shared finite volume fixture settings; original physical bounds unchanged.
use crate::cdt::{DelaunayVolumeMeshOptions, DelaunayVolumeQualityOptions};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};
pub(in crate::cdt) fn metric_request() -> MetricFieldRequest {
    MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(2.0).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
        cells_across_gap: None,
    }
}

pub(in crate::cdt) fn volume_options() -> DelaunayVolumeMeshOptions {
    DelaunayVolumeMeshOptions {
        quality: DelaunayVolumeQualityOptions {
            maximum_metric_edge_length: 2.0,
            maximum_radius_edge_ratio: 10.0,
            minimum_metric_scaled_jacobian: 0.01,
            ..DelaunayVolumeQualityOptions::default()
        },
        ..DelaunayVolumeMeshOptions::default()
    }
}

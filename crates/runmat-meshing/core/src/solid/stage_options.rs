use crate::{
    curve::CurveDiscretizationOptions, options::VolumeMeshingOptions,
    source_topology::SourceTopologyModel, surface::SurfaceDiscretizationOptions,
};

use super::{target_size_for_mesh, thin_low_face_topology, topology_min_span};

pub(super) fn curve_options_for_mesh(
    topology: &SourceTopologyModel,
    options: &VolumeMeshingOptions,
) -> CurveDiscretizationOptions {
    let mut target_size_m = target_size_for_mesh(topology, options);
    if thin_low_face_topology(topology) {
        if let Some(min_span_m) = topology_min_span(topology) {
            target_size_m = target_size_m.min(min_span_m.max(1.0e-6));
        }
    }
    CurveDiscretizationOptions {
        target_size_m,
        min_segments_per_edge: 1,
        max_segments_per_edge: options.max_elements.max(1).min(4096),
    }
}

pub(super) fn surface_options_for_mesh(
    topology: &SourceTopologyModel,
) -> SurfaceDiscretizationOptions {
    SurfaceDiscretizationOptions {
        max_curve_segments_per_edge: if thin_low_face_topology(topology) {
            20
        } else {
            8
        },
        ..SurfaceDiscretizationOptions::default()
    }
}

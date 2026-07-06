use runmat_meshing_core::contracts::MeshBackendSummary;
use runmat_meshing_tetrahedron::{generate::TetrahedronMesh, recover::TetrahedronRecoveryQueue};

use super::backend_counts::{
    tetrahedron_material_region_count, tetrahedron_unclassified_material_element_count,
};
use super::backend_generation::plc_input_and_generation_summary;
use super::backend_optimization::optimization_summary;
use super::backend_quality::BackendQualityEvidence;
use super::backend_recovery::recovery_summary;

pub(super) const SOLID_PLC_TETRAHEDRON_ALGORITHM: &str = "plc_tetrahedron/v1";

pub(super) struct BackendSummaryInput<'a> {
    pub(super) surface_element_count: usize,
    pub(super) tetrahedron_mesh: &'a TetrahedronMesh,
    pub(super) recovery_queue: &'a TetrahedronRecoveryQueue,
    pub(super) initial_backend_quality: &'a BackendQualityEvidence,
    pub(super) backend_quality: BackendQualityEvidence,
}

pub(super) fn build_backend_summary(input: BackendSummaryInput<'_>) -> MeshBackendSummary {
    let tetrahedron_mesh = input.tetrahedron_mesh;
    let backend_quality = input.backend_quality;
    let optimization_summary = optimization_summary(
        tetrahedron_mesh,
        input.initial_backend_quality,
        &backend_quality,
    );
    let generation_summary =
        plc_input_and_generation_summary(tetrahedron_mesh, optimization_summary);
    let recovery_summary = recovery_summary(input.recovery_queue, generation_summary);

    MeshBackendSummary {
        backend: "solid".to_string(),
        algorithm: SOLID_PLC_TETRAHEDRON_ALGORITHM.to_string(),
        surface_element_count: input.surface_element_count,
        tetrahedron_element_count: tetrahedron_mesh.elements.len(),
        tetrahedron_material_region_count: tetrahedron_material_region_count(tetrahedron_mesh),
        tetrahedron_unclassified_material_element_count:
            tetrahedron_unclassified_material_element_count(tetrahedron_mesh),
        tetrahedron_min_exact_scaled_jacobian: backend_quality.min_exact_scaled_jacobian,
        tetrahedron_exact_scaled_jacobian_below_threshold_count: backend_quality
            .exact_scaled_jacobian_below_threshold_count,
        tetrahedron_exact_scaled_jacobian_bins: backend_quality.exact_scaled_jacobian_bins,
        ..recovery_summary
    }
}

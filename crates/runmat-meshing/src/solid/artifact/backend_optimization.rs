use runmat_meshing_core::contracts::{
    MeshBackendSummary, TETRAHEDRON_EXACT_QUALITY_REPAIR_PASS_COUNT,
    TETRAHEDRON_EXACT_QUALITY_SEED_STAR_RELOCATION_COUNT,
    TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_INTERIOR_SEED_COUNT,
    TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_TOTAL_COUNT,
    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT,
    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT,
    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX,
    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT,
    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT,
    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX,
    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT,
    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT,
    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX,
    TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ACCEPTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ATTEMPT_COUNT,
    TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_BUDGET_LIMIT_COUNT,
    TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTED_COUNT,
    TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTION_PREFIX,
    TETRAHEDRON_UNTANGLING_FINAL_NEAR_SINGULAR_COUNT,
    TETRAHEDRON_UNTANGLING_INITIAL_NEAR_SINGULAR_COUNT, TETRAHEDRON_UNTANGLING_PASS_COUNT,
    TETRAHEDRON_UNTANGLING_RELOCATED_SEED_COUNT,
};
use runmat_meshing_tetrahedron::generate::TetrahedronMesh;

use super::backend_counts::{tetrahedron_entity_count, tetrahedron_rejection_counts_by_prefix};
use super::backend_quality::{optimization_target_evidence, BackendQualityEvidence};

pub(super) fn optimization_summary(
    tetrahedron_mesh: &TetrahedronMesh,
    initial_backend_quality: &BackendQualityEvidence,
    backend_quality: &BackendQualityEvidence,
) -> MeshBackendSummary {
    let optimization_targets =
        optimization_target_evidence(initial_backend_quality, backend_quality);

    MeshBackendSummary {
        tetrahedron_optimization_pass_count: usize::from(tetrahedron_mesh.quality_optimized),
        tetrahedron_optimization_budget_limited_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT,
        ) + tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT,
        ) + tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT,
        ) + tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_BUDGET_LIMIT_COUNT,
        ),
        tetrahedron_smoothed_point_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
        ) + tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT,
        ),
        tetrahedron_sliver_count: backend_quality.sliver_count,
        tetrahedron_sliver_removed_count: optimization_targets.sliver_removed_count,
        tetrahedron_optimization_sliver_removal_attempt_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ATTEMPT_COUNT,
        ),
        tetrahedron_optimization_sliver_removal_accepted_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_ACCEPTED_COUNT,
        ),
        tetrahedron_optimization_sliver_removal_rejected_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTED_COUNT,
        ),
        tetrahedron_optimization_sliver_removal_budget_limited_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_BUDGET_LIMIT_COUNT,
        ),
        tetrahedron_optimization_sliver_removal_rejected_by_reason:
            tetrahedron_rejection_counts_by_prefix(
                tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_SLIVER_REMOVAL_REJECTION_PREFIX,
            ),
        tetrahedron_optimization_target_seed_count: optimization_targets.target_seed_count,
        tetrahedron_optimization_skipped_target_seed_count: optimization_targets
            .skipped_target_seed_count,
        tetrahedron_optimization_interior_smoothing_attempt_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ATTEMPT_COUNT,
        ),
        tetrahedron_optimization_interior_smoothing_accepted_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_ACCEPTED_COUNT,
        ),
        tetrahedron_optimization_interior_smoothing_rejected_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTED_COUNT,
        ),
        tetrahedron_optimization_interior_smoothing_budget_limited_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_BUDGET_LIMIT_COUNT,
        ),
        tetrahedron_optimization_interior_smoothing_rejected_by_reason:
            tetrahedron_rejection_counts_by_prefix(
                tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_INTERIOR_SMOOTHING_REJECTION_PREFIX,
            ),
        tetrahedron_optimization_boundary_smoothing_attempt_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ATTEMPT_COUNT,
        ),
        tetrahedron_optimization_boundary_smoothing_accepted_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_ACCEPTED_COUNT,
        ),
        tetrahedron_optimization_boundary_smoothing_rejected_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTED_COUNT,
        ),
        tetrahedron_optimization_boundary_smoothing_budget_limited_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_BUDGET_LIMIT_COUNT,
        ),
        tetrahedron_optimization_boundary_smoothing_rejected_by_reason:
            tetrahedron_rejection_counts_by_prefix(
                tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_BOUNDARY_SMOOTHING_REJECTION_PREFIX,
            ),
        tetrahedron_optimization_local_reconnection_attempt_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ATTEMPT_COUNT,
        ),
        tetrahedron_optimization_local_reconnection_accepted_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_ACCEPTED_COUNT,
        ),
        tetrahedron_optimization_local_reconnection_rejected_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTED_COUNT,
        ),
        tetrahedron_optimization_local_reconnection_budget_limited_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_BUDGET_LIMIT_COUNT,
        ),
        tetrahedron_optimization_local_reconnection_rejected_by_reason:
            tetrahedron_rejection_counts_by_prefix(
                tetrahedron_mesh,
                TETRAHEDRON_OPTIMIZATION_LOCAL_RECONNECTION_REJECTION_PREFIX,
            ),
        tetrahedron_optimization_initial_max_aspect_ratio: initial_backend_quality.max_aspect_ratio,
        tetrahedron_optimization_final_max_aspect_ratio: backend_quality.max_aspect_ratio,
        tetrahedron_optimization_initial_min_exact_scaled_jacobian: initial_backend_quality
            .min_exact_scaled_jacobian,
        tetrahedron_optimization_final_min_exact_scaled_jacobian: backend_quality
            .min_exact_scaled_jacobian,
        tetrahedron_untangling_pass_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_UNTANGLING_PASS_COUNT,
        ),
        tetrahedron_untangling_initial_near_singular_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_UNTANGLING_INITIAL_NEAR_SINGULAR_COUNT,
        ),
        tetrahedron_untangling_final_near_singular_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_UNTANGLING_FINAL_NEAR_SINGULAR_COUNT,
        ),
        tetrahedron_untangling_relocated_seed_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_UNTANGLING_RELOCATED_SEED_COUNT,
        ),
        tetrahedron_exact_quality_repair_pass_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_EXACT_QUALITY_REPAIR_PASS_COUNT,
        ),
        tetrahedron_exact_quality_seed_star_relocation_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_EXACT_QUALITY_SEED_STAR_RELOCATION_COUNT,
        ),
        tetrahedron_exact_quality_unrepaired_total_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_TOTAL_COUNT,
        ),
        tetrahedron_exact_quality_unrepaired_interior_seed_count: tetrahedron_entity_count(
            tetrahedron_mesh,
            TETRAHEDRON_EXACT_QUALITY_UNREPAIRED_INTERIOR_SEED_COUNT,
        ),
        ..MeshBackendSummary::default()
    }
}

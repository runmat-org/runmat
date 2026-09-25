//! Family-specific operator-demand validation kept outside the resource estimator.

use super::{
    prepared_output_is_valid, PartitioningPolicy, ResourcePlanningError, SolveResourcePolicy,
};

pub(super) fn validate_buckling_demand(
    demand: &super::super::contract::StructuralBucklingResourceDemand,
) -> Result<(), ResourcePlanningError> {
    let SolveResourcePolicy::SymmetricSubspaceIteration {
        reduced_dimension,
        requested_mode_count,
        ..
    } = demand.operator.solve
    else {
        return Err(ResourcePlanningError::Invalid(
            "structural buckling demand requires a symmetric-subspace solve policy".into(),
        ));
    };
    if demand.requested_mode_count != requested_mode_count
        || demand.requested_mode_count.0 == 0
        || demand.requested_mode_count.0 > reduced_dimension.0
        || demand.phase_value_count_upper_bound.0 < demand.output_value_count_upper_bound.0
        || demand.output_value_count_upper_bound.0 == 0
        || demand.operator.partitioning != PartitioningPolicy::Monolithic
        || !prepared_output_is_valid(&demand.result_output)
        || !prepared_output_is_valid(&demand.receipt_output)
    {
        return Err(ResourcePlanningError::Invalid(
            "structural buckling output, subspace, or placement bounds are inconsistent".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_modal_demand(
    demand: &super::super::contract::StructuralModalResourceDemand,
) -> Result<(), ResourcePlanningError> {
    let SolveResourcePolicy::SymmetricSubspaceIteration {
        reduced_dimension,
        maximum_subspace_dimension,
        ..
    } = demand.operator.solve
    else {
        return Err(ResourcePlanningError::Invalid(
            "structural modal demand requires a modal-subspace solve policy".into(),
        ));
    };
    let expected_residual_iterations = if demand.residual_vector_count.0 == 0 {
        0
    } else {
        let dimension = usize::try_from(reduced_dimension.0)
            .map_err(|_| ResourcePlanningError::Overflow("modal reduced dimension"))?;
        u64::try_from(
            crate::pipeline::homogeneous_static::homogeneous_static_iteration_limit(dimension),
        )
        .map_err(|_| ResourcePlanningError::Overflow("modal static iteration limit"))?
    };
    let expected_residual_orthogonalization_passes = if demand.residual_vector_count.0 == 0 {
        0
    } else {
        crate::pipeline::modal_residual::RESIDUAL_MASS_ORTHOGONALIZATION_PASSES as u64
    };
    if demand.retained_mode_count_upper_bound.0 == 0
        || demand.retained_mode_count_upper_bound.0 > reduced_dimension.0
        || demand.modal_phase_value_count_upper_bound.0
            < demand.modal_output_value_count_upper_bound.0
        || demand.modal_output_value_count_upper_bound.0 == 0
        || maximum_subspace_dimension != reduced_dimension
        || demand.reduced_tangent_nonzero_count_upper_bound.0 == 0
        || demand.residual_vector_count.0 > 3
        || demand.residual_vector_static_iteration_count_upper_bound.0
            != expected_residual_iterations
        || demand.residual_vector_mass_orthogonalization_passes.0
            != expected_residual_orthogonalization_passes
        || (demand.prestress_linearization.is_none()
            && demand.geometric_stiffness_nonzero_count.0 > 0)
        || demand.operator.partitioning != PartitioningPolicy::Monolithic
        || !prepared_output_is_valid(&demand.result_output)
        || !prepared_output_is_valid(&demand.receipt_output)
    {
        return Err(ResourcePlanningError::Invalid(
            "structural modal output, subspace, or placement bounds are inconsistent".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_transient_block_demand(
    demand: &super::super::contract::StructuralTransientBlockResourceDemand,
) -> Result<(), ResourcePlanningError> {
    let SolveResourcePolicy::TransientBlock {
        physical_dimension,
        coordinate_dimension,
        workspace_bytes_upper_bound,
    } = demand.operator.solve
    else {
        return Err(ResourcePlanningError::Invalid(
            "transient block requires its block-solve workspace policy".into(),
        ));
    };
    let projected_segment_bytes =
        super::super::transient_block::structural_transient_block_payload_bytes_upper_bound(
            &demand.inputs,
            demand.time_point_count.0,
            physical_dimension.0,
        )?;
    if demand.time_point_count.0 == 0
        || physical_dimension != demand.operator.sparse.rows
        || coordinate_dimension.0 == 0
        || coordinate_dimension.0 > physical_dimension.0
        || workspace_bytes_upper_bound.0 == 0
        || demand.operator.partitioning != PartitioningPolicy::Monolithic
        || !prepared_output_is_valid(&demand.segment_output)
        || demand.segment_output.encoded_payload_bytes_upper_bound.0 != projected_segment_bytes
    {
        return Err(ResourcePlanningError::Invalid(
            "transient block dimensions, output, or placement are inconsistent".into(),
        ));
    }
    Ok(())
}

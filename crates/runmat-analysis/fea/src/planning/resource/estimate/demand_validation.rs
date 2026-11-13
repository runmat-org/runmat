//! Family-specific operator-demand validation kept outside the resource estimator.

use super::{add, mul, prepared_output_is_valid, PartitioningPolicy, ResourcePlanningError};
use crate::planning::resource::SolveResourcePolicy;

pub(super) fn validate_random_response_interval_demand(
    demand: &super::super::contract::StructuralRandomResponseIntervalResourceDemand,
) -> Result<(), ResourcePlanningError> {
    let minimum_spectral_values = mul(
        mul(
            demand.channel_count.0,
            demand.channel_count.0,
            "random-response channel matrix",
        )?,
        2,
        "random-response minimum spectral samples",
    )?;
    let expected_response_values = add(
        demand.physical_dof_count.0,
        demand.linear_field_value_count.0,
        "random-response response values",
    )?;
    let minimum_evaluations = mul(
        demand.interval_count.0,
        3,
        "random-response minimum interval evaluations",
    )?;
    if demand.interval_count.0 == 0
        || demand.channel_count.0 == 0
        || demand.spectral_matrix_complex_value_count.0 < minimum_spectral_values
        || demand.selected_mode_count.0 == 0
        || demand.physical_dof_count.0 == 0
        || demand.linear_field_value_count.0 == 0
        || demand.response_value_count.0 != expected_response_values
        || demand.maximum_frequency_evaluation_count.0 < minimum_evaluations
        || demand.operator.sparse.rows.0 != demand.physical_dof_count.0
        || demand.output.encoded_payload_bytes_upper_bound.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "random-response interval resource dimensions are inconsistent".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_buckling_demand(
    demand: &super::super::contract::StructuralBucklingResourceDemand,
) -> Result<(), ResourcePlanningError> {
    let SolveResourcePolicy::SymmetricSubspaceIteration {
        reduced_dimension,
        requested_mode_count,
        maximum_subspace_dimension,
        ..
    } = demand.operator.solve
    else {
        return Err(ResourcePlanningError::Invalid(
            "structural buckling demand requires a symmetric-subspace solve policy".into(),
        ));
    };
    if maximum_subspace_dimension != reduced_dimension
        || demand.requested_mode_count != requested_mode_count
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
    let expected_result = super::super::modal::output::result_output(
        demand.modal_output_value_count_upper_bound.0,
        demand.retained_mode_count_upper_bound.0,
        demand.residual_vector_count.0,
        demand.operator.sparse.rows.0,
        demand.operator.topology.nodes.0,
    )?;
    if demand.result_output != expected_result {
        return Err(ResourcePlanningError::Invalid(
            "modal result profile differs from its numeric and metadata inventory".into(),
        ));
    }
    let SolveResourcePolicy::SymmetricSubspaceIteration {
        reduced_dimension,
        maximum_subspace_dimension,
        workspace_bytes_upper_bound,
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
    if workspace_bytes_upper_bound.0 < demand.original_energy_source_bytes.0
        || demand.operator.phase_resources.resident_bytes_upper_bound.0
            < demand.original_energy_source_bytes.0
        || (demand.prestress_linearization.is_some() && demand.original_energy_source_bytes.0 != 0)
        || demand.retained_mode_count_upper_bound.0 == 0
        || demand.retained_mode_count_upper_bound.0 > reduced_dimension.0
        || demand.modal_phase_value_count_upper_bound.0
            < demand.modal_output_value_count_upper_bound.0
        || demand.modal_output_value_count_upper_bound.0 == 0
        || demand.retained_mode_count_upper_bound != maximum_subspace_dimension
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
        || !prepared_output_is_valid(&demand.state_profiles_output)
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

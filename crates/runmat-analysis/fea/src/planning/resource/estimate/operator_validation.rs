use super::{
    add, bytes, phase_policy::requires_no_solve_policy, sparse_validation::validate_sparse, sum,
};
use crate::planning::resource::contract::{
    FeaResourcePlanningInput, PartitioningPolicy, ResourcePlanningError, SolveResourcePolicy,
    StructuralOperatorResourceDemand,
};
use runmat_execution::Digest;

pub(super) fn validate_operator_demand(
    input: &FeaResourcePlanningInput,
    demand: &StructuralOperatorResourceDemand,
) -> Result<(), ResourcePlanningError> {
    if demand.topology_digest == Digest::sha256([])
        || demand.numerical_scope_digest == Digest::sha256([])
    {
        return Err(ResourcePlanningError::Invalid(
            "operator resource scope and identities must be nonzero".into(),
        ));
    }
    if demand.topology.nodes.0 == 0
        || demand.topology.elements.0 == 0
        || demand.topology.estimated_resident_bytes.0 == 0
        || demand.constraints.estimated_resident_bytes.0 == 0
        || demand.sparse.rows.0 == 0
        || demand.sparse.columns.0 == 0
        || demand.sparse.rows != demand.sparse.columns
    {
        return Err(ResourcePlanningError::Invalid(
            "operator phases require topology, resident bounds, constraints, and a square sparse structure"
                .into(),
        ));
    }
    validate_sparse(&demand.sparse)?;
    let excitation = &demand.excitation_workspace;
    let minimum_field_payload = bytes(
        excitation.scalar_component_count.0,
        16,
        "frozen excitation field payload",
    )?;
    if excitation.active_field_count.0 > excitation.scalar_component_count.0
        || (excitation.active_field_count.0 == 0 && excitation.scalar_component_count.0 != 0)
        || excitation.resident_bytes_upper_bound.0 < minimum_field_payload
    {
        return Err(ResourcePlanningError::Invalid(
            "excitation workspace inventory or frozen field payload is underreserved".into(),
        ));
    }
    let topology_payload_lower_bound = sum(
        &[
            bytes(demand.topology.nodes.0, 24, "topology coordinate payload")?,
            bytes(
                demand.topology.element_node_references.0,
                8,
                "element index payload",
            )?,
            bytes(
                demand.topology.boundary_node_references.0,
                8,
                "boundary index payload",
            )?,
            bytes(
                demand.topology.node_neighbor_references.0,
                8,
                "neighbor index payload",
            )?,
        ],
        "topology structural lower bound",
    )?;
    if demand.topology.estimated_resident_bytes.0 < topology_payload_lower_bound {
        return Err(ResourcePlanningError::Invalid(
            "topology resident bound is below its coordinate/index payload".into(),
        ));
    }
    let constraint_payload_lower_bound = sum(
        &[
            bytes(
                demand.constraints.rows.0,
                8,
                "constraint right-hand-side payload",
            )?,
            bytes(
                demand.constraints.coefficients.0,
                8,
                "constraint coefficient payload",
            )?,
            bytes(
                demand.constraints.reference_points.0,
                24,
                "constraint point payload",
            )?,
        ],
        "constraint structural lower bound",
    )?;
    if demand.constraints.estimated_resident_bytes.0 < constraint_payload_lower_bound {
        return Err(ResourcePlanningError::Invalid(
            "constraint resident bound is below its numerical payload".into(),
        ));
    }
    let contribution_payload_lower_bound = bytes(
        demand.assembly.contribution_nonzero_upper_bound.0,
        24,
        "assembly row, column, and real-f64 contribution payload",
    )?;
    let phase_payload_lower_bound = bytes(
        demand.phase_resources.resident_value_count.0,
        8,
        "phase value payload",
    )?;
    if demand.phase_resources.resident_bytes_upper_bound.0 < phase_payload_lower_bound {
        return Err(ResourcePlanningError::Invalid(
            "phase resident bound is below its scalar payload".into(),
        ));
    }
    let recovery = &demand.field_recovery;
    let recovery_is_empty = recovery.requirement_count.0 == 0
        && recovery.source_entity_count.0 == 0
        && recovery.output_entity_count.0 == 0
        && recovery.topology_incidence_count.0 == 0
        && recovery.arithmetic_operation_count.0 == 0;
    if !recovery_is_empty
        && (recovery.requirement_count.0 == 0
            || recovery.source_entity_count.0 == 0
            || recovery.output_entity_count.0 == 0
            || recovery.arithmetic_operation_count.0 == 0)
    {
        return Err(ResourcePlanningError::Invalid(
            "field recovery resource facts are incomplete".into(),
        ));
    }
    if demand.assembly.contribution_nonzero_upper_bound.0 < demand.sparse.maximum_nonzeros()
        || demand.assembly.contribution_bytes_upper_bound.0 == 0
        || demand.assembly.contribution_bytes_upper_bound.0 < contribution_payload_lower_bound
    {
        return Err(ResourcePlanningError::Invalid(
            "assembly contribution count and byte bounds must cover canonical nonzeros".into(),
        ));
    }
    match demand.solve {
        SolveResourcePolicy::GeneralSparse(general) => {
            general
                .validate()
                .map_err(|e| ResourcePlanningError::Invalid(e.to_string()))?;
            if general.dimension().0 > demand.sparse.rows.0
                || demand.partitioning != crate::planning::resource::PartitioningPolicy::Monolithic
                || !input.worker.accelerators.is_empty()
            {
                return Err(ResourcePlanningError::Invalid("GENERAL solve requires matching reduced dimension and serial host operator placement".into()));
            }
        }
        SolveResourcePolicy::NoSolve { .. } => {}
        SolveResourcePolicy::Iterative { vector_count, .. }
        | SolveResourcePolicy::DirectFactorization { vector_count, .. }
            if vector_count.0 == 0 =>
        {
            return Err(ResourcePlanningError::Invalid(
                "solve policy requires at least one vector".into(),
            ));
        }
        SolveResourcePolicy::RestartedGmres {
            block_dimension,
            restart,
            maximum_iterations,
            workspace_bytes_upper_bound,
        }
        | SolveResourcePolicy::SaddlePointGmres {
            block_dimension,
            restart,
            maximum_iterations,
            workspace_bytes_upper_bound,
            ..
        } if maximum_iterations.0 == 0
            || (block_dimension.0 == 0 && restart.0 != 0)
            || (block_dimension.0 != 0 && restart.0 == 0)
            || restart.0 > block_dimension.0
            || restart.0 > maximum_iterations.0
            || workspace_bytes_upper_bound.0 == 0 =>
        {
            return Err(ResourcePlanningError::Invalid(
                "restarted GMRES dimensions, iteration bounds, and workspace must be valid".into(),
            ));
        }
        SolveResourcePolicy::SaddlePointGmres {
            block_dimension,
            restart,
            maximum_iterations,
            workspace_bytes_upper_bound,
            ..
        } if block_dimension.0
            != add(
                demand.sparse.rows.0,
                demand.constraints.rows.0,
                "saddle dimension",
            )?
            || restart.0 != block_dimension.0.min(32)
            || maximum_iterations.0
                != block_dimension
                    .0
                    .checked_mul(8)
                    .ok_or(ResourcePlanningError::Overflow(
                        "saddle GMRES iteration bound",
                    ))?
                    .max(256)
            || workspace_bytes_upper_bound.0
                < super::super::preflight::saddle_gmres_workspace_bytes(
                    demand.sparse.rows.0,
                    demand.constraints.rows.0,
                )? =>
        {
            return Err(ResourcePlanningError::Invalid(
                "saddle GMRES dimensions or workspace disagree with structural facts".into(),
            ));
        }
        SolveResourcePolicy::ModalDenseHarmonic {
            full_dimension,
            mode_count,
            workspace_bytes_upper_bound,
        } if full_dimension.0 == 0
            || mode_count.0 == 0
            || mode_count.0 > full_dimension.0
            || workspace_bytes_upper_bound.0 == 0 =>
        {
            return Err(ResourcePlanningError::Invalid(
                "modal harmonic dimensions and workspace must be valid".into(),
            ));
        }
        SolveResourcePolicy::TransientBlock {
            physical_dimension,
            coordinate_dimension,
            workspace_bytes_upper_bound,
        } if physical_dimension != demand.sparse.rows
            || coordinate_dimension.0 == 0
            || coordinate_dimension.0 > physical_dimension.0
            || workspace_bytes_upper_bound.0 == 0 =>
        {
            return Err(ResourcePlanningError::Invalid(
                "transient block workspace and dimensions must be valid".into(),
            ));
        }
        SolveResourcePolicy::SymmetricSubspaceIteration {
            full_dimension,
            reduced_dimension,
            requested_mode_count,
            initial_subspace_dimension,
            maximum_subspace_dimension,
            boundary_probe_width,
            subspace_expansion_count_upper_bound,
            maximum_iterations_per_expansion,
            orthogonalization_passes_per_block,
            dense_projected_value_count_upper_bound,
            workspace_bytes_upper_bound,
            ..
        } if full_dimension.0 == 0
            || reduced_dimension.0 == 0
            || reduced_dimension.0 > full_dimension.0
            || requested_mode_count.0 == 0
            || requested_mode_count.0 > reduced_dimension.0
            || initial_subspace_dimension.0 < requested_mode_count.0
            || initial_subspace_dimension.0 > maximum_subspace_dimension.0
            || maximum_subspace_dimension.0 > reduced_dimension.0
            || boundary_probe_width.0 == 0
            || subspace_expansion_count_upper_bound.0 == 0
            || maximum_iterations_per_expansion.0 == 0
            || orthogonalization_passes_per_block.0 == 0
            || dense_projected_value_count_upper_bound.0
                != maximum_subspace_dimension
                    .0
                    .saturating_mul(maximum_subspace_dimension.0)
            || workspace_bytes_upper_bound.0 == 0 =>
        {
            return Err(ResourcePlanningError::Invalid(
                "modal extraction dimensions, probe bounds, and workspace must be valid".into(),
            ));
        }
        _ => {}
    }
    if requires_no_solve_policy(&input.phase)
        != matches!(demand.solve, SolveResourcePolicy::NoSolve { .. })
    {
        return Err(ResourcePlanningError::Invalid(
            "solve and post-processing phases require their matching solve policy".into(),
        ));
    }
    match (&demand.partitioning, &demand.solve) {
        (
            PartitioningPolicy::OperatorDomain {
                maximum_partitions, ..
            },
            _,
        ) if *maximum_partitions < 2 => {
            return Err(ResourcePlanningError::Invalid(
                "operator/domain partitioning requires at least two partitions".into(),
            ))
        }
        (
            PartitioningPolicy::OperatorDomain { .. },
            SolveResourcePolicy::DirectFactorization { .. },
        ) => {
            return Err(ResourcePlanningError::Invalid(
                "direct factorization lacks a partition-local fill contract".into(),
            ))
        }
        _ => {}
    }
    Ok(())
}

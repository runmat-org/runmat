use runmat_execution::{resource::ResourceRequest, Digest};

use super::contract::{
    ByteCount, FeaResourcePlan, FeaResourcePlanIdentity, FeaResourcePlanningInput,
    PartitioningPolicy, PreconditionerStorage, ResourceBreakdown, ResourcePlanningError,
    SolveResourcePolicy, SparseResourceFacts, RESOURCE_ESTIMATION_MODEL_REVISION,
};

pub fn build_resource_plan(
    input: FeaResourcePlanningInput,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    validate_input(&input)?;
    let topology_bytes = input.topology.resident_bytes_upper_bound;
    let csr_row_offset_bytes = ByteCount(bytes(
        add(input.sparse.rows.0, 1, "CSR row offsets")?,
        8,
        "CSR row offsets",
    )?);
    let csr_column_index_bytes =
        ByteCount(bytes(input.sparse.nonzeros.0, 8, "CSR column indices")?);
    let csr_value_bytes = ByteCount(bytes(
        mul(
            input.sparse.nonzeros.0,
            input.sparse.value_array_count.0,
            "CSR values",
        )?,
        8,
        "CSR values",
    )?);
    let assembly_contribution_bytes = input.assembly.contribution_bytes_upper_bound;
    let assembly_scratch_bytes = ByteCount(add(
        input.assembly.fixed_scratch_bytes_per_worker.0,
        mul(
            input.topology.elements.0,
            input.assembly.scratch_bytes_per_element.0,
            "assembly element scratch",
        )?,
        "assembly scratch",
    )?);
    let solve_vector_bytes = ByteCount(bytes(
        mul(
            input.sparse.rows.0,
            input.solve.vector_count(),
            "solve vector values",
        )?,
        input.solve.scalar_bytes(),
        "solve vectors",
    )?);
    let (preconditioner_bytes, factorization_bytes) = solve_storage(&input)?;
    let constraint_bytes = input.constraints.resident_bytes_upper_bound;
    let recovery_resident_bytes = input.recovery.resident_bytes_upper_bound;
    let persistent = sum(
        &[
            topology_bytes.0,
            csr_row_offset_bytes.0,
            csr_column_index_bytes.0,
            csr_value_bytes.0,
            constraint_bytes.0,
        ],
        "persistent numerical memory",
    )?;
    let solve_phase = sum(
        &[
            solve_vector_bytes.0,
            preconditioner_bytes.0,
            factorization_bytes.0,
        ],
        "solve phase memory",
    )?;
    let recovery_phase = sum(
        &[
            recovery_resident_bytes.0,
            input.observations.peak_resident_bytes.0,
        ],
        "recovery phase memory",
    )?;
    let numerical_memory = add(
        persistent,
        assembly_contribution_bytes
            .0
            .max(solve_phase)
            .max(recovery_phase),
        "numerical peak memory",
    )?;
    let total_memory = add(
        input.worker.fixed_memory_bytes.0,
        numerical_memory,
        "worker peak memory",
    )?;
    let phase_scratch = assembly_scratch_bytes.0.max(input.recovery.scratch_bytes.0);
    let total_scratch = add(
        input.worker.fixed_scratch_bytes.0,
        phase_scratch,
        "worker scratch",
    )?;
    let artifact_bytes = sum(
        &[
            input.observations.stored_bytes.0,
            input.recovery.retained_bytes.0,
            input.publication.checkpoint_bytes.0,
            input.publication.artifact_overhead_bytes.0,
        ],
        "published artifacts",
    )?;
    let total_egress = sum(
        &[
            input.observations.transfer_bytes.0,
            input.publication.result_egress_bytes.0,
            input.publication.checkpoint_egress_bytes.0,
        ],
        "result and checkpoint egress",
    )?;

    if input.worker.maximum_wall_milliseconds
        > input.accuracy_ceiling.maximum_wall_time_milliseconds
    {
        return Err(ResourcePlanningError::AccuracyCeiling("wall time"));
    }
    validate_accuracy_demand(&input)?;
    let partition = partition_facts(
        &input,
        numerical_memory,
        total_memory,
        phase_scratch,
        total_scratch,
    )?;
    let request = ResourceRequest {
        cpu_millicores: input.worker.cpu_millicores,
        memory_bytes: total_memory,
        scratch_bytes: total_scratch,
        max_wall_millis: input.worker.maximum_wall_milliseconds,
        max_artifact_bytes: artifact_bytes,
        max_egress_bytes: total_egress,
        max_relay_bytes: input.publication.relay_bytes.0,
        accelerators: input.worker.accelerators.iter().map(Into::into).collect(),
        required_capabilities: input.worker.required_capabilities.clone(),
    };
    request
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let breakdown = ResourceBreakdown {
        topology_bytes,
        csr_row_offset_bytes,
        csr_column_index_bytes,
        csr_value_bytes,
        assembly_contribution_bytes,
        assembly_scratch_bytes,
        solve_vector_bytes,
        preconditioner_bytes,
        factorization_bytes,
        constraint_bytes,
        recovery_resident_bytes,
        recovery_scratch_bytes: input.recovery.scratch_bytes,
        recovery_retained_bytes: input.recovery.retained_bytes,
        observation_sample_count: input.observations.sample_count,
        observation_operation_count: input.observations.operation_count,
        observation_peak_bytes: input.observations.peak_resident_bytes,
        observation_stored_bytes: input.observations.stored_bytes,
        observation_transfer_bytes: input.observations.transfer_bytes,
        result_egress_bytes: input.publication.result_egress_bytes,
        checkpoint_egress_bytes: input.publication.checkpoint_egress_bytes,
        total_egress_bytes: ByteCount(total_egress),
        observation_collectives: input.observations.collective_count,
        checkpoint_bytes: input.publication.checkpoint_bytes,
        artifact_overhead_bytes: input.publication.artifact_overhead_bytes,
        worker_overhead_bytes: input.worker.fixed_memory_bytes,
        partition_replicated_memory_bytes: ByteCount(partition.replicated_memory),
        partition_halo_memory_bytes: ByteCount(partition.halo_memory),
        partition_replicated_scratch_bytes: ByteCount(partition.replicated_scratch),
        partition_halo_scratch_bytes: ByteCount(partition.halo_scratch),
        partitionable_memory_bytes: ByteCount(partition.partitionable_memory),
        fixed_memory_bytes: ByteCount(partition.fixed_memory),
        partitionable_scratch_bytes: ByteCount(partition.partitionable_scratch),
        fixed_scratch_bytes: ByteCount(partition.fixed_scratch),
    };
    let mut plan = FeaResourcePlan {
        schema_version: 1,
        identity: FeaResourcePlanIdentity(Digest::sha256([])),
        input,
        breakdown,
        request,
    };
    plan.identity = plan.derive_identity()?;
    plan.validate_identity()?;
    Ok(plan)
}

struct PartitionFacts {
    partitionable_memory: u64,
    fixed_memory: u64,
    partitionable_scratch: u64,
    fixed_scratch: u64,
    replicated_memory: u64,
    halo_memory: u64,
    replicated_scratch: u64,
    halo_scratch: u64,
}

fn partition_facts(
    input: &FeaResourcePlanningInput,
    numerical_memory: u64,
    total_memory: u64,
    phase_scratch: u64,
    total_scratch: u64,
) -> Result<PartitionFacts, ResourcePlanningError> {
    match input.partitioning {
        PartitioningPolicy::Monolithic => {
            if total_memory > input.accuracy_ceiling.maximum_resident_bytes {
                return Err(ResourcePlanningError::AccuracyCeiling("resident memory"));
            }
            Ok(PartitionFacts {
                partitionable_memory: 0,
                fixed_memory: total_memory,
                partitionable_scratch: 0,
                fixed_scratch: total_scratch,
                replicated_memory: 0,
                halo_memory: 0,
                replicated_scratch: 0,
                halo_scratch: 0,
            })
        }
        PartitioningPolicy::OperatorDomain {
            replicated_memory_bytes_per_partition,
            halo_memory_bytes_per_partition,
            replicated_scratch_bytes_per_partition,
            halo_scratch_bytes_per_partition,
            ..
        } => Ok(PartitionFacts {
            partitionable_memory: numerical_memory,
            fixed_memory: sum(
                &[
                    input.worker.fixed_memory_bytes.0,
                    replicated_memory_bytes_per_partition.0,
                    halo_memory_bytes_per_partition.0,
                ],
                "partition fixed memory",
            )?,
            partitionable_scratch: phase_scratch,
            fixed_scratch: sum(
                &[
                    input.worker.fixed_scratch_bytes.0,
                    replicated_scratch_bytes_per_partition.0,
                    halo_scratch_bytes_per_partition.0,
                ],
                "partition fixed scratch",
            )?,
            replicated_memory: replicated_memory_bytes_per_partition.0,
            halo_memory: halo_memory_bytes_per_partition.0,
            replicated_scratch: replicated_scratch_bytes_per_partition.0,
            halo_scratch: halo_scratch_bytes_per_partition.0,
        }),
    }
}

fn validate_input(input: &FeaResourcePlanningInput) -> Result<(), ResourcePlanningError> {
    if input.estimation_model_revision != RESOURCE_ESTIMATION_MODEL_REVISION {
        return Err(ResourcePlanningError::Invalid(
            "unsupported resource estimation model revision".into(),
        ));
    }
    if input.topology_digest == Digest::sha256([])
        || input.operator_pattern_digest == Digest::sha256([])
        || input.topology.nodes.0 == 0
        || input.topology.elements.0 == 0
        || input.topology.resident_bytes_upper_bound.0 == 0
        || input.constraints.resident_bytes_upper_bound.0 == 0
        || input.sparse.rows.0 == 0
        || input.sparse.columns.0 == 0
        || input.sparse.rows != input.sparse.columns
        || input.worker.cpu_millicores == 0
        || input.worker.maximum_wall_milliseconds == 0
    {
        return Err(ResourcePlanningError::Invalid("topology, resident bounds, square operator, CPU, wall time, and identities must be nonzero".into()));
    }
    validate_sparse(&input.sparse)?;
    let topology_payload_lower_bound = sum(
        &[
            bytes(input.topology.nodes.0, 24, "topology coordinate payload")?,
            bytes(
                input.topology.element_node_references.0,
                8,
                "element index payload",
            )?,
            bytes(
                input.topology.boundary_node_references.0,
                8,
                "boundary index payload",
            )?,
            bytes(
                input.topology.node_neighbor_references.0,
                8,
                "neighbor index payload",
            )?,
        ],
        "topology structural lower bound",
    )?;
    if input.topology.resident_bytes_upper_bound.0 < topology_payload_lower_bound {
        return Err(ResourcePlanningError::Invalid(
            "topology resident bound is below its coordinate/index payload".into(),
        ));
    }
    let constraint_payload_lower_bound = sum(
        &[
            bytes(
                input.constraints.rows.0,
                8,
                "constraint right-hand-side payload",
            )?,
            bytes(
                input.constraints.coefficients.0,
                8,
                "constraint coefficient payload",
            )?,
            bytes(
                input.constraints.reference_points.0,
                24,
                "constraint point payload",
            )?,
        ],
        "constraint structural lower bound",
    )?;
    if input.constraints.resident_bytes_upper_bound.0 < constraint_payload_lower_bound {
        return Err(ResourcePlanningError::Invalid(
            "constraint resident bound is below its numerical payload".into(),
        ));
    }
    let contribution_payload_lower_bound = bytes(
        input.assembly.contribution_nonzero_upper_bound.0,
        24,
        "assembly row, column, and real-f64 contribution payload",
    )?;
    let recovery_payload_lower_bound = bytes(
        input.recovery.resident_value_count.0,
        8,
        "recovery value payload",
    )?;
    if input.recovery.resident_bytes_upper_bound.0 < recovery_payload_lower_bound {
        return Err(ResourcePlanningError::Invalid(
            "recovery resident bound is below its scalar payload".into(),
        ));
    }
    if input.assembly.contribution_nonzero_upper_bound.0 < input.sparse.nonzeros.0
        || input.assembly.contribution_bytes_upper_bound.0 == 0
        || input.assembly.contribution_bytes_upper_bound.0 < contribution_payload_lower_bound
    {
        return Err(ResourcePlanningError::Invalid(
            "assembly contribution count and byte bounds must cover canonical nonzeros".into(),
        ));
    }
    if input.solve.vector_count() == 0 {
        return Err(ResourcePlanningError::Invalid(
            "solve policy requires at least one vector".into(),
        ));
    }
    match (&input.partitioning, &input.solve) {
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
    for accelerator in &input.worker.accelerators {
        if accelerator.class.trim().is_empty()
            || accelerator.count == 0
            || accelerator.memory_bytes_each.0 == 0
        {
            return Err(ResourcePlanningError::Invalid(
                "accelerator class, count, and memory must be nonzero".into(),
            ));
        }
    }
    Ok(())
}

fn validate_sparse(facts: &SparseResourceFacts) -> Result<(), ResourcePlanningError> {
    if facts.rows.0 == 0
        || facts.columns.0 == 0
        || facts.nonzeros.0 == 0
        || facts.value_array_count.0 == 0
    {
        return Err(ResourcePlanningError::Invalid(
            "sparse dimensions, nonzeros, and value arrays must be nonzero".into(),
        ));
    }
    if facts.nonzeros.0 > mul(facts.rows.0, facts.columns.0, "sparse dense bound")? {
        return Err(ResourcePlanningError::Invalid(
            "sparse nonzeros exceed matrix dimensions".into(),
        ));
    }
    Ok(())
}

fn validate_accuracy_demand(input: &FeaResourcePlanningInput) -> Result<(), ResourcePlanningError> {
    let refinements = u64::try_from(input.accuracy_ceiling.maximum_refinement_cycles)
        .map_err(|_| ResourcePlanningError::Overflow("accuracy refinement ceiling"))?;
    let tightenings = u64::try_from(input.accuracy_ceiling.maximum_solver_tightenings)
        .map_err(|_| ResourcePlanningError::Overflow("accuracy tightening ceiling"))?;
    let samples = u64::try_from(input.accuracy_ceiling.maximum_samples)
        .map_err(|_| ResourcePlanningError::Overflow("accuracy sample ceiling"))?;
    if input.accuracy_demand.refinement_cycles.0 > refinements {
        return Err(ResourcePlanningError::AccuracyCeiling("refinement cycles"));
    }
    if input.accuracy_demand.solver_tightenings.0 > tightenings {
        return Err(ResourcePlanningError::AccuracyCeiling("solver tightenings"));
    }
    if input.accuracy_demand.samples.0 > samples || input.observations.sample_count.0 > samples {
        return Err(ResourcePlanningError::AccuracyCeiling("samples"));
    }
    Ok(())
}

fn solve_storage(
    input: &FeaResourcePlanningInput,
) -> Result<(ByteCount, ByteCount), ResourcePlanningError> {
    match input.solve {
        SolveResourcePolicy::Iterative {
            ref preconditioner, ..
        } => {
            let value = match preconditioner {
                PreconditionerStorage::None => 0,
                PreconditionerStorage::Diagonal => bytes(
                    input.sparse.rows.0,
                    input.solve.scalar_bytes(),
                    "diagonal preconditioner",
                )?,
                PreconditionerStorage::SparseCsr {
                    nonzero_upper_bound,
                    resident_bytes_upper_bound,
                } => {
                    let payload_lower_bound = sum(
                        &[
                            bytes(
                                add(input.sparse.rows.0, 1, "preconditioner row offsets")?,
                                8,
                                "preconditioner row offsets",
                            )?,
                            bytes(nonzero_upper_bound.0, 8, "preconditioner column indices")?,
                            bytes(
                                nonzero_upper_bound.0,
                                input.solve.scalar_bytes(),
                                "preconditioner values",
                            )?,
                        ],
                        "sparse preconditioner CSR payload",
                    )?;
                    if nonzero_upper_bound.0 == 0
                        || resident_bytes_upper_bound.0 < payload_lower_bound
                    {
                        return Err(ResourcePlanningError::Invalid(
                            "sparse preconditioner count and resident bound must be nonzero".into(),
                        ));
                    }
                    resident_bytes_upper_bound.0
                }
            };
            Ok((ByteCount(value), ByteCount(0)))
        }
        SolveResourcePolicy::DirectFactorization {
            factor_nonzero_upper_bound,
            factor_bytes_upper_bound,
            ..
        } => {
            let payload_lower_bound = sum(
                &[
                    bytes(
                        add(input.sparse.rows.0, 1, "factor row offsets")?,
                        8,
                        "factor row offsets",
                    )?,
                    bytes(factor_nonzero_upper_bound.0, 8, "factor column indices")?,
                    bytes(
                        factor_nonzero_upper_bound.0,
                        input.solve.scalar_bytes(),
                        "factor values",
                    )?,
                ],
                "factorization sparse payload",
            )?;
            if factor_nonzero_upper_bound.0 == 0 || factor_bytes_upper_bound.0 < payload_lower_bound
            {
                return Err(ResourcePlanningError::Invalid(
                    "factorization count and resident bound must be nonzero".into(),
                ));
            }
            Ok((ByteCount(0), factor_bytes_upper_bound))
        }
    }
}

fn bytes(count: u64, width: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    mul(count, width, label)
}
fn add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
fn mul(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_mul(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}
fn sum(values: &[u64], label: &'static str) -> Result<u64, ResourcePlanningError> {
    values
        .iter()
        .try_fold(0, |total, value| add(total, *value, label))
}

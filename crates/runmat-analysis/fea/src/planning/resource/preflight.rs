use crate::{
    algebra::CoercivitySymbolicPlan,
    assembly::{dofs::StructuralDofKind, mixed_dof_layout},
    constraints::{
        CanonicalStructuralConstraintPlan, CanonicalStructuralDofOwner,
        CanonicalStructuralTopologyIdentity, ResolvedConstraintMatrix,
    },
    execution::StructuralLinearWorkload,
    physics::structural::topology::node_space::CanonicalStructuralNodeSpace,
    results::observation::ResolvedObservationPlan,
    solve::preconditioner::SpdPreconditionerKind,
    topology::AnalysisProjection,
};
use runmat_meshing_core::SolverMeshArtifact;

use super::{
    build_resource_plan, AccuracyResourceDemand, AssemblyResourcePolicy, ByteCount,
    CheckpointArtifactPolicy, ConstraintResourceFacts, Count, FeaResourcePlan,
    FeaResourcePlanningInput, NumericalScalar, ObservationResourceFacts, PartitioningPolicy,
    PreconditionerStorage, RecoveryResourcePolicy, ResourcePlanningError, SolveResourcePolicy,
    SolverAdmissionResourcePolicy, SparseResourceFacts, StructuralSymbolicOperators,
    TopologyResourceFacts, WorkerResourcePolicy, RESOURCE_ESTIMATION_MODEL_REVISION,
};

const STRUCTURAL_ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT: u64 = 8 * 1024;
const STRUCTURAL_FIXED_ASSEMBLY_SCRATCH_BYTES: u64 = 64 * 1024;
const STRUCTURAL_FIXED_WORKER_MEMORY_BYTES: u64 = 16 * 1024 * 1024;
const STRUCTURAL_FIXED_WORKER_SCRATCH_BYTES: u64 = 8 * 1024 * 1024;
const STRUCTURAL_ARTIFACT_OVERHEAD_BYTES: u64 = 64 * 1024;
const STRUCTURAL_SOLVE_VECTOR_COUNT: u64 = 8;

/// Builds the canonical resource plan from authoritative post-projection numerical inputs.
///
/// This function does no assembly and accepts no caller-supplied topology, sparse, constraint, or
/// observation counts. It is therefore safe to run before allocations and expensive kernels.
pub fn build_structural_resource_plan(
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    observations: &ResolvedObservationPlan,
    workload: &StructuralLinearWorkload,
) -> Result<FeaResourcePlan, ResourcePlanningError> {
    workload
        .validate()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let topology_digest = constraints.topology.discretization_digest();
    if observations.topology_digest != topology_digest {
        return Err(ResourcePlanningError::Invalid(
            "observation plan and structural constraints name different topology".into(),
        ));
    }
    let projection = mesh
        .map(AnalysisProjection::derive)
        .transpose()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let symbolic = match constraints.topology {
        CanonicalStructuralTopologyIdentity::Explicit { .. } => {
            StructuralSymbolicOperators::derive_explicit(constraints, &workload.model)?
        }
        CanonicalStructuralTopologyIdentity::Solid { .. }
        | CanonicalStructuralTopologyIdentity::ConformalMixed { .. } => {
            let mesh = mesh.ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "solid structural resource planning requires a solver mesh".into(),
                )
            })?;
            let projection = projection.as_ref().ok_or_else(|| {
                ResourcePlanningError::Invalid(
                    "solid structural resource planning requires an analysis projection".into(),
                )
            })?;
            StructuralSymbolicOperators::derive(projection, constraints, &workload.model, mesh)?
        }
    };
    let sparse = SparseResourceFacts::from_patterns(symbolic.patterns())?;
    let contribution_nonzeros =
        u64::try_from(symbolic.stiffness_upper_bound().column_indices().len())
            .map_err(|_| ResourcePlanningError::Overflow("assembly contribution count"))?;
    let contribution_bytes =
        contribution_nonzeros
            .checked_mul(24)
            .ok_or(ResourcePlanningError::Overflow(
                "assembly contribution bytes",
            ))?;
    let observations = ObservationResourceFacts::from_plan(observations)?;
    let dofs = sparse.rows.0;
    let stiffness_nonzeros = sparse.maximum_nonzeros();
    let constraint_matrix = resolve_constraint_matrix(mesh, constraints, workload)?;
    let admission_pattern = if constraints.rows.iter().any(|row| row.terms.len() > 1) {
        constraint_matrix
            .affine_reduction()
            .and_then(|reduction| reduction.reduce_sparse_pattern(symbolic.stiffness_upper_bound()))
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?
    } else {
        symbolic.stiffness_upper_bound().clone()
    };
    let admission = CoercivitySymbolicPlan::derive(&admission_pattern)
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let solve = solve_policy(
        workload.preconditioner,
        dofs,
        stiffness_nonzeros,
        &admission,
    )?;
    let mixed_topology = if matches!(
        constraints.topology,
        CanonicalStructuralTopologyIdentity::ConformalMixed { .. }
    ) {
        let structural = workload.model.structural.as_ref().ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "mixed resource planning requires explicit structural topology".into(),
            )
        })?;
        Some((
            structural,
            CanonicalStructuralNodeSpace::derive(
                mesh.ok_or_else(|| {
                    ResourcePlanningError::Invalid(
                        "mixed resource planning requires a solver mesh".into(),
                    )
                })?,
                structural,
            )
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
        ))
    } else {
        None
    };
    let recovery_node_count = mixed_topology
        .as_ref()
        .map(|(_, topology)| topology.nodes.len())
        .or_else(|| {
            workload
                .model
                .structural
                .as_ref()
                .filter(|structural| !structural.elements.is_empty())
                .map(|structural| structural.nodes.len())
        })
        .or_else(|| projection.as_ref().map(|projection| projection.nodes.len()))
        .unwrap_or(0);
    let solid_element_count = projection
        .as_ref()
        .map(|projection| projection.elements.len())
        .unwrap_or(0);
    let recovery_element_count = solid_element_count
        .checked_add(
            workload
                .model
                .structural
                .as_ref()
                .map(|structural| structural.elements.len())
                .unwrap_or(0),
        )
        .ok_or(ResourcePlanningError::Overflow(
            "structural recovery element count",
        ))?;
    let recovery_values = recovery_node_count
        .checked_mul(6)
        .and_then(|value| value.checked_add(recovery_element_count.checked_mul(13)?))
        .ok_or(ResourcePlanningError::Overflow(
            "structural recovery value count",
        ))?;
    let recovery_values = u64::try_from(recovery_values)
        .map_err(|_| ResourcePlanningError::Overflow("structural recovery value count"))?;
    let recovery_bytes = checked_bytes(recovery_values, 8, "structural recovery bytes")?;
    let recovery_resident = checked_add(
        recovery_bytes,
        STRUCTURAL_ARTIFACT_OVERHEAD_BYTES,
        "structural recovery resident bytes",
    )?;
    let checkpoint_bytes = checked_add(
        checked_bytes(dofs, 8, "structural checkpoint values")?,
        STRUCTURAL_ARTIFACT_OVERHEAD_BYTES,
        "structural checkpoint bytes",
    )?;
    let step = workload
        .procedure_step()
        .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let topology_facts = if let Some((structural, topology)) = mixed_topology.as_ref() {
        let projection = projection.as_ref().ok_or_else(|| {
            ResourcePlanningError::Invalid("mixed resource planning lost its projection".into())
        })?;
        TopologyResourceFacts::from_mixed_projection(projection, structural, topology)?
    } else if matches!(
        constraints.topology,
        CanonicalStructuralTopologyIdentity::Explicit { .. }
    ) {
        let structural = workload.model.structural.as_ref().ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "explicit resource planning requires structural topology".into(),
            )
        })?;
        let topology = CanonicalStructuralNodeSpace::derive_explicit(structural)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        TopologyResourceFacts::from_explicit(structural, &topology)?
    } else {
        TopologyResourceFacts::from_projection(projection.as_ref().ok_or_else(|| {
            ResourcePlanningError::Invalid("solid resource planning lost its projection".into())
        })?)?
    };
    build_resource_plan(FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        topology_digest,
        operator_pattern_digest: symbolic.identity(),
        topology: topology_facts,
        sparse,
        constraints: ConstraintResourceFacts::from_plan(constraints)?,
        assembly: AssemblyResourcePolicy {
            contribution_nonzero_upper_bound: Count(contribution_nonzeros),
            contribution_bytes_upper_bound: ByteCount(contribution_bytes),
            scratch_bytes_per_element: ByteCount(STRUCTURAL_ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
            fixed_scratch_bytes_per_worker: ByteCount(STRUCTURAL_FIXED_ASSEMBLY_SCRATCH_BYTES),
        },
        solve,
        recovery: RecoveryResourcePolicy {
            resident_value_count: Count(recovery_values),
            resident_bytes_upper_bound: ByteCount(recovery_resident),
            scratch_bytes: ByteCount(recovery_resident),
            retained_bytes: ByteCount(recovery_bytes),
        },
        observations: observations.clone(),
        publication: CheckpointArtifactPolicy {
            checkpoint_bytes: ByteCount(checkpoint_bytes),
            artifact_overhead_bytes: ByteCount(STRUCTURAL_ARTIFACT_OVERHEAD_BYTES),
            relay_bytes: ByteCount(0),
            result_egress_bytes: ByteCount(recovery_bytes),
            checkpoint_egress_bytes: ByteCount(checkpoint_bytes),
        },
        worker: WorkerResourcePolicy {
            cpu_millicores: 1_000,
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            fixed_memory_bytes: ByteCount(STRUCTURAL_FIXED_WORKER_MEMORY_BYTES),
            fixed_scratch_bytes: ByteCount(STRUCTURAL_FIXED_WORKER_SCRATCH_BYTES),
            required_capabilities: Default::default(),
            accelerators: Vec::new(),
        },
        // The current structural kernel is monolithic. A recommendation would be dishonest until
        // the operator-domain execution path can consume it.
        partitioning: PartitioningPolicy::Monolithic,
        accuracy_demand: AccuracyResourceDemand {
            refinement_cycles: Count(0),
            solver_tightenings: Count(0),
            samples: observations.maximum_samples_per_observation,
        },
        accuracy_ceiling: step.accuracy.resources,
    })
}

fn resolve_constraint_matrix(
    mesh: Option<&SolverMeshArtifact>,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralLinearWorkload,
) -> Result<ResolvedConstraintMatrix, ResourcePlanningError> {
    if matches!(
        constraints.topology,
        CanonicalStructuralTopologyIdentity::Solid { .. }
    ) {
        let mesh = mesh.ok_or_else(|| {
            ResourcePlanningError::Invalid(
                "solid constraint planning requires a solver mesh".into(),
            )
        })?;
        let node_identities = mesh
            .topology
            .nodes
            .iter()
            .map(|node| node.stable_identity)
            .collect::<Vec<_>>();
        return ResolvedConstraintMatrix::from_solid_plan(constraints, &node_identities)
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()));
    }
    let structural = workload.model.structural.as_ref().ok_or_else(|| {
        ResourcePlanningError::Invalid(
            "mixed constraint planning requires explicit structural topology".into(),
        )
    })?;
    let topology = match constraints.topology {
        CanonicalStructuralTopologyIdentity::Explicit { .. } => {
            CanonicalStructuralNodeSpace::derive_explicit(structural)
        }
        CanonicalStructuralTopologyIdentity::ConformalMixed { .. } => {
            CanonicalStructuralNodeSpace::derive(
                mesh.ok_or_else(|| {
                    ResourcePlanningError::Invalid(
                        "mixed constraint planning requires a solver mesh".into(),
                    )
                })?,
                structural,
            )
        }
        CanonicalStructuralTopologyIdentity::Solid { .. } => unreachable!(),
    }
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
    let layout = mixed_dof_layout(&topology);
    let nodes = topology
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.identity, index))
        .collect::<std::collections::BTreeMap<_, _>>();
    ResolvedConstraintMatrix::from_plan(constraints, layout.total_dof_count(), |owner, dof| {
        match owner {
            CanonicalStructuralDofOwner::StructuralNode(identity) => nodes
                .get(&identity)
                .and_then(|node| layout.index(*node, StructuralDofKind::from_constraint(dof))),
            CanonicalStructuralDofOwner::ReferencePoint(_) => None,
        }
    })
    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))
}

fn solve_policy(
    preconditioner: SpdPreconditionerKind,
    dofs: u64,
    stiffness_nonzeros: u64,
    admission: &CoercivitySymbolicPlan,
) -> Result<SolveResourcePolicy, ResourcePlanningError> {
    let storage = match preconditioner {
        SpdPreconditionerKind::Jacobi => PreconditionerStorage::Diagonal,
        SpdPreconditionerKind::Ilu0 => {
            let row_offsets = checked_bytes(
                dofs.checked_add(1)
                    .ok_or(ResourcePlanningError::Overflow("ILU row offsets"))?,
                8,
                "ILU row offsets",
            )?;
            let entries = checked_bytes(stiffness_nonzeros, 16, "ILU columns and values")?;
            PreconditionerStorage::SparseCsr {
                nonzero_upper_bound: Count(stiffness_nonzeros),
                resident_bytes_upper_bound: ByteCount(checked_add(
                    row_offsets,
                    entries,
                    "ILU resident bytes",
                )?),
            }
        }
    };
    Ok(SolveResourcePolicy::Iterative {
        vector_count: Count(STRUCTURAL_SOLVE_VECTOR_COUNT),
        scalar: NumericalScalar::RealF64,
        preconditioner: storage,
        admission: SolverAdmissionResourcePolicy {
            dimension: Count(admission.dimension()),
            factor_nonzero_upper_bound: Count(admission.factor_nonzero_count()),
            update_count_upper_bound: Count(admission.update_count()),
            resident_bytes_upper_bound: ByteCount(admission.resident_bytes_upper_bound()),
        },
    })
}

fn checked_bytes(
    count: u64,
    width: u64,
    label: &'static str,
) -> Result<u64, ResourcePlanningError> {
    count
        .checked_mul(width)
        .ok_or(ResourcePlanningError::Overflow(label))
}

fn checked_add(left: u64, right: u64, label: &'static str) -> Result<u64, ResourcePlanningError> {
    left.checked_add(right)
        .ok_or(ResourcePlanningError::Overflow(label))
}

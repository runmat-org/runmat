//! Original physical topology, operator support and recovery projection.
use super::*;

impl StructuralFrequencyEntryResourceProjection {
    pub(super) fn derive_entry(
        mesh: &SolverMeshArtifact,
        constraints: &CanonicalStructuralConstraintPlan,
        compiler_peak: ByteCount,
        workload: &StructuralFrequencyWorkload,
        states: StructuralFrequencyStateInputs<'_>,
        accelerator_requests: &[AcceleratorRequest],
    ) -> Result<Self, ResourcePlanningError> {
        workload
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        mesh.validate_canonical()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if workload.backend() != ComputeBackend::Cpu || !accelerator_requests.is_empty() {
            return Err(ResourcePlanningError::Invalid(
                "structural frequency execution currently requires CPU resources without accelerators"
                    .into(),
            ));
        }
        let topology_digest = runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
        let linearization = match (workload.prestress_transition(), states.prestress) {
            (None, None) => None,
            (Some(transition), Some(state)) => Some(
                crate::pipeline::prestress::structural_linearization_from_state(
                    transition,
                    state,
                    topology_digest,
                )
                .map_err(ResourcePlanningError::Invalid)?,
            ),
            _ => {
                return Err(ResourcePlanningError::Invalid(
                    "frequency workload and prestress-state presence differ".into(),
                ))
            }
        };
        let facts = derive_structural_resource_facts(workload.model(), Some(mesh), constraints)?;
        let tangent_pattern = match linearization {
            Some(linearization) => facts
                .symbolic
                .stiffness_upper_bound()
                .union(linearization.geometric_stiffness().pattern())
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            None => facts.symbolic.stiffness_upper_bound().clone(),
        };
        let sparse = super::super::SparseResourceFacts::from_patterns([
            &tangent_pattern,
            facts.symbolic.mass_upper_bound(),
            facts.symbolic.damping_upper_bound(),
        ])?;
        let numerical_scope_digest = match linearization {
            Some(linearization) => runmat_execution::Digest::sha256(
                [
                    b"runmat-structural-prestressed-frequency-operator/v1\0".as_slice(),
                    facts.symbolic.identity().bytes().as_slice(),
                    linearization.identity().digest().bytes().as_slice(),
                ]
                .concat(),
            ),
            None => facts.symbolic.identity(),
        };
        let full_dofs = facts.sparse.rows.0;
        let (solve, modal_operator_digest) = match (workload.solver(), states.modal_basis) {
            (StructuralFrequencySolverPlan::Direct { policy }, None) => {
                let reduction = facts
                    .constraint_matrix
                    .affine_reduction()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
                if reduction.offset.iter().any(|value| *value != 0.0) {
                    return Err(ResourcePlanningError::Invalid(
                        "direct harmonic resource admission requires homogeneous constraints"
                            .into(),
                    ));
                }
                let reduced_dofs = count(reduction.reduced_dof_count, "reduced frequency DOFs")?;
                let block_dimension = multiply(reduced_dofs, 2, "real block dimension")?;
                let restart = count(policy.restart, "GMRES restart")?
                    .min(block_dimension)
                    .min(count(
                        policy.maximum_iterations,
                        "GMRES maximum iterations",
                    )?);
                let reduced_patterns = [
                    &tangent_pattern,
                    facts.symbolic.mass_upper_bound(),
                    facts.symbolic.damping_upper_bound(),
                ]
                .into_iter()
                .map(|pattern| reduction.reduce_sparse_pattern(pattern))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
                let reduced_nonzeros =
                    reduced_patterns.iter().try_fold(0_u64, |total, pattern| {
                        add(
                            total,
                            count(pattern.column_indices().len(), "reduced operator nonzeros")?,
                            "reduced operator nonzeros",
                        )
                    })?;
                (
                    SolveResourcePolicy::RestartedGmres {
                        block_dimension: Count(block_dimension),
                        restart: Count(restart),
                        maximum_iterations: Count(count(
                            policy.maximum_iterations,
                            "GMRES maximum iterations",
                        )?),
                        workspace_bytes_upper_bound: ByteCount(direct_workspace_bytes(
                            full_dofs,
                            reduced_dofs,
                            reduced_nonzeros,
                            &reduction,
                            restart,
                        )?),
                    },
                    None,
                )
            }
            (StructuralFrequencySolverPlan::ModalReduced { basis_state }, Some(state)) => {
                state
                    .validate()
                    .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
                let topology_digest =
                    runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
                let basis = state.payload.modal_basis().ok_or_else(|| {
                    ResourcePlanningError::Invalid(
                        "modal harmonic resource admission requires a modal basis state".into(),
                    )
                })?;
                if state.identity != *basis_state
                    || state.topology_digest != topology_digest
                    || basis
                        .modes()
                        .iter()
                        .any(|mode| u64::try_from(mode.dof_shape().len()).ok() != Some(full_dofs))
                {
                    return Err(ResourcePlanningError::Invalid(
                        "modal harmonic basis identity or topology differs from the frequency task"
                            .into(),
                    ));
                }
                let mode_count = count(basis.modes().len(), "modal harmonic mode count")?;
                (
                    SolveResourcePolicy::ModalDenseHarmonic {
                        full_dimension: Count(full_dofs),
                        mode_count: Count(mode_count),
                        workspace_bytes_upper_bound: ByteCount(modal_workspace_bytes(
                            full_dofs, mode_count,
                        )?),
                    },
                    Some(basis.operator_digest()),
                )
            }
            (StructuralFrequencySolverPlan::Direct { .. }, Some(_)) => {
                return Err(ResourcePlanningError::Invalid(
                    "direct frequency resource admission received a modal state".into(),
                ));
            }
            (StructuralFrequencySolverPlan::ModalReduced { .. }, None) => {
                return Err(ResourcePlanningError::Invalid(
                    "modal frequency resource admission is missing its basis state".into(),
                ));
            }
        };
        let contribution_nonzeros = count(
            tangent_pattern.column_indices().len(),
            "frequency assembly contributions",
        )?;
        let contribution_bytes =
            multiply(contribution_nonzeros, 24, "assembly contribution bytes")?;
        let recovery_layout = structural_frequency_recovery_layout(mesh, constraints, workload)?;
        let step = workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        Ok(Self {
            constraint_compilation_peak_bytes: compiler_peak,
            workload: workload.clone(),
            excitation_workspace: super::super::excitation_workspace::derive_excitation_workspace(
                workload.model(),
                step,
                Some(mesh),
            )?,
            topology_digest: facts.topology_digest,
            structural_topology: constraints.topology,
            numerical_scope_digest,
            topology: facts.topology,
            sparse,
            constraints: facts.constraints,
            assembly: AssemblyResourcePolicy {
                contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                contribution_bytes_upper_bound: ByteCount(
                    super::super::connector_buffers::include_connector_buffers(
                        contribution_bytes,
                        facts.connector_assembly_buffer_bytes,
                    )?,
                ),
                scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                fixed_scratch_bytes_per_worker: ByteCount(FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER),
            },
            solve,
            modal_operator_digest,
            recovery_layout,
            constraint_group_count: constraints.groups.len(),
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            accuracy_ceiling: step.accuracy.resources,
        })
    }

    pub(super) fn derive_recovery(
        mesh: &SolverMeshArtifact,
        constraints: &CanonicalStructuralConstraintPlan,
        compiler_peak: ByteCount,
        workload: &StructuralFrequencyWorkload,
        states: StructuralFrequencyStateInputs<'_>,
        accelerator_requests: &[AcceleratorRequest],
    ) -> Result<Self, ResourcePlanningError> {
        workload
            .validate()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        mesh.validate_canonical()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        if workload.backend() != ComputeBackend::Cpu || !accelerator_requests.is_empty() {
            return Err(ResourcePlanningError::Invalid(
                "structural frequency recovery currently requires CPU resources without accelerators"
                    .into(),
            ));
        }
        let topology_digest = runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes());
        if states.modal_basis.is_some() {
            return Err(ResourcePlanningError::Invalid(
                "frequency recovery does not consume a modal-basis state".into(),
            ));
        }
        let linearization = match (workload.prestress_transition(), states.prestress) {
            (None, None) => None,
            (Some(transition), Some(state)) => Some(
                crate::pipeline::prestress::structural_linearization_from_state(
                    transition,
                    state,
                    topology_digest,
                )
                .map_err(ResourcePlanningError::Invalid)?,
            ),
            _ => {
                return Err(ResourcePlanningError::Invalid(
                    "frequency recovery workload and prestress-state presence differ".into(),
                ))
            }
        };
        let facts = derive_structural_resource_facts(workload.model(), Some(mesh), constraints)?;
        let tangent_pattern = match linearization {
            Some(linearization) => facts
                .symbolic
                .stiffness_upper_bound()
                .union(linearization.geometric_stiffness().pattern())
                .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?,
            None => facts.symbolic.stiffness_upper_bound().clone(),
        };
        let sparse = super::super::SparseResourceFacts::from_patterns([
            &tangent_pattern,
            facts.symbolic.mass_upper_bound(),
            facts.symbolic.damping_upper_bound(),
        ])?;
        let numerical_scope_digest = match linearization {
            Some(linearization) => runmat_execution::Digest::sha256(
                [
                    b"runmat-structural-prestressed-frequency-recovery/v1\0".as_slice(),
                    facts.symbolic.identity().bytes().as_slice(),
                    linearization.identity().digest().bytes().as_slice(),
                ]
                .concat(),
            ),
            None => facts.symbolic.identity(),
        };
        let contribution_nonzeros = count(
            tangent_pattern.column_indices().len(),
            "frequency recovery assembly contributions",
        )?;
        let recovery_layout = structural_frequency_recovery_layout(mesh, constraints, workload)?;
        let step = workload
            .procedure_step()
            .map_err(|error| ResourcePlanningError::Invalid(error.to_string()))?;
        let full_dofs = facts.sparse.rows.0;
        Ok(Self {
            constraint_compilation_peak_bytes: compiler_peak,
            workload: workload.clone(),
            excitation_workspace: Default::default(),
            topology_digest: facts.topology_digest,
            structural_topology: constraints.topology,
            numerical_scope_digest,
            topology: facts.topology,
            sparse,
            constraints: facts.constraints,
            assembly: AssemblyResourcePolicy {
                contribution_nonzero_upper_bound: Count(contribution_nonzeros),
                contribution_bytes_upper_bound: ByteCount(
                    super::super::connector_buffers::include_connector_buffers(
                        multiply(
                            contribution_nonzeros,
                            24,
                            "frequency recovery contribution bytes",
                        )?,
                        facts.connector_assembly_buffer_bytes,
                    )?,
                ),
                scratch_bytes_per_element: ByteCount(ASSEMBLY_SCRATCH_BYTES_PER_ELEMENT),
                fixed_scratch_bytes_per_worker: ByteCount(FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER),
            },
            solve: SolveResourcePolicy::NoSolve {
                workspace_bytes_upper_bound: ByteCount(multiply(
                    full_dofs,
                    32,
                    "complex recovery operator workspace",
                )?),
            },
            modal_operator_digest: None,
            recovery_layout,
            constraint_group_count: constraints.groups.len(),
            maximum_wall_milliseconds: step.accuracy.resources.maximum_wall_time_milliseconds,
            accuracy_ceiling: step.accuracy.resources,
        })
    }
}

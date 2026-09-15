use runmat_analysis_core::{
    validate_model_sequence, AnalysisModel, AnalysisProcedure, FormulationRequirement,
    PhysicsFamily, ProcedureStep, ProcedureStepId, SolutionSpace, StateKind, StateSource, StateUse,
};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
use serde::{Deserialize, Serialize};

use crate::{
    assembly::AssemblySummary,
    contracts::ComputeBackend,
    solve::{
        dynamics::ModalReducedSystem,
        transient::{StructuralTimeStepPolicy, TransientSolveOptions},
    },
};

use super::{
    AnalysisExecutionError, AnalysisExecutionResult, AnalysisWorkKind, AnalysisWorkloadIdentity,
    CanonicalSemanticState, CanonicalStateTransition, ConstraintPlanIdentity,
    DiscretizationIdentity, ExecutionPlanIdentity, NumericalPlanIdentity, OperatorIdentity,
    ProblemIdentity, SequenceIdentity, StateIdentity, StepIdentity,
    StructuralDampingOperatorIdentity, StructuralDofLayoutIdentity,
    StructuralExcitationSetIdentity, StructuralFormulationIdentity, StructuralKinematicsIdentity,
    StructuralMassOperatorIdentity, StructuralStiffnessOperatorIdentity,
    StructuralTransientCheckpointScope, StructuralTransientCoordinateSpace,
    StructuralTransientIntegrator, StructuralTransientPrecision, StructuralTransientSchedule,
    StructuralTransientSolverOptionsIdentity, StudyIdentity,
};

const TRANSIENT_WORKLOAD_PREFIX: &[u8] = b"runmat-structural-transient-workload/v1\0";
const TRANSIENT_WORKLOAD_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
const TRANSIENT_IDENTITY_PREFIX: &[u8] = b"runmat-structural-transient-identity/v1\0";
const TRANSIENT_IDENTITY_LIMITS: CanonicalLimits =
    CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

/// Numerical coordinate space and typed upstream state for structural transient response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", deny_unknown_fields)]
pub enum StructuralTransientSolverPlan {
    Direct,
    ModalReduced {
        basis_transition: Box<CanonicalStateTransition>,
    },
}

impl StructuralTransientSolverPlan {
    pub const fn solution_space(&self) -> SolutionSpace {
        match self {
            Self::Direct => SolutionSpace::Direct,
            Self::ModalReduced { .. } => SolutionSpace::ModalReduced,
        }
    }

    pub const fn basis_state(&self) -> Option<StateIdentity> {
        match self {
            Self::Direct => None,
            Self::ModalReduced { basis_transition } => Some(basis_transition.source_state),
        }
    }

    pub const fn basis_transition(&self) -> Option<&CanonicalStateTransition> {
        match self {
            Self::Direct => None,
            Self::ModalReduced { basis_transition } => Some(basis_transition),
        }
    }
}

/// Portable authored input shared by every temporal block in one structural transient solve.
///
/// The solver mesh remains a separate numerical object so workers can cache and transfer it
/// independently. Block bounds and predecessor state remain in the block request/checkpoint
/// objects; changing either therefore never changes this root workload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTransientWorkload {
    pub schema_version: u16,
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub backend: ComputeBackend,
    pub formulation_requirement: FormulationRequirement,
    pub solver: StructuralTransientSolverPlan,
    pub time_step_policy: StructuralTimeStepPolicy,
    pub residual_target: f64,
}

impl StructuralTransientWorkload {
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        if self.schema_version != 1
            || !self.residual_target.is_finite()
            || self.residual_target <= 0.0
            || self.time_step_policy.interval_count() == 0
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient workload has invalid schema or solver options".into(),
            ));
        }
        if matches!(
            self.solver,
            StructuralTransientSolverPlan::ModalReduced { .. }
        ) && self.backend != ComputeBackend::Cpu
        {
            return Err(AnalysisExecutionError::Invalid(
                "modal transient execution currently requires the CPU backend".into(),
            ));
        }
        validate_model_sequence(&self.model)
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        let step = self.procedure_step()?;
        if step.procedure != AnalysisProcedure::TransientResponse
            || step.solution_space != self.solver.solution_space()
            || step.family != PhysicsFamily::Structural
        {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient workload requires a matching structural transient-response step"
                    .into(),
            ));
        }
        match (&self.solver, step.state_inputs.as_slice()) {
            (StructuralTransientSolverPlan::Direct, []) => {}
            (StructuralTransientSolverPlan::ModalReduced { basis_transition }, [input])
                if input.use_as == StateUse::ModalBasis
                    && input.required_kind == StateKind::ModalBasis
                    && matches!(input.source, StateSource::Upstream { .. }) =>
            {
                basis_transition.validate()?;
                let StateSource::Upstream {
                    producer_step,
                    state_id,
                } = &input.source
                else {
                    unreachable!()
                };
                if basis_transition.producer_step_id != *producer_step
                    || basis_transition.consumer_step_id != step.step_id
                    || basis_transition.state_id != *state_id
                    || basis_transition.method != runmat_analysis_core::StateTransitionMethod::Reuse
                {
                    return Err(AnalysisExecutionError::Identity(
                        "modal transient basis transition differs from its authored state input",
                    ));
                }
            }
            _ => {
                return Err(AnalysisExecutionError::Invalid(
                    "structural transient solver plan differs from its authored state inputs"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    pub fn procedure_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        let mut matches = self
            .model
            .sequence
            .steps
            .iter()
            .filter(|step| step.step_id == self.step_id);
        let step = matches.next().ok_or_else(|| {
            AnalysisExecutionError::Invalid(format!(
                "structural transient workload references absent step `{}`",
                self.step_id.as_str()
            ))
        })?;
        if matches.next().is_some() {
            return Err(AnalysisExecutionError::Invalid(format!(
                "structural transient workload references duplicate step `{}`",
                self.step_id.as_str()
            )));
        }
        Ok(step)
    }

    pub fn solve_options(&self, solver_mesh: SolverMeshArtifact) -> TransientSolveOptions {
        TransientSolveOptions {
            formulation_requirement: self.formulation_requirement,
            time_step_policy: self.time_step_policy,
            residual_target: self.residual_target,
            progress_operation: "fea.solve_structural_transient_block".into(),
            solver_mesh: Some(solver_mesh),
        }
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            TRANSIENT_WORKLOAD_PREFIX,
            "structural-transient-workload",
            self,
            TRANSIENT_WORKLOAD_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let workload: Self = decode_contract(
            TRANSIENT_WORKLOAD_PREFIX,
            "structural-transient-workload",
            bytes,
            TRANSIENT_WORKLOAD_LIMITS,
        )?;
        workload.validate()?;
        Ok(workload)
    }
}

pub(crate) struct StructuralTransientCheckpointScopeInput<'a> {
    pub workload: &'a StructuralTransientWorkload,
    pub solver_mesh: &'a SolverMeshArtifact,
    pub initial_assembly: &'a AssemblySummary,
    pub modal_state: Option<&'a CanonicalSemanticState>,
    pub schedule: StructuralTransientSchedule,
}

impl StructuralTransientCheckpointScope {
    pub(crate) fn from_prepared_workload(
        input: StructuralTransientCheckpointScopeInput<'_>,
    ) -> AnalysisExecutionResult<Self> {
        input.workload.validate()?;
        input
            .solver_mesh
            .validate_canonical()
            .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
        input.schedule.validate()?;
        if input.initial_assembly.dof_count == 0 {
            return Err(AnalysisExecutionError::Invalid(
                "structural transient checkpoint scope requires physical assembled DOFs".into(),
            ));
        }
        let model = &input.workload.model;
        let step_contract = input.workload.procedure_step()?;
        let study = StudyIdentity::from_authored_contract(model, None)?;
        let sequence = SequenceIdentity::from_solver_contract(&model.sequence)?;
        let step = StepIdentity::from_solver_contract(sequence, step_contract)?;
        let mesh_digest = Digest::from_bytes(*input.solver_mesh.canonical_digest.bytes());
        let (coordinate_space, modal_dependency) = match &input.workload.solver {
            StructuralTransientSolverPlan::Direct => {
                if input.modal_state.is_some() {
                    return Err(AnalysisExecutionError::Invalid(
                        "direct structural transient scope received a modal-basis state".into(),
                    ));
                }
                (StructuralTransientCoordinateSpace::Direct, None)
            }
            StructuralTransientSolverPlan::ModalReduced { basis_transition } => {
                let basis_state = basis_transition.source_state;
                let state = input.modal_state.ok_or_else(|| {
                    AnalysisExecutionError::Invalid(
                        "modal transient scope is missing its exact modal-basis state".into(),
                    )
                })?;
                state.validate()?;
                if state.identity != basis_state
                    || state.kind != StateKind::ModalBasis
                    || state.topology_digest != mesh_digest
                    || basis_transition.target_topology_digest != mesh_digest
                {
                    return Err(AnalysisExecutionError::Identity(
                        "modal transient basis state differs from its workload or solver mesh",
                    ));
                }
                let basis = state.payload.modal_basis().ok_or_else(|| {
                    AnalysisExecutionError::Invalid(
                        "modal transient state does not contain a modal basis".into(),
                    )
                })?;
                let reduced = ModalReducedSystem::project(input.initial_assembly, basis)
                    .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?;
                (
                    StructuralTransientCoordinateSpace::ModalReduced {
                        basis_state,
                        basis: basis.identity(),
                        projection: reduced.evidence().clone(),
                        coordinate_count: reduced.mode_count(),
                    },
                    Some([
                        basis_transition.source_state.digest(),
                        basis_transition.identity.digest(),
                    ]),
                )
            }
        };
        let constraint_digest = contract_digest(
            "constraint-plan",
            &(
                &input.initial_assembly.constraint_matrix,
                &input.initial_assembly.affine_constraint_reduction,
                input.initial_assembly.operator.constrained_dofs(),
            ),
        )?;
        let constraint_plan = ConstraintPlanIdentity::new(step, mesh_digest, constraint_digest)?;
        let excitations = StructuralExcitationSetIdentity::from_authored_contract(
            step,
            &(
                &model.signals,
                &model.excitations,
                &model.excitation_cases,
                &input.schedule,
            ),
        )?;
        let model_digest = contract_digest("model", model)?;
        let problem = ProblemIdentity::from_structural_contract(
            step,
            model_digest,
            None,
            constraint_plan,
            excitations,
        )?;
        let formulation_digest = contract_digest(
            "formulation",
            &input.initial_assembly.structural_formulations,
        )?;
        let discretization = DiscretizationIdentity::new(problem, mesh_digest, formulation_digest)?;
        let operators = input.initial_assembly.operator.operators();
        let stiffness_operator = OperatorIdentity::new(
            discretization,
            operators
                .stiffness()
                .numerical_identity()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
        )?;
        let mass_operator = OperatorIdentity::new(
            discretization,
            operators
                .mass()
                .numerical_identity()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
        )?;
        let damping_operator = OperatorIdentity::new(
            discretization,
            operators
                .damping()
                .numerical_identity()
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
        )?;
        let solver_options_digest = contract_digest(
            "solver-options",
            &(
                input.workload.formulation_requirement,
                &input.workload.solver,
                input.workload.time_step_policy,
                input.workload.residual_target,
            ),
        )?;
        let combined_operator = OperatorIdentity::new(
            discretization,
            contract_digest(
                "second-order-operators",
                &(stiffness_operator, mass_operator, damping_operator),
            )?,
        )?;
        let numerical_plan = NumericalPlanIdentity::new(combined_operator, solver_options_digest)?;
        let execution_plan = ExecutionPlanIdentity::new(
            numerical_plan,
            contract_digest("temporal-block-execution", &(1_u16, "accepted-boundary"))?,
        )?;
        let workload_bytes = input.workload.canonical_bytes()?;
        let mut dependencies = vec![mesh_digest];
        dependencies.extend(modal_dependency.into_iter().flatten());
        let source_workload = AnalysisWorkloadIdentity::new(
            study,
            step,
            execution_plan,
            AnalysisWorkKind::SolveStructuralTransientBlock,
            Digest::sha256(&workload_bytes),
            dependencies,
        )?;
        let dof_layout = StructuralDofLayoutIdentity::new(
            discretization,
            contract_digest(
                "structural-dof-layout",
                &input.initial_assembly.structural_dof_layout,
            )?,
        )?;
        let formulation = StructuralFormulationIdentity::new(discretization, formulation_digest)?;
        let kinematics = StructuralKinematicsIdentity::new(
            constraint_plan,
            excitations,
            contract_digest(
                "structural-kinematics",
                &(&model.constraints, &input.schedule),
            )?,
        )?;
        Ok(Self {
            study,
            step,
            execution_plan,
            source_workload,
            numerical_plan,
            discretization,
            constraint_plan,
            stiffness_operator: StructuralStiffnessOperatorIdentity::new(stiffness_operator)?,
            mass_operator: StructuralMassOperatorIdentity::new(mass_operator)?,
            damping_operator: StructuralDampingOperatorIdentity::new(damping_operator)?,
            excitations,
            dof_layout,
            formulation,
            kinematics,
            dof_count: input.initial_assembly.dof_count,
            coordinate_space,
            integrator: StructuralTransientIntegrator::NewmarkAverageAcceleration,
            precision: StructuralTransientPrecision::Real64,
            solver_options: StructuralTransientSolverOptionsIdentity::new(solver_options_digest)?,
            schedule: input.schedule,
        })
    }
}

fn contract_digest<T: Serialize>(
    domain: &'static str,
    value: &T,
) -> AnalysisExecutionResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        TRANSIENT_IDENTITY_PREFIX,
        domain,
        value,
        TRANSIENT_IDENTITY_LIMITS,
    )?))
}

#[cfg(test)]
mod tests {
    use runmat_analysis_core::{
        FormulationRequirement, IndependentDomain, ProcedureStep, ProcedureStepId,
        ReferenceStatePolicy, SequenceDependency, SequenceDependencyReason, StateId,
        StateTransition, StateTransitionMethod, TopologyTransitionPolicy, TransitionId,
    };

    use crate::{
        assembly::{assemble_structural_system, StructuralFormulationRequirement},
        excitation::{realize_step_excitations, ExcitationSamplePoint},
        fixtures::{fixture_model, FixtureId},
        ComputeBackend, ModalSolveOptions, ModalTruncationPolicy,
    };

    use super::*;
    use crate::execution::{
        CanonicalSemanticStateInput, CanonicalSemanticStatePayload, CanonicalStateTransitionInput,
        SolverResultIdentity, StructuralTransientBlockRequest,
        StructuralTransientContinuationApproach, StructuralTransientScheduleArrival,
        StructuralTransientSchedulePoint,
    };

    fn modal_case() -> (
        StructuralTransientWorkload,
        SolverMeshArtifact,
        CanonicalSemanticState,
        AssemblySummary,
        StructuralTransientSchedule,
    ) {
        let mut model = fixture_model(FixtureId::CantileverLinearStatic);
        let modal_step = ProcedureStep::structural_modal("modes");
        let mut transient_step =
            ProcedureStep::structural_modal_transient("response", "modes", "modal_basis");
        transient_step.independent_domain = IndependentDomain::TimeSeconds {
            origin: 0.0,
            start: 0.0,
            end: 1.0,
        };
        model.sequence.steps = vec![modal_step, transient_step];
        model.sequence.dependencies = vec![SequenceDependency {
            predecessor: ProcedureStepId("modes".into()),
            successor: ProcedureStepId("response".into()),
            reason: SequenceDependencyReason::StateTransfer,
        }];
        model.sequence.transitions = vec![StateTransition {
            transition_id: TransitionId("modal-basis-to-transient".into()),
            producer_step: ProcedureStepId("modes".into()),
            consumer_step: ProcedureStepId("response".into()),
            state_id: StateId("modal_basis".into()),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: ReferenceStatePolicy::Preserve,
        }];
        let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
        let modal = crate::run_modal_with_options(
            &model,
            ComputeBackend::Cpu,
            ModalSolveOptions {
                truncation: ModalTruncationPolicy::requested(3),
                solver_mesh: Some(mesh.clone()),
                formulation_requirement: FormulationRequirement::Experimental,
                ..ModalSolveOptions::default()
            },
        )
        .unwrap();
        let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
        let producer =
            StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
        let source_result =
            SolverResultIdentity::from_step_result(producer, modal.basis.identity().digest())
                .unwrap();
        let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
            producer,
            declaration: model.sequence.steps[0].state_outputs[0].clone(),
            topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            source_result,
            payload: CanonicalSemanticStatePayload::ModalBasis {
                basis: Box::new(modal.basis),
            },
        })
        .unwrap();
        let basis_transition = CanonicalStateTransition::new(CanonicalStateTransitionInput {
            sequence,
            producer,
            consumer: StepIdentity::from_solver_contract(sequence, &model.sequence.steps[1])
                .unwrap(),
            authored: &model.sequence.transitions[0],
            source: &state,
            target_topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        })
        .unwrap();
        let workload = StructuralTransientWorkload {
            schema_version: 1,
            model: model.clone(),
            step_id: ProcedureStepId("response".into()),
            backend: ComputeBackend::Cpu,
            formulation_requirement: FormulationRequirement::Experimental,
            solver: StructuralTransientSolverPlan::ModalReduced {
                basis_transition: Box::new(basis_transition),
            },
            time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 2 },
            residual_target: 1.0e-8,
        };
        workload.validate().unwrap();
        let excitations = realize_step_excitations(
            &model,
            &model.sequence.steps[1],
            ExcitationSamplePoint::time(0.0),
        )
        .unwrap();
        let assembly = assemble_structural_system(
            &model,
            &excitations,
            Some(mesh.clone()),
            StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .unwrap();
        let schedule = StructuralTransientSchedule::new(vec![
            StructuralTransientSchedulePoint {
                time_s: 0.0,
                arrival: StructuralTransientScheduleArrival::Initial,
                continuation: StructuralTransientContinuationApproach::At,
            },
            StructuralTransientSchedulePoint {
                time_s: 0.5,
                arrival: StructuralTransientScheduleArrival::Continuous,
                continuation: StructuralTransientContinuationApproach::At,
            },
            StructuralTransientSchedulePoint {
                time_s: 1.0,
                arrival: StructuralTransientScheduleArrival::Continuous,
                continuation: StructuralTransientContinuationApproach::At,
            },
        ])
        .unwrap();
        (workload, mesh, state, assembly, schedule)
    }

    #[test]
    fn modal_transient_scope_binds_exact_basis_projection_and_coordinate_count() {
        let (workload, mesh, state, assembly, schedule) = modal_case();
        let scope = StructuralTransientCheckpointScope::from_prepared_workload(
            StructuralTransientCheckpointScopeInput {
                workload: &workload,
                solver_mesh: &mesh,
                initial_assembly: &assembly,
                modal_state: Some(&state),
                schedule,
            },
        )
        .unwrap();
        let StructuralTransientCoordinateSpace::ModalReduced {
            basis_state,
            basis,
            projection,
            coordinate_count,
        } = &scope.coordinate_space
        else {
            panic!("modal workload produced direct coordinates");
        };
        assert_eq!(*basis_state, state.identity);
        assert_eq!(*basis, state.payload.modal_basis().unwrap().identity());
        assert_eq!(*coordinate_count, projection.mode_count);
        assert_eq!(
            scope.coordinate_space.coordinate_count(scope.dof_count),
            *coordinate_count
        );
        assert!(scope.identity().is_ok());

        let physical_state = crate::execution::StructuralTransientContinuationState {
            displacement: vec![0.0; scope.dof_count],
            velocity: vec![0.0; scope.dof_count],
            acceleration: vec![0.0; scope.dof_count],
        };
        let numerical_state = crate::execution::StructuralTransientNumericalState::ModalReduced {
            basis_state: state.identity,
            coordinates: crate::execution::ModalTransientCoordinateState {
                displacement: vec![0.0; *coordinate_count],
                velocity: vec![0.0; *coordinate_count],
                acceleration: vec![0.0; *coordinate_count],
            },
        };
        let checkpoint = crate::execution::StructuralTransientCheckpoint::new(
            crate::execution::StructuralTransientCheckpointInput {
                scope: scope.clone(),
                block_ordinal: 0,
                continuation: crate::execution::StructuralTransientContinuationPoint {
                    schedule_ordinal: 1,
                    time_s: 0.5,
                    approach: crate::execution::StructuralTransientContinuationApproach::At,
                },
                state: physical_state.clone(),
                numerical_state: numerical_state.clone(),
                balance: crate::execution::StructuralTransientBalanceAccumulator {
                    initial_mechanical_energy_j: 0.0,
                    applied_load_work_j: 0.0,
                    support_work_j: 0.0,
                    damping_dissipation_j: 0.0,
                },
                counters: crate::execution::StructuralTransientSolverCounters {
                    dense_factorization_count: 1,
                    dense_solve_count: 1,
                    ..crate::execution::StructuralTransientSolverCounters::default()
                },
                observation_cursors: Vec::new(),
            },
            None,
        )
        .unwrap();
        assert_eq!(
            crate::execution::StructuralTransientCheckpoint::from_canonical_bytes(
                &checkpoint.canonical_bytes().unwrap()
            )
            .unwrap(),
            checkpoint
        );
        assert!(crate::execution::StructuralTransientCheckpoint::new(
            crate::execution::StructuralTransientCheckpointInput {
                scope,
                block_ordinal: 0,
                continuation: checkpoint.continuation.clone(),
                state: physical_state,
                numerical_state: crate::execution::StructuralTransientNumericalState::Direct,
                balance: checkpoint.balance.clone(),
                counters: checkpoint.counters.clone(),
                observation_cursors: Vec::new(),
            },
            None,
        )
        .is_err());
    }

    #[test]
    fn modal_transient_physical_blocks_restart_from_exact_reduced_coordinates() {
        let (workload, mesh, state, _, _) = modal_case();
        let scope = crate::pipeline::transient::prepare_transient_checkpoint_scope(
            &workload,
            &mesh,
            Some(&state),
        )
        .unwrap();
        let terminal = u64::try_from(scope.schedule.points.len() - 1).unwrap();
        assert!(terminal >= 2);
        let solver_mesh = mesh;
        let run =
            |request: &StructuralTransientBlockRequest,
             predecessor: Option<&crate::execution::StructuralTransientCheckpoint>| {
                crate::pipeline::transient::run_transient_block(
                    crate::pipeline::transient::StructuralTransientRunBlockInput {
                        workload: &workload,
                        solver_mesh: &solver_mesh,
                        request,
                        predecessor,
                        modal_state: Some(&state),
                        observation_cursors: Vec::new(),
                    },
                )
                .unwrap()
            };

        let complete_request =
            StructuralTransientBlockRequest::new(scope.clone(), None, terminal).unwrap();
        let complete = run(&complete_request, None);
        let first_request = StructuralTransientBlockRequest::new(scope.clone(), None, 1).unwrap();
        let first = run(&first_request, None);
        let second_request = StructuralTransientBlockRequest::new(
            scope,
            Some(&first.manifest().terminal_checkpoint),
            terminal,
        )
        .unwrap();
        let second = run(&second_request, Some(&first.manifest().terminal_checkpoint));

        assert_eq!(
            complete.manifest().terminal_checkpoint.state,
            second.manifest().terminal_checkpoint.state
        );
        assert_eq!(
            complete.manifest().terminal_checkpoint.numerical_state,
            second.manifest().terminal_checkpoint.numerical_state
        );
        assert_eq!(
            complete.manifest().terminal_checkpoint.balance,
            second.manifest().terminal_checkpoint.balance
        );
        assert_eq!(
            complete.manifest().terminal_checkpoint.counters,
            second.manifest().terminal_checkpoint.counters
        );
        assert_eq!(
            complete.manifest().solver_evidence.solver_method,
            crate::contracts::SolverMethod::StructuralModalTransientNewmarkDenseFactorization
        );
        assert_eq!(
            complete
                .manifest()
                .solver_evidence
                .counters
                .dense_factorization_count,
            terminal + 1
        );
        assert_eq!(first.samples()[0].physical, complete.samples()[0].physical);
        assert_eq!(
            second.samples().last().unwrap().physical,
            complete.samples().last().unwrap().physical
        );
        assert!(complete.samples().iter().all(|sample| {
            sample.physical.projected_residual_norm <= workload.residual_target
                && sample
                    .physical
                    .dynamic_balance
                    .equilibrium_residual
                    .iter()
                    .all(|value| value.is_finite())
        }));
    }

    #[test]
    fn modal_transient_scope_rejects_missing_or_substituted_basis_state() {
        let (workload, mesh, state, assembly, schedule) = modal_case();
        let input = |modal_state| StructuralTransientCheckpointScopeInput {
            workload: &workload,
            solver_mesh: &mesh,
            initial_assembly: &assembly,
            modal_state,
            schedule: schedule.clone(),
        };
        assert!(StructuralTransientCheckpointScope::from_prepared_workload(input(None)).is_err());
        let mut substituted = state;
        substituted.topology_digest = Digest::sha256(b"other-topology");
        assert!(
            StructuralTransientCheckpointScope::from_prepared_workload(input(Some(&substituted)))
                .is_err()
        );
    }

    #[test]
    fn modal_transient_workload_is_canonical_and_rejects_accelerator_substitution() {
        let (workload, ..) = modal_case();
        assert_eq!(
            StructuralTransientWorkload::from_canonical_bytes(&workload.canonical_bytes().unwrap())
                .unwrap(),
            workload
        );
        let mut accelerated = workload;
        accelerated.backend = ComputeBackend::Gpu;
        assert!(accelerated.validate().is_err());
    }
}

use runmat_analysis_core::{AnalysisModel, AnalysisSequence, ProcedureStep};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{AnalysisExecutionError, AnalysisExecutionResult, SweepDefinition};

const IDENTITY_PREFIX: &[u8] = b"runmat-analysis-identity/v1\0";
const IDENTITY_LIMITS: CanonicalLimits = CanonicalLimits::new(1024 * 1024, 65_536, 65_536, 32);

macro_rules! identity_type {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(Digest);

        impl $name {
            pub const fn digest(self) -> Digest {
                self.0
            }

            pub(super) fn derive<T: Serialize>(
                domain: &'static str,
                input: &T,
            ) -> AnalysisExecutionResult<Self> {
                Ok(Self(derive_digest(domain, input)?))
            }
        }
    };
}

identity_type!(StudyIdentity);
identity_type!(SweepIdentity);
identity_type!(SequenceIdentity);
identity_type!(StepIdentity);
identity_type!(StateIdentity);
identity_type!(StateTransitionIdentity);
identity_type!(ConstraintPlanIdentity);
identity_type!(GeneratorIdentity);
identity_type!(SignalMaterializationIdentity);
identity_type!(SignalRealizationIdentity);
identity_type!(SolverExcitationIdentity);
identity_type!(ProblemIdentity);
identity_type!(DiscretizationIdentity);
identity_type!(OperatorIdentity);
identity_type!(NumericalPlanIdentity);
identity_type!(ExecutionPlanIdentity);
identity_type!(AnalysisWorkloadIdentity);
identity_type!(AnalysisTaskIdentity);
identity_type!(AnalysisAttemptIdentity);
identity_type!(ObservationIdentity);
identity_type!(AccuracyIdentity);
identity_type!(AnalysisResultIdentity);
identity_type!(SolverResultIdentity);
identity_type!(NumericalReceiptIdentity);
identity_type!(FieldRecoveryRequestIdentity);
identity_type!(ObservationEvaluationRequestIdentity);
identity_type!(ObservationPartitionEvaluationRequestIdentity);
identity_type!(ObservationPartitionReductionRequestIdentity);
identity_type!(ObservationAssemblyRequestIdentity);
identity_type!(AccuracyEvaluationRequestIdentity);
identity_type!(SweepResultIdentity);
identity_type!(PublicationIdentity);
identity_type!(PublicationRequestIdentity);
identity_type!(ObservationHistoryCheckpointRequestIdentity);
identity_type!(StructuralTransientScheduleIdentity);
identity_type!(StructuralDofLayoutIdentity);
identity_type!(StructuralFormulationIdentity);
identity_type!(StructuralStiffnessOperatorIdentity);
identity_type!(StructuralMassOperatorIdentity);
identity_type!(StructuralDampingOperatorIdentity);
identity_type!(StructuralExcitationSetIdentity);
identity_type!(StructuralKinematicsIdentity);
identity_type!(StructuralTransientSolverOptionsIdentity);
identity_type!(StructuralTransientCheckpointIdentity);
identity_type!(StructuralTransientCheckpointScopeIdentity);
identity_type!(StructuralTransientBlockRequestIdentity);
identity_type!(StructuralTransientSegmentIdentity);
identity_type!(StructuralTransientSegmentSampleIdentity);
identity_type!(StructuralTransientPhysicalSampleIdentity);
identity_type!(StructuralTransientResultAssemblyRequestIdentity);
identity_type!(StructuralFrequencyScopeIdentity);
identity_type!(StructuralFrequencyProblemIdentity);
identity_type!(StructuralFrequencyWorkloadIdentity);
identity_type!(StructuralFrequencyPhysicalSampleIdentity);
identity_type!(StructuralFrequencyEntryRequestIdentity);
identity_type!(StructuralFrequencyEntryResultIdentity);
identity_type!(StructuralFrequencyResultAssemblyRequestIdentity);
identity_type!(StructuralFrequencyPublicationIdentity);
identity_type!(StructuralModalWorkloadIdentity);
identity_type!(StructuralModalResponseSpectrumWorkloadIdentity);
identity_type!(StructuralModalResponseSpectrumScopeIdentity);
identity_type!(StructuralModalResponseSpectrumDirectionRequestIdentity);
identity_type!(StructuralModalResponseSpectrumDirectionResultIdentity);
identity_type!(StructuralModalResponseSpectrumAssemblyRequestIdentity);
identity_type!(StructuralBucklingWorkloadIdentity);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisWorkKind {
    MaterializeSignal,
    CompileConstraints,
    AssembleOperator,
    Solve,
    ExtractStructuralModes,
    SolveStructuralModalResponseSpectrumDirection,
    AssembleStructuralModalResponseSpectrumResult,
    SolveStructuralBucklingModes,
    RecoverField,
    PublishState,
    CheckpointObservationHistory,
    SolveStructuralTransientBlock,
    AssembleStructuralTransientResult,
    SolveStructuralFrequencyEntry,
    AssembleStructuralFrequencyResult,
    EvaluateObservationPartition,
    ReduceObservationPartitions,
    AssembleObservations,
    EvaluateAccuracy,
    PublishResult,
}

impl StudyIdentity {
    pub fn from_authored_digest(
        authored_study_digest: Digest,
        frozen_project_digest: Option<Digest>,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "authored-study",
            &(authored_study_digest, frozen_project_digest),
        )
    }

    pub fn from_authored_contract(
        model: &AnalysisModel,
        frozen_project_digest: Option<Digest>,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("authored-study", &(model, frozen_project_digest))
    }
}

impl SweepIdentity {
    pub fn new(
        sweep_contract_digest: Digest,
        studies: Vec<StudyIdentity>,
        composition: super::SweepCompositionPolicy,
    ) -> AnalysisExecutionResult<Self> {
        SweepDefinition::new(sweep_contract_digest, studies, composition)?.identity()
    }

    pub(super) fn from_definition(definition: &SweepDefinition) -> AnalysisExecutionResult<Self> {
        Self::derive("sweep", definition)
    }
}

impl SequenceIdentity {
    pub fn new(sequence_digest: Digest) -> AnalysisExecutionResult<Self> {
        Self::derive("solver-sequence", &sequence_digest)
    }

    pub fn from_solver_contract(sequence: &AnalysisSequence) -> AnalysisExecutionResult<Self> {
        Self::new(derive_digest(
            "solver-sequence-contract",
            &sequence.solver_identity_contract(),
        )?)
    }
}

impl StepIdentity {
    pub fn new(sequence: SequenceIdentity, step_digest: Digest) -> AnalysisExecutionResult<Self> {
        Self::derive("step", &(sequence, step_digest))
    }

    pub fn from_solver_contract(
        sequence: SequenceIdentity,
        step: &ProcedureStep,
    ) -> AnalysisExecutionResult<Self> {
        Self::new(
            sequence,
            derive_digest("solver-step", &step.solver_identity_contract())?,
        )
    }
}

impl StateTransitionIdentity {
    pub fn new(
        sequence: SequenceIdentity,
        predecessor: StepIdentity,
        successor: StepIdentity,
        source_state: StateIdentity,
        source_topology_digest: Digest,
        target_topology_digest: Digest,
        transition_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "state-transition",
            &(
                sequence,
                predecessor,
                successor,
                source_state,
                source_topology_digest,
                target_topology_digest,
                transition_digest,
            ),
        )
    }
}

impl StateIdentity {
    pub fn new(
        producer: StepIdentity,
        state_contract_digest: Digest,
        topology_digest: Digest,
        source_result: SolverResultIdentity,
        content_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "semantic-state",
            &(
                producer,
                state_contract_digest,
                topology_digest,
                source_result,
                content_digest,
            ),
        )
    }
}

impl ConstraintPlanIdentity {
    pub fn new(
        step: StepIdentity,
        mesh_digest: Digest,
        constraint_plan_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "constraint-plan",
            &(step, mesh_digest, constraint_plan_digest),
        )
    }
}

impl GeneratorIdentity {
    pub fn new(
        callable_digest: Digest,
        capture_digest: Digest,
        generator_contract_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "generator",
            &(callable_digest, capture_digest, generator_contract_digest),
        )
    }
}

impl SignalMaterializationIdentity {
    pub fn new(
        authored_signal_digest: Digest,
        generator: Option<GeneratorIdentity>,
        realization_policy_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "signal-materialization",
            &(authored_signal_digest, generator, realization_policy_digest),
        )
    }
}

impl SignalRealizationIdentity {
    pub fn new(
        materialization: SignalMaterializationIdentity,
        realized_content_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "signal-realization",
            &(materialization, realized_content_digest),
        )
    }
}

impl SolverExcitationIdentity {
    pub fn new(
        step: StepIdentity,
        signal: SignalRealizationIdentity,
        target_digest: Digest,
        projection_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "solver-excitation",
            &(step, signal, target_digest, projection_digest),
        )
    }
}

impl ProblemIdentity {
    pub fn new(
        step: StepIdentity,
        model_digest: Digest,
        input_state: Option<StateTransitionIdentity>,
        constraint_plan: ConstraintPlanIdentity,
        mut excitations: Vec<SolverExcitationIdentity>,
    ) -> AnalysisExecutionResult<Self> {
        excitations.sort_unstable();
        let count = excitations.len();
        excitations.dedup();
        if excitations.len() != count {
            return Err(AnalysisExecutionError::Invalid(
                "problem contains duplicate solver excitations".into(),
            ));
        }
        Self::derive(
            "problem",
            &(
                step,
                model_digest,
                input_state,
                constraint_plan,
                excitations,
            ),
        )
    }

    pub fn from_structural_contract(
        step: StepIdentity,
        model_digest: Digest,
        input_state: Option<StateTransitionIdentity>,
        constraint_plan: ConstraintPlanIdentity,
        excitations: StructuralExcitationSetIdentity,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "structural-problem-contract",
            &(
                step,
                model_digest,
                input_state,
                constraint_plan,
                excitations,
            ),
        )
    }
}

impl DiscretizationIdentity {
    pub fn new(
        problem: ProblemIdentity,
        mesh_digest: Digest,
        formulation_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "discretization",
            &(problem, mesh_digest, formulation_digest),
        )
    }
}

impl StructuralDofLayoutIdentity {
    pub fn new(
        discretization: DiscretizationIdentity,
        layout_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("structural-dof-layout", &(discretization, layout_digest))
    }
}

impl StructuralFormulationIdentity {
    pub fn new(
        discretization: DiscretizationIdentity,
        realization_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "structural-formulation-realization",
            &(discretization, realization_digest),
        )
    }
}

impl StructuralStiffnessOperatorIdentity {
    pub fn new(operator: OperatorIdentity) -> AnalysisExecutionResult<Self> {
        Self::derive("structural-stiffness-operator", &operator)
    }
}

impl StructuralMassOperatorIdentity {
    pub fn new(operator: OperatorIdentity) -> AnalysisExecutionResult<Self> {
        Self::derive("structural-mass-operator", &operator)
    }
}

impl StructuralDampingOperatorIdentity {
    pub fn new(operator: OperatorIdentity) -> AnalysisExecutionResult<Self> {
        Self::derive("structural-damping-operator", &operator)
    }
}

impl StructuralExcitationSetIdentity {
    pub fn new(mut excitations: Vec<SolverExcitationIdentity>) -> AnalysisExecutionResult<Self> {
        excitations.sort_unstable();
        let original_count = excitations.len();
        excitations.dedup();
        if excitations.len() != original_count {
            return Err(AnalysisExecutionError::Invalid(
                "structural excitation set contains duplicate identities".into(),
            ));
        }
        Self::derive("structural-excitation-set", &excitations)
    }

    pub fn from_authored_contract<T: Serialize>(
        step: StepIdentity,
        contract: &T,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("structural-excitation-set-contract", &(step, contract))
    }
}

impl StructuralKinematicsIdentity {
    pub fn new(
        constraint_plan: ConstraintPlanIdentity,
        excitations: StructuralExcitationSetIdentity,
        kinematics_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "structural-kinematics",
            &(constraint_plan, excitations, kinematics_digest),
        )
    }
}

impl StructuralTransientSolverOptionsIdentity {
    pub fn new(options_digest: Digest) -> AnalysisExecutionResult<Self> {
        Self::derive("structural-transient-solver-options", &options_digest)
    }
}

impl StructuralFrequencyProblemIdentity {
    pub(crate) fn from_physical_digest(digest: Digest) -> AnalysisExecutionResult<Self> {
        Self::derive("structural-frequency-physical-problem", &digest)
    }
}

impl OperatorIdentity {
    pub fn new(
        discretization: DiscretizationIdentity,
        operator_contract_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("operator", &(discretization, operator_contract_digest))
    }
}

impl AccuracyIdentity {
    pub fn new(step: StepIdentity, accuracy_plan_digest: Digest) -> AnalysisExecutionResult<Self> {
        Self::derive("accuracy", &(step, accuracy_plan_digest))
    }
}

impl NumericalPlanIdentity {
    pub fn new(
        operator: OperatorIdentity,
        solver_contract_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("numerical-plan", &(operator, solver_contract_digest))
    }
}

impl ExecutionPlanIdentity {
    /// Identifies portable execution semantics. Pool, worker, and concrete device placement must
    /// not be included in `execution_contract_digest`.
    pub fn new(
        numerical_plan: NumericalPlanIdentity,
        execution_contract_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "execution-plan",
            &(numerical_plan, execution_contract_digest),
        )
    }
}

impl AnalysisWorkloadIdentity {
    pub fn new(
        study: StudyIdentity,
        step: StepIdentity,
        execution_plan: ExecutionPlanIdentity,
        kind: AnalysisWorkKind,
        work_contract_digest: Digest,
        input_digests: Vec<Digest>,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "workload",
            &(
                study,
                step,
                execution_plan,
                kind,
                work_contract_digest,
                input_digests,
            ),
        )
    }

    #[cfg(test)]
    pub(crate) fn for_test(label: &str) -> AnalysisExecutionResult<Self> {
        Self::derive("test-workload", &label)
    }
}

impl AnalysisTaskIdentity {
    pub fn new(
        workload: AnalysisWorkloadIdentity,
        mut dependencies: Vec<AnalysisTaskIdentity>,
    ) -> AnalysisExecutionResult<Self> {
        dependencies.sort_unstable();
        let count = dependencies.len();
        dependencies.dedup();
        if dependencies.len() != count {
            return Err(AnalysisExecutionError::Invalid(
                "task contains duplicate dependencies".into(),
            ));
        }
        Self::derive("task", &(workload, dependencies))
    }
}

impl AnalysisAttemptIdentity {
    pub fn new(
        task: AnalysisTaskIdentity,
        attempt_number: u16,
        execution_environment_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "attempt",
            &(task, attempt_number, execution_environment_digest),
        )
    }
}

impl ObservationIdentity {
    pub fn new(
        step: StepIdentity,
        observation_contract_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("observation", &(step, observation_contract_digest))
    }
}

impl AnalysisResultIdentity {
    pub fn new(
        workload: AnalysisWorkloadIdentity,
        output_digests: Vec<Digest>,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("result", &(workload, output_digests))
    }
}

impl SolverResultIdentity {
    pub fn new(
        workload: AnalysisWorkloadIdentity,
        content_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("solver-result", &(workload, content_digest))
    }

    /// Identifies an in-process solver result when no portable scheduler workload exists.
    pub fn from_step_result(
        step: StepIdentity,
        content_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("in-process-solver-result", &(step, content_digest))
    }

    pub(super) fn from_structural_frequency_response(
        scope: StructuralFrequencyScopeIdentity,
        content_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "structural-frequency-physical-result",
            &(scope, content_digest),
        )
    }
}

impl NumericalReceiptIdentity {
    pub(super) fn new(
        source_result: SolverResultIdentity,
        receipt_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("numerical-receipt", &(source_result, receipt_digest))
    }
}

impl PublicationIdentity {
    pub fn new(
        request: PublicationRequestIdentity,
        source_result: SolverResultIdentity,
        observations: Option<Digest>,
        accuracy: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "publication",
            &(request, source_result, observations, accuracy),
        )
    }
}

fn derive_digest<T: Serialize>(domain: &'static str, input: &T) -> AnalysisExecutionResult<Digest> {
    Ok(Digest::sha256(encode_contract(
        IDENTITY_PREFIX,
        domain,
        input,
        IDENTITY_LIMITS,
    )?))
}

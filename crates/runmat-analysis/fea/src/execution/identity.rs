use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use serde::{Deserialize, Serialize};

use super::{AnalysisExecutionError, AnalysisExecutionResult};

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

            fn derive<T: Serialize>(
                domain: &'static str,
                input: &T,
            ) -> AnalysisExecutionResult<Self> {
                Ok(Self(derive_digest(domain, input)?))
            }
        }
    };
}

identity_type!(StudyIdentity);
identity_type!(SequenceIdentity);
identity_type!(StepIdentity);
identity_type!(StateTransitionIdentity);
identity_type!(ConstraintPlanIdentity);
identity_type!(GeneratorIdentity);
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
identity_type!(PublicationIdentity);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisWorkKind {
    MaterializeSignal,
    CompileConstraints,
    AssembleOperator,
    Solve,
    RecoverField,
    EvaluateObservation,
    EvaluateAccuracy,
    PublishResult,
}

impl StudyIdentity {
    pub fn new(
        authored_study_digest: Digest,
        frozen_project_digest: Option<Digest>,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("study", &(authored_study_digest, frozen_project_digest))
    }
}

impl SequenceIdentity {
    pub fn new(study: StudyIdentity, sequence_digest: Digest) -> AnalysisExecutionResult<Self> {
        Self::derive("sequence", &(study, sequence_digest))
    }
}

impl StepIdentity {
    pub fn new(sequence: SequenceIdentity, step_digest: Digest) -> AnalysisExecutionResult<Self> {
        Self::derive("step", &(sequence, step_digest))
    }
}

impl StateTransitionIdentity {
    pub fn new(
        sequence: SequenceIdentity,
        predecessor: StepIdentity,
        successor: StepIdentity,
        transition_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "state-transition",
            &(sequence, predecessor, successor, transition_digest),
        )
    }
}

impl ConstraintPlanIdentity {
    pub fn new(
        step: StepIdentity,
        mesh_digest: Digest,
        authored_constraints_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "constraint-plan",
            &(step, mesh_digest, authored_constraints_digest),
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

impl SignalRealizationIdentity {
    pub fn new(
        authored_signal_digest: Digest,
        generator: Option<GeneratorIdentity>,
        sampling_policy_digest: Digest,
        realized_content_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "signal-realization",
            &(
                authored_signal_digest,
                generator,
                sampling_policy_digest,
                realized_content_digest,
            ),
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
        accuracy: AccuracyIdentity,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "numerical-plan",
            &(operator, solver_contract_digest, accuracy),
        )
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
        execution_plan: ExecutionPlanIdentity,
        kind: AnalysisWorkKind,
        work_contract_digest: Digest,
        input_digests: Vec<Digest>,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive(
            "workload",
            &(
                study,
                execution_plan,
                kind,
                work_contract_digest,
                input_digests,
            ),
        )
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

impl PublicationIdentity {
    pub fn new(
        result: AnalysisResultIdentity,
        accuracy: AccuracyIdentity,
        evidence_digest: Digest,
    ) -> AnalysisExecutionResult<Self> {
        Self::derive("publication", &(result, accuracy, evidence_digest))
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

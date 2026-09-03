use serde::{Deserialize, Serialize};

use super::{
    accuracy::AccuracyPlan, output::ObservationRequest, AnalysisProcedure, CoupledPhysicsFamily,
    PhysicsFamily, SolutionSpace,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProcedureStepId(pub String);

impl ProcedureStepId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StateId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TransitionId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SequenceBranchId(pub String);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisSequence {
    pub sequence_id: String,
    pub steps: Vec<ProcedureStep>,
    pub dependencies: Vec<SequenceDependency>,
    pub transitions: Vec<StateTransition>,
    pub branches: Vec<SequenceBranch>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureStep {
    pub step_id: ProcedureStepId,
    pub family: PhysicsFamily,
    pub procedure: AnalysisProcedure,
    pub solution_space: SolutionSpace,
    pub independent_domain: IndependentDomain,
    pub activations: Vec<StepObjectActivation>,
    pub state_inputs: Vec<StateInput>,
    pub state_outputs: Vec<StateOutput>,
    pub observations: Vec<ObservationRequest>,
    pub accuracy: AccuracyPlan,
    pub restart: RestartPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndependentDomain {
    Static,
    TimeSeconds {
        origin: f64,
        start: f64,
        end: f64,
    },
    FrequencyHertz(Vec<f64>),
    LoadFactor {
        start: f64,
        end: f64,
    },
    Eigenvalue {
        requested_count: usize,
    },
    StudyParameter {
        parameter_id: String,
        canonical_values: Vec<f64>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceDependency {
    pub predecessor: ProcedureStepId,
    pub successor: ProcedureStepId,
    pub reason: SequenceDependencyReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SequenceDependencyReason {
    StateTransfer,
    AuthoredOrdering,
    SharedInterface,
    Adaptation,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepObjectRef {
    Constraint(String),
    Boundary(String),
    Connector(String),
    Interface(String),
    Excitation(String),
    InitialCondition(String),
    MaterialParameter(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepObjectActivation {
    pub object: StepObjectRef,
    pub lifecycle: ActivationLifecycle,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationLifecycle {
    Activate,
    Continue {
        from_step: ProcedureStepId,
    },
    Modify {
        prior_step: ProcedureStepId,
        prior_object: StepObjectRef,
    },
    Deactivate {
        from_step: ProcedureStepId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateInput {
    pub input_id: String,
    pub required_kind: StateKind,
    pub source: StateSource,
    pub use_as: StateUse,
    pub conflict: InitialStateConflictPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateSource {
    InitialConditions,
    Upstream {
        producer_step: ProcedureStepId,
        state_id: StateId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateUse {
    PhysicalInitialState,
    LinearizationPoint,
    Prestress,
    InitialGuess,
    ModalBasis,
    CoupledInterface,
    Restart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitialStateConflictPolicy {
    Reject,
    ReplaceInitialCondition,
    AugmentDisjointQuantities,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateOutput {
    pub state_id: StateId,
    pub kind: StateKind,
    pub completeness: StateCompleteness,
    pub persistence: StatePersistence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateKind {
    StructuralPrimary,
    StructuralKinematics,
    StructuralStressStrain,
    GeometricStiffness,
    MaterialHistory,
    ContactHistory,
    Thermal,
    ElectromagneticStatic,
    ElectromagneticComplex,
    AcousticComplex,
    Fluid,
    MovingMesh,
    ModalBasis,
    CoupledInterface,
    Integrator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateCompleteness {
    Complete,
    Checkpoint,
    Partial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatePersistence {
    Canonical,
    UntilDependentsComplete,
    CheckpointOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateTransition {
    pub transition_id: TransitionId,
    pub producer_step: ProcedureStepId,
    pub consumer_step: ProcedureStepId,
    pub state_id: StateId,
    pub method: StateTransitionMethod,
    pub topology: TopologyTransitionPolicy,
    pub reference: ReferenceStatePolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateTransitionMethod {
    Reuse,
    ConservativeProjection,
    Interpolation,
    Linearize,
    DeriveOperatorContribution,
    InitialGuessOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopologyTransitionPolicy {
    RequireIdentical,
    ProjectWithEvidence,
    AdaptedMeshLineage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceStatePolicy {
    Preserve,
    ReceivingStepOrigin,
    ExplicitLinearizationPoint,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceBranch {
    pub branch_id: SequenceBranchId,
    pub producer_step: ProcedureStepId,
    pub state_ids: Vec<StateId>,
    pub consumer_steps: Vec<ProcedureStepId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RestartPolicy {
    Fresh,
    ContinueFromCompletedState {
        producer_step: ProcedureStepId,
        state_id: StateId,
    },
    ResumeCheckpoint {
        checkpoint_id: String,
        required_state_digest: String,
    },
}

impl ProcedureStep {
    pub fn experimental(
        step_id: impl Into<String>,
        family: PhysicsFamily,
        procedure: AnalysisProcedure,
        solution_space: SolutionSpace,
        independent_domain: IndependentDomain,
        state_outputs: Vec<StateOutput>,
    ) -> Self {
        let step_id = step_id.into();
        Self {
            accuracy: AccuracyPlan::experimental_baseline(format!("{step_id}.accuracy")),
            step_id: ProcedureStepId(step_id),
            family,
            procedure,
            solution_space,
            independent_domain,
            activations: Vec::new(),
            state_inputs: Vec::new(),
            state_outputs,
            observations: Vec::new(),
            restart: RestartPolicy::Fresh,
        }
    }

    pub fn structural_static(step_id: impl Into<String>) -> Self {
        Self::experimental(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::Static,
            SolutionSpace::Direct,
            IndependentDomain::Static,
            vec![state("structural_primary", StateKind::StructuralPrimary)],
        )
    }

    pub fn structural_modal(step_id: impl Into<String>) -> Self {
        Self::experimental(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::ModalExtraction,
            SolutionSpace::Direct,
            IndependentDomain::Eigenvalue { requested_count: 6 },
            vec![state("modal_basis", StateKind::ModalBasis)],
        )
    }

    pub fn structural_transient(step_id: impl Into<String>) -> Self {
        Self::experimental(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
            IndependentDomain::TimeSeconds {
                origin: 0.0,
                start: 0.0,
                end: 1.0,
            },
            vec![
                state("structural_primary", StateKind::StructuralPrimary),
                state("structural_kinematics", StateKind::StructuralKinematics),
                state("integrator", StateKind::Integrator),
            ],
        )
    }

    pub fn thermal_static(step_id: impl Into<String>) -> Self {
        Self::experimental(
            step_id,
            PhysicsFamily::Thermal,
            AnalysisProcedure::Static,
            SolutionSpace::Direct,
            IndependentDomain::Static,
            vec![state("thermal", StateKind::Thermal)],
        )
    }

    pub fn structural_nonlinear(step_id: impl Into<String>) -> Self {
        Self::experimental(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::Nonlinear,
            SolutionSpace::Direct,
            IndependentDomain::LoadFactor {
                start: 0.0,
                end: 1.0,
            },
            vec![
                state("structural_primary", StateKind::StructuralPrimary),
                state("stress_strain", StateKind::StructuralStressStrain),
                state("material_history", StateKind::MaterialHistory),
            ],
        )
    }

    pub fn electromagnetic_static(step_id: impl Into<String>) -> Self {
        Self::experimental(
            step_id,
            PhysicsFamily::Electromagnetic,
            AnalysisProcedure::Static,
            SolutionSpace::Direct,
            IndependentDomain::Static,
            vec![state("electromagnetic", StateKind::ElectromagneticStatic)],
        )
    }

    pub fn fluid_static(step_id: impl Into<String>) -> Self {
        Self::experimental(
            step_id,
            PhysicsFamily::Fluid,
            AnalysisProcedure::Static,
            SolutionSpace::Direct,
            IndependentDomain::Static,
            vec![state("fluid", StateKind::Fluid)],
        )
    }

    pub fn acoustic_harmonic(step_id: impl Into<String>) -> Self {
        Self::experimental(
            step_id,
            PhysicsFamily::Acoustic,
            AnalysisProcedure::HarmonicResponse,
            SolutionSpace::Direct,
            IndependentDomain::FrequencyHertz(vec![1_000.0]),
            vec![state("acoustic_complex", StateKind::AcousticComplex)],
        )
    }

    pub fn fluid_transient(step_id: impl Into<String>) -> Self {
        Self::experimental(
            step_id,
            PhysicsFamily::Fluid,
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
            IndependentDomain::TimeSeconds {
                origin: 0.0,
                start: 0.0,
                end: 1.0,
            },
            vec![state("fluid", StateKind::Fluid)],
        )
    }

    pub fn coupled(
        step_id: impl Into<String>,
        family: CoupledPhysicsFamily,
        procedure: AnalysisProcedure,
    ) -> Self {
        let domain = match procedure {
            AnalysisProcedure::TransientResponse => IndependentDomain::TimeSeconds {
                origin: 0.0,
                start: 0.0,
                end: 1.0,
            },
            AnalysisProcedure::Nonlinear => IndependentDomain::LoadFactor {
                start: 0.0,
                end: 1.0,
            },
            _ => IndependentDomain::Static,
        };
        Self::experimental(
            step_id,
            PhysicsFamily::Coupled(family),
            procedure,
            SolutionSpace::Direct,
            domain,
            vec![state("coupled_interface", StateKind::CoupledInterface)],
        )
    }
}

impl AnalysisSequence {
    pub fn from_steps(sequence_id: impl Into<String>, steps: Vec<ProcedureStep>) -> Self {
        let dependencies = steps
            .windows(2)
            .map(|pair| SequenceDependency {
                predecessor: pair[0].step_id.clone(),
                successor: pair[1].step_id.clone(),
                reason: SequenceDependencyReason::AuthoredOrdering,
            })
            .collect();
        Self {
            sequence_id: sequence_id.into(),
            steps,
            dependencies,
            transitions: Vec::new(),
            branches: Vec::new(),
        }
    }
}

fn state(id: &str, kind: StateKind) -> StateOutput {
    StateOutput {
        state_id: StateId(id.to_string()),
        kind,
        completeness: StateCompleteness::Complete,
        persistence: StatePersistence::UntilDependentsComplete,
    }
}

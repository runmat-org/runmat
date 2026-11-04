use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ResponseSpectrumId;

mod activation;
mod electrical;
pub use activation::{
    ActivationLifecycle, ExcitationCaseSelection, StepObjectActivation, StepObjectRef,
};

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

/// An owned analysis sequence whose semantic graph has passed all model validation.
///
/// The execution layer consumes this value instead of reconstructing procedure
/// ordering from generic scheduler edges. Its fields are private so callers cannot
/// manufacture a value that bypasses the analysis validation boundary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ValidatedAnalysisSequence {
    sequence: AnalysisSequence,
    topological_order: Vec<ProcedureStepId>,
    parallel_levels: Vec<Vec<ProcedureStepId>>,
    direct_predecessors: BTreeMap<ProcedureStepId, Vec<ProcedureStepId>>,
}

impl ValidatedAnalysisSequence {
    pub(crate) fn new(
        sequence: AnalysisSequence,
        topological_order: Vec<ProcedureStepId>,
        parallel_levels: Vec<Vec<ProcedureStepId>>,
        direct_predecessors: BTreeMap<ProcedureStepId, Vec<ProcedureStepId>>,
    ) -> Self {
        Self {
            sequence,
            topological_order,
            parallel_levels,
            direct_predecessors,
        }
    }

    pub fn sequence(&self) -> &AnalysisSequence {
        &self.sequence
    }

    pub fn topological_order(&self) -> &[ProcedureStepId] {
        &self.topological_order
    }

    pub fn parallel_levels(&self) -> &[Vec<ProcedureStepId>] {
        &self.parallel_levels
    }

    pub fn direct_predecessors(&self) -> &BTreeMap<ProcedureStepId, Vec<ProcedureStepId>> {
        &self.direct_predecessors
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcedureStep {
    pub step_id: ProcedureStepId,
    pub family: PhysicsFamily,
    pub procedure: AnalysisProcedure,
    pub solution_space: SolutionSpace,
    pub independent_domain: IndependentDomain,
    pub settings: ProcedureSettings,
    pub excitation_case_selection: ExcitationCaseSelection,
    pub activations: Vec<StepObjectActivation>,
    pub state_inputs: Vec<StateInput>,
    pub state_outputs: Vec<StateOutput>,
    pub observations: Vec<ObservationRequest>,
    pub accuracy: AccuracyPlan,
    pub restart: RestartPolicy,
}

/// Procedure-specific analysis inputs that are neither an independent domain nor an execution
/// policy. Keeping this union closed prevents procedure controls from degrading into string-keyed
/// solver options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcedureSettings {
    None,
    StructuralRandomResponse(StructuralRandomResponseSettings),
    StructuralNonlinear(StructuralNonlinearSettings),
    /// Static-correction (residual) vectors a structural modal extraction appends to its basis.
    StructuralModalBasis(StructuralModalBasisSettings),
    /// Enrichment of the reduced space a modal-reduced frequency sweep solves in.
    StructuralModalFrequency(StructuralModalFrequencySettings),
}

/// Inertia residual vectors of a structural modal basis: for each global direction, the static
/// response to a unit acceleration load in that direction with the retained modes removed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalBasisSettings {
    pub residual_vector_directions: Vec<crate::SpectrumDirection>,
}

/// Static correction of a modal-reduced frequency sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalStaticCorrection {
    /// Solve in the retained modes (and any residual vectors of the basis) only.
    None,
    /// Also span the static response to the sweep's applied load, so the reduced space contains
    /// the exact quasi-static solution and the modal truncation error falls with frequency.
    AppliedLoads,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralModalFrequencySettings {
    pub static_correction: ModalStaticCorrection,
}

/// Kinematic description of a structural nonlinear step. Small strain keeps the reference
/// configuration for strain and equilibrium; large deformation is the total Lagrangian form
/// (Green-Lagrange strain, second Piola-Kirchhoff stress, consistent material plus geometric
/// tangent) for large displacement and large rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralKinematics {
    SmallStrain,
    LargeDeformation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralNonlinearSettings {
    pub kinematics: StructuralKinematics,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralRandomResponseSettings {
    pub modal_damping_ratio: f64,
    pub minimum_cumulative_mass_ratio: f64,
}

impl ProcedureStep {
    /// Residual-vector directions an authored structural modal extraction requests.
    pub fn modal_residual_vector_directions(&self) -> &[crate::SpectrumDirection] {
        match &self.settings {
            ProcedureSettings::StructuralModalBasis(settings) => {
                &settings.residual_vector_directions
            }
            _ => &[],
        }
    }

    /// Static correction of a modal-reduced frequency step; every other step has none.
    pub fn modal_static_correction(&self) -> ModalStaticCorrection {
        match self.settings {
            ProcedureSettings::StructuralModalFrequency(settings) => settings.static_correction,
            _ => ModalStaticCorrection::None,
        }
    }

    /// Kinematics of a structural nonlinear step; every other step is small strain.
    pub fn structural_kinematics(&self) -> StructuralKinematics {
        match self.settings {
            ProcedureSettings::StructuralNonlinear(settings) => settings.kinematics,
            _ => StructuralKinematics::SmallStrain,
        }
    }

    /// Resolves whether an authored model object participates in this step.
    ///
    /// Empty activation inventories retain the concise single-step convention in which authored
    /// objects are active. Once a step declares activations, the inventory is closed and a
    /// deactivation explicitly removes its object.
    pub fn object_is_active(&self, object: &StepObjectRef) -> bool {
        self.activations.is_empty()
            || self.activations.iter().any(|activation| {
                &activation.object == object
                    && !matches!(activation.lifecycle, ActivationLifecycle::Deactivate { .. })
            })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndependentDomain {
    Static,
    TimeSeconds {
        origin: f64,
        start: f64,
        end: f64,
        #[serde(with = "runmat_types::wire_integer")]
        interval_count: usize,
    },
    FrequencyHertz(Vec<f64>),
    LoadFactor {
        start: f64,
        end: f64,
    },
    Eigenvalue {
        #[serde(with = "runmat_types::wire_integer")]
        requested_count: usize,
    },
    StudyParameter {
        parameter_id: String,
        canonical_values: Vec<f64>,
    },
    ResponseSpectra {
        spectrum_ids: Vec<ResponseSpectrumId>,
    },
    RandomExcitation {
        excitation_id: crate::RandomExcitationId,
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
    /// An upstream electromagnetic state whose (cycle-averaged) Joule loss density is the
    /// consumer's volumetric heat source.
    HeatSource,
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
    StructuralComplex,
    StructuralResponseEnvelope,
    StructuralRandomStatistics,
    GeometricStiffness,
    MaterialHistory,
    ContactHistory,
    Thermal,
    ElectromagneticStatic,
    ElectromagneticComplex,
    AcousticComplex,
    /// Cavity pressure mode shapes and their natural frequencies.
    AcousticModes,
    /// Real time-domain pressure field (and its rate) at the end of a transient.
    AcousticPressure,
    Fluid,
    MovingMesh,
    ModalBasis,
    BucklingModes,
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
    pub fn new(
        step_id: impl Into<String>,
        family: PhysicsFamily,
        procedure: AnalysisProcedure,
        solution_space: SolutionSpace,
        independent_domain: IndependentDomain,
        state_outputs: Vec<StateOutput>,
    ) -> Self {
        let step_id = step_id.into();
        Self {
            accuracy: AccuracyPlan::baseline(format!("{step_id}.accuracy")),
            step_id: ProcedureStepId(step_id),
            family,
            procedure,
            solution_space,
            independent_domain,
            settings: ProcedureSettings::None,
            excitation_case_selection: ExcitationCaseSelection::All,
            activations: Vec::new(),
            state_inputs: Vec::new(),
            state_outputs,
            observations: Vec::new(),
            restart: RestartPolicy::Fresh,
        }
    }

    pub fn structural_static(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::Static,
            SolutionSpace::Direct,
            IndependentDomain::Static,
            vec![state("structural_primary", StateKind::StructuralPrimary)],
        )
    }

    pub fn structural_preload(step_id: impl Into<String>) -> Self {
        let mut step = Self::structural_static(step_id);
        step.state_outputs
            .push(state("geometric_stiffness", StateKind::GeometricStiffness));
        step
    }

    pub fn thermomechanical_preload(
        step_id: impl Into<String>,
        thermal_producer: impl Into<String>,
        state_id: impl Into<String>,
    ) -> Self {
        let mut step = Self::structural_preload(step_id);
        step.family = PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical);
        let mut input = prestress_input(thermal_producer, state_id);
        input.required_kind = StateKind::Thermal;
        step.state_inputs.push(input);
        step
    }

    pub fn structural_modal(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::ModalExtraction,
            SolutionSpace::Direct,
            IndependentDomain::Eigenvalue { requested_count: 6 },
            vec![state("modal_basis", StateKind::ModalBasis)],
        )
    }

    pub fn structural_buckling(
        step_id: impl Into<String>,
        producer_step: impl Into<String>,
        state_id: impl Into<String>,
    ) -> Self {
        let mut step = Self::new(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::EigenvalueBuckling,
            SolutionSpace::Direct,
            IndependentDomain::Eigenvalue { requested_count: 6 },
            vec![state("buckling_modes", StateKind::BucklingModes)],
        );
        step.state_inputs
            .push(prestress_input(producer_step, state_id));
        step
    }

    pub fn structural_prestressed_modal(
        step_id: impl Into<String>,
        producer_step: impl Into<String>,
        state_id: impl Into<String>,
    ) -> Self {
        let mut step = Self::structural_modal(step_id);
        step.procedure = AnalysisProcedure::PrestressedModalExtraction;
        step.state_inputs
            .push(prestress_input(producer_step, state_id));
        step
    }

    pub fn structural_modal_response_spectrum(
        step_id: impl Into<String>,
        spectrum_ids: Vec<ResponseSpectrumId>,
        producer_step: impl Into<String>,
        state_id: impl Into<String>,
    ) -> Self {
        let mut step = Self::new(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::ModalResponseSpectrum,
            SolutionSpace::ModalReduced,
            IndependentDomain::ResponseSpectra { spectrum_ids },
            vec![state(
                "response_envelope",
                StateKind::StructuralResponseEnvelope,
            )],
        );
        step.state_inputs
            .push(modal_basis_input(producer_step, state_id));
        step.accuracy =
            AccuracyPlan::modal_response_spectrum(format!("{}.accuracy", step.step_id.0));
        step
    }

    pub fn structural_random_response(
        step_id: impl Into<String>,
        excitation_id: crate::RandomExcitationId,
        settings: StructuralRandomResponseSettings,
        producer_step: impl Into<String>,
        state_id: impl Into<String>,
    ) -> Self {
        let mut step = Self::new(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::RandomResponse,
            SolutionSpace::ModalReduced,
            IndependentDomain::RandomExcitation { excitation_id },
            vec![state(
                "random_statistics",
                StateKind::StructuralRandomStatistics,
            )],
        );
        step.state_inputs
            .push(modal_basis_input(producer_step, state_id));
        step.settings = ProcedureSettings::StructuralRandomResponse(settings);
        step.accuracy = AccuracyPlan::random_response(format!("{}.accuracy", step.step_id.0));
        step
    }

    pub fn structural_transient(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
            IndependentDomain::TimeSeconds {
                origin: 0.0,
                start: 0.0,
                end: 1.0,
                interval_count: 10,
            },
            vec![
                state("structural_primary", StateKind::StructuralPrimary),
                state("structural_kinematics", StateKind::StructuralKinematics),
                state("integrator", StateKind::Integrator),
            ],
        )
    }

    pub fn structural_harmonic(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Structural,
            AnalysisProcedure::HarmonicResponse,
            SolutionSpace::Direct,
            IndependentDomain::FrequencyHertz(vec![60.0]),
            vec![state("structural_complex", StateKind::StructuralComplex)],
        )
    }

    pub fn structural_prestressed_harmonic(
        step_id: impl Into<String>,
        producer_step: impl Into<String>,
        state_id: impl Into<String>,
    ) -> Self {
        let mut step = Self::structural_harmonic(step_id);
        step.procedure = AnalysisProcedure::PrestressedHarmonicResponse;
        step.state_inputs
            .push(prestress_input(producer_step, state_id));
        step
    }

    pub fn structural_modal_transient(
        step_id: impl Into<String>,
        producer_step: impl Into<String>,
        state_id: impl Into<String>,
    ) -> Self {
        let mut step = Self::structural_transient(step_id);
        step.solution_space = SolutionSpace::ModalReduced;
        step.state_inputs
            .push(modal_basis_input(producer_step, state_id));
        step
    }

    pub fn structural_modal_harmonic(
        step_id: impl Into<String>,
        producer_step: impl Into<String>,
        state_id: impl Into<String>,
    ) -> Self {
        let mut step = Self::structural_harmonic(step_id);
        step.solution_space = SolutionSpace::ModalReduced;
        step.state_inputs
            .push(modal_basis_input(producer_step, state_id));
        step
    }

    pub fn thermal_transient(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Thermal,
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
            IndependentDomain::TimeSeconds {
                origin: 0.0,
                start: 0.0,
                end: 1.0,
                interval_count: 10,
            },
            vec![state("thermal", StateKind::Thermal)],
        )
    }

    pub fn thermal_steady(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Thermal,
            AnalysisProcedure::Static,
            SolutionSpace::Direct,
            IndependentDomain::Static,
            vec![state("thermal", StateKind::Thermal)],
        )
    }

    pub fn structural_nonlinear(step_id: impl Into<String>) -> Self {
        Self::new(
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

    pub fn structural_nonlinear_with(
        step_id: impl Into<String>,
        settings: StructuralNonlinearSettings,
    ) -> Self {
        let mut step = Self::structural_nonlinear(step_id);
        step.settings = ProcedureSettings::StructuralNonlinear(settings);
        step
    }

    pub fn fluid_static(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Fluid,
            AnalysisProcedure::Static,
            SolutionSpace::Direct,
            IndependentDomain::Static,
            vec![state("fluid", StateKind::Fluid)],
        )
    }

    pub fn acoustic_harmonic(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Acoustic,
            AnalysisProcedure::HarmonicResponse,
            SolutionSpace::Direct,
            IndependentDomain::FrequencyHertz(vec![1_000.0]),
            vec![state("acoustic_complex", StateKind::AcousticComplex)],
        )
    }

    pub fn acoustic_modal(step_id: impl Into<String>, requested_count: usize) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Acoustic,
            AnalysisProcedure::ModalExtraction,
            SolutionSpace::Direct,
            IndependentDomain::Eigenvalue { requested_count },
            vec![state("acoustic_modes", StateKind::AcousticModes)],
        )
    }

    pub fn acoustic_transient(step_id: impl Into<String>, end: f64, interval_count: usize) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Acoustic,
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
            IndependentDomain::TimeSeconds {
                origin: 0.0,
                start: 0.0,
                end,
                interval_count,
            },
            vec![state("acoustic_pressure", StateKind::AcousticPressure)],
        )
    }

    pub fn fluid_transient(step_id: impl Into<String>) -> Self {
        Self::new(
            step_id,
            PhysicsFamily::Fluid,
            AnalysisProcedure::TransientResponse,
            SolutionSpace::Direct,
            IndependentDomain::TimeSeconds {
                origin: 0.0,
                start: 0.0,
                end: 1.0,
                interval_count: 10,
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
                interval_count: 10,
            },
            AnalysisProcedure::Nonlinear => IndependentDomain::LoadFactor {
                start: 0.0,
                end: 1.0,
            },
            _ => IndependentDomain::Static,
        };
        Self::new(
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

fn modal_basis_input(producer_step: impl Into<String>, state_id: impl Into<String>) -> StateInput {
    let producer_step = ProcedureStepId(producer_step.into());
    let state_id = StateId(state_id.into());
    StateInput {
        input_id: format!("{}_{}_input", producer_step.as_str(), state_id.0),
        required_kind: StateKind::ModalBasis,
        source: StateSource::Upstream {
            producer_step,
            state_id,
        },
        use_as: StateUse::ModalBasis,
        conflict: InitialStateConflictPolicy::Reject,
    }
}

fn prestress_input(producer_step: impl Into<String>, state_id: impl Into<String>) -> StateInput {
    let producer_step = ProcedureStepId(producer_step.into());
    let state_id = StateId(state_id.into());
    StateInput {
        input_id: format!("{}_{}_prestress_input", producer_step.as_str(), state_id.0),
        required_kind: StateKind::GeometricStiffness,
        source: StateSource::Upstream {
            producer_step,
            state_id,
        },
        use_as: StateUse::Prestress,
        conflict: InitialStateConflictPolicy::Reject,
    }
}

#[cfg(test)]
mod tests;

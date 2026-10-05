use super::{invalid, AnalysisExecutionResult, FluidSteadySolverResult, LIMITS, WORKLOAD_PREFIX};
use crate::{
    execution::{AnalysisWorkloadIdentity, SequenceIdentity, StepIdentity},
    physics::cfd::canonical::CanonicalFluidPolicy,
    topology::FiniteVolumeProjectionPolicy,
};
use runmat_analysis_core::{
    validate_model, AnalysisConstraint, AnalysisModel, AnalysisProcedure, CoordinateFrame,
    ExcitationAction, ExcitationCaseActivation, ExcitationCaseSelection, FormulationRequirement,
    IndependentDomain, PhysicalSignal, PhysicsFamily, ProcedureSettings, ProcedureStepId,
    RestartPolicy, SolutionSpace, StateCompleteness, StateKind, StatePersistence,
};
use runmat_canonical_codec::{
    decode_contract, encode_contract, serialized_materialization_resident_bytes_estimate,
};
use runmat_execution::Digest;
use runmat_meshing_core::SolverMeshArtifact;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FluidSteadyWorkload {
    pub model: AnalysisModel,
    pub step_id: ProcedureStepId,
    pub formulation_requirement: FormulationRequirement,
    pub policy: CanonicalFluidPolicy,
    pub geometry_policy: FiniteVolumeProjectionPolicy,
}

pub(crate) use super::source::PreparedFluidSource;

impl FluidSteadyWorkload {
    pub fn formulation(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<crate::admission::AnalysisRunIdentity> {
        self.validate()?;
        crate::admission::admit_canonical_fluid_steady(mesh, self.formulation_requirement)
            .map_err(invalid)
    }
    pub(crate) fn resident_ceiling(&self) -> AnalysisExecutionResult<u64> {
        let [step] = self.model.sequence.steps.as_slice() else {
            return Err(invalid(
                "steady fluid requires exactly one resource-owning step",
            ));
        };
        if step.step_id != self.step_id {
            return Err(invalid("steady fluid resource step differs from producer"));
        }
        Ok(self
            .policy
            .maximum_resident_bytes
            .min(step.accuracy.resources.maximum_resident_bytes))
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        validate_model(&self.model).map_err(invalid)?;
        let [step] = self.model.sequence.steps.as_slice() else {
            return Err(invalid("steady fluid requires exactly one authored step"));
        };
        self.policy.validate().map_err(invalid)?;
        crate::results::observation::validate_sampled_requests(step).map_err(invalid)?;
        let definitions = super::fields::fluid_native_field_definition_inventory()?;
        if step
            .observations
            .iter()
            .filter_map(|request| request.spec())
            .any(|spec| {
                !definitions.iter().any(|definition| {
                    definition.quantity() == spec.quantity
                        && definition.location() == spec.source_location
                        && definition.value_shape() == &spec.value_shape
                }) || spec.frame != CoordinateFrame::Global
                    || spec.sign_convention != runmat_analysis_core::SignConvention::Native
            })
        {
            return Err(invalid(
                "fluid observations require original native cell U/p or owner-oriented face q",
            ));
        }
        if self
            .model
            .excitation_cases
            .iter()
            .any(|c| c.activation != ExcitationCaseActivation::Always)
            || self
                .model
                .excitations
                .iter()
                .any(|a| !matches!(a, ExcitationAction::BodyForce(_)))
        {
            return Err(invalid(
                "steady fluid has no scheduled/alternate excitation-case authority",
            ));
        }
        if self.formulation_requirement != FormulationRequirement::Experimental
            || step.step_id != self.step_id
            || step.family != PhysicsFamily::Fluid
            || step.procedure != AnalysisProcedure::Static
            || step.solution_space != SolutionSpace::Direct
            || step.independent_domain != IndependentDomain::Static
            || step.settings != ProcedureSettings::None
            || step.restart != RestartPolicy::Fresh
            || step.excitation_case_selection != ExcitationCaseSelection::All
            || !step.activations.is_empty()
            || !step.state_inputs.is_empty()
            || step.state_outputs.len() != 1
            || step.state_outputs.iter().any(|output| {
                output.kind != StateKind::Fluid
                    || output.completeness != StateCompleteness::Complete
                    || output.persistence != StatePersistence::UntilDependentsComplete
            })
            || !self.model.sequence.dependencies.is_empty()
            || !self.model.sequence.transitions.is_empty()
            || !self.model.sequence.branches.is_empty()
            || self.model.frame != CoordinateFrame::Global
            || self.model.structural.is_some()
            || self.model.thermal.is_some()
            || self.model.thermo_mechanical.is_some()
            || self.model.electro_thermal.is_some()
            || self.model.electromagnetic.is_some()
            || !self.model.connectors.is_empty()
            || !self.model.interfaces.is_empty()
            || !self.model.initial_conditions.is_empty()
            || !self.model.response_spectra.is_empty()
            || !self.model.random_excitations.is_empty()
            || self
                .model
                .constraints
                .iter()
                .any(|c| !matches!(c, AnalysisConstraint::Fluid(_)))
            || self.model.signals.iter().any(|s| {
                !matches!(
                    s,
                    PhysicalSignal::Velocity(_)
                        | PhysicalSignal::Pressure(_)
                        | PhysicalSignal::BodyForceDensity(_)
                )
            })
        {
            return Err(invalid(
                "unsupported standalone experimental CPU steady fluid authority",
            ));
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisExecutionResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            WORKLOAD_PREFIX,
            "steady-fluid-workload",
            self,
            LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisExecutionResult<Self> {
        let value: Self = decode_contract(WORKLOAD_PREFIX, "steady-fluid-workload", bytes, LIMITS)?;
        value.validate()?;
        Ok(value)
    }

    pub fn identity(&self) -> AnalysisExecutionResult<AnalysisWorkloadIdentity> {
        AnalysisWorkloadIdentity::derive(
            "steady-fluid-workload",
            &Digest::sha256(self.canonical_bytes()?),
        )
    }

    pub fn producer_identity(&self) -> AnalysisExecutionResult<StepIdentity> {
        self.validate()?;
        StepIdentity::from_solver_contract(
            SequenceIdentity::from_solver_contract(&self.model.sequence)?,
            &self.model.sequence.steps[0],
        )
    }

    /// Local numerical execution only; no scheduler/resource-accounting attestation.
    pub fn execute(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<FluidSteadySolverResult> {
        let source = self.prepare(mesh)?;
        let original = source.problem(self)?.solve().map_err(invalid)?;
        FluidSteadySolverResult::from_original(self, &source, original)
    }

    pub(crate) fn prepare(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<PreparedFluidSource> {
        let ceiling = self.resident_ceiling()?;
        let metadata =
            serialized_materialization_resident_bytes_estimate(self, LIMITS).map_err(invalid)?;
        super::source::require_bytes(metadata, ceiling)?;
        self.validate()?;
        PreparedFluidSource::prepare(
            self,
            &self.model,
            mesh,
            self.geometry_policy,
            self.policy,
            ceiling,
        )
    }
}

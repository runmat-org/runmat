use super::{invalid, AnalysisExecutionResult, FluidSteadySolverResult, LIMITS, WORKLOAD_PREFIX};
use crate::{
    admission,
    execution::{AnalysisWorkloadIdentity, SequenceIdentity, StepIdentity},
    physics::cfd::canonical::CanonicalFluidPolicy,
    topology::FiniteVolumeProjectionPolicy,
};
use runmat_analysis_core::{
    validate_model, AnalysisConstraint, AnalysisModel, AnalysisProcedure, CoordinateFrame,
    CoupledPhysicsFamily, ExcitationCaseActivation, ExcitationCaseSelection, IndependentDomain,
    PhysicsFamily, ProcedureSettings, ProcedureStepId, RestartPolicy, SolutionSpace,
    StateCompleteness, StateKind, StatePersistence,
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
    pub policy: CanonicalFluidPolicy,
    pub geometry_policy: FiniteVolumeProjectionPolicy,
}

pub(crate) use super::source::PreparedFluidSource;

/// The physics solved together with the authored flow in this workload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluidCoupling {
    Standalone,
    /// Coolant flow and conduction in the cooled solid, one energy equation over both.
    ConjugateHeat,
    /// Flow loading a small-strain elastic structure whose wall motion reshapes the flow.
    FluidStructure,
}

impl FluidSteadyWorkload {
    pub fn formulation(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<crate::admission::AnalysisRunIdentity> {
        self.validate()?;
        if self.coupling() == FluidCoupling::ConjugateHeat && self.is_transient() {
            crate::admission::admit_canonical_conjugate_heat_transient(mesh)
        } else if self.coupling() == FluidCoupling::ConjugateHeat {
            crate::admission::admit_canonical_conjugate_heat_steady(mesh)
        } else if self.coupling() == FluidCoupling::FluidStructure && self.is_transient() {
            crate::admission::admit_canonical_fluid_structure_transient(mesh)
        } else if self.coupling() == FluidCoupling::FluidStructure {
            crate::admission::admit_canonical_fluid_structure_steady(mesh)
        } else if self.is_transient() {
            crate::admission::admit_canonical_fluid_transient(mesh)
        } else {
            crate::admission::admit_canonical_fluid_steady(mesh)
        }
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
            .resident_byte_budget()
            .min(step.accuracy.resources.resident_byte_budget()))
    }
    pub fn coupling(&self) -> FluidCoupling {
        match self.model.sequence.steps.as_slice() {
            [step]
                if step.family
                    == PhysicsFamily::Coupled(CoupledPhysicsFamily::ConjugateHeatTransfer) =>
            {
                FluidCoupling::ConjugateHeat
            }
            [step]
                if step.family
                    == PhysicsFamily::Coupled(CoupledPhysicsFamily::FluidStructureInteraction) =>
            {
                FluidCoupling::FluidStructure
            }
            _ => FluidCoupling::Standalone,
        }
    }

    /// Authored procedure of the singleton fluid step. Steady and backward-Euler transient
    /// studies share this workload, its task lifecycle and its publication.
    pub fn is_transient(&self) -> bool {
        matches!(
            self.model.sequence.steps.as_slice(),
            [step] if step.procedure == AnalysisProcedure::TransientResponse
        )
    }

    /// The same authored intent seen by the shared backward-Euler kernel. A fresh authored
    /// root carries no upstream initial-field bindings: initial state comes from the model.
    pub(crate) fn transient_view(&self) -> super::FluidTransientWorkload {
        super::FluidTransientWorkload {
            model: self.model.clone(),
            step_id: self.step_id.clone(),
            policy: self.policy,
            geometry_policy: self.geometry_policy,
            initial_fields: Vec::new(),
        }
    }

    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        validate_model(&self.model).map_err(invalid)?;
        let [step] = self.model.sequence.steps.as_slice() else {
            return Err(invalid("fluid study requires exactly one authored step"));
        };
        self.policy.validate().map_err(invalid)?;
        // Fluid outputs observe native fields on the solver mesh through the shared
        // volume-mesh engine. Histories sample a transient's physical time; a steady flow has
        // only its final state. Solver diagnostics and monitors are not fluid outputs.
        for request in &step.observations {
            match request {
                runmat_analysis_core::ObservationRequest::Diagnostic(_)
                | runmat_analysis_core::ObservationRequest::Monitor(_) => {
                    return Err(invalid("fluid outputs observe native fields only"));
                }
                _ if !self.is_transient()
                    && request.spec().is_some_and(|spec| {
                        spec.sampling != runmat_analysis_core::SamplingPolicy::Final
                    }) =>
                {
                    return Err(invalid(
                        "a steady fluid output observes the final state; time histories need a transient analysis",
                    ));
                }
                _ => {}
            }
        }
        let definitions = super::fields::field_definition_inventory(self.coupling())?;
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
        if step.step_id != self.step_id {
            return Err(invalid("fluid workload producer differs from its step"));
        }
        if self.coupling() == FluidCoupling::ConjugateHeat {
            return self.validate_conjugate_heat(step);
        }
        if self.coupling() == FluidCoupling::FluidStructure {
            return self.validate_fluid_structure(step);
        }
        if self.is_transient() {
            return self.transient_view().validate();
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
                .any(|a| !crate::physics::cfd::canonical::supports_action(a))
        {
            return Err(invalid(
                "steady fluid has no scheduled/alternate excitation-case authority",
            ));
        }
        if step.step_id != self.step_id
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
            || self
                .model
                .signals
                .iter()
                .any(|s| !crate::physics::cfd::canonical::supports_signal(s))
        {
            return Err(invalid("unsupported standalone CPU steady fluid authority"));
        }
        Ok(())
    }

    /// Authored uniform starting temperature of a transient conjugate study, kelvin. The
    /// steady flow it is advected by needs no initial state.
    pub(crate) fn initial_temperature(&self) -> AnalysisExecutionResult<f64> {
        use runmat_analysis_core::{InitialCondition, InitialConditionTarget, InitialValue};
        let [step] = self.model.sequence.steps.as_slice() else {
            return Err(invalid("conjugate heat requires one authored step"));
        };
        let mut found = None;
        for condition in self
            .model
            .initial_conditions
            .iter()
            .filter(|c| c.applies_to_step(step))
        {
            let InitialCondition::Temperature(value) = condition else {
                return Err(invalid(admission::CONJUGATE_HEAT_INITIAL_KIND.message));
            };
            let (InitialConditionTarget::WholeModel, InitialValue::Uniform(t)) =
                (&value.target, &value.value)
            else {
                return Err(invalid(admission::CONJUGATE_HEAT_INITIAL_UNIFORM.message));
            };
            if found.replace(t.canonical_value()).is_some()
                || !t.canonical_value().is_finite()
                || t.canonical_value() <= 0.0
            {
                return Err(invalid(admission::CONJUGATE_HEAT_INITIAL_UNIFORM.message));
            }
        }
        found.ok_or_else(|| invalid(admission::CONJUGATE_HEAT_INITIAL_TEMPERATURE.message))
    }

    /// Steady conjugate heat transfer: fluid and thermal authority only, with every
    /// fluid–solid interface declared.
    fn validate_conjugate_heat(
        &self,
        step: &runmat_analysis_core::ProcedureStep,
    ) -> AnalysisExecutionResult<()> {
        use runmat_analysis_core::{
            AnalysisInterfaceKind, BoundaryOperator, ExcitationAction, ThermalConstraintKind,
        };
        let transient = match (step.procedure, &step.independent_domain) {
            (AnalysisProcedure::Static, IndependentDomain::Static) => Some(false),
            (AnalysisProcedure::TransientResponse, IndependentDomain::TimeSeconds { .. }) => {
                super::FluidTransientSchedule::derive(&self.model, step)?;
                self.initial_temperature()?;
                Some(true)
            }
            _ => None,
        };
        if transient.is_none()
            || step.solution_space != SolutionSpace::Direct
            || step.settings != ProcedureSettings::None
            || step.restart != RestartPolicy::Fresh
            || step.excitation_case_selection != ExcitationCaseSelection::All
            || !crate::admission::fluid_step_initial_selection_supported(step)
            || step.state_outputs.is_empty()
            || step.state_outputs.iter().any(|output| {
                !matches!(output.kind, StateKind::CoupledInterface | StateKind::Fluid)
                    || output.completeness != StateCompleteness::Complete
                    || output.persistence != StatePersistence::UntilDependentsComplete
            })
            || !self.model.sequence.dependencies.is_empty()
            || !self.model.sequence.transitions.is_empty()
            || !self.model.sequence.branches.is_empty()
            || self.model.frame != CoordinateFrame::Global
            || self.model.structural.is_some()
            || self.model.thermo_mechanical.is_some()
            || self.model.electro_thermal.is_some()
            || self.model.electromagnetic.is_some()
            || !self.model.connectors.is_empty()
            || (transient == Some(false) && !self.model.initial_conditions.is_empty())
            || !self.model.response_spectra.is_empty()
            || !self.model.random_excitations.is_empty()
            || self
                .model
                .excitation_cases
                .iter()
                .any(|c| c.activation != ExcitationCaseActivation::Always)
            || self.model.constraints.iter().any(|c| match c {
                AnalysisConstraint::Fluid(_) => false,
                AnalysisConstraint::Thermal(t) => {
                    !matches!(t.kind, ThermalConstraintKind::Prescribed { .. })
                }
                _ => true,
            })
            || self.model.boundaries.iter().any(|b| {
                !matches!(
                    b.operator,
                    BoundaryOperator::FluidOutletPressure { .. }
                        | BoundaryOperator::ThermalHeatFlux { .. }
                        | BoundaryOperator::ThermalConvection { .. }
                )
            })
            || self.model.excitations.iter().any(|a| {
                !crate::physics::cfd::canonical::supports_action(a)
                    && !matches!(
                        a,
                        ExcitationAction::HeatGeneration(_) | ExcitationAction::HeatFlux(_)
                    )
            })
            || self.model.interfaces.is_empty()
            || self
                .model
                .interfaces
                .iter()
                .any(|i| !matches!(i.kind, AnalysisInterfaceKind::ConjugateHeatTransfer(_)))
        {
            return Err(invalid(
                "unsupported steady conjugate heat transfer authority",
            ));
        }
        Ok(())
    }

    /// Steady fluid–structure interaction: flow, small-strain structure and declared walls.
    fn validate_fluid_structure(
        &self,
        step: &runmat_analysis_core::ProcedureStep,
    ) -> AnalysisExecutionResult<()> {
        use runmat_analysis_core::{
            AnalysisInterfaceKind, BoundaryOperator, StructuralConstraintKind,
        };
        if self.model.interfaces.is_empty()
            || self
                .model
                .interfaces
                .iter()
                .any(|i| !matches!(i.kind, AnalysisInterfaceKind::FluidStructure(_)))
        {
            return Err(invalid(admission::FLUID_STRUCTURE_INTERFACE.message));
        }
        if !self.model.initial_conditions.is_empty() {
            return Err(invalid(
                admission::FLUID_STRUCTURE_INITIAL_CONDITION.message,
            ));
        }
        let timing = match (step.procedure, &step.independent_domain) {
            (AnalysisProcedure::Static, IndependentDomain::Static) => true,
            (AnalysisProcedure::TransientResponse, IndependentDomain::TimeSeconds { .. }) => {
                super::FluidTransientSchedule::derive(&self.model, step)?;
                true
            }
            _ => false,
        };
        if !timing
            || step.solution_space != SolutionSpace::Direct
            || step.settings != ProcedureSettings::None
            || step.restart != RestartPolicy::Fresh
            || step.excitation_case_selection != ExcitationCaseSelection::All
            || !step.activations.is_empty()
            || !step.state_inputs.is_empty()
            || step.state_outputs.is_empty()
            || step.state_outputs.iter().any(|output| {
                !matches!(output.kind, StateKind::CoupledInterface | StateKind::Fluid)
                    || output.completeness != StateCompleteness::Complete
                    || output.persistence != StatePersistence::UntilDependentsComplete
            })
            || !self.model.sequence.dependencies.is_empty()
            || !self.model.sequence.transitions.is_empty()
            || !self.model.sequence.branches.is_empty()
            || self.model.frame != CoordinateFrame::Global
            || self.model.thermal.is_some()
            || self.model.thermo_mechanical.is_some()
            || self.model.electro_thermal.is_some()
            || self.model.electromagnetic.is_some()
            || !self.model.connectors.is_empty()
            || !self.model.response_spectra.is_empty()
            || !self.model.random_excitations.is_empty()
            || self
                .model
                .excitation_cases
                .iter()
                .any(|c| c.activation != ExcitationCaseActivation::Always)
            || self.model.constraints.iter().any(|c| match c {
                AnalysisConstraint::Fluid(_) => false,
                AnalysisConstraint::Structural(s) => !matches!(
                    s.kind,
                    StructuralConstraintKind::Fixed { .. }
                        | StructuralConstraintKind::Symmetry { .. }
                ),
                _ => true,
            })
            || self
                .model
                .boundaries
                .iter()
                .any(|b| !matches!(b.operator, BoundaryOperator::FluidOutletPressure { .. }))
            || self
                .model
                .excitations
                .iter()
                .any(|a| !crate::physics::cfd::canonical::supports_action(a))
        {
            return Err(invalid("unsupported fluid–structure interaction authority"));
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
        if self.coupling() == FluidCoupling::FluidStructure && self.is_transient() {
            let schedule =
                super::FluidTransientSchedule::derive(&self.model, &self.model.sequence.steps[0])?;
            let coupled = source
                .fluid_structure(self, mesh)?
                .solve_transient(schedule.points_s())
                .map_err(super::fluid_structure_error)?;
            return FluidSteadySolverResult::from_fluid_structure_transient(
                self, &source, mesh, coupled,
            );
        }
        if self.coupling() == FluidCoupling::FluidStructure {
            let coupled = source
                .fluid_structure(self, mesh)?
                .solve()
                .map_err(super::fluid_structure_error)?;
            return FluidSteadySolverResult::from_fluid_structure(self, &source, mesh, coupled);
        }
        if self.coupling() == FluidCoupling::ConjugateHeat && self.is_transient() {
            let flow = source.problem(self)?.solve().map_err(super::fluid_error)?;
            let schedule =
                super::FluidTransientSchedule::derive(&self.model, &self.model.sequence.steps[0])?;
            let times = schedule.points_s();
            let first = source.conjugate_heat_at(
                self,
                flow.face_volume_flux_m3_per_s.canonical_values(),
                times[0],
            )?;
            let mut state = first.uniform_state(self.initial_temperature()?);
            drop(first);
            let mut steps = Vec::with_capacity(times.len() - 1);
            let mut departure = state.clone();
            for window in times.windows(2) {
                let problem = source.conjugate_heat_at(
                    self,
                    flow.face_volume_flux_m3_per_s.canonical_values(),
                    window[1],
                )?;
                let next = problem
                    .step_backward_euler(&state, window[1] - window[0])
                    .map_err(invalid)?;
                steps.push(crate::physics::conjugate_heat::ConjugateHeatStep {
                    time_s: window[1],
                    peak_temperature_k: next.temperature_k.iter().copied().fold(f64::MIN, f64::max),
                    stored_heat_rate_w: next.evidence.energy.stored_heat_rate_w,
                    relative_imbalance: next.evidence.energy.relative_imbalance,
                });
                departure = std::mem::replace(&mut state, next);
            }
            return FluidSteadySolverResult::from_conjugate_heat_transient(
                self,
                &source,
                crate::physics::conjugate_heat::ConjugateHeatTransientResult {
                    flow,
                    departure_time_s: times[times.len() - 2],
                    departure,
                    energy: state,
                    steps,
                },
            );
        }
        if self.coupling() == FluidCoupling::ConjugateHeat {
            let flow = source.problem(self)?.solve().map_err(super::fluid_error)?;
            let energy = source
                .conjugate_heat(self, flow.face_volume_flux_m3_per_s.canonical_values())?
                .solve()
                .map_err(invalid)?;
            return FluidSteadySolverResult::from_conjugate_heat(self, &source, flow, energy);
        }
        if self.is_transient() {
            let view = self.transient_view();
            let original = view.prepare(mesh)?.execute()?;
            return FluidSteadySolverResult::from_transient(self, &source, original);
        }
        let original = source.problem(self)?.solve().map_err(super::fluid_error)?;
        FluidSteadySolverResult::from_original(self, &source, original)
    }

    pub(crate) fn prepare(
        &self,
        mesh: &SolverMeshArtifact,
    ) -> AnalysisExecutionResult<std::sync::Arc<PreparedFluidSource>> {
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

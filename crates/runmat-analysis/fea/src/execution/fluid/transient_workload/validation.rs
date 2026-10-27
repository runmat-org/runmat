use super::*;
use runmat_analysis_core::*;
impl FluidTransientWorkload {
    pub(crate) fn selected_step(&self) -> AnalysisExecutionResult<&ProcedureStep> {
        let [step] = self.model.sequence.steps.as_slice() else {
            return Err(invalid(
                "portable transient fluid requires one authored resource-owning step",
            ));
        };
        if step.step_id != self.step_id {
            return Err(invalid(
                "transient fluid producer differs from selected step",
            ));
        }
        Ok(step)
    }
    pub(crate) fn resident_ceiling(&self) -> AnalysisExecutionResult<u64> {
        Ok(self.policy.resident_byte_budget().min(
            self.selected_step()?
                .accuracy
                .resources
                .resident_byte_budget(),
        ))
    }
    pub fn validate(&self) -> AnalysisExecutionResult<()> {
        let profile = runmat_canonical_codec::profile_serialized(self, LIMITS)?;
        super::super::source::require_bytes(
            profile.materialization_resident_bytes_estimate,
            self.resident_ceiling()?,
        )?;
        self.policy.validate().map_err(invalid)?;
        runmat_analysis_core::validate_model(&self.model).map_err(invalid)?;
        let step = self.selected_step()?;
        crate::execution::validate_initial_field_declarations(
            &self.model,
            step,
            &self.initial_fields,
        )?;
        if step.family != PhysicsFamily::Fluid
            || step.procedure != AnalysisProcedure::TransientResponse
            || step.solution_space != SolutionSpace::Direct
            || !matches!(
                step.independent_domain,
                IndependentDomain::TimeSeconds { .. }
            )
            || step.settings != ProcedureSettings::None
            || step.restart != RestartPolicy::Fresh
            || !step.activations.is_empty()
            || !step.state_inputs.is_empty()
            || step.state_outputs.len() != 1
            || step.state_outputs.iter().any(|o| {
                o.kind != StateKind::Fluid
                    || o.completeness != StateCompleteness::Complete
                    || o.persistence != StatePersistence::UntilDependentsComplete
            })
            || !self.model.sequence.dependencies.is_empty()
            || !self.model.sequence.transitions.is_empty()
            || !self.model.sequence.branches.is_empty()
            || self.model.frame != CoordinateFrame::Global
            || self.model.structural.is_some()
            || self.model.thermal.is_some()
            || self.model.electromagnetic.is_some()
            || self.model.thermo_mechanical.is_some()
            || self.model.electro_thermal.is_some()
            || !self.model.connectors.is_empty()
            || !self.model.interfaces.is_empty()
            || !self.model.response_spectra.is_empty()
            || !self.model.random_excitations.is_empty()
            || self
                .model
                .constraints
                .iter()
                .any(|c| !matches!(c, AnalysisConstraint::Fluid(_)))
            || self.model.initial_conditions.iter().any(|c| {
                c.applies_to_step(step)
                    && !matches!(
                        c,
                        InitialCondition::FluidVelocity(_) | InitialCondition::FluidPressure(_)
                    )
            })
        {
            return Err(invalid(
                "unsupported standalone fresh transient fluid authority",
            ));
        }
        // Keep unsupported actions explicit; selection itself belongs to the shared case owner.
        if crate::excitation::active_actions(&self.model, Some(step))
            .map_err(invalid)?
            .any(|a| !crate::physics::cfd::canonical::supports_action(a.action))
        {
            return Err(invalid(
                "transient fluid action lacks canonical realization",
            ));
        }
        if step.observations.iter().any(|request| {
            matches!(
                request,
                runmat_analysis_core::ObservationRequest::Diagnostic(_)
                    | runmat_analysis_core::ObservationRequest::Monitor(_)
            )
        }) {
            return Err(invalid("fluid outputs observe native fields only"));
        }
        super::super::FluidTransientSchedule::derive(&self.model, step)?;
        Ok(())
    }
}

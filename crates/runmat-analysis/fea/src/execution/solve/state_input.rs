//! Closed state-input authority for an exact structural static workload.

use super::*;
use runmat_analysis_core::{StateInput, StateKind, StateSource, StateTransitionMethod, StateUse};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "transition",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum StructuralLinearStateInput {
    InitialGuess(CanonicalStateTransition),
    ThermalPrestress(CanonicalStateTransition),
}

impl StructuralLinearStateInput {
    pub fn transition(&self) -> &CanonicalStateTransition {
        match self {
            Self::InitialGuess(transition) | Self::ThermalPrestress(transition) => transition,
        }
    }

    fn matches_input(&self, input: &StateInput) -> bool {
        match self {
            Self::InitialGuess(_) => {
                input.use_as == StateUse::InitialGuess
                    && input.required_kind == StateKind::StructuralPrimary
            }
            Self::ThermalPrestress(_) => {
                input.use_as == StateUse::Prestress && input.required_kind == StateKind::Thermal
            }
        }
    }
}

impl StructuralLinearWorkload {
    pub(super) fn validate_authored_state_input<'a>(
        &self,
        step: &'a ProcedureStep,
    ) -> AnalysisExecutionResult<Option<&'a StateInput>> {
        if !step.activations.is_empty() || step.restart != RestartPolicy::Fresh {
            return Err(AnalysisExecutionError::Invalid(
                "structural sequence execution requires fresh steps without object activation"
                    .into(),
            ));
        }
        let input = match step.state_inputs.as_slice() {
            [] => return Ok(None),
            [input] => input,
            _ => {
                return Err(AnalysisExecutionError::Invalid(
                    "structural execution requires at most one upstream state input".into(),
                ))
            }
        };
        let method = match (input.use_as, input.required_kind) {
            (StateUse::InitialGuess, StateKind::StructuralPrimary) => StateTransitionMethod::InitialGuessOnly,
            (StateUse::Prestress, StateKind::Thermal)
                if step.family == PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical) => StateTransitionMethod::Reuse,
            _ => return Err(AnalysisExecutionError::Invalid("structural state input must be a structural initial guess or solved thermal prestress for thermomechanical static execution".into())),
        };
        let StateSource::Upstream {
            producer_step,
            state_id,
        } = &input.source
        else {
            return Err(AnalysisExecutionError::Invalid(
                "structural state input must name an upstream producer".into(),
            ));
        };
        let transitions = self
            .model
            .sequence
            .transitions
            .iter()
            .filter(|transition| {
                &transition.producer_step == producer_step
                    && transition.consumer_step == step.step_id
                    && &transition.state_id == state_id
            })
            .collect::<Vec<_>>();
        let [transition] = transitions.as_slice() else {
            return Err(AnalysisExecutionError::Invalid(
                "structural state input requires exactly one authored transition".into(),
            ));
        };
        if transition.method != method
            || transition.topology
                != runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical
            || transition.reference != runmat_analysis_core::ReferenceStatePolicy::Preserve
        {
            return Err(AnalysisExecutionError::Invalid("structural state transition requires its exact method, identical topology and preserved reference".into()));
        }
        Ok(Some(input))
    }

    pub(super) fn validate_state_input(&self, step: &ProcedureStep) -> AnalysisExecutionResult<()> {
        let input = self.validate_authored_state_input(step)?;
        let (binding, input) = match (&self.state_input, input) {
            (None, None) => return Ok(()),
            (Some(binding), Some(input)) => (binding, input),
            _ => {
                return Err(AnalysisExecutionError::Invalid(
                    "structural authored input and bound state must be supplied together".into(),
                ))
            }
        };
        let transition = binding.transition();
        transition.validate()?;
        let StateSource::Upstream {
            producer_step,
            state_id,
        } = &input.source
        else {
            return Err(AnalysisExecutionError::Invalid(
                "structural state input is not upstream".into(),
            ));
        };
        let authored = self
            .model
            .sequence
            .transitions
            .iter()
            .find(|authored| {
                authored.producer_step == *producer_step
                    && authored.consumer_step == step.step_id
                    && authored.state_id == *state_id
            })
            .ok_or(AnalysisExecutionError::Identity(
                "structural transition is absent",
            ))?;
        if !binding.matches_input(input)
            || transition.producer_step_id != *producer_step
            || transition.consumer_step_id != step.step_id
            || transition.state_id != *state_id
            || transition.transition_id != authored.transition_id
            || transition.method != authored.method
            || transition.topology != authored.topology
            || transition.reference != authored.reference
        {
            return Err(AnalysisExecutionError::Identity(
                "bound structural state differs from its exact authored input",
            ));
        }
        let producer = self
            .model
            .sequence
            .steps
            .iter()
            .find(|candidate| &candidate.step_id == producer_step)
            .ok_or(AnalysisExecutionError::Identity(
                "structural state producer is absent",
            ))?;
        let sequence = super::super::SequenceIdentity::from_solver_contract(&self.model.sequence)?;
        if transition.sequence != sequence
            || transition.producer
                != super::super::StepIdentity::from_solver_contract(sequence, producer)?
            || transition.consumer
                != super::super::StepIdentity::from_solver_contract(sequence, step)?
        {
            return Err(AnalysisExecutionError::Identity(
                "bound structural state identities differ from the full authored sequence",
            ));
        }
        Ok(())
    }
}

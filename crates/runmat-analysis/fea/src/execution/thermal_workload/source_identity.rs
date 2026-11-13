//! Original workload identities admitted and bound before geometry retention.
use super::*;
use crate::execution::{SequenceIdentity, StepIdentity, ThermalNumericalIntentIdentity};
use runmat_canonical_codec::profile_serialized;

pub(crate) struct ThermalSourceIdentities<'a> {
    pub step: &'a ProcedureStep,
    pub workload: AnalysisWorkloadIdentity,
    pub numerical_intent: ThermalNumericalIntentIdentity,
    pub producer: StepIdentity,
    pub resident_bytes: u64,
}

impl ThermalWorkload {
    /// Additional canonical identity and dependency-index workspace; borrowed source storage
    /// and other model validation are excluded. Solver-semantic collection views are borrowed, not copied inventories.
    pub(crate) fn source_identity_resident_bytes_upper_bound(
        &self,
    ) -> AnalysisExecutionResult<u64> {
        let limits = CanonicalLimits::new(usize::MAX, usize::MAX, usize::MAX, 64);
        let workload = profile_serialized(self, limits)?;
        let intent = profile_serialized(
            &(
                "shared-solver-relative-residual/v2",
                self.model.solver_identity_contract(),
                &self.step_id,
                self.procedure,
                self.residual_target,
                &self.initial_fields,
            ),
            limits,
        )?;
        let sequence = profile_serialized(&self.model.sequence.solver_identity_contract(), limits)?;
        Ok(workload
            .materialization_resident_bytes_estimate
            .max(intent.materialization_resident_bytes_estimate)
            .max(sequence.materialization_resident_bytes_estimate)
            .max(
                runmat_analysis_core::dependency_graph_resident_bytes_upper_bound(
                    &self.model.sequence,
                )
                .map_err(|error| AnalysisExecutionError::Invalid(error.to_string()))?,
            ))
    }

    pub(crate) fn bind_source_identities_admitted(
        &self,
        maximum_resident_bytes: u64,
    ) -> AnalysisExecutionResult<ThermalSourceIdentities<'_>> {
        let resident_bytes = self.source_identity_resident_bytes_upper_bound()?;
        let require = |maximum| {
            if resident_bytes > maximum {
                Err(AnalysisExecutionError::Limit(
                    "thermal source identity workspace",
                ))
            } else {
                Ok(())
            }
        };
        require(maximum_resident_bytes)?;
        let step = self.validated_step_with_dependency_graph_budget(maximum_resident_bytes)?;
        require(step.accuracy.resources.resident_byte_budget())?;
        let sequence = SequenceIdentity::from_solver_contract(&self.model.sequence)?;
        let producer = StepIdentity::from_solver_contract(sequence, step)?;
        let workload = Self::identity_for(
            &self.model,
            &self.step_id,
            self.backend,
            self.procedure,
            self.residual_target,
            &self.initial_fields,
        )?;
        let numerical_intent = Self::numerical_intent_identity_for(
            &self.model,
            &self.step_id,
            self.procedure,
            self.residual_target,
            &self.initial_fields,
        )?;
        Ok(ThermalSourceIdentities {
            step,
            workload,
            numerical_intent,
            producer,
            resident_bytes,
        })
    }
}

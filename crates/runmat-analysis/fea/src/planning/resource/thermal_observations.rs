//! Shared history-plan costs plus the thermal evaluator's complete sample catalog copy.
use super::{ByteCount, ObservationResourceFacts, ResourcePlanningError};
use crate::{
    physics::thermal::history_fields,
    results::observation::{
        resolve_observation_plan, ObservationExecutionPolicy, ObservationLimits,
    },
    topology::AnalysisProjection,
};
use runmat_analysis_core::{ObservationRequest, ProcedureStep, SamplingPolicy};
use runmat_meshing_core::SolverMeshArtifact;

#[derive(Clone, Debug)]
pub struct ThermalObservationResourceFacts {
    pub shared: Option<ObservationResourceFacts>,
    pub catalog_copy_bytes: ByteCount,
    pub evaluation_peak_bytes: ByteCount,
}

impl ThermalObservationResourceFacts {
    pub fn derive(
        mesh: &SolverMeshArtifact,
        step: &ProcedureStep,
    ) -> Result<Self, ResourcePlanningError> {
        // Final fields and spatial reductions are evaluated after the solve, but their selectors and components
        // must still be admissible before numerical assembly. Keep their validation
        // separate from the inline-history reservation below.
        if step.observations.iter().any(|request| {
            matches!(
                request,
                ObservationRequest::Field(_) | ObservationRequest::Reduction(_)
            )
        }) {
            let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
            let definitions = history_fields::definitions().map_err(invalid)?;
            resolve_observation_plan(
                step,
                &projection,
                &definitions,
                ObservationLimits::default(),
            )
            .map_err(invalid)?;
        }
        let mut history_step = step.clone();
        history_step
            .observations
            .retain(|request| matches!(request, ObservationRequest::History(_)));
        if history_step.observations.is_empty() {
            return Ok(Self {
                shared: None,
                catalog_copy_bytes: ByteCount(0),
                evaluation_peak_bytes: ByteCount(0),
            });
        }
        if history_step.observations.iter().any(|request| {
            matches!(request,
            ObservationRequest::History(spec) if matches!(spec.sampling, SamplingPolicy::Event(_)))
        }) {
            return Err(ResourcePlanningError::Invalid(
                "thermal history pipeline has no event detector".into(),
            ));
        }
        let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
        let definitions = history_fields::definitions().map_err(invalid)?;
        let plan = resolve_observation_plan(
            &history_step,
            &projection,
            &definitions,
            ObservationLimits::default(),
        )
        .map_err(invalid)?;
        let shared =
            ObservationResourceFacts::from_plan(&plan, ObservationExecutionPolicy::scalable())?;
        // Current sample evaluator is monolithic; do not substitute the smaller partition peak.
        let evaluation_peak_bytes = ByteCount(
            plan.observations
                .iter()
                .map(|observation| observation.resolved_cost.peak_resident_bytes)
                .max()
                .unwrap_or(0)
                .max(shared.peak_resident_bytes.0),
        );
        // Evaluation clones the complete source inventory, even for one requested quantity.
        // Derive widths from the same definitions used by the evaluator (including storage rate).
        let counts = crate::results::field::RecoveredFieldEntityCounts {
            oriented_faces: 0,
            nodes: projection.nodes.len(),
            volume_elements: projection.elements.len(),
            volume_integration_points: 0,
            oriented_edges: 0,
            beam_elements: 0,
            shell_elements: 0,
            connectors: 0,
            constraint_groups: 0,
        };
        let catalog_copy_bytes = ByteCount(definitions.iter().try_fold(0u64, |total, field| {
            let entities =
                u64::try_from(counts.entity_count(field.entity_domain())).map_err(invalid)?;
            if entities == 0 {
                return Err(invalid(
                    "thermal catalog field has no admitted entity domain",
                ));
            }
            let components = u64::try_from(field.storage_component_count().map_err(invalid)?)
                .map_err(invalid)?;
            entities
                .checked_mul(components)
                .and_then(|values| values.checked_mul(8))
                .and_then(|bytes| total.checked_add(bytes))
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal history catalog copy",
                ))
        })?);
        Ok(Self {
            shared: Some(shared),
            catalog_copy_bytes,
            evaluation_peak_bytes,
        })
    }

    /// Only selector validation occurs in the numerical task. Observation execution has its
    /// own source-bound task/resource plan and must retain the original authored dynamic cap.
    pub(crate) fn derive_deferred(
        mesh: &SolverMeshArtifact,
        step: &ProcedureStep,
    ) -> Result<Self, ResourcePlanningError> {
        if step.observations.iter().any(|r| matches!(r, ObservationRequest::History(s) if matches!(s.sampling, SamplingPolicy::Event(_)))) { return Err(invalid("thermal history pipeline has no event detector")); }
        if !step.observations.is_empty() {
            let projection = AnalysisProjection::derive(mesh).map_err(invalid)?;
            let definitions = history_fields::definitions().map_err(invalid)?;
            resolve_observation_plan(
                step,
                &projection,
                &definitions,
                ObservationLimits::default(),
            )
            .map_err(invalid)?;
        }
        Ok(Self {
            shared: None,
            catalog_copy_bytes: ByteCount(0),
            evaluation_peak_bytes: ByteCount(0),
        })
    }

    /// Conservative composition of shared stored-history cost and evaluation peak; not total RSS.
    pub fn reservation_bytes(&self) -> Result<ByteCount, ResourcePlanningError> {
        self.shared
            .as_ref()
            .map_or(0, |facts| facts.stored_bytes.0)
            .checked_add(self.evaluation_peak_bytes.0)
            .and_then(|bytes| bytes.checked_add(self.catalog_copy_bytes.0))
            .map(ByteCount)
            .ok_or(ResourcePlanningError::Overflow(
                "thermal observation reservation",
            ))
    }
}

fn invalid(error: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(error.to_string())
}

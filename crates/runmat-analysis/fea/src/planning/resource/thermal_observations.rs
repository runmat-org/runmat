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
        // evaluate_field_histories clones all three source fields, even for one requested quantity.
        let nodes = u64::try_from(projection.nodes.len()).map_err(invalid)?;
        let elements = u64::try_from(projection.elements.len()).map_err(invalid)?;
        let catalog_copy_bytes = ByteCount(
            elements
                .checked_mul(6)
                .and_then(|values| values.checked_add(nodes))
                .and_then(|values| values.checked_mul(8))
                .ok_or(ResourcePlanningError::Overflow(
                    "thermal history catalog copy",
                ))?,
        );
        Ok(Self {
            shared: Some(shared),
            catalog_copy_bytes,
            evaluation_peak_bytes,
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

    pub(crate) fn require_fits(
        &self,
        numerical: ByteCount,
        maximum: u64,
    ) -> Result<(), ResourcePlanningError> {
        let required = numerical.0.checked_add(self.reservation_bytes()?.0).ok_or(
            ResourcePlanningError::Overflow("thermal producer observation reservation"),
        )?;
        if required > maximum {
            return Err(ResourcePlanningError::AccuracyCeiling {
                resource: "thermal numerics and observation reservation",
                required,
                maximum,
            });
        }
        Ok(())
    }
}

fn invalid(error: impl std::fmt::Display) -> ResourcePlanningError {
    ResourcePlanningError::Invalid(error.to_string())
}

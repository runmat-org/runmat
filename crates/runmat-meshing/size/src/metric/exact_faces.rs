//! Shared constant-face resolution for surface consumers and boundary traces.
use super::*;
use crate::{grading::grade_metric_evaluations, incidence::TopologyMetricIncidence};
use runmat_geometry_core::{ExactBRepTopology, GeometryEvaluationControl, GeometryEvaluationError};

#[derive(Debug)]
pub enum ExactFaceMetricResolutionError {
    Metric(MetricContractError),
    Control(GeometryEvaluationError),
}
impl std::fmt::Display for ExactFaceMetricResolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Metric(e) => e.fmt(f),
            Self::Control(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for ExactFaceMetricResolutionError {}
impl From<MetricContractError> for ExactFaceMetricResolutionError {
    fn from(value: MetricContractError) -> Self {
        Self::Metric(value)
    }
}
impl ResolvedMetricField {
    /// Resolve the exact same face incidence and graph grading for every
    /// consumer. The supplied incidence belongs to the admitted topology.
    pub fn resolve_exact_faces(
        &self,
        topology: &ExactBRepTopology,
        incidence: &TopologyMetricIncidence,
        control: Option<&dyn GeometryEvaluationControl>,
    ) -> Result<
        BTreeMap<PersistentEntityId, ResolvedMetricEvaluation>,
        ExactFaceMetricResolutionError,
    > {
        let checkpoint = || match control {
            Some(c) => c
                .checkpoint()
                .map_err(ExactFaceMetricResolutionError::Control),
            None => Ok(()),
        };
        checkpoint()?;
        if topology
            .faces
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
        {
            return Err(MetricContractError::invalid(
                "exact face inventory",
                "must be canonical and unique",
            )
            .into());
        }
        for contribution in &self.request.contributions {
            checkpoint()?;
            let entity_id = match &contribution.scope {
                MetricContributionScope::Region { region_id } => region_id,
                MetricContributionScope::Entity { entity_id } => entity_id,
            };
            if !incidence.knows(entity_id) {
                return Err(MetricContractError::invalid(
                    "metric contribution scope",
                    format!("references unknown exact entity {entity_id:?}"),
                )
                .into());
            }
        }
        let mut faces = BTreeMap::new();
        for face in &topology.faces {
            checkpoint()?;
            let evaluation = self.resolve(&incidence.incident_face_entities(face))?;
            if faces.insert(face.id.clone(), evaluation).is_some() {
                return Err(MetricContractError::invalid(
                    "exact face inventory",
                    "contains duplicate identities",
                )
                .into());
            }
        }
        checkpoint()?;
        grade_metric_evaluations(
            self.request.maximum_grading_ratio,
            &TopologyMetricIncidence::face_adjacency(topology),
            &mut faces,
        )?;
        checkpoint()?;
        Ok(faces)
    }
}

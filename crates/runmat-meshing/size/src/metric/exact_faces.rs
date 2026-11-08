//! Shared face sizing for surface consumers, boundary traces and validation.
//!
//! Without entity supports every face has one constant metric and fine faces constrain their
//! graph neighbors to at most the grading ratio times their size. With supports, a face keeps
//! only its own uniform contributions; every entity-local source of the request (whether or not
//! it lies on the face) and the uniform sizing of each finer neighbor across their shared edges
//! are graded by distance at each evaluated point (`ResolvedMetricField::evaluate_at`). A fine
//! fillet then refines its neighbors near the fillet, not across their whole extent, and a face
//! near a thin gap is as fine there as the volume beside it.
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
/// Uniform sizing of one exact face plus the face's own sources graded pointwise: the uniform
/// sizing of finer neighbors from their shared edges. Consumers evaluate the face with
/// `ResolvedMetricField::evaluate_at(uniform.metric, &local, point)`, which adds every
/// entity-local source of the request.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactFaceSizing {
    pub uniform: ResolvedMetricEvaluation,
    pub local: Vec<LocalMetricSource>,
}

impl ResolvedMetricField {
    /// Resolve the exact same face sizing for every consumer. The supplied incidence belongs to
    /// the admitted topology.
    pub fn resolve_exact_faces(
        &self,
        topology: &ExactBRepTopology,
        incidence: &TopologyMetricIncidence,
        control: Option<&dyn GeometryEvaluationControl>,
    ) -> Result<BTreeMap<PersistentEntityId, ExactFaceSizing>, ExactFaceMetricResolutionError> {
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
        let mut uniform = BTreeMap::new();
        for face in &topology.faces {
            checkpoint()?;
            let evaluation = self.resolve_constant(&incidence.incident_face_entities(face))?;
            if uniform.insert(face.id.clone(), evaluation).is_some() {
                return Err(MetricContractError::invalid(
                    "exact face inventory",
                    "contains duplicate identities",
                )
                .into());
            }
        }
        checkpoint()?;
        if !self.has_supports() {
            grade_metric_evaluations(
                self.request.maximum_grading_ratio,
                &TopologyMetricIncidence::face_adjacency(topology),
                &mut uniform,
            )?;
            return Ok(uniform
                .into_iter()
                .map(|(face, uniform)| {
                    (
                        face,
                        ExactFaceSizing {
                            uniform,
                            local: Vec::new(),
                        },
                    )
                })
                .collect());
        }
        let mut sizing = BTreeMap::new();
        for face in &topology.faces {
            checkpoint()?;
            sizing.insert(
                face.id.clone(),
                ExactFaceSizing {
                    uniform: uniform[&face.id].clone(),
                    local: Vec::new(),
                },
            );
        }
        // A finer neighbor constrains a face from their shared edges, graded by distance.
        for edge in &topology.edges {
            checkpoint()?;
            let mut faces = incidence.edge_face_ids(&edge.id).to_vec();
            faces.sort_unstable();
            faces.dedup();
            for receiver in &faces {
                for neighbor in &faces {
                    if receiver == neighbor {
                        continue;
                    }
                    let neighbor_metric = uniform[neighbor].metric;
                    if intersection::dominates(uniform[receiver].metric, neighbor_metric) {
                        continue;
                    }
                    if let Some(source) = self.source_on(&edge.id, neighbor_metric)? {
                        sizing
                            .get_mut(receiver)
                            .expect("resolved every exact face")
                            .local
                            .push(source);
                    }
                }
            }
        }
        Ok(sizing)
    }
}

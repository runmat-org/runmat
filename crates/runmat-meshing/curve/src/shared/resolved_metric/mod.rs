mod incidence;

use std::collections::BTreeMap;

use incidence::TopologyIncidence;
use runmat_geometry_core::{ExactBRepTopology, PersistentEntityId};
use runmat_meshing_core::{MetricContributionScope, MetricFieldRequest};
use runmat_meshing_size::grading::grade_metric_evaluations;
use runmat_meshing_size::metric::{ResolvedMetricEvaluation, ResolvedMetricField};

use super::{
    CurveMetricEvaluation, CurveMetricField, CurveMetricQuery, SharedCurveError,
    SharedCurveErrorKind,
};

/// Exact-topology projection of the canonical resolved metric request for curve evaluation.
///
/// Every applicable contribution is combined in canonical request order. Matrix addition is a
/// conservative SPD intersection: its directional density is at least that of every operand.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCurveMetricField {
    by_edge: BTreeMap<PersistentEntityId, ResolvedMetricEvaluation>,
}

impl ResolvedCurveMetricField {
    pub fn new(
        topology: &ExactBRepTopology,
        request: &MetricFieldRequest,
    ) -> Result<Self, SharedCurveError> {
        request.validate().map_err(|error| {
            SharedCurveError::invalid_request("resolved curve metric", error.to_string())
        })?;
        let resolver = ResolvedMetricField::new(request).map_err(|error| {
            SharedCurveError::invalid_request("resolved curve metric", error.to_string())
        })?;
        let incidence = TopologyIncidence::new(topology);
        for contribution in &request.contributions {
            let entity_id = match &contribution.scope {
                MetricContributionScope::Region { region_id } => region_id,
                MetricContributionScope::Entity { entity_id } => entity_id,
            };
            if !incidence.knows(entity_id) {
                return Err(SharedCurveError::invalid_request(
                    "metric contribution scope",
                    format!("references unknown exact entity {entity_id:?}"),
                ));
            }
        }
        let mut by_edge = BTreeMap::new();
        for edge in &topology.edges {
            let incident = incidence.incident_entities(edge);
            let resolved = resolver.resolve(&incident).map_err(|error| {
                SharedCurveError::invalid_request(
                    "resolved curve metric intersection",
                    error.to_string(),
                )
                .for_edge(&edge.id)
            })?;
            by_edge.insert(edge.id.clone(), resolved);
        }
        grade_metric_evaluations(
            request.maximum_grading_ratio,
            &TopologyIncidence::edge_adjacency(topology),
            &mut by_edge,
        )
        .map_err(|error| {
            SharedCurveError::invalid_request("resolved curve metric grading", error.to_string())
        })?;
        Ok(Self { by_edge })
    }
}

impl CurveMetricField for ResolvedCurveMetricField {
    fn evaluate(
        &self,
        query: CurveMetricQuery<'_>,
    ) -> Result<CurveMetricEvaluation, SharedCurveError> {
        if query.point_m.iter().any(|value| !value.is_finite())
            || query.unit_tangent.iter().any(|value| !value.is_finite())
        {
            return Err(SharedCurveError::new(
                SharedCurveErrorKind::MetricEvaluation,
                "resolved curve metric query",
                "point and tangent must be finite",
            )
            .for_edge(query.edge_id));
        }
        let resolved = self.by_edge.get(query.edge_id).ok_or_else(|| {
            SharedCurveError::new(
                SharedCurveErrorKind::MetricEvaluation,
                "resolved curve metric query",
                "edge is absent from the admitted exact topology",
            )
            .for_edge(query.edge_id)
        })?;
        Ok(CurveMetricEvaluation {
            metric: resolved.metric,
            active_sources: resolved.active_sources.clone(),
            applied_contribution_count: resolved.applied_contribution_count,
            clipped_contribution_count: resolved.clipped_contribution_count,
            rejected_contribution_count: resolved.rejected_contribution_count,
        })
    }
}

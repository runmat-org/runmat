//! Retain edge-local volume restrictions separately from the face envelope.
//! A face envelope combines several boundary features; it cannot stand in for
//! the metric of a volume cell touching only one of those features.
use super::*;
use runmat_geometry_core::{ExactFace, GeometryEvaluationControl, PersistentEntityKind};
use runmat_meshing_size::{incidence::TopologyMetricIncidence, metric::ResolvedMetricField};
use std::collections::BTreeSet;
#[derive(Clone, Debug, PartialEq)]
pub(super) struct EdgeMetric {
    edge_id: PersistentEntityId,
    metric: MetricTensor3,
}
pub(super) fn prepare(
    face: &ExactFace,
    incidence: &TopologyMetricIncidence,
    resolver: &ResolvedMetricField,
    control: Option<&dyn GeometryEvaluationControl>,
) -> Result<Vec<EdgeMetric>, ExactFaceMetricError> {
    if resolver
        .uniform_evaluation()
        .map_err(|error| invalid_request(error.to_string()))?
        .is_some()
    {
        return Ok(Vec::new());
    }
    let entities = incidence.incident_face_entities(face);
    let regions = || {
        entities
            .iter()
            .filter(|id| id.kind == PersistentEntityKind::Region)
    };
    let region_count = regions().count();
    if region_count == 0 {
        return Ok(Vec::new());
    }
    let edges = || {
        entities
            .iter()
            .filter(|id| id.kind == PersistentEntityKind::Edge)
    };
    let invalid = |reason| invalid_request(reason);
    let count = edges()
        .count()
        .checked_mul(region_count)
        .ok_or_else(|| invalid("boundary metric inventory overflow"))?;
    let mut bytes = count
        .checked_mul(std::mem::size_of::<EdgeMetric>())
        .ok_or_else(|| invalid("boundary metric storage overflow"))?;
    for edge in edges() {
        let strings = edge
            .assembly_path
            .iter()
            .try_fold(edge.source_topology_id.len(), |total, path| {
                total.checked_add(path.len())
            })
            .ok_or_else(|| invalid("boundary metric identity overflow"))?;
        let owned = edge
            .assembly_path
            .len()
            .checked_mul(std::mem::size_of::<String>())
            .and_then(|slots| slots.checked_add(strings))
            .and_then(|size| size.checked_mul(region_count))
            .ok_or_else(|| invalid("boundary metric identity overflow"))?;
        bytes = bytes
            .checked_add(owned)
            .ok_or_else(|| invalid("boundary metric storage overflow"))?;
    }
    if let Some(control) = control {
        control
            .consume_allocation_bytes(
                u64::try_from(bytes).map_err(|_| invalid("boundary metric storage overflow"))?,
            )
            .map_err(|error| geometry_error(&face.id, error))?;
    }
    let mut metrics = Vec::with_capacity(count);
    for edge in edges() {
        for region in regions() {
            if let Some(control) = control {
                control
                    .checkpoint()
                    .map_err(|error| geometry_error(&face.id, error))?;
            }
            let metric = resolver
                .resolve(&BTreeSet::from([edge.clone(), region.clone()]))
                .map_err(|error| invalid_request(error.to_string()))?
                .metric;
            metrics.push(EdgeMetric {
                edge_id: edge.clone(),
                metric,
            });
        }
    }
    Ok(metrics)
}
impl ResolvedFaceMetricField {
    pub(crate) fn has_volume_edge_metrics(&self, face: &PersistentEntityId) -> bool {
        self.by_face
            .get(face)
            .is_some_and(|record| !record.boundary_metrics.is_empty())
    }
    pub(crate) fn volume_edge_metrics(
        &self,
        face: &PersistentEntityId,
        edge: &PersistentEntityId,
    ) -> impl Iterator<Item = MetricTensor3> + '_ {
        let metrics = self
            .by_face
            .get(face)
            .map(|record| record.boundary_metrics.as_slice())
            .unwrap_or(&[]);
        let begin = metrics.partition_point(|record| record.edge_id < *edge);
        let end = metrics.partition_point(|record| record.edge_id <= *edge);
        metrics[begin..end].iter().map(|record| record.metric)
    }
}

#[cfg(test)]
mod tests;

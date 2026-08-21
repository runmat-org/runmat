use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;

use crate::{ExactFaceChart, ExactFaceChartRefinedMesh, ExactFacePslg, ExactFacePslgSegmentSource};

use super::{ExactFaceJoinError, ExactFaceJoinErrorKind, ExactFaceMeshEdgeParameter};

pub(super) fn validate_trim_lineage(
    chart: &ExactFaceChart,
    mesh: &ExactFaceChartRefinedMesh,
) -> Result<(), ExactFaceJoinError> {
    let original = exact_trim_lineage(&chart.pslg);
    let refined = exact_trim_lineage(&mesh.mesh.topology.pslg);
    if original != refined {
        return Err(invalid(
            chart,
            "refined chart changed its exact-trim segment lineage",
        ));
    }
    let admitted_cut_ids = chart
        .pslg
        .segments
        .iter()
        .filter_map(|segment| match segment.source {
            ExactFacePslgSegmentSource::ChartCut { cut_id } => Some(cut_id),
            ExactFacePslgSegmentSource::ExactTrim { .. } => None,
        })
        .collect::<BTreeSet<_>>();
    if mesh.mesh.topology.pslg.segments.iter().any(|segment| {
        matches!(segment.source, ExactFacePslgSegmentSource::ChartCut { cut_id } if !admitted_cut_ids.contains(&cut_id))
    }) {
        return Err(invalid(
            chart,
            "refined chart introduced an unknown chart cut",
        ));
    }
    Ok(())
}

#[derive(PartialEq)]
struct ExactTrimLineage {
    source_coedge_id: PersistentEntityId,
    source_edge_id: PersistentEntityId,
    node_ids: [StableDigest; 2],
    edge_parameters: Option<[f64; 2]>,
}

fn exact_trim_lineage(pslg: &ExactFacePslg) -> Vec<ExactTrimLineage> {
    pslg.segments
        .iter()
        .filter_map(|segment| {
            let ExactFacePslgSegmentSource::ExactTrim {
                source_coedge_id,
                source_edge_id,
            } = &segment.source
            else {
                return None;
            };
            Some(ExactTrimLineage {
                source_coedge_id: source_coedge_id.clone(),
                source_edge_id: source_edge_id.clone(),
                node_ids: segment
                    .vertex_indices
                    .map(|index| pslg.vertices[index as usize].node_id),
                edge_parameters: segment.edge_parameters,
            })
        })
        .collect()
}

pub(super) fn node_edge_parameters(
    mesh: &ExactFaceChartRefinedMesh,
) -> Result<BTreeMap<u32, Vec<ExactFaceMeshEdgeParameter>>, ExactFaceJoinError> {
    let mut result = BTreeMap::<u32, Vec<ExactFaceMeshEdgeParameter>>::new();
    for segment in &mesh.mesh.topology.pslg.segments {
        let ExactFacePslgSegmentSource::ExactTrim {
            source_coedge_id,
            source_edge_id,
        } = &segment.source
        else {
            continue;
        };
        let Some(parameters) = segment.edge_parameters else {
            return Err(ExactFaceJoinError::new(
                ExactFaceJoinErrorKind::InvalidInput,
                &mesh.mesh.geometry.source_face_id,
                "exact trim is missing endpoint parameters",
            )
            .with_chart(mesh.chart_id));
        };
        for (endpoint, parameter) in parameters.into_iter().enumerate() {
            result
                .entry(segment.vertex_indices[endpoint])
                .or_default()
                .push(ExactFaceMeshEdgeParameter {
                    source_coedge_id: source_coedge_id.clone(),
                    source_edge_id: source_edge_id.clone(),
                    parameter,
                });
        }
    }
    for parameters in result.values_mut() {
        parameters.sort_by(|left, right| {
            (&left.source_coedge_id, &left.source_edge_id)
                .cmp(&(&right.source_coedge_id, &right.source_edge_id))
                .then_with(|| left.parameter.total_cmp(&right.parameter))
        });
        parameters.dedup();
    }
    Ok(result)
}

fn invalid(chart: &ExactFaceChart, reason: &str) -> ExactFaceJoinError {
    ExactFaceJoinError::new(
        ExactFaceJoinErrorKind::InvalidInput,
        &chart.source_face_id,
        reason,
    )
    .with_chart(chart.chart_id)
}

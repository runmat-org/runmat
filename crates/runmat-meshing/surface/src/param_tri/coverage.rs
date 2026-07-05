use std::collections::{BTreeMap, BTreeSet};

use runmat_meshing_cad::SourceTopologyFace;
use runmat_meshing_curve::CadCurveEdgeProvenance;

use super::{
    boundary::FaceCurveSegment, SurfaceCadCurveBoundaryEdgeProvenance,
    SurfaceCadCurveBoundaryProvenanceReport, SurfaceDiscretizationError, SurfaceLoopCoverageReport,
};

pub(super) struct SurfaceLoopCoverageAccumulator {
    source_face_count: usize,
    recovered_face_count: usize,
    boundary_loop_count: usize,
    boundary_segment_count: usize,
    max_loops_per_face: usize,
    recovered_source_edge_ids: BTreeSet<u32>,
}

impl SurfaceLoopCoverageAccumulator {
    pub(super) fn new(source_face_count: usize) -> Self {
        Self {
            source_face_count,
            recovered_face_count: 0,
            boundary_loop_count: 0,
            boundary_segment_count: 0,
            max_loops_per_face: 0,
            recovered_source_edge_ids: BTreeSet::new(),
        }
    }

    pub(super) fn record_face(
        &mut self,
        face: &SourceTopologyFace,
        segment_loops: &[Vec<FaceCurveSegment>],
    ) {
        self.record_face_edges(&face.edge_ids, segment_loops);
    }

    pub(super) fn record_face_edges(
        &mut self,
        source_edge_ids: &[u32],
        segment_loops: &[Vec<FaceCurveSegment>],
    ) {
        self.recovered_face_count += 1;
        self.boundary_loop_count += segment_loops.len();
        self.max_loops_per_face = self.max_loops_per_face.max(segment_loops.len());
        for segment in segment_loops.iter().flatten() {
            self.boundary_segment_count += 1;
            if source_edge_ids.contains(&segment.source_edge_id) {
                self.recovered_source_edge_ids
                    .insert(segment.source_edge_id);
            }
        }
    }

    pub(super) fn finish(self) -> SurfaceLoopCoverageReport {
        SurfaceLoopCoverageReport {
            source_face_count: self.source_face_count,
            recovered_face_count: self.recovered_face_count,
            boundary_loop_count: self.boundary_loop_count,
            recovered_source_edge_count: self.recovered_source_edge_ids.len(),
            boundary_segment_count: self.boundary_segment_count,
            max_loops_per_face: self.max_loops_per_face,
        }
    }
}

pub(super) struct SurfaceCadCurveBoundaryProvenanceAccumulator {
    edges: BTreeMap<u32, SurfaceCadCurveBoundaryEdgeProvenance>,
}

impl SurfaceCadCurveBoundaryProvenanceAccumulator {
    pub(super) fn new() -> Self {
        Self {
            edges: BTreeMap::new(),
        }
    }

    pub(super) fn record_segments<'a>(
        &mut self,
        segment_loops: &[Vec<FaceCurveSegment>],
        provenance_by_source_edge: &BTreeMap<u32, &'a CadCurveEdgeProvenance>,
    ) -> Result<(), SurfaceDiscretizationError> {
        for segment in segment_loops.iter().flatten() {
            let source = provenance_by_source_edge
                .get(&segment.source_edge_id)
                .ok_or(SurfaceDiscretizationError::MissingCadCurveProvenance {
                    source_edge_id: segment.source_edge_id,
                })?;
            self.edges
                .entry(segment.source_edge_id)
                .and_modify(|entry| entry.boundary_segment_count += 1)
                .or_insert_with(|| SurfaceCadCurveBoundaryEdgeProvenance {
                    source_edge_id: source.source_edge_id,
                    cad_edge_id: source.cad_edge_id.clone(),
                    imported_curve_id: source.imported_curve_id,
                    evaluator_id: source.evaluator_id.clone(),
                    evaluator_supports_point_evaluation: source.evaluator_supports_point_evaluation,
                    evaluator_supports_projection: source.evaluator_supports_projection,
                    evaluator_supports_tangent: source.evaluator_supports_tangent,
                    evaluator_supports_curvature: source.evaluator_supports_curvature,
                    evaluator_sample_count: source.evaluator_sample_count,
                    live_query_backed: source.live_query_backed,
                    live_query_sample_count: source.live_query_sample_count,
                    rejected_evaluator_sample_count: source.rejected_evaluator_sample_count,
                    boundary_segment_count: 1,
                });
        }
        Ok(())
    }

    pub(super) fn finish(self) -> SurfaceCadCurveBoundaryProvenanceReport {
        let edges = self.edges.into_values().collect::<Vec<_>>();
        SurfaceCadCurveBoundaryProvenanceReport {
            recovered_source_edge_count: edges.len(),
            boundary_segment_count: edges.iter().map(|edge| edge.boundary_segment_count).sum(),
            imported_curve_edge_count: edges
                .iter()
                .filter(|edge| edge.imported_curve_id.is_some())
                .count(),
            evaluator_curve_edge_count: edges
                .iter()
                .filter(|edge| edge.evaluator_id.is_some())
                .count(),
            evaluator_sample_count: edges.iter().map(|edge| edge.evaluator_sample_count).sum(),
            live_query_edge_count: edges.iter().filter(|edge| edge.live_query_backed).count(),
            live_query_sample_count: edges.iter().map(|edge| edge.live_query_sample_count).sum(),
            rejected_evaluator_sample_count: edges
                .iter()
                .map(|edge| edge.rejected_evaluator_sample_count)
                .sum(),
            edges,
        }
    }
}

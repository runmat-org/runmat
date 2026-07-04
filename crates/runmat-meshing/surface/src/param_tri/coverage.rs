use std::collections::BTreeSet;

use runmat_meshing_cad::SourceTopologyFace;

use super::{boundary::FaceCurveSegment, SurfaceLoopCoverageReport};

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
        self.recovered_face_count += 1;
        self.boundary_loop_count += segment_loops.len();
        self.max_loops_per_face = self.max_loops_per_face.max(segment_loops.len());
        for segment in segment_loops.iter().flatten() {
            self.boundary_segment_count += 1;
            if face.edge_ids.contains(&segment.source_edge_id) {
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

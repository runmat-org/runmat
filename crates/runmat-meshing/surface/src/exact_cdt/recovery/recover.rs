use std::collections::BTreeSet;

use runmat_meshing_core::{
    predicate::incircle2d_symbolic, MeshingCancellationSignal, PredicateSign,
};

use crate::{
    validate_exact_face_delaunay, ExactFaceBoundary, ExactFaceDelaunay, ExactFaceDelaunayError,
    ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions, ExactFacePslg,
};

use super::{
    cavity::recover_segment_cavity, ExactFaceConstrainedDelaunay, ExactFaceRecoveredSegment,
};
use crate::exact_cdt::topology::{edge_uses, flip_edge, properly_crosses, sorted_edge};

pub fn recover_exact_face_segments(
    delaunay: &ExactFaceDelaunay,
    pslg: &ExactFacePslg,
    boundary: &ExactFaceBoundary,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceConstrainedDelaunay, ExactFaceDelaunayError> {
    validate_exact_face_delaunay(delaunay, pslg, boundary, cancellation, options)?;
    recover_validated_face_segments(delaunay, pslg, cancellation, options)
}

pub(crate) fn recover_validated_face_segments(
    delaunay: &ExactFaceDelaunay,
    pslg: &ExactFacePslg,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceConstrainedDelaunay, ExactFaceDelaunayError> {
    crate::exact_cdt::validate_face_delaunay_topology(delaunay, pslg, cancellation, options)?;
    let mut control = RecoveryControl::new(pslg, cancellation, options);
    let mut triangles = delaunay.triangles.clone();
    let mut protected = BTreeSet::new();
    let mut protected_segments = Vec::with_capacity(pslg.segments.len());
    let mut recovery_edge_flip_count = 0u64;
    let mut cavity_retriangulation_count = 0u64;

    for (pslg_segment_index, segment) in pslg.segments.iter().enumerate() {
        control.checkpoint()?;
        let target = sorted_edge(segment.vertex_indices);
        while !edge_uses(&triangles).contains_key(&target) {
            let edges = edge_uses(&triangles);
            let mut flipped = false;
            for (edge, uses) in &edges {
                control.consume_predicates(4)?;
                if uses.len() != 2
                    || protected.contains(edge)
                    || !properly_crosses(*edge, target, pslg)
                        .map_err(|e| control.predicate_error(e))?
                {
                    continue;
                }
                let candidate = sorted_edge([uses[0].opposite_vertex, uses[1].opposite_vertex]);
                control.consume_predicates(4)?;
                if properly_crosses(candidate, target, pslg)
                    .map_err(|e| control.predicate_error(e))?
                {
                    continue;
                }
                control.consume_predicates(2)?;
                if flip_edge(&mut triangles, pslg, *edge, uses, &edges)
                    .map_err(|e| control.predicate_error(e))?
                    .is_some()
                {
                    control.consume_flip()?;
                    recovery_edge_flip_count = recovery_edge_flip_count.saturating_add(1);
                    flipped = true;
                    break;
                }
            }
            if !flipped {
                if recover_segment_cavity(&mut triangles, pslg, target, &protected, &mut control)? {
                    cavity_retriangulation_count = cavity_retriangulation_count.saturating_add(1);
                } else {
                    return Err(control.unsatisfied(
                        "protected segment has no valid deterministic recovery cavity",
                    ));
                }
            }
        }
        protected.insert(target);
        protected_segments.push(ExactFaceRecoveredSegment {
            pslg_segment_index: pslg_segment_index as u32,
            source: segment.source.clone(),
            vertex_indices: segment.vertex_indices,
        });
    }

    let delaunay_restoration_flip_count =
        restore_delaunay(&mut triangles, pslg, &protected, &mut control)?;
    let result = ExactFaceConstrainedDelaunay {
        source_face_id: pslg.source_face_id.clone(),
        triangles,
        protected_segments,
        recovery_edge_flip_count,
        cavity_retriangulation_count,
        delaunay_restoration_flip_count,
    };
    super::validate::validate_face_constrained_topology(&result, pslg, cancellation, options)?;
    Ok(result)
}

fn restore_delaunay(
    triangles: &mut [crate::ExactFaceDelaunayTriangle],
    pslg: &ExactFacePslg,
    protected: &BTreeSet<[u32; 2]>,
    control: &mut RecoveryControl<'_>,
) -> Result<u64, ExactFaceDelaunayError> {
    let mut flip_count = 0u64;
    loop {
        control.checkpoint()?;
        let edges = edge_uses(triangles);
        let mut changed = false;
        for (edge, uses) in &edges {
            if protected.contains(edge) || uses.len() != 2 {
                continue;
            }
            control.consume_predicates(1)?;
            let triangle = triangles[uses[0].triangle_index].vertex_indices;
            let query = [
                triangle[0],
                triangle[1],
                triangle[2],
                uses[1].opposite_vertex,
            ]
            .map(|index| {
                super::super::triangulation::predicate_point(pslg.vertices[index as usize], index)
            });
            if incircle2d_symbolic(query).map_err(|e| control.predicate_error(e))?
                != PredicateSign::Positive
            {
                continue;
            }
            control.consume_predicates(2)?;
            if flip_edge(triangles, pslg, *edge, uses, &edges)
                .map_err(|e| control.predicate_error(e))?
                .is_some()
            {
                control.consume_flip()?;
                flip_count = flip_count.saturating_add(1);
                changed = true;
                break;
            }
            return Err(control.unsatisfied(
                "unprotected illegal edge cannot be restored by a strict convex flip",
            ));
        }
        if !changed {
            return Ok(flip_count);
        }
    }
}

pub(super) struct RecoveryControl<'a> {
    pslg: &'a ExactFacePslg,
    cancellation: &'a dyn MeshingCancellationSignal,
    predicates_remaining: u64,
    flips_remaining: u64,
    cavities_remaining: u64,
    maximum_triangles: usize,
    check_interval: u64,
    work_since_check: u64,
}

impl<'a> RecoveryControl<'a> {
    pub(super) fn new(
        pslg: &'a ExactFacePslg,
        cancellation: &'a dyn MeshingCancellationSignal,
        options: ExactFaceDelaunayOptions,
    ) -> Self {
        Self {
            pslg,
            cancellation,
            predicates_remaining: options.maximum_predicate_evaluations,
            flips_remaining: options.maximum_edge_flips,
            cavities_remaining: options.maximum_cavity_retriangulations,
            maximum_triangles: options.maximum_triangles,
            check_interval: options.cancellation_check_interval,
            work_since_check: 0,
        }
    }

    pub(super) fn consume_predicates(&mut self, count: u64) -> Result<(), ExactFaceDelaunayError> {
        self.predicates_remaining =
            self.predicates_remaining
                .checked_sub(count)
                .ok_or_else(|| {
                    self.error(
                        ExactFaceDelaunayErrorKind::ResourceLimit,
                        "segment recovery predicate hard limit exceeded",
                    )
                })?;
        self.work_since_check = self.work_since_check.saturating_add(count);
        if self.work_since_check >= self.check_interval {
            self.checkpoint()?;
        }
        Ok(())
    }

    fn consume_flip(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.flips_remaining = self.flips_remaining.checked_sub(1).ok_or_else(|| {
            self.error(
                ExactFaceDelaunayErrorKind::ResourceLimit,
                "segment recovery edge-flip hard limit exceeded",
            )
        })?;
        Ok(())
    }

    pub(super) fn consume_cavity(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.cavities_remaining = self.cavities_remaining.checked_sub(1).ok_or_else(|| {
            self.error(
                ExactFaceDelaunayErrorKind::ResourceLimit,
                "segment recovery cavity hard limit exceeded",
            )
        })?;
        Ok(())
    }

    pub(super) fn ensure_triangle_limit(
        &self,
        triangle_count: usize,
    ) -> Result<(), ExactFaceDelaunayError> {
        if triangle_count > self.maximum_triangles {
            Err(self.error(
                ExactFaceDelaunayErrorKind::ResourceLimit,
                "segment recovery triangle hard limit exceeded",
            ))
        } else {
            Ok(())
        }
    }

    pub(super) fn checkpoint(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.work_since_check = 0;
        if self.cancellation.is_cancelled() {
            Err(self.error(
                ExactFaceDelaunayErrorKind::Cancelled,
                "surface segment recovery cancelled",
            ))
        } else {
            Ok(())
        }
    }

    pub(super) fn predicate_error(
        &self,
        error: runmat_meshing_core::PlanarPredicateError,
    ) -> ExactFaceDelaunayError {
        self.error(
            ExactFaceDelaunayErrorKind::InvalidTopology,
            format!("invalid recovery predicate input: {error:?}"),
        )
    }

    pub(super) fn unsatisfied(&self, reason: &str) -> ExactFaceDelaunayError {
        self.error(ExactFaceDelaunayErrorKind::UnsatisfiedConstraint, reason)
    }

    fn error(
        &self,
        kind: ExactFaceDelaunayErrorKind,
        reason: impl Into<String>,
    ) -> ExactFaceDelaunayError {
        ExactFaceDelaunayError::new(kind, &self.pslg.source_face_id, reason)
    }
}

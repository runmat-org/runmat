use std::collections::BTreeSet;

use runmat_meshing_core::{predicate::incircle2d, MeshingCancellationSignal, PredicateSign};

use crate::{
    validate_exact_face_delaunay, ExactFaceBoundary, ExactFaceDelaunay, ExactFaceDelaunayError,
    ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions, ExactFacePslg,
};

use super::{
    cavity::recover_segment_cavity, ExactFaceConstrainedDelaunay, ExactFaceRecoveredSegment,
};
use crate::exact_cdt::topology::{
    edge_uses_with_checkpoint, flip_edge, properly_crosses, sorted_edge,
};

pub fn recover_exact_face_segments(
    delaunay: &ExactFaceDelaunay,
    pslg: &ExactFacePslg,
    boundary: &ExactFaceBoundary,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceConstrainedDelaunay, ExactFaceDelaunayError> {
    validate_exact_face_delaunay(delaunay, pslg, boundary, cancellation, options)?;
    recover_validated_face_segments(delaunay, pslg, cancellation, options, None)
}

pub(crate) fn recover_validated_face_segments(
    delaunay: &ExactFaceDelaunay,
    pslg: &ExactFacePslg,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
    geometry_control: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<ExactFaceConstrainedDelaunay, ExactFaceDelaunayError> {
    crate::exact_cdt::validate_face_delaunay_topology(delaunay, pslg, cancellation, options)?;
    let mut control = RecoveryControl::new(pslg, cancellation, options);
    let mut triangles = delaunay.triangles.clone();
    let mut protected = BTreeSet::new();
    let mut protected_segments = Vec::with_capacity(pslg.segments.len());
    let mut recovery_edge_flip_count = 0u64;
    let mut cavity_retriangulation_count = 0u64;

    let mut edges = edge_uses_with_checkpoint(&triangles, || control.checkpoint())?;
    for (pslg_segment_index, segment) in pslg.segments.iter().enumerate() {
        control.checkpoint()?;
        let target = sorted_edge(segment.vertex_indices);
        while !edges.contains_key(&target) {
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
                match recover_segment_cavity(
                    &mut triangles,
                    pslg,
                    target,
                    &protected,
                    &mut control,
                )? {
                    Ok(()) => {
                        cavity_retriangulation_count =
                            cavity_retriangulation_count.saturating_add(1);
                    }
                    Err(reason) => {
                        let uv = |vertex: u32| pslg.vertices[vertex as usize].uv;
                        let [a, b] = [uv(target[0]), uv(target[1])];
                        let direction = [b[0] - a[0], b[1] - a[1]];
                        let length_squared = direction[0].powi(2) + direction[1].powi(2);
                        let on_segment = pslg
                            .vertices
                            .iter()
                            .enumerate()
                            .filter(|(index, vertex)| {
                                let offset = [vertex.uv[0] - a[0], vertex.uv[1] - a[1]];
                                let along = (offset[0] * direction[0] + offset[1] * direction[1])
                                    / length_squared;
                                let across = offset[0] * direction[1] - offset[1] * direction[0];
                                !target.contains(&(*index as u32))
                                    && (-1e-6..=1.0 + 1e-6).contains(&along)
                                    && across.abs() <= 1e-6 * length_squared
                            })
                            .map(|(index, vertex)| {
                                let index = index as u32;
                                let segment_count = pslg
                                    .segments
                                    .iter()
                                    .filter(|segment| segment.vertex_indices.contains(&index))
                                    .count();
                                let triangle_count = triangles
                                    .iter()
                                    .filter(|triangle| triangle.vertex_indices.contains(&index))
                                    .count();
                                (index, vertex.uv, segment_count, triangle_count)
                            })
                            .collect::<Vec<_>>();
                        let endpoint_triangles = target.map(|vertex| {
                            triangles
                                .iter()
                                .filter(|triangle| triangle.vertex_indices.contains(&vertex))
                                .map(|triangle| {
                                    triangle.vertex_indices.map(|corner| (corner, uv(corner)))
                                })
                                .collect::<Vec<_>>()
                        });
                        return Err(control.unsatisfied(&format!(
                            "protected segment {target:?} {a:?} -> {b:?} (endpoint triangles {endpoint_triangles:?}, {} vertices, {} triangles) has no valid deterministic recovery cavity: {reason}; vertices on or near the closed segment (index, uv, segment uses, triangle uses) {on_segment:?}",
                            pslg.vertices.len(),
                            triangles.len(),
                        )));
                    }
                }
            }
            // Both flips and cavity replacement invalidate triangle-index incidence.
            edges = edge_uses_with_checkpoint(&triangles, || control.checkpoint())?;
        }
        protected.insert(target);
        protected_segments.push(ExactFaceRecoveredSegment {
            pslg_segment_index: pslg_segment_index as u32,
            source: segment.source.clone(),
            vertex_indices: segment.vertex_indices,
        });
    }

    drop(edges); // Restoration owns its own index; retain only one complete map.
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
    super::validate::validate_face_constrained_topology(
        &result,
        pslg,
        cancellation,
        options,
        geometry_control,
    )?;
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
        let edges = edge_uses_with_checkpoint(triangles, || control.checkpoint())?;
        let mut changed = false;
        for (edge, uses) in &edges {
            // Robust incircle work can be much more expensive than one cheap
            // bookkeeping unit. Probe per edge as well as the existing work cap.
            control.checkpoint()?;
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
            if incircle2d(query.map(|point| point.coordinates))
                .map_err(|e| control.predicate_error(e))?
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
            control.checkpoint()?;
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
                        ExactFaceDelaunayErrorKind::SearchWorkLimit,
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
                ExactFaceDelaunayErrorKind::SearchWorkLimit,
                "segment recovery edge-flip hard limit exceeded",
            )
        })?;
        Ok(())
    }

    pub(super) fn consume_cavity(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.cavities_remaining = self.cavities_remaining.checked_sub(1).ok_or_else(|| {
            self.error(
                ExactFaceDelaunayErrorKind::IterationLimit,
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
                ExactFaceDelaunayErrorKind::ElementLimit,
                "segment recovery triangle hard limit exceeded",
            ))
        } else {
            Ok(())
        }
    }

    #[track_caller]
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

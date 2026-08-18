use std::collections::BTreeSet;

use runmat_meshing_core::{predicate::orient2d, MeshingCancellationSignal, PredicateSign};

use crate::{
    validate_exact_face_constrained_delaunay, ExactFaceBoundary, ExactFaceConstrainedDelaunay,
    ExactFaceDelaunayError, ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions, ExactFacePslg,
};

use super::{super::topology::edge_uses, ExactFaceTrimmedDelaunay};

pub fn carve_exact_face_domain(
    constrained: &ExactFaceConstrainedDelaunay,
    pslg: &ExactFacePslg,
    boundary: &ExactFaceBoundary,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceTrimmedDelaunay, ExactFaceDelaunayError> {
    validate_exact_face_constrained_delaunay(constrained, pslg, boundary, cancellation, options)?;
    carve_validated_face_domain(constrained, pslg, cancellation, options)
}

pub(crate) fn carve_validated_face_domain(
    constrained: &ExactFaceConstrainedDelaunay,
    pslg: &ExactFacePslg,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<ExactFaceTrimmedDelaunay, ExactFaceDelaunayError> {
    crate::exact_cdt::validate_face_constrained_topology(constrained, pslg, cancellation, options)?;
    let mut control = CarvingControl::new(pslg, cancellation, options);
    control.checkpoint()?;
    let protected = constrained
        .protected_segments
        .iter()
        .map(|segment| sorted_edge(segment.vertex_indices))
        .collect::<BTreeSet<_>>();
    let edges = edge_uses(&constrained.triangles);
    let (component_of, components) = components(
        constrained.triangles.len(),
        &edges,
        &protected,
        &mut control,
    )?;

    // A recovered segment is an impermeable topological barrier. Any component
    // touching an unprotected convex-hull edge is therefore outside the outer wire.
    let exterior_components = edges
        .iter()
        .filter(|(edge, uses)| uses.len() == 1 && !protected.contains(*edge))
        .map(|(_, uses)| component_of[uses[0].triangle_index])
        .collect::<BTreeSet<_>>();
    let mut hole_components = BTreeSet::new();
    for loop_index in 1..pslg.loops.len() {
        let loop_record = &pslg.loops[loop_index];
        let range = loop_record.first_segment as usize
            ..loop_record.first_segment as usize + loop_record.segment_count as usize;
        // The exact loop winding identifies its geometric interior side regardless
        // of face orientation or whether its pcurve traversal is clockwise.
        let winding = loop_winding(pslg, range.clone(), &mut control)?;
        let mut loop_components = BTreeSet::new();
        for segment in &pslg.segments[range] {
            let edge = sorted_edge(segment.vertex_indices);
            let uses = edges.get(&edge).ok_or_else(|| {
                control.invalid("hole boundary segment is absent from constrained topology")
            })?;
            if uses.len() != 2 {
                return Err(control.invalid(
                    "hole boundary segment does not separate two constrained components",
                ));
            }
            let mut interior_use = None;
            for edge_use in uses {
                control.consume_predicate()?;
                let sign = orient2d([
                    pslg.vertices[segment.vertex_indices[0] as usize].uv,
                    pslg.vertices[segment.vertex_indices[1] as usize].uv,
                    pslg.vertices[edge_use.opposite_vertex as usize].uv,
                ])
                .map_err(|error| control.predicate_error(error))?;
                if sign == winding && interior_use.replace(edge_use.triangle_index).is_some() {
                    return Err(control
                        .invalid("hole boundary has multiple triangles on its interior side"));
                }
            }
            let interior_use = interior_use.ok_or_else(|| {
                control.invalid("hole boundary has no triangle on its interior side")
            })?;
            loop_components.insert(component_of[interior_use]);
        }
        if loop_components.len() != 1 {
            return Err(control.invalid("hole boundary does not enclose one topology component"));
        }
        hole_components.extend(loop_components);
    }
    if !exterior_components.is_disjoint(&hole_components) {
        return Err(control.invalid("exterior and hole topology components are connected"));
    }

    let removed_exterior_triangle_count = exterior_components
        .iter()
        .map(|component| components[*component].len() as u64)
        .sum();
    let removed_hole_triangle_count = hole_components
        .iter()
        .map(|component| components[*component].len() as u64)
        .sum();
    let removed = exterior_components
        .union(&hole_components)
        .copied()
        .collect::<BTreeSet<_>>();
    let triangles = constrained
        .triangles
        .iter()
        .enumerate()
        .filter_map(|(index, triangle)| {
            (!removed.contains(&component_of[index])).then_some(*triangle)
        })
        .collect::<Vec<_>>();
    if triangles.is_empty() {
        return Err(control.invalid("trim-domain carving removed every triangle"));
    }

    let result = ExactFaceTrimmedDelaunay {
        source_face_id: pslg.source_face_id.clone(),
        triangles,
        boundary_segments: constrained.protected_segments.clone(),
        removed_exterior_triangle_count,
        removed_hole_triangle_count,
    };
    super::validate::validate_face_trimmed_topology(
        &result,
        constrained,
        pslg,
        cancellation,
        options,
    )?;
    Ok(result)
}

fn components(
    triangle_count: usize,
    edges: &std::collections::BTreeMap<[u32; 2], Vec<super::super::topology::EdgeUse>>,
    protected: &BTreeSet<[u32; 2]>,
    control: &mut CarvingControl<'_>,
) -> Result<(Vec<usize>, Vec<Vec<usize>>), ExactFaceDelaunayError> {
    let mut adjacency = vec![Vec::new(); triangle_count];
    for (edge, uses) in edges {
        if !protected.contains(edge) && uses.len() == 2 {
            adjacency[uses[0].triangle_index].push(uses[1].triangle_index);
            adjacency[uses[1].triangle_index].push(uses[0].triangle_index);
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut component_of = vec![usize::MAX; triangle_count];
    let mut result = Vec::new();
    for seed in 0..triangle_count {
        if component_of[seed] != usize::MAX {
            continue;
        }
        let component_index = result.len();
        let mut stack = vec![seed];
        let mut component = Vec::new();
        component_of[seed] = component_index;
        while let Some(triangle) = stack.pop() {
            control.visit()?;
            component.push(triangle);
            for neighbor in adjacency[triangle].iter().rev().copied() {
                if component_of[neighbor] == usize::MAX {
                    component_of[neighbor] = component_index;
                    stack.push(neighbor);
                }
            }
        }
        component.sort_unstable();
        result.push(component);
    }
    Ok((component_of, result))
}

fn loop_winding(
    pslg: &ExactFacePslg,
    range: std::ops::Range<usize>,
    control: &mut CarvingControl<'_>,
) -> Result<PredicateSign, ExactFaceDelaunayError> {
    let vertices = pslg.segments[range]
        .iter()
        .map(|segment| segment.vertex_indices[0])
        .collect::<Vec<_>>();
    let extreme = (0..vertices.len())
        .min_by(|left, right| compare_vertex(vertices[*left], vertices[*right], pslg))
        .ok_or_else(|| control.invalid("trim loop has no vertices"))?;
    for backward in 1..vertices.len() {
        for forward in 1..vertices.len() {
            let previous = vertices[(extreme + vertices.len() - backward) % vertices.len()];
            let next = vertices[(extreme + forward) % vertices.len()];
            if previous == next {
                continue;
            }
            control.consume_predicate()?;
            let sign = orient2d([
                pslg.vertices[previous as usize].uv,
                pslg.vertices[vertices[extreme] as usize].uv,
                pslg.vertices[next as usize].uv,
            ])
            .map_err(|error| control.predicate_error(error))?;
            if sign != PredicateSign::Zero {
                return Ok(sign);
            }
        }
    }
    Err(control.invalid("trim loop has zero exact parametric area"))
}

fn compare_vertex(left: u32, right: u32, pslg: &ExactFacePslg) -> std::cmp::Ordering {
    let left_uv = pslg.vertices[left as usize].uv;
    let right_uv = pslg.vertices[right as usize].uv;
    left_uv[0]
        .total_cmp(&right_uv[0])
        .then_with(|| left_uv[1].total_cmp(&right_uv[1]))
        .then_with(|| left.cmp(&right))
}

fn sorted_edge(mut edge: [u32; 2]) -> [u32; 2] {
    edge.sort_unstable();
    edge
}

struct CarvingControl<'a> {
    pslg: &'a ExactFacePslg,
    cancellation: &'a dyn MeshingCancellationSignal,
    predicates_remaining: u64,
    visits_remaining: usize,
    check_interval: u64,
    work_since_check: u64,
}

impl<'a> CarvingControl<'a> {
    fn new(
        pslg: &'a ExactFacePslg,
        cancellation: &'a dyn MeshingCancellationSignal,
        options: ExactFaceDelaunayOptions,
    ) -> Self {
        Self {
            pslg,
            cancellation,
            predicates_remaining: options.maximum_predicate_evaluations,
            visits_remaining: options.maximum_triangles,
            check_interval: options.cancellation_check_interval,
            work_since_check: 0,
        }
    }

    fn consume_predicate(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.predicates_remaining = self.predicates_remaining.checked_sub(1).ok_or_else(|| {
            self.error(
                ExactFaceDelaunayErrorKind::SearchWorkLimit,
                "trim-domain predicate hard limit exceeded",
            )
        })?;
        self.work(1)
    }

    fn visit(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.visits_remaining = self.visits_remaining.checked_sub(1).ok_or_else(|| {
            self.error(
                ExactFaceDelaunayErrorKind::SearchWorkLimit,
                "trim-domain traversal hard limit exceeded",
            )
        })?;
        self.work(1)
    }

    fn work(&mut self, amount: u64) -> Result<(), ExactFaceDelaunayError> {
        self.work_since_check = self.work_since_check.saturating_add(amount);
        if self.work_since_check >= self.check_interval {
            self.work_since_check = 0;
            if self.cancellation.is_cancelled() {
                return Err(self.error(
                    ExactFaceDelaunayErrorKind::Cancelled,
                    "trim-domain carving cancelled",
                ));
            }
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.work_since_check = 0;
        if self.cancellation.is_cancelled() {
            Err(self.error(
                ExactFaceDelaunayErrorKind::Cancelled,
                "trim-domain carving cancelled",
            ))
        } else {
            Ok(())
        }
    }

    fn predicate_error(
        &self,
        error: runmat_meshing_core::PlanarPredicateError,
    ) -> ExactFaceDelaunayError {
        self.invalid(format!("invalid trim-domain predicate input: {error:?}"))
    }

    fn invalid(&self, reason: impl Into<String>) -> ExactFaceDelaunayError {
        self.error(ExactFaceDelaunayErrorKind::InvalidTopology, reason)
    }

    fn error(
        &self,
        kind: ExactFaceDelaunayErrorKind,
        reason: impl Into<String>,
    ) -> ExactFaceDelaunayError {
        ExactFaceDelaunayError::new(kind, &self.pslg.source_face_id, reason)
    }
}

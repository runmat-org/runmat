use std::collections::{BTreeMap, BTreeSet, VecDeque};

use runmat_meshing_core::{predicate::orient2d, MeshingCancellationSignal, PredicateSign};

use crate::{
    validate_exact_face_constrained_delaunay, ExactFaceBoundary, ExactFaceConstrainedDelaunay,
    ExactFaceDelaunayError, ExactFaceDelaunayErrorKind, ExactFaceDelaunayOptions, ExactFacePslg,
};

use super::ExactFaceTrimmedDelaunay;

pub fn validate_exact_face_trimmed_delaunay(
    trimmed: &ExactFaceTrimmedDelaunay,
    constrained: &ExactFaceConstrainedDelaunay,
    pslg: &ExactFacePslg,
    boundary: &ExactFaceBoundary,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<(), ExactFaceDelaunayError> {
    validate_exact_face_constrained_delaunay(constrained, pslg, boundary, cancellation, options)?;
    validate_face_trimmed_topology(trimmed, constrained, pslg, cancellation, options)
}

pub(crate) fn validate_face_trimmed_topology(
    trimmed: &ExactFaceTrimmedDelaunay,
    constrained: &ExactFaceConstrainedDelaunay,
    pslg: &ExactFacePslg,
    cancellation: &dyn MeshingCancellationSignal,
    options: ExactFaceDelaunayOptions,
) -> Result<(), ExactFaceDelaunayError> {
    crate::exact_cdt::validate_face_constrained_topology(constrained, pslg, cancellation, options)?;
    let mut control = ValidationControl::new(pslg, cancellation, options);
    control.checkpoint()?;
    if trimmed.source_face_id != pslg.source_face_id
        || trimmed.triangles.is_empty()
        || trimmed.triangles.len() > constrained.triangles.len()
        || trimmed.boundary_segments != constrained.protected_segments
    {
        return Err(control.invalid("trimmed Delaunay inventory is inconsistent"));
    }

    let constrained_inventory = constrained
        .triangles
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if trimmed.triangles.windows(2).any(|pair| pair[0] >= pair[1])
        || trimmed
            .triangles
            .iter()
            .any(|triangle| !constrained_inventory.contains(triangle))
    {
        return Err(control
            .invalid("trimmed triangles are noncanonical or absent from constrained topology"));
    }

    let protected = pslg
        .segments
        .iter()
        .map(|segment| ordered_edge(segment.vertex_indices))
        .collect::<BTreeSet<_>>();
    let retained_edges = independent_edge_uses(&trimmed.triangles);
    if retained_edges.values().any(|uses| uses.len() > 2) {
        return Err(control.invalid("trimmed topology contains a nonmanifold edge"));
    }
    let actual_boundary = retained_edges
        .iter()
        .filter_map(|(edge, uses)| (uses.len() == 1).then_some(*edge))
        .collect::<BTreeSet<_>>();
    if actual_boundary != protected {
        return Err(
            control.invalid("trimmed topology boundary differs from the authoritative PSLG")
        );
    }
    let referenced = trimmed
        .triangles
        .iter()
        .flat_map(|triangle| triangle.vertex_indices)
        .collect::<BTreeSet<_>>();
    if referenced.len() != pslg.vertices.len() {
        return Err(control.invalid("trimmed topology does not retain every PSLG vertex"));
    }
    validate_euler_characteristic(trimmed, &retained_edges, pslg, &control)?;

    // Reconstruct incidence and topological flood regions independently of the
    // carving implementation; sharing only the immutable stage contracts is intentional.
    let full_edges = independent_edge_uses(&constrained.triangles);
    let adjacency = unconstrained_adjacency(constrained.triangles.len(), &full_edges, &protected);
    let exterior_seeds = full_edges
        .iter()
        .filter(|(edge, uses)| uses.len() == 1 && !protected.contains(*edge))
        .map(|(_, uses)| uses[0].triangle_index)
        .collect::<BTreeSet<_>>();
    let exterior = flood(&adjacency, exterior_seeds, &mut control)?;

    let mut hole_seeds = BTreeSet::new();
    for loop_record in pslg.loops.iter().skip(1) {
        let start = loop_record.first_segment as usize;
        let end = start + loop_record.segment_count as usize;
        let winding = independent_loop_winding(pslg, start..end, &mut control)?;
        let mut loop_regions = BTreeSet::new();
        for segment in &pslg.segments[start..end] {
            let uses = full_edges
                .get(&ordered_edge(segment.vertex_indices))
                .ok_or_else(|| control.invalid("validator cannot find hole boundary edge"))?;
            if uses.len() != 2 {
                return Err(control.invalid("validator hole edge is not two-sided"));
            }
            let mut seed = None;
            for edge_use in uses {
                control.consume_predicate()?;
                let sign = orient2d([
                    pslg.vertices[segment.vertex_indices[0] as usize].uv,
                    pslg.vertices[segment.vertex_indices[1] as usize].uv,
                    pslg.vertices[edge_use.opposite_vertex as usize].uv,
                ])
                .map_err(|error| control.predicate_error(error))?;
                if sign == winding && seed.replace(edge_use.triangle_index).is_some() {
                    return Err(control.invalid("validator found two hole-interior sides"));
                }
            }
            loop_regions.insert(
                seed.ok_or_else(|| control.invalid("validator found no hole-interior side"))?,
            );
        }
        let flooded = flood(&adjacency, loop_regions, &mut control)?;
        if flooded.is_empty() {
            return Err(control.invalid("validator found an empty hole component"));
        }
        hole_seeds.extend(flooded);
    }
    if !exterior.is_disjoint(&hole_seeds) {
        return Err(control.invalid("validator found a hole connected to the exterior"));
    }

    let expected_removed = exterior
        .union(&hole_seeds)
        .copied()
        .collect::<BTreeSet<_>>();
    let expected_retained = constrained
        .triangles
        .iter()
        .enumerate()
        .filter_map(|(index, triangle)| (!expected_removed.contains(&index)).then_some(*triangle))
        .collect::<Vec<_>>();
    if trimmed.triangles != expected_retained {
        return Err(
            control.invalid("trimmed topology does not match independent oriented-wire carving")
        );
    }
    if trimmed.removed_exterior_triangle_count != exterior.len() as u64
        || trimmed.removed_hole_triangle_count != hole_seeds.len() as u64
    {
        return Err(control.invalid("trim-domain removal evidence is inconsistent"));
    }
    Ok(())
}

fn unconstrained_adjacency(
    triangle_count: usize,
    edges: &BTreeMap<[u32; 2], Vec<ValidatorEdgeUse>>,
    protected: &BTreeSet<[u32; 2]>,
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); triangle_count];
    for (edge, uses) in edges {
        if protected.contains(edge) || uses.len() != 2 {
            continue;
        }
        adjacency[uses[0].triangle_index].push(uses[1].triangle_index);
        adjacency[uses[1].triangle_index].push(uses[0].triangle_index);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adjacency
}

fn flood(
    adjacency: &[Vec<usize>],
    seeds: BTreeSet<usize>,
    control: &mut ValidationControl<'_>,
) -> Result<BTreeSet<usize>, ExactFaceDelaunayError> {
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::from_iter(seeds);
    while let Some(triangle) = queue.pop_front() {
        if !visited.insert(triangle) {
            continue;
        }
        control.visit()?;
        for neighbor in &adjacency[triangle] {
            if !visited.contains(neighbor) {
                queue.push_back(*neighbor);
            }
        }
    }
    Ok(visited)
}

fn independent_loop_winding(
    pslg: &ExactFacePslg,
    range: std::ops::Range<usize>,
    control: &mut ValidationControl<'_>,
) -> Result<PredicateSign, ExactFaceDelaunayError> {
    let vertices = pslg.segments[range]
        .iter()
        .map(|segment| segment.vertex_indices[0])
        .collect::<Vec<_>>();
    let extreme = (0..vertices.len())
        .max_by(|left, right| compare(vertices[*left], vertices[*right], pslg))
        .ok_or_else(|| control.invalid("validator trim loop has no vertices"))?;
    for forward in 1..vertices.len() {
        for backward in 1..vertices.len() {
            let next = vertices[(extreme + forward) % vertices.len()];
            let previous = vertices[(extreme + vertices.len() - backward) % vertices.len()];
            if next == previous {
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
    Err(control.invalid("validator trim loop has zero exact parametric area"))
}

fn validate_euler_characteristic(
    trimmed: &ExactFaceTrimmedDelaunay,
    edges: &BTreeMap<[u32; 2], Vec<ValidatorEdgeUse>>,
    pslg: &ExactFacePslg,
    control: &ValidationControl<'_>,
) -> Result<(), ExactFaceDelaunayError> {
    let vertices = trimmed
        .triangles
        .iter()
        .flat_map(|triangle| triangle.vertex_indices)
        .collect::<BTreeSet<_>>()
        .len() as i128;
    let characteristic = vertices - edges.len() as i128 + trimmed.triangles.len() as i128;
    let expected = 1i128 - (pslg.loops.len() as i128 - 1);
    if characteristic != expected {
        Err(control.invalid("trimmed topology has the wrong Euler characteristic"))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct ValidatorEdgeUse {
    triangle_index: usize,
    opposite_vertex: u32,
}

fn independent_edge_uses(
    triangles: &[crate::ExactFaceDelaunayTriangle],
) -> BTreeMap<[u32; 2], Vec<ValidatorEdgeUse>> {
    let mut edges = BTreeMap::<[u32; 2], Vec<ValidatorEdgeUse>>::new();
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        let [first, second, third] = triangle.vertex_indices;
        for (edge, opposite_vertex) in [
            (ordered_edge([first, second]), third),
            (ordered_edge([second, third]), first),
            (ordered_edge([third, first]), second),
        ] {
            edges.entry(edge).or_default().push(ValidatorEdgeUse {
                triangle_index,
                opposite_vertex,
            });
        }
    }
    edges
}

fn ordered_edge(mut edge: [u32; 2]) -> [u32; 2] {
    edge.sort_unstable();
    edge
}

fn compare(left: u32, right: u32, pslg: &ExactFacePslg) -> std::cmp::Ordering {
    let left_uv = pslg.vertices[left as usize].uv;
    let right_uv = pslg.vertices[right as usize].uv;
    left_uv[0]
        .total_cmp(&right_uv[0])
        .then_with(|| left_uv[1].total_cmp(&right_uv[1]))
        .then_with(|| left.cmp(&right))
}

struct ValidationControl<'a> {
    pslg: &'a ExactFacePslg,
    cancellation: &'a dyn MeshingCancellationSignal,
    predicates_remaining: u64,
    visits_remaining: usize,
    check_interval: u64,
    work_since_check: u64,
}

impl<'a> ValidationControl<'a> {
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
                ExactFaceDelaunayErrorKind::ResourceLimit,
                "trim validation predicate hard limit exceeded",
            )
        })?;
        self.work()
    }

    fn visit(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.visits_remaining = self.visits_remaining.checked_sub(1).ok_or_else(|| {
            self.error(
                ExactFaceDelaunayErrorKind::ResourceLimit,
                "trim validation traversal hard limit exceeded",
            )
        })?;
        self.work()
    }

    fn work(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.work_since_check = self.work_since_check.saturating_add(1);
        if self.work_since_check >= self.check_interval {
            self.checkpoint()?;
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), ExactFaceDelaunayError> {
        self.work_since_check = 0;
        if self.cancellation.is_cancelled() {
            Err(self.error(
                ExactFaceDelaunayErrorKind::Cancelled,
                "trim-domain validation cancelled",
            ))
        } else {
            Ok(())
        }
    }

    fn predicate_error(
        &self,
        error: runmat_meshing_core::PlanarPredicateError,
    ) -> ExactFaceDelaunayError {
        self.invalid(format!(
            "invalid trim validation predicate input: {error:?}"
        ))
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

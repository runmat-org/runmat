//! Merge canonical face-node streams with one heap entry per face. Output has
//! exact unique-node capacity; original node/use buffers move without cloning.
use super::*;
use crate::{ExactFaceMesh, ExactFaceMeshNodeUse};
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};
use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

fn allocation_error(error: runmat_geometry_core::GeometryEvaluationError) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(
        if error.kind == runmat_geometry_core::GeometryEvaluationErrorKind::Cancelled {
            ExactSurfaceMeshErrorKind::Cancelled
        } else {
            ExactSurfaceMeshErrorKind::ResourceLimit
        },
        error.to_string(),
    )
}
pub(in crate::surface_mesh) fn reserve<'a, T>(
    count: usize,
    allocations: Option<&'a dyn GeometryEvaluationControl>,
) -> Result<Option<GeometryAllocationReservation<'a>>, ExactSurfaceMeshError> {
    let bytes = (count as u64)
        .checked_mul(std::mem::size_of::<T>() as u64)
        .ok_or_else(|| limit("surface join buffer inventory overflow"))?;
    allocations
        .map(|control| GeometryAllocationReservation::new(control, bytes).map_err(allocation_error))
        .transpose()
}
fn unique_count(
    faces: &[ExactFaceMesh],
    options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
    allocations: Option<&dyn GeometryEvaluationControl>,
) -> Result<usize, ExactSurfaceMeshError> {
    let _heap_storage = reserve::<Reverse<(StableDigest, usize, usize)>>(faces.len(), allocations)?;
    let mut heap = BinaryHeap::with_capacity(faces.len());
    for (face_index, face) in faces.iter().enumerate() {
        checkpoint(control)?;
        if let Some(node) = face.nodes.first() {
            heap.push(Reverse((node.node_id, face_index, 0)));
        }
    }
    let mut previous = None;
    let mut count = 0usize;
    while let Some(Reverse((id, face_index, node_index))) = heap.pop() {
        checkpoint(control)?;
        if let Some(allocations) = allocations {
            allocations
                .consume_search_work(1)
                .map_err(allocation_error)?;
        }
        if previous != Some(id) {
            count = count
                .checked_add(1)
                .ok_or_else(|| limit("surface unique node count overflow"))?;
            previous = Some(id);
        }
        if count as u64 > options.maximum_nodes {
            return Err(limit("surface join exceeded its node hard limit"));
        }
        if let Some(node) = faces[face_index].nodes.get(node_index + 1) {
            heap.push(Reverse((node.node_id, face_index, node_index + 1)));
        }
    }
    Ok(count)
}
struct Head {
    node: ExactFaceMeshNode,
    face_index: usize,
}
impl PartialEq for Head {
    fn eq(&self, other: &Self) -> bool {
        (self.node.node_id, self.face_index) == (other.node.node_id, other.face_index)
    }
}
impl Eq for Head {}
impl PartialOrd for Head {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Head {
    fn cmp(&self, other: &Self) -> Ordering {
        (other.node.node_id, other.face_index).cmp(&(self.node.node_id, self.face_index))
    }
}

pub(super) fn merge(
    topology: &ExactBRepTopology,
    faces: &mut [ExactFaceMesh],
    options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
    allocations: Option<&dyn GeometryEvaluationControl>,
) -> Result<Vec<ExactFaceMeshNode>, ExactSurfaceMeshError> {
    let count = unique_count(faces, options, control, allocations)?;
    let mut output_storage = reserve::<ExactFaceMeshNode>(count, allocations)?;
    let _heads_storage = reserve::<Head>(faces.len(), allocations)?;
    let _iterators_storage =
        reserve::<std::vec::IntoIter<ExactFaceMeshNode>>(faces.len(), allocations)?;
    let mut output = Vec::<ExactFaceMeshNode>::with_capacity(count);
    let mut iterators = Vec::with_capacity(faces.len());
    let mut heap = BinaryHeap::with_capacity(faces.len());
    for (face_index, face) in faces.iter_mut().enumerate() {
        checkpoint(control)?;
        let mut nodes = std::mem::take(&mut face.nodes).into_iter();
        if let Some(node) = nodes.next() {
            heap.push(Head { node, face_index });
        }
        iterators.push(nodes);
    }
    while let Some(Head {
        node: mut next,
        face_index,
    }) = heap.pop()
    {
        checkpoint(control)?;
        if let Some(allocations) = allocations {
            allocations
                .consume_search_work(1)
                .map_err(allocation_error)?;
        }
        if let Some(previous) = output
            .last_mut()
            .filter(|node| node.node_id == next.node_id)
        {
            let tolerance = previous
                .uses
                .iter()
                .chain(&next.uses)
                .flat_map(|usage| &usage.exact_edge_parameters)
                .filter_map(|parameter| {
                    topology
                        .edges
                        .binary_search_by(|edge| edge.id.cmp(&parameter.source_edge_id))
                        .ok()
                        .map(|index| topology.edges[index].tolerance_m)
                })
                .map(|tolerance| tolerance.canonical_value())
                .fold(options.coordinate_tolerance_m.canonical_value(), f64::max);
            let discrepancy = distance(previous.point_m, next.point_m);
            if discrepancy > tolerance {
                return Err(invalid(format!("shared face node {:?} disagrees by {discrepancy:e} m, exceeding admitted {tolerance:e} m",next.node_id)).with_face(&faces[face_index].source_face_id));
            }
            for usage in &next.uses {
                checkpoint(control)?;
                if previous.uses.iter().any(|existing| {
                    existing.source_face_id == usage.source_face_id
                        && existing.chart_id == usage.chart_id
                        && existing.uv == usage.uv
                }) {
                    return Err(invalid("surface join contains a duplicate node use")
                        .with_face(&faces[face_index].source_face_id));
                }
            }
            let required = previous
                .uses
                .len()
                .checked_add(next.uses.len())
                .ok_or_else(|| limit("joined node-use inventory overflow"))?;
            if previous.uses.capacity() < required {
                let mut storage = reserve::<ExactFaceMeshNodeUse>(required, allocations)?;
                previous
                    .uses
                    .try_reserve_exact(required - previous.uses.len())
                    .map_err(|error| limit(&error.to_string()))?;
                if let Some(storage) = &mut storage {
                    storage.retain_bytes(
                        (previous.uses.capacity() * std::mem::size_of::<ExactFaceMeshNodeUse>())
                            as u64,
                    );
                }
            }
            previous.uses.append(&mut next.uses);
        } else {
            output.push(next);
        }
        if let Some(node) = iterators[face_index].next() {
            heap.push(Head { node, face_index });
        }
    }
    if output.len() != count {
        return Err(invalid(
            "surface stream merge differs from its unique-node inventory",
        ));
    }
    if let Some(storage) = &mut output_storage {
        storage.retain_bytes((output.capacity() * std::mem::size_of::<ExactFaceMeshNode>()) as u64);
    }
    Ok(output)
}

#[cfg(test)]
mod tests;

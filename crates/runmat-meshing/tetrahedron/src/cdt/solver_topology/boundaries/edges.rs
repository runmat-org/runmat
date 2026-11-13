//! Flat boundary incidence borrows original provenance until final ownership.
use super::*;
use runmat_geometry_core::{GeometryAllocationReservation, PersistentEntityId};
use runmat_meshing_core::{solver_boundary_edge_identity, BoundaryEdgeOrder, SolverBoundaryEdge};

#[derive(Clone, Copy)]
struct Row<'a> {
    nodes: [u64; 2],
    face: u64,
    provenance: &'a [PersistentEntityId],
}

pub(in crate::cdt::solver_topology) fn build_edges(
    input: &ProjectionInput<'_>,
    faces: &[SolverBoundaryFace],
    options: DelaunaySolverTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<SolverBoundaryEdge>, DelaunaySolverTopologyError> {
    let count = faces.len().checked_mul(3).ok_or_else(overflow)?;
    let bytes = (count as u64)
        .checked_mul(std::mem::size_of::<Row<'_>>() as u64)
        .ok_or_else(overflow)?;
    let control = input.exact_evaluation.map(|evaluation| evaluation.control);
    let _rows_storage = control
        .map(|control| GeometryAllocationReservation::new(control, bytes))
        .transpose()
        .map_err(error::geometry)?;
    let mut rows = Vec::with_capacity(count);
    let identity = |id: u64| {
        id.checked_sub(1)
            .and_then(|index| usize::try_from(index).ok())
            .and_then(|index| input.volume_mesh.topology.nodes.get(index))
            .map(|node| node.identity)
            .ok_or_else(|| invalid_mesh("boundary edge references an absent canonical node"))
    };
    for (work, face) in faces.iter().enumerate() {
        checkpoint(work as u64, options, cancellation)?;
        let corners: [u64; 3] =
            face.node_ids.as_slice().try_into().map_err(|_| {
                invalid_mesh("boundary edge preparation requires three corner nodes")
            })?;
        for pair in [[0, 1], [0, 2], [1, 2]] {
            let mut nodes = [corners[pair[0]], corners[pair[1]]];
            nodes.sort_unstable();
            let mut identities = [identity(nodes[0])?, identity(nodes[1])?];
            identities.sort_unstable();
            let provenance = match input
                .volume_mesh
                .provenance
                .segments
                .binary_search_by_key(&identities, |segment| segment.node_identities)
            {
                Ok(index) => input.volume_mesh.provenance.segments[index]
                    .entity_ids
                    .as_slice(),
                Err(_) => face.provenance.as_slice(),
            };
            rows.push(Row {
                nodes,
                face: face.face_id,
                provenance,
            });
        }
    }
    let scratch_storage = control
        .map(|control| GeometryAllocationReservation::new(control, bytes))
        .transpose()
        .map_err(error::geometry)?;
    let mut scratch = rows.clone();
    runmat_cpu_execution::sort_copy_with_checkpoint(
        &mut rows,
        &mut scratch,
        |a: &Row<'_>, b: &Row<'_>| a.nodes.cmp(&b.nodes).then_with(|| a.face.cmp(&b.face)),
        || checkpoint(0, options, cancellation),
    )?;
    drop(scratch);
    drop(scratch_storage);
    let mut groups = 0usize;
    let mut previous = None;
    for (work, row) in rows.iter().enumerate() {
        checkpoint(work as u64, options, cancellation)?;
        if previous != Some(row.nodes) {
            groups += 1;
            previous = Some(row.nodes);
        }
    }
    if groups as u64 > options.maximum_boundary_edges {
        return Err(error::failure(
            DelaunaySolverTopologyErrorKind::ResourceLimit,
            format!(
                "boundary edge inventory exceeds its hard limit of {}",
                options.maximum_boundary_edges
            ),
        ));
    }
    let references = super::super::construction::MAX_PROVENANCE_PER_ENTITY;
    let reference_bytes = (references * std::mem::size_of::<&PersistentEntityId>()) as u64;
    let _reference_storage = control
        .map(|control| GeometryAllocationReservation::new(control, reference_bytes))
        .transpose()
        .map_err(error::geometry)?;
    let mut provenance = Vec::with_capacity(references);
    let storage = super::super::storage::Storage::for_input(input);
    let mut result = storage.vector(groups)?;
    let mut start = 0;
    while start < rows.len() {
        checkpoint(start as u64, options, cancellation)?;
        let nodes = rows[start].nodes;
        let end = start + rows[start..].partition_point(|row| row.nodes == nodes);
        provenance.clear();
        let mut adjacent = storage.vector(end - start)?;
        for row in &rows[start..end] {
            checkpoint(adjacent.len() as u64, options, cancellation)?;
            if adjacent.last() != Some(&row.face) {
                adjacent.push(row.face);
            }
            for entity in row.provenance {
                super::super::construction::insert_entity(&mut provenance, entity)?;
            }
        }
        if provenance.is_empty() {
            return Err(invalid_mesh(
                "boundary edge provenance is empty or exceeds its bound",
            ));
        }
        let owned_provenance = storage.entities(provenance.iter().copied())?;
        let capacity = match input.request.element_order {
            runmat_meshing_core::ElementOrder::Tet4 => 2,
            runmat_meshing_core::ElementOrder::Tet10 => 3,
        };
        let mut node_ids = storage.vector(capacity)?;
        node_ids.extend(nodes);
        result.push(SolverBoundaryEdge {
            edge_id: result.len() as u64 + 1,
            stable_identity: solver_boundary_edge_identity([
                identity(nodes[0])?,
                identity(nodes[1])?,
            ]),
            order: BoundaryEdgeOrder::Line2,
            node_ids,
            adjacent_boundary_face_ids: adjacent,
            provenance: owned_provenance,
        });
        start = end;
    }
    Ok(result)
}
fn overflow() -> DelaunaySolverTopologyError {
    error::failure(
        DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded,
        "boundary edge storage inventory overflow",
    )
}

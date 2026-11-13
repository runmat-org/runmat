//! Flat boundary indexes borrow original lineage; no repeated owned entity trees.
use super::*;
use runmat_geometry_core::{GeometryAllocationReservation, GeometryEvaluationControl};
use std::ops::Range;
type SurfaceRow<'a> = ([u64; 2], SurfaceUse<'a>);
type SurfaceGroup = ([u64; 2], Range<usize>);
pub(in crate::cdt::solver_topology::order_elevation) struct BoundarySurfaceUses<'a> {
    uses: Vec<SurfaceUse<'a>>,
    groups: Vec<SurfaceGroup>,
    _storage: GeometryAllocationReservation<'a>,
}
impl<'a> BoundarySurfaceUses<'a> {
    pub fn get(&self, edge: &[u64; 2]) -> Option<&[SurfaceUse<'a>]> {
        let index = self
            .groups
            .binary_search_by_key(edge, |group| group.0)
            .ok()?;
        Some(&self.uses[self.groups[index].1.clone()])
    }
    pub fn iter(&self) -> impl Iterator<Item = (&[u64; 2], &[SurfaceUse<'a>])> {
        self.groups
            .iter()
            .map(|(edge, range)| (edge, &self.uses[range.clone()]))
    }
}
pub(in crate::cdt::solver_topology::order_elevation) struct BoundaryExactEdges<'a> {
    edges: Vec<([u64; 2], &'a PersistentEntityId)>,
    _storage: GeometryAllocationReservation<'a>,
}
impl<'a> BoundaryExactEdges<'a> {
    pub fn get(&self, edge: &[u64; 2]) -> Option<&'a PersistentEntityId> {
        self.edges
            .binary_search_by_key(edge, |entry| entry.0)
            .ok()
            .map(|index| self.edges[index].1)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&[u64; 2], &'a PersistentEntityId)> {
        self.edges.iter().map(|(edge, id)| (edge, *id))
    }
}
pub(in crate::cdt::solver_topology::order_elevation) fn surface_uses<'a>(
    input: &ProjectionInput<'a>,
    topology: &SolverMeshTopology,
    control: &'a dyn GeometryEvaluationControl,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<BoundarySurfaceUses<'a>, DelaunaySolverTopologyError> {
    let count = topology
        .boundary_faces
        .len()
        .checked_mul(3)
        .ok_or_else(overflow)?;
    let rows_storage = GeometryAllocationReservation::new(control, bytes::<SurfaceRow<'_>>(count)?)
        .map_err(error::geometry)?;
    let mut rows = Vec::with_capacity(count);
    for face in &topology.boundary_faces {
        boundary_parameter_checkpoint(cancellation)?;
        let facet = input
            .volume_mesh
            .provenance
            .facets
            .get(face.face_id as usize - 1)
            .ok_or_else(|| invalid("boundary face has no recovered facet lineage"))?;
        let source_face_id = facet
            .entity_ids
            .iter()
            .find(|id| id.kind == PersistentEntityKind::Face)
            .ok_or_else(|| invalid("recovered facet has no exact face"))?;
        let usage = SurfaceUse {
            source_face_id,
            chart_id: facet.chart_id,
        };
        for [a, b] in [[0, 1], [1, 2], [2, 0]] {
            rows.push((sorted_edge([face.node_ids[a], face.node_ids[b]]), usage));
        }
    }
    let sort_storage = GeometryAllocationReservation::new(control, bytes::<SurfaceRow<'_>>(count)?)
        .map_err(error::geometry)?;
    let mut scratch = rows.clone();
    runmat_cpu_execution::sort_copy_with_checkpoint(
        &mut rows,
        &mut scratch,
        |a: &SurfaceRow<'_>, b: &SurfaceRow<'_>| a.cmp(b),
        || control.checkpoint().map_err(error::geometry),
    )?;
    drop(scratch);
    drop(sort_storage);
    rows.dedup();
    let mut groups_count = 0;
    let mut previous = None;
    for (edge, _) in &rows {
        control.checkpoint().map_err(error::geometry)?;
        if previous != Some(*edge) {
            groups_count += 1;
            previous = Some(*edge);
        }
    }
    let requested = bytes::<SurfaceUse<'_>>(rows.len())?
        .checked_add(bytes::<SurfaceGroup>(groups_count)?)
        .ok_or_else(overflow)?;
    let storage =
        GeometryAllocationReservation::new(control, requested).map_err(error::geometry)?;
    let mut uses = Vec::with_capacity(rows.len());
    let mut groups = Vec::with_capacity(groups_count);
    let mut start = 0;
    while start < rows.len() {
        control.checkpoint().map_err(error::geometry)?;
        let end = start + rows[start..].partition_point(|row| row.0 == rows[start].0);
        groups.push((rows[start].0, start..end));
        uses.extend(rows[start..end].iter().map(|row| row.1));
        start = end;
    }
    drop(rows);
    drop(rows_storage);
    Ok(BoundarySurfaceUses {
        uses,
        groups,
        _storage: storage,
    })
}
pub(in crate::cdt::solver_topology::order_elevation) fn exact_edges<'a>(
    input: &ProjectionInput<'a>,
    topology: &SolverMeshTopology,
    control: &'a dyn GeometryEvaluationControl,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<BoundaryExactEdges<'a>, DelaunaySolverTopologyError> {
    let count = topology
        .boundary_edges
        .iter()
        .filter(|edge| {
            edge.provenance
                .iter()
                .any(|id| id.kind == PersistentEntityKind::Edge)
        })
        .count();
    let storage = GeometryAllocationReservation::new(
        control,
        bytes::<([u64; 2], &PersistentEntityId)>(count)?,
    )
    .map_err(error::geometry)?;
    let mut edges = Vec::with_capacity(count);
    for edge in &topology.boundary_edges {
        boundary_parameter_checkpoint(cancellation)?;
        let mut owners = edge
            .provenance
            .iter()
            .filter(|id| id.kind == PersistentEntityKind::Edge);
        let Some(owner) = owners.next() else {
            continue;
        };
        if owners.next().is_some() {
            return Err(invalid(
                "one boundary mesh edge has multiple exact edge owners",
            ));
        }
        let key = sorted_edge([edge.node_ids[0], edge.node_ids[1]]);
        let mut identities = key.map(|node| topology.nodes[(node - 1) as usize].stable_identity);
        identities.sort_unstable();
        let index = input
            .volume_mesh
            .provenance
            .segments
            .binary_search_by_key(&identities, |segment| segment.node_identities)
            .map_err(|_| invalid("exact boundary edge has no original segment lineage"))?;
        let id = input.volume_mesh.provenance.segments[index]
            .entity_ids
            .iter()
            .find(|id| *id == owner)
            .ok_or_else(|| invalid("exact boundary edge owner differs from original lineage"))?;
        edges.push((key, id));
    }
    edges.sort_unstable_by_key(|edge| edge.0);
    Ok(BoundaryExactEdges {
        edges,
        _storage: storage,
    })
}
fn bytes<T>(count: usize) -> Result<u64, DelaunaySolverTopologyError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(overflow)
}
fn overflow() -> DelaunaySolverTopologyError {
    error::failure(
        DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded,
        "boundary source inventory overflow",
    )
}

//! Borrow original exact sources in bounded flat indexes; storage follows ownership.
use super::SurfaceUse;
use crate::cdt::solver_topology::{
    error, DelaunaySolverTopologyError, DelaunaySolverTopologyErrorKind, ProjectionInput,
};
use runmat_geometry_core::{
    model::PreparedOccurrenceTransforms, ExactCoedge, ExactEdge, ExactFace,
    GeometryAllocationReservation, GeometryEvaluationControl, PersistentEntityId,
};
use runmat_meshing_core::StableDigest;
use runmat_meshing_surface::{ExactFaceMeshNode, ExactSurfaceMesh};
mod spatial;
use spatial::SurfaceLookup;
type ImageKey<'a> = (&'a PersistentEntityId, StableDigest, &'a PersistentEntityId);
type ImageRow<'a> = (ImageKey<'a>, &'a PersistentEntityId);
type FallbackRow<'a> = (
    (&'a PersistentEntityId, &'a PersistentEntityId),
    &'a PersistentEntityId,
);

pub(in crate::cdt::solver_topology::order_elevation) struct NodeSources<'a> {
    rows: Vec<(StableDigest, &'a ExactFaceMeshNode)>,
    _storage: GeometryAllocationReservation<'a>,
}
impl<'a> NodeSources<'a> {
    fn new(
        surface: &'a ExactSurfaceMesh,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        let storage = GeometryAllocationReservation::new(
            control,
            bytes::<(StableDigest, &ExactFaceMeshNode)>(surface.nodes.len())?,
        )
        .map_err(error::geometry)?;
        let mut rows = Vec::with_capacity(surface.nodes.len());
        for node in &surface.nodes {
            work(control)?;
            rows.push((node.node_id, node));
        }
        sort(&mut rows, |a, b| a.0.cmp(&b.0), control)?;
        if rows.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(invalid("duplicate source node identity"));
        }
        Ok(Self {
            rows,
            _storage: storage,
        })
    }
    pub(in crate::cdt::solver_topology::order_elevation) fn get(
        &self,
        id: &StableDigest,
    ) -> Option<&&'a ExactFaceMeshNode> {
        self.rows
            .binary_search_by_key(id, |row| row.0)
            .ok()
            .map(|index| &self.rows[index].1)
    }
}

pub(in crate::cdt::solver_topology::order_elevation) struct ParameterSources<'a> {
    pub(super) nodes: NodeSources<'a>,
    pub(super) transforms: PreparedOccurrenceTransforms<'a>,
    edges: Vec<&'a ExactEdge>,
    faces: Vec<&'a ExactFace>,
    coedges: Vec<&'a ExactCoedge>,
    images: Vec<ImageRow<'a>>,
    fallback_images: Vec<FallbackRow<'a>>,
    surface: std::sync::OnceLock<Result<SurfaceLookup<'a>, DelaunaySolverTopologyError>>,
    source_surface: &'a ExactSurfaceMesh,
    source_control: &'a dyn GeometryEvaluationControl,
    surface_tolerance: f64,
    _storage: GeometryAllocationReservation<'a>,
}
impl<'a> ParameterSources<'a> {
    pub(in crate::cdt::solver_topology::order_elevation) fn prepare(
        input: &ProjectionInput<'a>,
        control: &'a dyn GeometryEvaluationControl,
    ) -> Result<Self, DelaunaySolverTopologyError> {
        work(control)?;
        let topology = input.exact_topology;
        let image_count = input
            .exact_surface
            .nodes
            .iter()
            .try_fold(0usize, |count, node| {
                node.uses.iter().try_fold(count, |count, image| {
                    work(control)?;
                    count
                        .checked_add(image.exact_edge_parameters.len())
                        .ok_or_else(|| resource("source image inventory overflow"))
                })
            })?;
        let requested = bytes::<&ExactEdge>(topology.edges.len())?
            .checked_add(bytes::<&ExactFace>(topology.faces.len())?)
            .and_then(|subtotal| {
                subtotal.checked_add(bytes::<&ExactCoedge>(topology.coedges.len()).ok()?)
            })
            .and_then(|subtotal| subtotal.checked_add(bytes::<ImageRow<'_>>(image_count).ok()?))
            .and_then(|subtotal| {
                subtotal.checked_add(bytes::<FallbackRow<'_>>(topology.coedges.len()).ok()?)
            })
            .ok_or_else(|| resource("source storage overflow"))?;
        let storage =
            GeometryAllocationReservation::new(control, requested).map_err(error::geometry)?;
        let nodes = NodeSources::new(input.exact_surface, control)?;
        let mut edges = Vec::with_capacity(topology.edges.len());
        edges.extend(topology.edges.iter());
        sort(&mut edges, |a, b| a.id.cmp(&b.id), control)?;
        let mut faces = Vec::with_capacity(topology.faces.len());
        faces.extend(topology.faces.iter());
        sort(&mut faces, |a, b| a.id.cmp(&b.id), control)?;
        let mut coedges = Vec::with_capacity(topology.coedges.len());
        coedges.extend(topology.coedges.iter());
        sort(&mut coedges, |a, b| a.id.cmp(&b.id), control)?;
        let mut images = Vec::with_capacity(image_count);
        for node in &input.exact_surface.nodes {
            for image in &node.uses {
                for binding in &image.exact_edge_parameters {
                    work(control)?;
                    let index = coedges
                        .binary_search_by(|coedge| coedge.id.cmp(&binding.source_coedge_id))
                        .map_err(|_| invalid("source image references an absent coedge"))?;
                    let coedge = coedges[index];
                    if coedge.face_id != image.source_face_id
                        || coedge.edge_id != binding.source_edge_id
                    {
                        return Err(invalid(
                            "source coedge image has the wrong face or edge owner",
                        ));
                    }
                    images.push((
                        (
                            &image.source_face_id,
                            image.chart_id,
                            &binding.source_edge_id,
                        ),
                        &binding.source_coedge_id,
                    ));
                }
            }
        }
        sort(&mut images, |a, b| a.cmp(b), control)?;
        images.dedup();
        let mut fallback_images = Vec::with_capacity(coedges.len());
        for coedge in &coedges {
            work(control)?;
            fallback_images.push(((&coedge.face_id, &coedge.edge_id), &coedge.id));
        }
        sort(&mut fallback_images, |a, b| a.cmp(b), control)?;
        fallback_images.dedup();
        let tolerance = input
            .request
            .tolerance
            .source_tolerance_m
            .canonical_value()
            .max(input.request.tolerance.absolute_floor_m.canonical_value())
            .max(
                input
                    .request
                    .tolerance
                    .maximum_healing_displacement_m
                    .canonical_value(),
            );
        let surface = std::sync::OnceLock::new();
        Ok(Self {
            nodes,
            edges,
            faces,
            coedges,
            images,
            fallback_images,
            surface,
            source_surface: input.exact_surface,
            source_control: control,
            surface_tolerance: tolerance,
            transforms: PreparedOccurrenceTransforms::new(topology, control)
                .map_err(error::geometry)?,
            _storage: storage,
        })
    }
    pub(super) fn edge(
        &self,
        id: &PersistentEntityId,
    ) -> Result<&'a ExactEdge, DelaunaySolverTopologyError> {
        self.edges
            .binary_search_by(|edge| edge.id.cmp(id))
            .ok()
            .map(|index| self.edges[index])
            .ok_or_else(|| invalid("boundary edge references an absent exact edge"))
    }
    pub(super) fn face(
        &self,
        id: &PersistentEntityId,
    ) -> Result<&'a ExactFace, DelaunaySolverTopologyError> {
        self.faces
            .binary_search_by(|face| face.id.cmp(id))
            .ok()
            .map(|index| self.faces[index])
            .ok_or_else(|| invalid("boundary use references an absent exact face"))
    }
    pub(super) fn coedge(
        &self,
        id: &PersistentEntityId,
    ) -> Result<&'a ExactCoedge, DelaunaySolverTopologyError> {
        self.coedges
            .binary_search_by(|coedge| coedge.id.cmp(id))
            .ok()
            .map(|index| self.coedges[index])
            .ok_or_else(|| invalid("exact chart references an absent coedge"))
    }
    pub(super) fn coedge_images<'query>(
        &'query self,
        edge: &'query PersistentEntityId,
        surface: &'query SurfaceUse,
    ) -> Result<
        impl Iterator<Item = &'a PersistentEntityId> + Clone + 'query,
        DelaunaySolverTopologyError,
    > {
        let key = (surface.source_face_id, surface.chart_id, edge);
        let start = self.images.partition_point(|row| row.0 < key);
        let end = start + self.images[start..].partition_point(|row| row.0 == key);
        let explicit = &self.images[start..end];
        let fallback = if explicit.is_empty() {
            let key = (surface.source_face_id, edge);
            let start = self.fallback_images.partition_point(|row| row.0 < key);
            let end = start + self.fallback_images[start..].partition_point(|row| row.0 == key);
            &self.fallback_images[start..end]
        } else {
            &[]
        };
        if explicit.is_empty() && fallback.is_empty() {
            return Err(invalid("exact chart has no coedge image"));
        }
        Ok(explicit
            .iter()
            .map(|row| row.1)
            .chain(fallback.iter().map(|row| row.1)))
    }
    pub(super) fn triangle_candidates(
        &self,
        surface: &SurfaceUse,
        point: [f64; 3],
        control: &dyn GeometryEvaluationControl,
        result: &mut Vec<usize>,
    ) -> Result<(), DelaunaySolverTopologyError> {
        let index = self.surface.get_or_init(|| {
            SurfaceLookup::prepare(
                self.source_surface,
                self.surface_tolerance,
                &self.nodes,
                self.source_control,
            )
        });
        match index {
            Ok(index) => index.candidates_into(surface, point, control, result),
            Err(error) => Err(error.clone()),
        }
    }
}
fn sort<T: Copy>(
    values: &mut [T],
    compare: impl Fn(&T, &T) -> std::cmp::Ordering,
    control: &dyn GeometryEvaluationControl,
) -> Result<(), DelaunaySolverTopologyError> {
    let reservation = GeometryAllocationReservation::new(control, bytes::<T>(values.len())?)
        .map_err(error::geometry)?;
    let mut scratch = values.to_vec();
    runmat_cpu_execution::sort_copy_with_checkpoint(values, &mut scratch, compare, || {
        work(control)
    })?;
    drop(scratch);
    drop(reservation);
    Ok(())
}
fn bytes<T>(count: usize) -> Result<u64, DelaunaySolverTopologyError> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| resource("source storage overflow"))
}
pub(super) fn work(
    control: &dyn GeometryEvaluationControl,
) -> Result<(), DelaunaySolverTopologyError> {
    control.checkpoint().map_err(error::geometry)?;
    control.consume_search_work(1).map_err(error::geometry)
}
fn invalid(reason: &'static str) -> DelaunaySolverTopologyError {
    error::failure(DelaunaySolverTopologyErrorKind::InvalidMesh, reason)
}
fn resource(reason: &'static str) -> DelaunaySolverTopologyError {
    error::failure(
        DelaunaySolverTopologyErrorKind::MemoryBudgetExceeded,
        reason,
    )
}
#[cfg(test)]
mod tests;

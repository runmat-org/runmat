//! Borrowed identity dictionaries and compact rows, reused by profiling and encoding.
use super::quality::QualityEncoding;
use super::rows::MeshRows;
use super::*;
use crate::*;
use serde::{ser::SerializeSeq, Serialize};

pub(crate) struct FaceEncoding<'a> {
    mesh: MeshRows<'a>,
    entities: Vec<&'a PersistentEntityId>,
    charts: Vec<StableDigest>,
    quality: QualityEncoding<'a>,
}
impl<'a> FaceEncoding<'a> {
    pub fn preparation_bytes(
        mesh: &ExactFaceMesh,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<u64, ExactSurfaceMeshError> {
        Self::preparation_for_rows(MeshRows::face(mesh), control)
    }
    pub(super) fn preparation_for_rows(
        mesh: MeshRows<'_>,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<u64, ExactSurfaceMeshError> {
        let references = Self::reference_count(mesh, control)?;
        let charts = Self::chart_count(mesh, control)?;
        references
            .checked_mul(std::mem::size_of::<&PersistentEntityId>())
            .and_then(|n| {
                charts
                    .checked_mul(std::mem::size_of::<StableDigest>())
                    .and_then(|c| n.checked_add(c))
            })
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| invalid("dictionary storage overflow"))
    }
    fn reference_count(
        mesh: MeshRows<'_>,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<usize, ExactSurfaceMeshError> {
        let mut count = 1usize
            .checked_add(mesh.triangles.len())
            .ok_or_else(|| invalid("entity inventory overflow"))?;
        for (index, node) in mesh.nodes.iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            for usage in &node.uses {
                count = count
                    .checked_add(1)
                    .and_then(|n| {
                        usage
                            .exact_edge_parameters
                            .len()
                            .checked_mul(2)
                            .and_then(|p| n.checked_add(p))
                    })
                    .ok_or_else(|| invalid("entity inventory overflow"))?;
            }
        }
        count
            .checked_add(
                mesh.boundary_segments
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| invalid("boundary inventory overflow"))?,
            )
            .ok_or_else(|| invalid("entity inventory overflow"))
    }
    fn chart_count(
        mesh: MeshRows<'_>,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<usize, ExactSurfaceMeshError> {
        let mut count = mesh.triangles.len();
        for (index, node) in mesh.nodes.iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            count = count
                .checked_add(node.uses.len())
                .ok_or_else(|| invalid("chart inventory overflow"))?;
        }
        Ok(count)
    }
    pub fn new(
        mesh: &'a ExactFaceMesh,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<Self, ExactSurfaceMeshError> {
        Self::prepare_rows(MeshRows::face(mesh), control)
    }
    pub(super) fn prepare_rows(
        mesh: MeshRows<'a>,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<Self, ExactSurfaceMeshError> {
        checkpoint(control)?;
        let mut entities = Vec::with_capacity(Self::reference_count(mesh, control)?);
        let charts_count = Self::chart_count(mesh, control)?;
        let mut charts = Vec::with_capacity(charts_count);
        entities.push(mesh.source_face_id);
        for (index, node) in mesh.nodes.iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            for usage in &node.uses {
                entities.push(&usage.source_face_id);
                charts.push(usage.chart_id);
                for edge in &usage.exact_edge_parameters {
                    entities.extend([&edge.source_coedge_id, &edge.source_edge_id]);
                }
            }
        }
        for (index, triangle) in mesh.triangles.iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            entities.push(&triangle.source_face_id);
            charts.push(triangle.chart_id);
        }
        for (index, edge) in mesh.boundary_segments.iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            entities.extend([&edge.source_coedge_id, &edge.source_edge_id]);
        }
        unique(&mut entities, control)?;
        unique(&mut charts, control)?;
        Ok(Self {
            mesh,
            entities,
            charts,
            quality: QualityEncoding::new(mesh.triangles, control)?,
        })
    }
    pub fn retained_preparation_bytes(&self) -> u64 {
        (self.entities.capacity() * std::mem::size_of::<&PersistentEntityId>()
            + self.charts.capacity() * std::mem::size_of::<StableDigest>()) as u64
    }
    fn entity<E: serde::ser::Error>(&self, id: &PersistentEntityId) -> Result<u32, E> {
        u32::try_from(
            self.entities
                .binary_search(&id)
                .map_err(|_| E::custom("entity absent from surface dictionary"))?,
        )
        .map_err(E::custom)
    }
    fn chart<E: serde::ser::Error>(&self, id: StableDigest) -> Result<u32, E> {
        u32::try_from(
            self.charts
                .binary_search(&id)
                .map_err(|_| E::custom("chart absent from surface dictionary"))?,
        )
        .map_err(E::custom)
    }
    fn node<E: serde::ser::Error>(&self, id: StableDigest) -> Result<NodeReference, E> {
        match self
            .mesh
            .nodes
            .binary_search_by_key(&id, |node| node.node_id)
        {
            Ok(index) => Ok(NodeReference::Index(
                u32::try_from(index).map_err(E::custom)?,
            )),
            Err(_) => Ok(NodeReference::Foreign(Box::new(DigestWords(id)))),
        }
    }
}
fn unique<T: Ord + Copy>(
    values: &mut Vec<T>,
    control: &dyn MeshingCancellationSignal,
) -> Result<(), ExactSurfaceMeshError> {
    runmat_cpu_execution::sort_in_place_with_checkpoint(values, Ord::cmp, || checkpoint(control))?;
    let mut retained = 0;
    for index in 0..values.len() {
        if index.is_multiple_of(128) {
            checkpoint(control)?;
        }
        if retained == 0 || values[index] != values[retained - 1] {
            values[retained] = values[index];
            retained += 1;
        }
    }
    values.truncate(retained);
    Ok(())
}
impl Serialize for FaceEncoding<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Face<'a, 'b> {
            schema_version: u16,
            source_face: u32,
            entities: &'a [&'b PersistentEntityId],
            charts: Charts<'a>,
            nodes: Nodes<'a, 'b>,
            triangles: Triangles<'a, 'b>,
            quality: &'a QualityEncoding<'b>,
            boundary_segments: Boundaries<'a, 'b>,
            joined_chart_cuts: &'b [ExactFaceMeshJoinedCut],
            #[serde(with = "runmat_types::physical::canonical_scalar")]
            maximum_chordal_deviation_m: runmat_types::physical::Length,
            maximum_normal_deviation_rad: f64,
        }
        Face {
            schema_version: self.mesh.schema_version,
            source_face: self.entity(self.mesh.source_face_id)?,
            entities: &self.entities,
            charts: Charts(&self.charts),
            nodes: Nodes(self),
            triangles: Triangles(self),
            quality: &self.quality,
            boundary_segments: Boundaries(self),
            joined_chart_cuts: self.mesh.joined_chart_cuts,
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(
                self.mesh.maximum_chordal_deviation_m.canonical_value(),
            ),
            maximum_normal_deviation_rad: self.mesh.maximum_normal_deviation_rad,
        }
        .serialize(serializer)
    }
}
struct Charts<'a>(&'a [StableDigest]);
impl Serialize for Charts<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for id in self.0 {
            seq.serialize_element(&DigestWords(*id))?;
        }
        seq.end()
    }
}
struct Nodes<'a, 'b>(&'a FaceEncoding<'b>);
impl Serialize for Nodes<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.mesh.nodes.len()))?;
        for node in self.0.mesh.nodes {
            seq.serialize_element(&(
                DigestWords(node.node_id),
                node.point_m,
                Uses(self.0, &node.uses),
            ))?;
        }
        seq.end()
    }
}
struct Uses<'a, 'b>(&'a FaceEncoding<'b>, &'b [ExactFaceMeshNodeUse]);
impl Serialize for Uses<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.1.len()))?;
        for usage in self.1 {
            seq.serialize_element(&(
                self.0.entity::<S::Error>(&usage.source_face_id)?,
                self.0.chart::<S::Error>(usage.chart_id)?,
                usage.uv,
                usage.evaluator_uv,
                Edges(self.0, &usage.exact_edge_parameters),
            ))?;
        }
        seq.end()
    }
}
struct Edges<'a, 'b>(&'a FaceEncoding<'b>, &'b [ExactFaceMeshEdgeParameter]);
impl Serialize for Edges<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.1.len()))?;
        for edge in self.1 {
            seq.serialize_element(&(
                self.0.entity::<S::Error>(&edge.source_coedge_id)?,
                self.0.entity::<S::Error>(&edge.source_edge_id)?,
                edge.parameter,
            ))?;
        }
        seq.end()
    }
}
struct Triangles<'a, 'b>(&'a FaceEncoding<'b>);
impl Serialize for Triangles<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.mesh.triangles.len()))?;
        for triangle in self.0.mesh.triangles {
            let node_ids = [
                self.0.node::<S::Error>(triangle.node_ids[0])?,
                self.0.node::<S::Error>(triangle.node_ids[1])?,
                self.0.node::<S::Error>(triangle.node_ids[2])?,
            ];
            seq.serialize_element(&(
                DigestWords(triangle.triangle_id),
                self.0.chart::<S::Error>(triangle.chart_id)?,
                self.0.entity::<S::Error>(&triangle.source_face_id)?,
                node_ids,
            ))?;
        }
        seq.end()
    }
}
struct Boundaries<'a, 'b>(&'a FaceEncoding<'b>);
impl Serialize for Boundaries<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.mesh.boundary_segments.len()))?;
        for edge in self.0.mesh.boundary_segments {
            let nodes = [
                self.0.node::<S::Error>(edge.node_ids[0])?,
                self.0.node::<S::Error>(edge.node_ids[1])?,
            ];
            seq.serialize_element(&(
                self.0.entity::<S::Error>(&edge.source_coedge_id)?,
                self.0.entity::<S::Error>(&edge.source_edge_id)?,
                nodes,
                edge.edge_parameters,
            ))?;
        }
        seq.end()
    }
}

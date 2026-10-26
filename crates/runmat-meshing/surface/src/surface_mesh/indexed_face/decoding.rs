//! Reconstruct original rich records only after checking dictionary and column inventories.
use super::quality::{QualityDecoded, TriangleIdentity};
use super::*;
use crate::*;
use runmat_canonical_codec::ExactVec;
use serde::Deserialize;

pub(super) type EdgeRow = (u32, u32, f64);
pub(super) type UseRow = (u32, u32, [f64; 2], [f64; 2], ExactVec<EdgeRow>);
pub(super) type NodeRow = (DigestWords, [f64; 3], ExactVec<UseRow>);
pub(super) type TriangleRow = (DigestWords, u32, u32, [NodeReference; 3]);
pub(super) type BoundaryRow = (u32, u32, [NodeReference; 2], [f64; 2]);
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DecodedFace {
    schema_version: u16,
    source_face: u32,
    entities: ExactVec<PersistentEntityId>,
    charts: ExactVec<DigestWords>,
    nodes: ExactVec<NodeRow>,
    triangles: ExactVec<TriangleRow>,
    quality: QualityDecoded,
    boundary_segments: ExactVec<BoundaryRow>,
    joined_chart_cuts: ExactVec<ExactFaceMeshJoinedCut>,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    maximum_chordal_deviation_m: runmat_types::physical::Length,
    maximum_normal_deviation_rad: f64,
}
impl DecodedFace {
    pub(super) fn validate_global_frame(
        &self,
        faces: &[PersistentEntityId],
    ) -> Result<(), ExactSurfaceMeshError> {
        if self.schema_version != crate::EXACT_FACE_MESH_SCHEMA_VERSION
            || !self.joined_chart_cuts.is_empty()
            || faces.first() != Some(self.entity(self.source_face)?)
        {
            return Err(invalid(
                "indexed global surface frame differs from its canonical face inventory",
            ));
        }
        Ok(())
    }
    fn entity(&self, index: u32) -> Result<&PersistentEntityId, ExactSurfaceMeshError> {
        self.entities
            .get(index as usize)
            .ok_or_else(|| invalid("entity index outside original dictionary"))
    }
    /// Additional rich ownership while the indexed DTO is still live. The decoder
    /// owner must admit its separate DTO storage before invoking Serde.
    pub fn expansion_bytes(
        &self,
        control: &dyn MeshingCancellationSignal,
    ) -> Result<u64, ExactSurfaceMeshError> {
        // Dictionary clones are handles sharing the decoded identity allocation.
        self.entity(self.source_face)?;
        let mut bytes = 0u64;
        let mut add = |value: u64| {
            bytes = bytes
                .checked_add(value)
                .ok_or_else(|| invalid("rich surface expansion inventory overflow"))?;
            Ok::<_, ExactSurfaceMeshError>(())
        };
        add((self.nodes.len() as u64)
            .checked_mul(std::mem::size_of::<ExactFaceMeshNode>() as u64)
            .ok_or_else(|| invalid("node expansion overflow"))?)?;
        add((self.triangles.len() as u64)
            .checked_mul(std::mem::size_of::<ExactFaceMeshTriangle>() as u64)
            .ok_or_else(|| invalid("triangle expansion overflow"))?)?;
        add((self.boundary_segments.len() as u64)
            .checked_mul(std::mem::size_of::<ExactFaceMeshBoundarySegment>() as u64)
            .ok_or_else(|| invalid("boundary expansion overflow"))?)?;
        for (index, (_, _, uses)) in self.nodes.iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            add((uses.len() as u64)
                .checked_mul(std::mem::size_of::<ExactFaceMeshNodeUse>() as u64)
                .ok_or_else(|| invalid("node-use expansion overflow"))?)?;
            for (face, _, _, _, edges) in uses {
                self.entity(*face)?;
                add((edges.len() as u64)
                    .checked_mul(std::mem::size_of::<ExactFaceMeshEdgeParameter>() as u64)
                    .ok_or_else(|| invalid("edge parameter expansion overflow"))?)?;
                for (coedge, edge, _) in edges {
                    self.entity(*coedge)?;
                    self.entity(*edge)?;
                }
            }
        }
        for (index, (_, _, face, _)) in self.triangles.iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            self.entity(*face)?;
        }
        for (index, (coedge, edge, _, _)) in self.boundary_segments.iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            self.entity(*coedge)?;
            self.entity(*edge)?;
        }
        Ok(bytes)
    }
    pub fn into_mesh(
        self,
        control: &dyn MeshingCancellationSignal,
        mut admit: impl FnMut(u64) -> Result<(), ExactSurfaceMeshError>,
    ) -> Result<ExactFaceMesh, ExactSurfaceMeshError> {
        checkpoint(control)?;
        for (index, pair) in self.entities.windows(2).enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            if pair[0] >= pair[1] {
                return Err(invalid(
                    "surface entity dictionary must be strictly canonical and unique",
                ));
            }
        }
        for (index, pair) in self.charts.windows(2).enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            if pair[0].0 >= pair[1].0 {
                return Err(invalid(
                    "surface chart dictionary must be strictly canonical and unique",
                ));
            }
        }
        self.quality.validate(self.triangles.len())?;
        admit(self.expansion_bytes(control)?)?;
        let entity = |index: u32| {
            self.entities
                .get(index as usize)
                .cloned()
                .ok_or_else(|| invalid("entity index outside original dictionary"))
        };
        let chart = |index: u32| {
            self.charts
                .get(index as usize)
                .map(|id| id.0)
                .ok_or_else(|| invalid("chart index outside original dictionary"))
        };
        let source_face_id = entity(self.source_face)?;
        let mut nodes = Vec::with_capacity(self.nodes.len());
        for (index, (id, point_m, uses)) in self.nodes.into_iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            let mut original_uses = Vec::with_capacity(uses.len());
            for (face, chart_index, uv, evaluator_uv, edges) in uses {
                let mut exact_edge_parameters = Vec::with_capacity(edges.len());
                for (coedge, edge, parameter) in edges {
                    exact_edge_parameters.push(ExactFaceMeshEdgeParameter {
                        source_coedge_id: entity(coedge)?,
                        source_edge_id: entity(edge)?,
                        parameter,
                    });
                }
                original_uses.push(ExactFaceMeshNodeUse {
                    source_face_id: entity(face)?,
                    chart_id: chart(chart_index)?,
                    uv,
                    evaluator_uv,
                    exact_edge_parameters,
                });
            }
            nodes.push(ExactFaceMeshNode {
                node_id: id.0,
                point_m,
                uses: original_uses,
            });
        }
        let node = |reference: &NodeReference| match reference {
            NodeReference::Index(index) => nodes
                .get(*index as usize)
                .map(|node| node.node_id)
                .ok_or_else(|| invalid("node index outside original inventory")),
            NodeReference::Foreign(id) => Ok(id.0),
        };
        let mut triangles = Vec::with_capacity(self.triangles.len());
        for (index, (id, chart_index, face, references)) in self.triangles.into_iter().enumerate() {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            let node_ids = [
                node(&references[0])?,
                node(&references[1])?,
                node(&references[2])?,
            ];
            triangles.push(self.quality.triangle(
                index,
                TriangleIdentity {
                    triangle_id: id.0,
                    chart_id: chart(chart_index)?,
                    source_face_id: entity(face)?,
                    node_ids,
                },
            ));
        }
        let mut boundary_segments = Vec::with_capacity(self.boundary_segments.len());
        for (index, (coedge, edge, references, edge_parameters)) in
            self.boundary_segments.into_iter().enumerate()
        {
            if index.is_multiple_of(128) {
                checkpoint(control)?;
            }
            boundary_segments.push(ExactFaceMeshBoundarySegment {
                source_coedge_id: entity(coedge)?,
                source_edge_id: entity(edge)?,
                node_ids: [node(&references[0])?, node(&references[1])?],
                edge_parameters,
            });
        }
        Ok(ExactFaceMesh {
            schema_version: self.schema_version,
            source_face_id,
            nodes,
            triangles,
            boundary_segments,
            joined_chart_cuts: self.joined_chart_cuts.into_vec(),
            maximum_chordal_deviation_m: runmat_types::physical::Length::new(
                self.maximum_chordal_deviation_m.canonical_value(),
            ),
            maximum_normal_deviation_rad: self.maximum_normal_deviation_rad,
        })
    }
}

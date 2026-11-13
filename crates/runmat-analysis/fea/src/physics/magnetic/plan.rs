//! Material-independent exact sparse support prepared before physical quadrature.
use super::MagnetostaticError;
use crate::{algebra::SparsePattern, topology::AnalysisProjection};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[cfg(test)]
mod tests;

const PREFIX: &[u8] = b"runmat-magnetic-system-plan/v2\0";
const LIMITS: CanonicalLimits =
    crate::contracts::canonical_digest::wire_bounded(CanonicalLimits::platform_content(256, 16));

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
/// Material-independent support only, not a stiffness operator or solver receipt.
/// Decoding proves shape; `validate_for` proves exact projected topology before use.
pub struct MagneticSystemPlan {
    data: PlanData,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanData {
    projection_digest: Digest,
    /// Exterior faces with tangential A = 0, sorted; the rest are insulated.
    grounded_face_identities: Vec<StableDigest>,
    edge_identities: Vec<[StableDigest; 2]>,
    free_edge_indices: Vec<usize>,
    gauge_node_identities: Vec<StableDigest>,
    physical_pattern: SparsePattern,
    saddle_pattern: SparsePattern,
}

impl<'de> Deserialize<'de> for MagneticSystemPlan {
    fn deserialize<D: serde::Deserializer<'de>>(decoder: D) -> Result<Self, D::Error> {
        let value = Self {
            data: PlanData::deserialize(decoder)?,
        };
        value.validate().map_err(serde::de::Error::custom)?;
        Ok(value)
    }
}

/// A plan derived in this process together with the exact edge and boundary spaces it was
/// derived from, so assembly reuses them instead of deriving the topology again.
pub(crate) struct DerivedMagneticSystem {
    pub plan: MagneticSystemPlan,
    pub edges: crate::topology::TetrahedralEdgeSpace,
    pub boundary: super::boundary::MagneticBoundarySpace,
}

impl MagneticSystemPlan {
    /// No material coefficients, current values, physical quadrature or matrices
    /// are constructed here. Existing canonical topology owns every edge/row.
    pub fn derive(
        projection: &AnalysisProjection,
        grounded_faces: &[StableDigest],
    ) -> Result<Self, MagnetostaticError> {
        Ok(Self::derive_with_spaces(projection, grounded_faces)?.plan)
    }

    /// Rows are built from element incidence: each row gathers the free-edge (and gauge)
    /// columns of its incident elements, sorted and deduplicated, straight into CSR.
    pub(crate) fn derive_with_spaces(
        projection: &AnalysisProjection,
        grounded_faces: &[StableDigest],
    ) -> Result<DerivedMagneticSystem, MagnetostaticError> {
        let mut grounded_face_identities = grounded_faces.to_vec();
        grounded_face_identities.sort_unstable();
        grounded_face_identities.dedup();
        let (edges, boundary) = super::validated_spaces(projection, &grounded_face_identities)?;
        let free = boundary.free_edges.len();
        let dimension = free
            .checked_add(boundary.gauge_nodes.len())
            .ok_or_else(|| invalid("dimension overflow"))?;
        const NONE: usize = usize::MAX;
        let mut edge_row = vec![NONE; edges.edges().len()];
        for (row, &edge) in boundary.free_edges.iter().enumerate() {
            *edge_row
                .get_mut(edge)
                .ok_or_else(|| invalid("free edge outside the edge space"))? = row;
        }
        let mut gauge_row = vec![NONE; projection.nodes.len()];
        for (index, &node) in boundary.gauge_nodes.iter().enumerate() {
            *gauge_row
                .get_mut(node)
                .ok_or_else(|| invalid("gauge node outside the projection"))? = free + index;
        }
        let local = |index: usize| {
            edges
                .element_edge_indices(index)
                .ok_or_else(|| invalid("missing element edge inventory"))
        };
        // Row -> incident element CSR: an edge or corner occurs once per element.
        let mut offsets = vec![0_usize; dimension + 1];
        for (index, element) in projection.elements.iter().enumerate() {
            for edge in local(index)? {
                if edge_row[edge] != NONE {
                    offsets[edge_row[edge] + 1] += 1;
                }
            }
            for &node in &element.node_indices[..4] {
                if gauge_row[node] != NONE {
                    offsets[gauge_row[node] + 1] += 1;
                }
            }
        }
        for row in 0..dimension {
            offsets[row + 1] += offsets[row];
        }
        let mut cursor = offsets[..dimension].to_vec();
        let mut incident = vec![0_u32; offsets[dimension]];
        for (index, element) in projection.elements.iter().enumerate() {
            let element_index = u32::try_from(index).map_err(invalid)?;
            let rows = local(index)?.into_iter().map(|edge| edge_row[edge]).chain(
                element.node_indices[..4]
                    .iter()
                    .map(|&node| gauge_row[node]),
            );
            for row in rows.filter(|&row| row != NONE) {
                incident[cursor[row]] = element_index;
                cursor[row] += 1;
            }
        }
        drop(cursor);
        let mut physical_offsets = Vec::with_capacity(free + 1);
        let mut physical_columns = Vec::new();
        let mut saddle_offsets = Vec::with_capacity(dimension + 1);
        let mut saddle_columns = Vec::new();
        physical_offsets.push(0_u64);
        saddle_offsets.push(0_u64);
        let mut edge_columns = Vec::new();
        let mut gauge_columns = Vec::new();
        for row in 0..dimension {
            edge_columns.clear();
            gauge_columns.clear();
            for &element in &incident[offsets[row]..offsets[row + 1]] {
                let element = element as usize;
                let edge_rows = local(element)?
                    .into_iter()
                    .map(|edge| edge_row[edge])
                    .filter(|&column| column != NONE);
                if row < free {
                    edge_columns.extend(edge_rows);
                    gauge_columns.extend(
                        projection.elements[element].node_indices[..4]
                            .iter()
                            .map(|&node| gauge_row[node])
                            .filter(|&column| column != NONE),
                    );
                } else {
                    // A gauge row couples to every free edge of its incident elements.
                    gauge_columns.extend(edge_rows);
                }
            }
            edge_columns.sort_unstable();
            edge_columns.dedup();
            gauge_columns.sort_unstable();
            gauge_columns.dedup();
            if row < free {
                for &column in &edge_columns {
                    physical_columns.push(u64::try_from(column).map_err(invalid)?);
                }
                physical_offsets.push(u64::try_from(physical_columns.len()).map_err(invalid)?);
            }
            // Free-edge columns precede every gauge column, so the union stays sorted.
            for &column in edge_columns.iter().chain(&gauge_columns) {
                saddle_columns.push(u64::try_from(column).map_err(invalid)?);
            }
            saddle_offsets.push(u64::try_from(saddle_columns.len()).map_err(invalid)?);
        }
        drop(incident);
        let square = |rows: usize, offsets, columns| {
            let rows = u64::try_from(rows).map_err(invalid)?;
            SparsePattern::try_new(rows, rows, offsets, columns).map_err(invalid)
        };
        let plan = Self {
            data: PlanData {
                projection_digest: projection.projection_digest,
                grounded_face_identities,
                edge_identities: edges.edges().iter().map(|e| e.node_identities).collect(),
                free_edge_indices: boundary.free_edges.clone(),
                gauge_node_identities: boundary
                    .gauge_nodes
                    .iter()
                    .map(|&n| projection.nodes[n].stable_identity)
                    .collect(),
                physical_pattern: square(free, physical_offsets, physical_columns)?,
                saddle_pattern: square(dimension, saddle_offsets, saddle_columns)?,
            },
        };
        plan.validate()?;
        Ok(DerivedMagneticSystem {
            plan,
            edges,
            boundary,
        })
    }
    pub fn validate(&self) -> Result<(), MagnetostaticError> {
        let d = &self.data;
        let free = d.free_edge_indices.len();
        let dimension = free
            .checked_add(d.gauge_node_identities.len())
            .ok_or_else(|| invalid("dimension overflow"))?;
        if d.edge_identities.is_empty()
            || d.grounded_face_identities.is_empty()
            || d.grounded_face_identities.windows(2).any(|p| p[0] >= p[1])
            || d.edge_identities.iter().any(|[a, b]| a >= b)
            || d.edge_identities.windows(2).any(|p| p[0] >= p[1])
            || d.free_edge_indices
                .iter()
                .any(|&i| i >= d.edge_identities.len())
            || d.free_edge_indices.windows(2).any(|p| p[0] >= p[1])
            || d.gauge_node_identities
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                != d.gauge_node_identities.len()
            || d.physical_pattern.rows() != free
            || d.physical_pattern.columns() != free
            || d.saddle_pattern.rows() != dimension
            || d.saddle_pattern.columns() != dimension
        {
            return Err(invalid("invalid plan inventory or shape"));
        }
        Ok(())
    }
    pub fn validate_for(&self, projection: &AnalysisProjection) -> Result<(), MagnetostaticError> {
        self.validate()?;
        if self != &Self::derive(projection, &self.data.grounded_face_identities)? {
            return Err(invalid("plan is not exact canonical magnetic support"));
        }
        Ok(())
    }
    pub(crate) fn projection_digest(&self) -> Digest {
        self.data.projection_digest
    }
    pub fn grounded_faces(&self) -> &[StableDigest] {
        &self.data.grounded_face_identities
    }
    pub fn physical_pattern(&self) -> &SparsePattern {
        &self.data.physical_pattern
    }
    pub fn saddle_pattern(&self) -> &SparsePattern {
        &self.data.saddle_pattern
    }
    pub fn free_edge_count(&self) -> usize {
        self.data.free_edge_indices.len()
    }
    pub fn gauge_node_count(&self) -> usize {
        self.data.gauge_node_identities.len()
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, MagnetostaticError> {
        self.validate()?;
        encode_contract(PREFIX, "magnetic-system-plan", self, LIMITS).map_err(invalid)
    }
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, MagnetostaticError> {
        let value: Self =
            decode_contract(PREFIX, "magnetic-system-plan", bytes, LIMITS).map_err(invalid)?;
        value.validate()?;
        Ok(value)
    }
    /// SHA-256 of the canonical encoding, streamed: the plan's wire is never materialized
    /// (a mesh-sized plan can exceed the transport cap yet still have an identity).
    pub fn identity(&self) -> Result<Digest, MagnetostaticError> {
        self.validate()?;
        crate::contracts::canonical_digest::canonical_contract_digest(
            PREFIX,
            "magnetic-system-plan",
            self,
            LIMITS,
        )
        .map_err(invalid)
    }
}

fn invalid(error: impl std::fmt::Display) -> MagnetostaticError {
    MagnetostaticError::InvalidInput(format!("magnetic system plan: {error}"))
}

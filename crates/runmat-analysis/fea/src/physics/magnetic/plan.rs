//! Material-independent exact sparse support prepared before physical quadrature.
use super::MagnetostaticError;
use crate::{algebra::SparsePattern, topology::AnalysisProjection};
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::StableDigest;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
mod tests;

const PREFIX: &[u8] = b"runmat-magnetic-system-plan/v2\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(512 * 1024 * 1024, 64_000_000, 256, 16);

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

impl MagneticSystemPlan {
    /// No material coefficients, current values, physical quadrature or matrices
    /// are constructed here. Existing canonical topology owns every edge/row.
    pub fn derive(
        projection: &AnalysisProjection,
        grounded_faces: &[StableDigest],
    ) -> Result<Self, MagnetostaticError> {
        let mut grounded_face_identities = grounded_faces.to_vec();
        grounded_face_identities.sort_unstable();
        grounded_face_identities.dedup();
        let (edges, boundary) = super::validated_spaces(projection, &grounded_face_identities)?;
        let free = boundary.free_edges.len();
        let dimension = free
            .checked_add(boundary.gauge_nodes.len())
            .ok_or_else(|| invalid("dimension overflow"))?;
        let edge_rows: BTreeMap<_, _> = boundary
            .free_edges
            .iter()
            .enumerate()
            .map(|(i, &edge)| (edge, i))
            .collect();
        let gauge_rows: BTreeMap<_, _> = boundary
            .gauge_nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, free + i))
            .collect();
        let mut physical = vec![BTreeSet::new(); free];
        let mut saddle = vec![BTreeSet::new(); dimension];
        for (index, element) in projection.elements.iter().enumerate() {
            let local = edges
                .element_edge_indices(index)
                .ok_or_else(|| invalid("missing element edge inventory"))?;
            for edge in local {
                let Some(&row) = edge_rows.get(&edge) else {
                    continue;
                };
                for other in local {
                    if let Some(&column) = edge_rows.get(&other) {
                        physical[row].insert(column);
                        saddle[row].insert(column);
                    }
                }
                for node in &element.node_indices[..4] {
                    if let Some(&gauge) = gauge_rows.get(node) {
                        saddle[row].insert(gauge);
                        saddle[gauge].insert(row);
                    }
                }
            }
        }
        let value = Self {
            data: PlanData {
                projection_digest: projection.projection_digest,
                grounded_face_identities,
                edge_identities: edges.edges().iter().map(|e| e.node_identities).collect(),
                free_edge_indices: boundary.free_edges,
                gauge_node_identities: boundary
                    .gauge_nodes
                    .iter()
                    .map(|&n| projection.nodes[n].stable_identity)
                    .collect(),
                physical_pattern: pattern(physical)?,
                saddle_pattern: pattern(saddle)?,
            },
        };
        value.validate()?;
        Ok(value)
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
    pub fn identity(&self) -> Result<Digest, MagnetostaticError> {
        Ok(Digest::sha256(self.canonical_bytes()?))
    }
}

fn pattern(rows: Vec<BTreeSet<usize>>) -> Result<SparsePattern, MagnetostaticError> {
    let dimension = u64::try_from(rows.len()).map_err(invalid)?;
    let mut offsets = vec![0];
    let mut columns = Vec::new();
    for row in rows {
        for column in row {
            columns.push(u64::try_from(column).map_err(invalid)?);
        }
        offsets.push(u64::try_from(columns.len()).map_err(invalid)?);
    }
    SparsePattern::try_new(dimension, dimension, offsets, columns).map_err(invalid)
}
fn invalid(error: impl std::fmt::Display) -> MagnetostaticError {
    MagnetostaticError::InvalidInput(format!("magnetic system plan: {error}"))
}

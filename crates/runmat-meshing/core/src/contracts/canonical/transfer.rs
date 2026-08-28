use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::{GeometryRevisionRef, MeshingContractError, SolverMeshArtifact, StableDigest};

pub const SOLVER_MESH_TRANSFER_SCHEMA_VERSION: u16 = 1;
const MAX_TRANSFER_SOURCES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverTransferMethod {
    BarycentricInterpolation,
    CentroidProjection,
    ConservativeProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverTransferSource {
    pub stable_identity: StableDigest,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverEntityTransfer {
    pub target_stable_identity: StableDigest,
    pub method: SolverTransferMethod,
    pub sources: Vec<SolverTransferSource>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverMeshTransferMap {
    pub schema_version: u16,
    pub source_artifact_digest: StableDigest,
    pub target_artifact_digest: StableDigest,
    pub geometry: GeometryRevisionRef,
    pub node_transfers: Vec<SolverEntityTransfer>,
    pub volume_element_transfers: Vec<SolverEntityTransfer>,
    pub boundary_face_transfers: Vec<SolverEntityTransfer>,
    pub boundary_edge_transfers: Vec<SolverEntityTransfer>,
}

impl SolverMeshTransferMap {
    pub fn validate(&self) -> Result<(), MeshingContractError> {
        if self.schema_version != SOLVER_MESH_TRANSFER_SCHEMA_VERSION {
            return Err(invalid(
                "solver mesh transfer schema version",
                format!("expected {SOLVER_MESH_TRANSFER_SCHEMA_VERSION}"),
            ));
        }
        self.source_artifact_digest
            .validate_nonzero("transfer.source_artifact_digest")?;
        self.target_artifact_digest
            .validate_nonzero("transfer.target_artifact_digest")?;
        if self.source_artifact_digest == self.target_artifact_digest {
            return Err(invalid(
                "solver mesh transfer",
                "source and target artifacts must be distinct",
            ));
        }
        self.geometry.validate()?;
        validate_transfers(
            "node transfers",
            &self.node_transfers,
            &[SolverTransferMethod::BarycentricInterpolation],
        )?;
        validate_transfers(
            "volume element transfers",
            &self.volume_element_transfers,
            &[
                SolverTransferMethod::CentroidProjection,
                SolverTransferMethod::ConservativeProjection,
            ],
        )?;
        validate_transfers(
            "boundary face transfers",
            &self.boundary_face_transfers,
            &[
                SolverTransferMethod::CentroidProjection,
                SolverTransferMethod::ConservativeProjection,
            ],
        )?;
        validate_transfers(
            "boundary edge transfers",
            &self.boundary_edge_transfers,
            &[
                SolverTransferMethod::CentroidProjection,
                SolverTransferMethod::ConservativeProjection,
            ],
        )
    }

    pub fn validate_against(
        &self,
        source: &SolverMeshArtifact,
        target: &SolverMeshArtifact,
    ) -> Result<(), MeshingContractError> {
        self.validate()?;
        source.validate()?;
        target.validate()?;
        if self.source_artifact_digest != source.canonical_digest
            || self.target_artifact_digest != target.canonical_digest
            || self.geometry != source.geometry
            || self.geometry != target.geometry
        {
            return Err(invalid(
                "solver mesh transfer binding",
                "artifact digests and exact geometry revision must match",
            ));
        }
        validate_closure(
            "node transfers",
            &self.node_transfers,
            source
                .topology
                .nodes
                .iter()
                .map(|entity| entity.stable_identity),
            target
                .topology
                .nodes
                .iter()
                .map(|entity| entity.stable_identity),
        )?;
        validate_closure(
            "volume element transfers",
            &self.volume_element_transfers,
            source
                .topology
                .volume_elements
                .iter()
                .map(|entity| entity.stable_identity),
            target
                .topology
                .volume_elements
                .iter()
                .map(|entity| entity.stable_identity),
        )?;
        validate_closure(
            "boundary face transfers",
            &self.boundary_face_transfers,
            source
                .topology
                .boundary_faces
                .iter()
                .map(|entity| entity.stable_identity),
            target
                .topology
                .boundary_faces
                .iter()
                .map(|entity| entity.stable_identity),
        )?;
        validate_closure(
            "boundary edge transfers",
            &self.boundary_edge_transfers,
            source
                .topology
                .boundary_edges
                .iter()
                .map(|entity| entity.stable_identity),
            target
                .topology
                .boundary_edges
                .iter()
                .map(|entity| entity.stable_identity),
        )
    }
}

fn validate_transfers(
    field: &str,
    transfers: &[SolverEntityTransfer],
    methods: &[SolverTransferMethod],
) -> Result<(), MeshingContractError> {
    let mut previous = None;
    for transfer in transfers {
        if transfer.target_stable_identity == StableDigest::ZERO
            || previous.is_some_and(|identity| identity >= transfer.target_stable_identity)
            || !methods.contains(&transfer.method)
            || transfer.sources.is_empty()
            || transfer.sources.len() > MAX_TRANSFER_SOURCES
        {
            return Err(invalid(
                field,
                "targets must be canonical and methods/source inventories valid",
            ));
        }
        let mut source_previous = None;
        let mut weight_sum = 0.0;
        for source in &transfer.sources {
            if source.stable_identity == StableDigest::ZERO
                || source_previous.is_some_and(|identity| identity >= source.stable_identity)
                || !source.weight.is_finite()
                || source.weight <= 0.0
            {
                return Err(invalid(
                    field,
                    "weighted sources must be positive, finite, unique, and canonical",
                ));
            }
            weight_sum += source.weight;
            source_previous = Some(source.stable_identity);
        }
        if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > 64.0 * f64::EPSILON {
            return Err(invalid(field, "weighted sources must sum to one"));
        }
        previous = Some(transfer.target_stable_identity);
    }
    Ok(())
}

fn validate_closure(
    field: &str,
    transfers: &[SolverEntityTransfer],
    source: impl Iterator<Item = StableDigest>,
    target: impl Iterator<Item = StableDigest>,
) -> Result<(), MeshingContractError> {
    let source = source.collect::<BTreeSet<_>>();
    let target = target.collect::<BTreeSet<_>>();
    let mapped = transfers
        .iter()
        .map(|transfer| transfer.target_stable_identity)
        .collect::<BTreeSet<_>>();
    let required = target.difference(&source).copied().collect::<BTreeSet<_>>();
    if !mapped.is_subset(&required)
        || transfers.iter().any(|transfer| {
            transfer
                .sources
                .iter()
                .any(|source_identity| !source.contains(&source_identity.stable_identity))
        })
    {
        return Err(invalid(
            field,
            "map may contain only new target identities from admitted source identities",
        ));
    }
    Ok(())
}

fn invalid(field: &str, reason: impl Into<String>) -> MeshingContractError {
    MeshingContractError::invalid(field, reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::canonical::{artifact_tests::artifact, CanonicalMeshingContract};

    #[test]
    fn transfer_map_is_canonical_bounded_and_weighted() {
        let source = artifact();
        let mut transfer = SolverMeshTransferMap {
            schema_version: SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
            source_artifact_digest: source.canonical_digest,
            target_artifact_digest: StableDigest::from_bytes([99; 32]),
            geometry: source.geometry,
            node_transfers: vec![SolverEntityTransfer {
                target_stable_identity: StableDigest::from_bytes([88; 32]),
                method: SolverTransferMethod::BarycentricInterpolation,
                sources: source
                    .topology
                    .nodes
                    .iter()
                    .map(|node| SolverTransferSource {
                        stable_identity: node.stable_identity,
                        weight: 0.25,
                    })
                    .collect(),
            }],
            volume_element_transfers: Vec::new(),
            boundary_face_transfers: Vec::new(),
            boundary_edge_transfers: Vec::new(),
        };
        let encoded = transfer.canonical_encode().unwrap();
        assert_eq!(
            SolverMeshTransferMap::canonical_decode(&encoded).unwrap(),
            transfer
        );

        let mut oversized = transfer.clone();
        oversized.node_transfers[0].sources = (1..=65)
            .map(|index| SolverTransferSource {
                stable_identity: StableDigest::from_bytes([index; 32]),
                weight: 1.0 / 65.0,
            })
            .collect();
        assert_eq!(oversized.validate().unwrap_err().field, "node transfers");

        transfer.node_transfers[0].sources[0].weight = 0.5;
        assert_eq!(transfer.validate().unwrap_err().field, "node transfers");
        transfer.node_transfers[0].sources.truncate(1);
        transfer.node_transfers[0].sources[0].weight = 1.0;
        transfer.node_transfers[0].method = SolverTransferMethod::CentroidProjection;
        assert_eq!(transfer.validate().unwrap_err().field, "node transfers");
    }
}

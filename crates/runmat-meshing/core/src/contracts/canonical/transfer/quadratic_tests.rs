use super::*;
use crate::{fixtures::canonical_tetrahedron_solver_mesh, CanonicalMeshingContract, ElementOrder};

fn map() -> SolverMeshTransferMap {
    let source = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let mut sources = source.topology.volume_elements[0]
        .node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| SolverTransferSource {
            stable_identity: source.topology.nodes[*id as usize - 1].stable_identity,
            weight: if i < 4 { -0.125 } else { 0.25 },
        })
        .collect::<Vec<_>>();
    sources.sort_by_key(|s| s.stable_identity);
    SolverMeshTransferMap {
        schema_version: SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
        source_artifact_digest: source.canonical_digest,
        target_artifact_digest: StableDigest::from_bytes([99; 32]),
        geometry: source.geometry,
        node_transfers: vec![SolverEntityTransfer {
            target_stable_identity: StableDigest::from_bytes([88; 32]),
            method: SolverTransferMethod::QuadraticLagrangeInterpolation,
            sources,
        }],
        volume_element_transfers: Vec::new(),
        boundary_face_transfers: Vec::new(),
        boundary_edge_transfers: Vec::new(),
    }
}

#[test]
fn quadratic_weights_are_explicit_canonical_and_cold_version_bound() {
    let valid = map();
    let bytes = valid.canonical_encode().unwrap();
    assert_eq!(
        SolverMeshTransferMap::canonical_decode(&bytes).unwrap(),
        valid
    );
    let legacy_bytes = runmat_canonical_codec::encode_contract(
        b"runmat-meshing-canonical-cbor/v1\0",
        "analysis.mesh.solver-transfer-map/v1",
        &valid,
        SolverMeshTransferMap::LIMITS.into(),
    )
    .unwrap();
    assert!(SolverMeshTransferMap::canonical_decode(&legacy_bytes).is_err());
    for change in 0..8 {
        let mut invalid = valid.clone();
        match change {
            0 => invalid.schema_version = 1,
            1 => invalid.node_transfers[0].method = SolverTransferMethod::BarycentricInterpolation,
            2 => invalid.node_transfers[0].sources[0].weight = -0.126,
            3 => invalid.node_transfers[0].sources[0].weight = f64::NAN,
            4 => invalid.node_transfers[0].sources[0].weight = 0.0,
            5 => invalid.node_transfers[0].sources.reverse(),
            6 => invalid.node_transfers[0].sources[0].weight += 0.01,
            _ => invalid.node_transfers[0]
                .sources
                .push(SolverTransferSource {
                    stable_identity: StableDigest::from_bytes([255; 32]),
                    weight: 0.01,
                }),
        }
        assert!(invalid.validate().is_err(), "mutation {change}");
        assert!(invalid.canonical_encode().is_err());
    }
}

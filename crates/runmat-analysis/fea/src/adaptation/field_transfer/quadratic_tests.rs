use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use super::*;
use runmat_meshing_core::{
    fixtures::{canonical_tetrahedron_solver_mesh, connected_quadratic_tetrahedron_solver_mesh},
    CanonicalMeshingContract, SolverEntityTransfer, SolverTransferMethod, SolverTransferSource,
    SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
};

#[test]
fn quadratic_h_transfer_retains_scalar_vector_modes_and_rejects_linear_fallback() {
    let source = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let target = connected_quadratic_tetrahedron_solver_mesh();
    let source_ids = source
        .topology
        .nodes
        .iter()
        .map(|n| n.stable_identity)
        .collect::<std::collections::BTreeSet<_>>();
    let mut map = SolverMeshTransferMap {
        schema_version: SOLVER_MESH_TRANSFER_SCHEMA_VERSION,
        source_artifact_digest: source.canonical_digest,
        target_artifact_digest: target.canonical_digest,
        geometry: source.geometry.clone(),
        node_transfers: Vec::new(),
        volume_element_transfers: Vec::new(),
        boundary_face_transfers: Vec::new(),
        boundary_edge_transfers: Vec::new(),
    };
    // Independent unit-simplex cardinal polynomial, not the production basis helper.
    for node in &target.topology.nodes {
        if source_ids.contains(&node.stable_identity) {
            continue;
        }
        let [x, y, z] = node.coordinates_m.canonical_value();
        let bary = [1.0 - x - y - z, x, y, z];
        let mut weights = bary.map(|l| 2.0 * l * l - l).to_vec();
        for [a, b] in [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]] {
            weights.push(4.0 * bary[a] * bary[b]);
        }
        let mut sources = source.topology.volume_elements[0]
            .node_ids
            .iter()
            .zip(weights)
            .filter(|(_, w)| *w != 0.0)
            .map(|(id, weight)| SolverTransferSource {
                stable_identity: source.topology.nodes[*id as usize - 1].stable_identity,
                weight,
            })
            .collect::<Vec<_>>();
        sources.sort_by_key(|s| s.stable_identity);
        map.node_transfers.push(SolverEntityTransfer {
            target_stable_identity: node.stable_identity,
            method: SolverTransferMethod::QuadraticLagrangeInterpolation,
            sources,
        });
    }
    map.node_transfers.sort_by_key(|t| t.target_stable_identity);
    let cold = SolverMeshTransferMap::canonical_decode(&map.canonical_encode().unwrap()).unwrap();
    for components in [1, 3] {
        let modes = |p: [f64; 3]| {
            [
                p[0] * p[0] + 2.0 * p[1] * p[2],
                p[1] * p[1] + p[0] * p[2],
                p[2] * p[2] + p[0] * p[1],
            ]
        };
        let field = AnalysisField::host_f64(
            "quadratic", AnalysisFieldLayout::components(ObservationQuantity::Displacement, FieldLocation::Node, components),
            vec![source.topology.nodes.len(), components],
            source
                .topology
                .nodes
                .iter()
                .flat_map(|n| modes(n.coordinates_m.canonical_value())[..components].to_vec())
                .collect(),
        );
        let reference = AnalysisField::host_f64(
            "quadratic", AnalysisFieldLayout::components(ObservationQuantity::Displacement, FieldLocation::Node, components),
            vec![target.topology.nodes.len(), components],
            target
                .topology
                .nodes
                .iter()
                .flat_map(|n| modes(n.coordinates_m.canonical_value())[..components].to_vec())
                .collect(),
        );
        let transferred =
            transfer_solver_field(&source, &target, Some(&cold), "nodes", &field).unwrap();
        assert!(
            measure_solver_field_transfer_error(&transferred, &reference)
                .unwrap()
                .absolute_l2_error
                < 1.0e-14
        );
        assert!(transferred
            .evidence
            .methods
            .contains(&SolverFieldTransferMethod::QuadraticLagrangeInterpolation));
        let mut incomplete = cold.clone();
        incomplete
            .node_transfers
            .retain(|t| t.target_stable_identity == target.topology.nodes[4].stable_identity);
        assert_eq!(
            transfer_solver_field(&source, &target, Some(&incomplete), "nodes", &field)
                .unwrap_err(),
            SolverFieldTransferError::UnsupportedTopologyChange
        );
        let mut linear = cold.clone();
        for t in &mut linear.node_transfers {
            t.method = SolverTransferMethod::BarycentricInterpolation;
            t.sources = source.topology.nodes[..4]
                .iter()
                .map(|n| SolverTransferSource {
                    stable_identity: n.stable_identity,
                    weight: 0.25,
                })
                .collect();
            t.sources.sort_by_key(|s| s.stable_identity);
        }
        assert!(matches!(
            transfer_solver_field(&source, &target, Some(&linear), "nodes", &field),
            Err(SolverFieldTransferError::InvalidTransferMap(_))
        ));
    }
}

#[test]
fn transfer_error_keeps_extreme_finite_scales_and_rejects_unrepresentable_norms() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for scale in [1.0e200, 1.0e-200] {
        let field = AnalysisField::host_f64("x", AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global), vec![4], vec![scale; 4]);
        let transfer = transfer_solver_field(&mesh, &mesh, None, "nodes", &field).unwrap();
        let reference = AnalysisField::host_f64("x", AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global), vec![4], vec![2.0 * scale; 4]);
        let evidence = measure_solver_field_transfer_error(&transfer, &reference).unwrap();
        assert!((evidence.absolute_l2_error / (2.0 * scale) - 1.0).abs() < 1.0e-14);
        assert!((evidence.relative_l2_error.unwrap() - 0.5).abs() < 1.0e-14);
    }
    let field = AnalysisField::host_f64("x", AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global), vec![4], vec![1.0e308; 4]);
    let transfer = transfer_solver_field(&mesh, &mesh, None, "nodes", &field).unwrap();
    let reference = AnalysisField::host_f64("x", AnalysisFieldLayout::scalar(ObservationQuantity::Residual, FieldLocation::Global), vec![4], vec![0.0; 4]);
    assert_eq!(
        measure_solver_field_transfer_error(&transfer, &reference).unwrap_err(),
        SolverFieldTransferError::InvalidReferenceField
    );
}

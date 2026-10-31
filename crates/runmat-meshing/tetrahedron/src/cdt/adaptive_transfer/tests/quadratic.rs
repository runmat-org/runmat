use super::*;
use runmat_meshing_core::{
    solver_midside_node_identity, SolverMeshTransferMap, TETRAHEDRON_MIDSIDE_EDGE_CORNERS,
};
use std::collections::BTreeMap;

// Test-only straight-sided elevation of the CDT fixture, without boundary charts.
fn elevate(mut mesh: SolverMeshArtifact) -> SolverMeshArtifact {
    let mut midsides = BTreeMap::new();
    let topo = &mut *mesh.topology;
    for element in &mut topo.volume_elements {
        let corners = element.node_ids.clone();
        for [a, b] in TETRAHEDRON_MIDSIDE_EDGE_CORNERS {
            let mut edge = [corners[a], corners[b]];
            edge.sort_unstable();
            let id = *midsides.entry(edge).or_insert_with(|| {
                let endpoints = edge.map(|id| &topo.nodes[id as usize - 1]);
                let node = SolverMeshNode {
                    node_id: topo.nodes.len() as u64 + 1,
                    stable_identity: solver_midside_node_identity(
                        endpoints.map(|n| n.stable_identity),
                    ),
                    coordinates_m: runmat_types::physical::Quantity::<
                        runmat_types::physical::kind::Position,
                        _,
                    >::new(std::array::from_fn(|i| {
                        0.5 * (endpoints[0].coordinates_m.canonical_value()[i]
                            + endpoints[1].coordinates_m.canonical_value()[i])
                    })),
                    provenance: vec![region()],
                    exact_parameters: Vec::new(),
                };
                let id = node.node_id;
                topo.nodes.push(node);
                id
            });
            element.node_ids.push(id);
        }
        element.order = ElementOrder::Tet10;
    }
    mesh.resolved_request.element_order = ElementOrder::Tet10;
    topo.field_topologies[0].ordered_entity_ids = topo.nodes.iter().map(|n| n.node_id).collect();
    mesh.seal_canonical_digest().unwrap();
    mesh
}

// witness: FEA-ADAPTATION-LOOP-001/qualified_transfer_governed_loop @
//   adaptive_analysis/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Quadratic refinement transfer preserves all polynomial modes and cold lineage for
//   Tet4 and Tet10; plastic, contact and coupled-interface state transfer are not covered.
#[test]
fn quadratic_refinement_preserves_all_polynomial_modes_and_cold_lineage() {
    let (topology, provenance, metric, quality_options, refinement) = adaptation();
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &metric,
        &provenance,
        quality_options,
        &NeverCancelled,
    )
    .unwrap();
    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &metric,
        provenance: &provenance,
        quality: &quality,
        quality_options,
        fixed_node_identities: &[],
    };
    for source_order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let source = if source_order == ElementOrder::Tet10 {
            elevate(artifact(&topology))
        } else {
            artifact(&topology)
        };
        let target = elevate(artifact(&refinement.topology));
        let build =
            |source: &SolverMeshArtifact, options, cancellation: &dyn MeshingCancellationSignal| {
                build_refinement_solver_adaptation(
                    input,
                    &refinement,
                    source,
                    &target,
                    DelaunayAdaptiveRefinementOptions::default(),
                    options,
                    cancellation,
                )
            };
        let result = build(
            &source,
            DelaunayAdaptiveTransferOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let nodes = source
            .topology
            .nodes
            .iter()
            .map(|n| (n.stable_identity, n.coordinates_m.canonical_value()))
            .collect::<BTreeMap<_, _>>();
        let target_nodes = target
            .topology
            .nodes
            .iter()
            .map(|n| (n.stable_identity, n.coordinates_m.canonical_value()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            result.transfer_map.node_transfers.len(),
            target_nodes
                .keys()
                .filter(|id| !nodes.contains_key(id))
                .count()
        );
        for transfer in &result.transfer_map.node_transfers {
            let point = target_nodes[&transfer.target_stable_identity];
            assert_eq!(
                transfer.method,
                if source_order == ElementOrder::Tet10 {
                    SolverTransferMethod::QuadraticLagrangeInterpolation
                } else {
                    SolverTransferMethod::BarycentricInterpolation
                }
            );
            for a in 0..4 {
                for b in 0..4 {
                    if source_order == ElementOrder::Tet4 && a != 3 && b != 3 {
                        continue;
                    }
                    let mode = |p: [f64; 3]| if a == 3 { 1.0 } else { p[a] } * if b == 3 { 1.0 } else { p[b] };
                    let actual: f64 = transfer
                        .sources
                        .iter()
                        .map(|s| s.weight * mode(nodes[&s.stable_identity]))
                        .sum();
                    assert!(
                        (actual - mode(point)).abs() < 2.0e-14,
                        "{a},{b}: {actual} vs {}",
                        mode(point)
                    );
                }
            }
        }
        if source_order == ElementOrder::Tet10 {
            assert!(result
                .transfer_map
                .node_transfers
                .iter()
                .any(|t| t.sources.iter().any(|s| s.weight < 0.0)));
            let mut curved = source.clone();
            {
                let quantity_change = 0.01;
                let quantity_index = 2;
                let quantity_slot = &mut curved.topology.nodes[4].coordinates_m;
                let mut quantity_coordinates = quantity_slot.canonical_value();
                quantity_coordinates[quantity_index] += quantity_change;
                *quantity_slot = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(quantity_coordinates);
            };
            curved.seal_canonical_digest().unwrap();
            assert_eq!(
                build(
                    &curved,
                    DelaunayAdaptiveTransferOptions::default(),
                    &NeverCancelled
                )
                .unwrap_err()
                .kind,
                DelaunayAdaptiveTransferErrorKind::UnsupportedGeometry
            );
        }
        // One shared ceiling covers source-affinity checks, nodal location and all
        // four child centroids; separate per-phase ceilings would incorrectly pass.
        let required = (if source_order == ElementOrder::Tet10 {
            6
        } else {
            0
        }) + 4 * result.transfer_map.node_transfers.len() as u64
            + 16;
        assert_eq!(
            build(
                &source,
                DelaunayAdaptiveTransferOptions {
                    maximum_point_location_predicates: required - 1,
                    ..Default::default()
                },
                &NeverCancelled
            )
            .unwrap_err()
            .kind,
            DelaunayAdaptiveTransferErrorKind::ResourceLimit
        );
        build(
            &source,
            DelaunayAdaptiveTransferOptions {
                maximum_point_location_predicates: required,
                ..Default::default()
            },
            &NeverCancelled,
        )
        .unwrap();
        let bytes = result.transfer_map.canonical_encode().unwrap();
        let cold = SolverMeshTransferMap::canonical_decode(&bytes).unwrap();
        assert_eq!(cold, result.transfer_map);
        result
            .lineage
            .validate_against(&source, &target, &cold)
            .unwrap();
        assert_eq!(
            build(
                &source,
                DelaunayAdaptiveTransferOptions {
                    maximum_point_location_predicates: 6,
                    ..Default::default()
                },
                &NeverCancelled
            )
            .unwrap_err()
            .kind,
            DelaunayAdaptiveTransferErrorKind::ResourceLimit
        );
        assert_eq!(
            build(
                &source,
                DelaunayAdaptiveTransferOptions::default(),
                &Cancelled
            )
            .unwrap_err()
            .kind,
            DelaunayAdaptiveTransferErrorKind::Cancelled
        );
    }
}

#[test]
fn quadratic_coarsening_preserves_original_nodal_inventory() {
    let (topology, provenance, metric, quality_options, refinement) = adaptation();
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &metric,
        &provenance,
        quality_options,
        &NeverCancelled,
    )
    .unwrap();
    let input = DelaunayVolumeRefinementInput {
        topology: &topology,
        metric_request: &metric,
        provenance: &provenance,
        quality: &quality,
        quality_options,
        fixed_node_identities: &[],
    };
    let original = elevate(artifact(&topology));
    let refined = elevate(artifact(&refinement.topology));
    let forward = build_refinement_solver_adaptation(
        input,
        &refinement,
        &original,
        &refined,
        DelaunayAdaptiveRefinementOptions::default(),
        DelaunayAdaptiveTransferOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let inserted = forward.lineage.mutations[0].node_identity;
    let removal = [inserted];
    let coarsening = coarsen_marked_delaunay_volume(
        input,
        &refinement,
        &removal,
        DelaunayAdaptiveCoarseningOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let restored = elevate(artifact(&coarsening.topology));
    let reverse = build_coarsening_solver_adaptation(
        DelaunayAdaptiveCoarseningInput {
            original: input,
            refinement: &refinement,
            removal_node_identities: &removal,
            coarsening: &coarsening,
            source_artifact: &refined,
            target_artifact: &restored,
        },
        DelaunayAdaptiveCoarseningOptions::default(),
        DelaunayAdaptiveTransferOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert!(reverse.transfer_map.node_transfers.is_empty());
    let retained = refined
        .topology
        .nodes
        .iter()
        .map(|n| (n.stable_identity, n.coordinates_m.canonical_value()))
        .collect::<BTreeMap<_, _>>();
    assert!(restored
        .topology
        .nodes
        .iter()
        .all(|n| retained.get(&n.stable_identity) == Some(&n.coordinates_m.canonical_value())));
    reverse
        .lineage
        .validate_against(&refined, &restored, &reverse.transfer_map)
        .unwrap();
}

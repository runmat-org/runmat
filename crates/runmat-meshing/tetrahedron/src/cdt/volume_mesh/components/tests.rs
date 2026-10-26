use super::*;
use crate::cdt::volume_mesh::tests::{metric_request, volume_options};
use crate::cdt::{
    construct_delaunay_volume_mesh, encode_delaunay_volume_mesh, evaluate_delaunay_volume_quality,
    DelaunayVolumeMesh,
};
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled, StableDigest};

fn component() -> DelaunayVolumeMesh {
    let (topology, surface) = crate::cdt::constraints::tests::tetrahedron();
    construct_delaunay_volume_mesh(
        &topology,
        &surface,
        &metric_request(),
        volume_options(),
        &NeverCancelled,
    )
    .unwrap()
}

fn reference(meshes: &[DelaunayVolumeMesh]) -> DelaunayVolumeMesh {
    let options = volume_options();
    let topology = super::reference::assemble_topology(
        meshes,
        options.refinement.step.insertion.topology,
        &NeverCancelled,
    )
    .unwrap();
    let provenance = super::reference::assemble_provenance(meshes);
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &metric_request(),
        &provenance,
        options.quality,
        &NeverCancelled,
    )
    .unwrap();
    DelaunayVolumeMesh {
        topology,
        provenance,
        quality,
        facet_recovery_insertions: super::reference::assemble_insertions(meshes),
        quality_mutations: super::reference::assemble_quality_mutations(meshes),
    }
}

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Single-region completion retains the original node, cell and quality allocations and
//   matches the former rebuild's complete output and independently validated canonical encoding.
//   This excludes whole-native-study memory peaks and multi-region aggregate allocation bounds.
#[test]
fn single_component_moves_storage_and_preserves_independent_encoding() {
    let mesh = component();
    let expected = reference(std::slice::from_ref(&mesh));
    let nodes = mesh.topology.nodes.as_ptr();
    let cells = mesh.topology.tetrahedra.as_ptr();
    let quality = mesh.quality.tetrahedra.as_ptr();
    let actual = assemble(
        vec![mesh],
        &metric_request(),
        volume_options(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(actual.topology.nodes.as_ptr(), nodes);
    assert_eq!(actual.topology.tetrahedra.as_ptr(), cells);
    assert_eq!(actual.quality.tetrahedra.as_ptr(), quality);
    assert_eq!(actual, expected);
    let (topology, surface) = crate::cdt::constraints::tests::tetrahedron();
    let encode = |mesh| {
        encode_delaunay_volume_mesh(
            mesh,
            &topology,
            &surface,
            &metric_request(),
            volume_options(),
            &NeverCancelled,
        )
        .unwrap()
    };
    assert_eq!(encode(&actual), encode(&expected));
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Consuming multi-region assembly matches the original independent rebuild including
//   merged shared-node provenance and complete quality. This does not establish arbitrary partition
//   scaling, native throughput or whole-task admission.
#[test]
fn multiple_components_match_original_assembly_and_merge_shared_provenance() {
    let first = component();
    let mut second = first.clone();
    for cell in &mut second.topology.tetrahedra {
        cell.region_id
            .as_mut()
            .unwrap()
            .source_topology_id
            .push_str("-second-region");
    }
    // Disjoint cells sharing an original boundary node exercise identity merging.
    // Reflect one tetrahedral component across its original corner.
    for node in &mut second.topology.nodes {
        if node.coordinates_m.canonical_value() != [0.0; 3] {
            node.coordinates_m =
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                    node.coordinates_m.canonical_value().map(|value| -value),
                );
            let mut bytes = *node.identity.bytes();
            bytes[0] ^= 0xff;
            node.identity = StableDigest::from_bytes(bytes);
        }
    }
    // Keep one common node binding; the remaining provenance belongs to the
    // first component. Both algorithms must preserve exactly the same inventory.
    second.provenance.nodes.retain(|binding| {
        second
            .topology
            .nodes
            .iter()
            .any(|node| node.identity == binding.node_identity)
    });
    second.provenance.segments.clear();
    second.provenance.facets.clear();
    let meshes = vec![first, second];
    let expected = reference(&meshes);
    let actual = assemble(meshes, &metric_request(), volume_options(), &NeverCancelled).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn conflicting_identity_and_cancelled_handoff_reject() {
    let first = component();
    let mut second = first.clone();
    {
        let quantity_change = 1.0;
        let quantity_index = 0;
        let quantity_slot = &mut second.topology.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] += quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert_eq!(
        assemble(
            vec![first, second],
            &metric_request(),
            volume_options(),
            &NeverCancelled
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeMeshErrorKind::InvalidTopology
    );
    let mut invalid = volume_options();
    invalid
        .refinement
        .step
        .insertion
        .topology
        .cancellation_check_interval = 0;
    assert_eq!(
        assemble(
            vec![component()],
            &metric_request(),
            invalid,
            &NeverCancelled
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeMeshErrorKind::InvalidOptions
    );
    let mut limited = volume_options();
    limited.refinement.step.insertion.topology.maximum_nodes = 1;
    assert_eq!(
        assemble(
            vec![component(), component()],
            &metric_request(),
            limited,
            &NeverCancelled
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeMeshErrorKind::ResourceLimit
    );
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    assert_eq!(
        assemble(
            vec![component()],
            &metric_request(),
            volume_options(),
            &Cancelled
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeMeshErrorKind::Cancelled
    );
    struct CancelDuringAssembly(std::sync::atomic::AtomicUsize);
    impl MeshingCancellationSignal for CancelDuringAssembly {
        fn is_cancelled(&self) -> bool {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed) > 0
        }
    }
    let cancellation = CancelDuringAssembly(std::sync::atomic::AtomicUsize::new(0));
    assert_eq!(
        assemble(
            vec![component(), component()],
            &metric_request(),
            volume_options(),
            &cancellation
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeMeshErrorKind::Cancelled
    );
}

// witness: FEA-MESH-DISCRETIZATION-001/volume_conformity @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Conformal multi-region interfaces mesh in one component so both sides share every
//   interface Steiner point (fsi-compliant-tube bore).
#[test]
fn regions_joined_by_an_interface_facet_form_one_component() {
    let constraints = crate::cdt::test_fixtures::same_facet_multi_steiner_constraints();
    let components = super::partition_constraints(&constraints).unwrap();
    assert_eq!(components.len(), 1);
    assert!(components[0].joined_regions);
    assert_eq!(
        components[0].constraints.facets.len(),
        constraints.facets.len()
    );
}

use super::*;
use crate::cdt::topology::build_delaunay_volume_topology_with_regions;
use crate::cdt::volume_quality::tests::{metric_contexts, metric_request, region, topology};
use crate::cdt::DelaunayTopologyOptions;
use runmat_meshing_core::NeverCancelled;

fn assert_reference(
    candidate: &DelaunayVolumeTopology,
    base: &DelaunayVolumeTopology,
    request: &MetricFieldRequest,
    provenance: &DelaunayVolumeProvenance,
    expected_reused: usize,
) {
    let options = DelaunayVolumeQualityOptions::default();
    let base_quality =
        evaluate_delaunay_volume_quality(base, request, provenance, options, &NeverCancelled)
            .unwrap();
    let (actual, reused) = evaluate(
        candidate,
        (base, &base_quality),
        request,
        provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let reference =
        evaluate_delaunay_volume_quality(candidate, request, provenance, options, &NeverCancelled)
            .unwrap();
    assert_eq!(reused, expected_reused);
    assert_eq!(actual, reference);
    validate_delaunay_volume_quality(
        candidate,
        request,
        provenance,
        &actual,
        options,
        &NeverCancelled,
    )
    .unwrap();
}

#[test]
fn constant_metric_reuses_only_physically_unchanged_cells() {
    let base = topology();
    let provenance = metric_contexts(&base);
    let mut request = metric_request();
    request.contributions.clear();
    assert_reference(&base, &base, &request, &provenance, 2);
    let mut moved = base.nodes.clone();
    {
        let quantity_change = -0.2;
        let quantity_index = 2;
        let quantity_slot = &mut moved[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    let candidate = build_delaunay_volume_topology_with_regions(
        moved,
        base.tetrahedra
            .iter()
            .map(|cell| (cell.vertex_indices, cell.region_id.clone()))
            .collect(),
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_reference(&candidate, &base, &request, &provenance, 1);
    let regions = build_delaunay_volume_topology_with_regions(
        base.nodes.clone(),
        base.tetrahedra
            .iter()
            .enumerate()
            .map(|(index, cell)| {
                (
                    cell.vertex_indices,
                    if index == 0 {
                        Some(region("replacement"))
                    } else {
                        cell.region_id.clone()
                    },
                )
            })
            .collect(),
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_reference(&regions, &base, &request, &provenance, 1);
    // Scoped metrics regrade the current graph, then retain only exact matches.
    assert_reference(&candidate, &base, &metric_request(), &provenance, 1);
}

#[test]
fn mutation_quality_preserves_cancellation_and_inventory_limits() {
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    let base = topology();
    let provenance = metric_contexts(&base);
    let mut request = metric_request();
    request.contributions.clear();
    let options = DelaunayVolumeQualityOptions::default();
    let quality =
        evaluate_delaunay_volume_quality(&base, &request, &provenance, options, &NeverCancelled)
            .unwrap();
    assert_eq!(
        evaluate(
            &base,
            (&base, &quality),
            &request,
            &provenance,
            options,
            &Cancelled
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeQualityErrorKind::Cancelled
    );
    let limited = DelaunayVolumeQualityOptions {
        maximum_tetrahedra: 1,
        ..options
    };
    assert_eq!(
        evaluate(
            &base,
            (&base, &quality),
            &request,
            &provenance,
            limited,
            &NeverCancelled
        )
        .unwrap_err()
        .kind,
        DelaunayVolumeQualityErrorKind::ResourceLimit
    );
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A physical four-cell centroid insertion reuses the unchanged neighboring cell while
//   recomputing the new cells, with inserted identities before, between and after the baseline node
//   identities. Complete quality evidence and independent validation match fresh evaluation under
//   unchanged constant-metric controls. This component evidence excludes general scoped-metric
//   reuse, arbitrary mutation locality and whole native qualification.
#[test]
fn insertion_recomputes_new_cells_and_preserves_original_quality_after_index_shifts() {
    use crate::cdt::DelaunayVolumeNode;
    let base = topology();
    let provenance = metric_contexts(&base);
    let mut request = metric_request();
    request.contributions.clear();
    let mut before = [0; 32];
    before[31] = 1;
    let mut between = [2; 32];
    between[31] = 3;
    for bytes in [before, between, [6; 32]] {
        let identity = StableDigest::from_bytes(bytes);
        let source = &base.tetrahedra[0];
        let centroid = std::array::from_fn(|axis| {
            source
                .vertex_indices
                .iter()
                .map(|vertex| base.nodes[*vertex as usize].coordinates_m.canonical_value()[axis])
                .sum::<f64>()
                / 4.0
        });
        let mut nodes = base.nodes.clone();
        nodes.push(DelaunayVolumeNode::explicit(identity, centroid));
        let old_order = nodes.iter().map(|node| node.identity).collect::<Vec<_>>();
        let inserted = (nodes.len() - 1) as u32;
        let mut cells = base
            .tetrahedra
            .iter()
            .skip(1)
            .map(|cell| (cell.vertex_indices, cell.region_id.clone()))
            .collect::<Vec<_>>();
        for opposite in 0..4 {
            let mut corners = source.vertex_indices;
            corners[opposite] = inserted;
            cells.push((corners, source.region_id.clone()));
        }
        nodes.sort_unstable_by_key(|node| node.identity);
        for (corners, _) in &mut cells {
            *corners = corners.map(|index| {
                nodes
                    .binary_search_by_key(&old_order[index as usize], |node| node.identity)
                    .unwrap() as u32
            });
        }
        let candidate = build_delaunay_volume_topology_with_regions(
            nodes,
            cells,
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        assert_reference(&candidate, &base, &request, &provenance, 1);
    }
}

#[test]
fn metric_changes_recompute_geometry_and_control_changes_reapply_policy() {
    let base = topology();
    let provenance = metric_contexts(&base);
    let mut request = metric_request();
    request.contributions.clear();
    let options = DelaunayVolumeQualityOptions::default();
    let baseline =
        evaluate_delaunay_volume_quality(&base, &request, &provenance, options, &NeverCancelled)
            .unwrap();
    let changed_controls = DelaunayVolumeQualityOptions {
        maximum_metric_edge_length: 0.1,
        maximum_radius_edge_ratio: 0.3,
        minimum_metric_scaled_jacobian: 0.9,
        ..options
    };
    let (actual, reused) = evaluate(
        &base,
        (&base, &baseline),
        &request,
        &provenance,
        changed_controls,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(reused, 2);
    let expected = evaluate_delaunay_volume_quality(
        &base,
        &request,
        &provenance,
        changed_controls,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(actual, expected);
    request.global_metric = MetricTensor3 {
        xx: 2.,
        yy: 3.,
        zz: 4.,
        xy: 0.1,
        xz: 0.2,
        yz: 0.3,
    };
    let (actual, reused) = evaluate(
        &base,
        (&base, &baseline),
        &request,
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(reused, 0);
    let expected =
        evaluate_delaunay_volume_quality(&base, &request, &provenance, options, &NeverCancelled)
            .unwrap();
    assert_eq!(actual, expected);
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: One physical node change recomputes its incident cell while reusing all unrelated
//   cells across one and 128-cell constant-metric inventories. Every quality record, metadata
//   field, aggregate and selected worst cell matches complete evaluation and independent quality
//   validation. Inventory traversal and metadata materialization remain global; this does not
//   establish sublinear work, scoped-metric propagation, measured native memory or accepted study
//   results.
#[test]
fn local_geometry_change_reuses_unrelated_cells_across_inventory_growth() {
    use crate::cdt::DelaunayVolumeNode;
    for count in [1, 128, 1024] {
        let mut nodes = Vec::new();
        let mut cells = Vec::new();
        for ordinal in 0..count {
            let x = ordinal as f64 * 3.0;
            let first = nodes.len() as u32;
            for point in [[x, 0., 0.], [x + 1., 0., 0.], [x, 1., 0.], [x, 0., 1.]] {
                let mut bytes = [0; 32];
                bytes[24..].copy_from_slice(&((nodes.len() + 1) as u64).to_be_bytes());
                nodes.push(DelaunayVolumeNode::explicit(
                    StableDigest::from_bytes(bytes),
                    point,
                ));
            }
            cells.push((
                [first, first + 1, first + 2, first + 3],
                Some(region("solid")),
            ));
        }
        let base = build_delaunay_volume_topology_with_regions(
            nodes,
            cells.clone(),
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let mut changed = base.nodes.clone();
        {
            let quantity_change = -0.2;
            let quantity_index = 2;
            let quantity_slot = &mut changed[0].coordinates_m;
            let mut quantity_coordinates = quantity_slot.canonical_value();
            quantity_coordinates[quantity_index] = quantity_change;
            *quantity_slot = runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Position,
                _,
            >::new(quantity_coordinates);
        };
        let candidate = build_delaunay_volume_topology_with_regions(
            changed,
            cells,
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let provenance = DelaunayVolumeProvenance {
            nodes: vec![],
            segments: vec![],
            facets: vec![],
        };
        let mut request = metric_request();
        request.contributions.clear();
        for scoped in [false, true] {
            if scoped {
                request.contributions = metric_request().contributions;
            }
            assert_reference(&candidate, &base, &request, &provenance, count - 1);
        }
    }
}

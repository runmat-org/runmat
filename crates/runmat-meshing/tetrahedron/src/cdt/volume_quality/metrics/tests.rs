use super::*;
use crate::cdt::volume_quality::tests::{metric_contexts, metric_request, topology};
use runmat_meshing_core::NeverCancelled;

#[test]
fn compact_metrics_match_complete_indexed_resolution_and_grading_exactly() {
    let topology = topology();
    let provenance = metric_contexts(&topology);
    let options = DelaunayVolumeQualityOptions::default();
    let contexts = validate_inputs(
        &topology,
        &metric_request(),
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    for ratio in [1.0, 1.25, f64::MAX] {
        for tensor in [
            MetricTensor3::isotropic_length_m(0.01).unwrap(),
            MetricTensor3 {
                xx: 4.0,
                yy: 3.0,
                zz: 2.0,
                xy: 0.2,
                xz: 0.1,
                yz: -0.1,
            },
        ] {
            for scoped in [false, true] {
                let mut request = metric_request();
                request.global_metric = tensor;
                request.maximum_grading_ratio = ratio;
                if !scoped {
                    request.contributions.clear();
                }
                let field = ResolvedMetricField::new(&request).unwrap();
                let expected = resolve_indexed_metrics(
                    &topology,
                    contexts.indexed().expect("scoped fixture inventory"),
                    &field,
                    &request,
                    options,
                    &NeverCancelled,
                )
                .unwrap();
                let observed = resolve_graded_metrics(
                    &topology,
                    &contexts,
                    &request,
                    options,
                    &NeverCancelled,
                )
                .unwrap();
                assert_eq!(observed.len(), topology.tetrahedra.len());
                assert_eq!(
                    matches!(observed, CellMetricEvaluations::Uniform { .. }),
                    !scoped
                );
                assert_eq!(observed.collect::<Vec<_>>(), expected);
            }
        }
    }
}

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Uniform metric contexts stream one cell at a time and match every original context
//   and complete quality report. This does not measure allocator footprint, scoped graph demand or
//   native study acceptance.
#[test]
fn streamed_uniform_contexts_preserve_every_original_cell_and_complete_quality() {
    let topology = topology();
    let provenance = metric_contexts(&topology);
    let mut request = metric_request();
    request.contributions.clear();
    let options = DelaunayVolumeQualityOptions::default();
    let streamed =
        validate_inputs(&topology, &request, &provenance, options, &NeverCancelled).unwrap();
    assert!(streamed.indexed().is_none());
    let original = crate::cdt::volume_provenance::derive_delaunay_volume_metric_contexts(
        &topology,
        &provenance,
        options.provenance,
        &NeverCancelled,
    )
    .unwrap();
    let observed = streamed.into_iter().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(observed, original);
    let expected = evaluate_from_metric_contexts(
        &topology,
        &request,
        options,
        MetricContexts::Indexed(original),
        stream::BoundaryProtection {
            nodes: &boundary_node_identities(&provenance),
            facets: &protected_facet_identities(&provenance),
            provenance: &provenance,
        },
        &NeverCancelled,
    )
    .unwrap();
    let actual = evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        options,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn scoped_resolution_work_counts_distinct_contexts_not_cells() {
    let topology = topology();
    let provenance = metric_contexts(&topology);
    let request = metric_request();
    let options = DelaunayVolumeQualityOptions::default();
    let contexts =
        validate_inputs(&topology, &request, &provenance, options, &NeverCancelled).unwrap();
    let contexts = contexts.indexed().unwrap();
    let field = ResolvedMetricField::new(&request).unwrap();
    let (observed, resolutions) = resolve_context_metrics(
        &topology,
        contexts.iter().map(|context| {
            (
                &context.tetrahedron_node_identities,
                context.incident_entity_ids.as_slice(),
            )
        }),
        &field,
        &request,
        options,
        &NeverCancelled,
    )
    .unwrap();
    let mut expected = contexts
        .iter()
        .map(|context| {
            field
                .resolve(&context.incident_entity_ids.iter().cloned().collect())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let identities = contexts
        .iter()
        .map(|context| &context.tetrahedron_node_identities)
        .collect::<Vec<_>>();
    let mut offsets = vec![0];
    let mut neighbors = vec![];
    for cell in &topology.tetrahedra {
        neighbors.extend(
            cell.neighbors
                .iter()
                .flatten()
                .map(|neighbor| *neighbor as usize),
        );
        offsets.push(neighbors.len());
    }
    grade_indexed_metric_evaluations(
        request.maximum_grading_ratio,
        &identities,
        &offsets,
        &neighbors,
        &mut expected,
    )
    .unwrap();
    assert_eq!(observed, expected);
    assert_eq!(
        resolutions,
        contexts
            .iter()
            .map(|context| &context.incident_entity_ids)
            .collect::<BTreeSet<_>>()
            .len()
    );
}

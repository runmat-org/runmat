//! Terminal snapshots preserve rejected work without acquiring mesh authority.
use super::*;
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_size::metric::{MetricCombinationRule, MetricTensor3};
use std::sync::Mutex;

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An immutable, rejected tetrahedron reaches a host-owned snapshot observer. An
//   injected optional observer failure preserves the original quality verdict; cold snapshot
//   decoding preserves bound controls and remains rejected. No accepted mesh or continuation
//   authority is created, and native cache retention under every budget is not asserted.
#[test]
fn stalled_state_reaches_the_host_observer_without_replacing_the_verdict() {
    struct Observer(Mutex<Option<Vec<u8>>>);
    impl VolumeQualityObserver for Observer {
        fn observe(&self, _: VolumeQualitySnapshot<'_>) -> Result<(), DelaunayVolumeMeshError> {
            Ok(())
        }
        fn observe_stalled(
            &self,
            snapshot: VolumeQualitySnapshot<'_>,
        ) -> Result<(), DelaunayVolumeMeshError> {
            *self.0.lock().unwrap() = Some(encode_volume_quality_snapshot(
                &snapshot,
                binding(),
                LIMITS,
                &Control,
            )?);
            Err(error(
                DelaunayVolumeMeshStage::Quality,
                DelaunayVolumeMeshErrorKind::Cancelled,
                "injected optional observer failure",
            ))
        }
    }
    let entity = |kind, name: String| PersistentEntityId::new(kind, name, Vec::new());
    let region = entity(PersistentEntityKind::Region, "immutable-tetrahedron".into());
    let nodes = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1e-4]]
        .into_iter()
        .enumerate()
        .map(|(index, point)| {
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([index as u8 + 1; 32]), point)
        })
        .collect();
    let topology = crate::cdt::build_delaunay_volume_topology(
        nodes,
        vec![[0, 1, 2, 3]],
        Default::default(),
        &NeverCancelled,
    )
    .unwrap();
    let topology = crate::cdt::assign_delaunay_volume_regions(
        topology,
        vec![region.clone()],
        Default::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut facets = topology
        .incidence
        .boundary_facets
        .iter()
        .enumerate()
        .map(|(index, facet)| {
            let mut identities = facet
                .vertex_indices
                .map(|vertex| topology.nodes[vertex as usize].identity);
            identities.sort_unstable();
            crate::cdt::DelaunayFacetProvenance {
                node_identities: identities,
                chart_id: StableDigest::from_bytes([42; 32]),
                entity_ids: vec![entity(
                    PersistentEntityKind::Face,
                    format!("immutable-face-{index}"),
                )],
                region_ids: vec![region.clone()],
            }
        })
        .collect::<Vec<_>>();
    facets.sort_by_key(|facet| facet.node_identities);
    let provenance = DelaunayVolumeProvenance {
        nodes: Vec::new(),
        segments: Vec::new(),
        facets,
    };
    let request = MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(10.).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
        cells_across_gap: None,
    };
    let options = DelaunayVolumeMeshOptions::default();
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        options.quality,
        &NeverCancelled,
    )
    .unwrap();
    assert!(quality.worst_refinement_tetrahedron.is_none());
    assert!(quality.worst_optimization_tetrahedron.is_some());
    let observer = Observer(Mutex::new(None));
    // All four faces are immutable and the sole cell has no neighbors. There is
    // no legal interior relocation, reconnection, or protected-boundary split.
    let failure = super::super::super::quality_convergence::converge_volume_quality(
        topology,
        quality,
        &request,
        provenance,
        &[],
        options,
        super::super::super::quality_convergence::QualityExecution {
            cancellation: &NeverCancelled,
            observer: &observer,
        },
    )
    .err()
    .expect("an immutable bad cell cannot acquire mesh acceptance");
    assert_eq!(failure.kind, DelaunayVolumeMeshErrorKind::InvalidQuality);
    assert!(failure.reason.contains("NoAdmissibleOptimization"));
    assert!(!failure
        .reason
        .contains("injected optional observer failure"));
    let captured = observer
        .0
        .into_inner()
        .unwrap()
        .expect("stalled state must reach its observer");
    let replay_observer = Observer(Mutex::new(None));
    let mut foreign = binding();
    foreign.geometry = StableDigest::from_bytes([99; 32]);
    assert!(replay_volume_quality_observed(
        &captured,
        foreign,
        options,
        LIMITS,
        &Control,
        &NeverCancelled,
        &replay_observer
    )
    .is_err());
    assert!(
        replay_observer.0.lock().unwrap().is_none(),
        "untrusted input must not reach observation"
    );
    let replay_failure = replay_volume_quality_observed(
        &captured,
        binding(),
        options,
        LIMITS,
        &Control,
        &NeverCancelled,
        &replay_observer,
    )
    .err()
    .expect("replay cannot promote rejected geometry");
    assert_eq!(replay_failure.kind, failure.kind);
    assert!(replay_failure.reason.contains("NoAdmissibleOptimization"));
    assert!(!replay_failure
        .reason
        .contains("injected optional observer failure"));
    assert!(
        replay_observer.0.lock().unwrap().is_some(),
        "checked replay must forward the terminal observation"
    );
    let decoded: DecodedEnvelope = runmat_canonical_codec::decode_contract_with_control(
        PREFIX, DOMAIN, &captured, LIMITS, &Control,
    )
    .unwrap();
    assert_eq!(decoded.binding, binding());
    assert_eq!(decoded.input.options, options);
    let expansion =
        runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate(
            [captured.as_slice()],
            &Control,
        )
        .unwrap();
    let cold = decoded
        .input
        .volume
        .into_parts(options, &NeverCancelled, expansion, None)
        .unwrap();
    assert_eq!(cold.topology.nodes.len(), 4);
    assert_eq!(cold.topology.tetrahedra.len(), 1);
    let quality = evaluate_delaunay_volume_quality(
        &cold.topology,
        &decoded.input.metric,
        &cold.provenance,
        options.quality,
        &NeverCancelled,
    )
    .unwrap();
    assert!(
        quality.worst_optimization_tetrahedron.is_some(),
        "cold diagnostics must not upgrade the rejected state"
    );
}

use super::*;
use crate::cdt::test_fixtures::coupled_quality::{fixture, metric};
use runmat_meshing_core::NeverCancelled;
mod compatibility;

struct Control;
impl CanonicalCodecControl for Control {
    fn is_cancelled(&self) -> bool {
        false
    }
}

fn binding() -> VolumeQualityBinding {
    VolumeQualityBinding {
        request: StableDigest::from_bytes([1; 32]),
        geometry: StableDigest::from_bytes([2; 32]),
        implementation: StableDigest::from_bytes([3; 32]),
    }
}
const LIMITS: CanonicalLimits = CanonicalLimits::new(4 * 1024 * 1024, 100_000, 4096, 64);

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The retained96-node/236-cell scoped collar has a certified fixed-boundary quality
//   conflict at the unchanged0.01 target. Increasing the explicitly bound search cap cannot make it
//   feasible. Exact protected-facet preservation remains mandatory; a generated surface must
//   instead be repaired by its surface owner. This witness does not establish positive convergence,
//   upstream surface repair or completed studies.
#[test]
fn thin_boundary_collar_requires_surface_remeshing() {
    compatibility::assert_incompatible(include_bytes!("fixtures/noncommuting-source-quality.cbor"));
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An outward common-metric edge-star bound proves that the retained108-node/292-cell
//   fixed boundary cannot satisfy its unchanged0.01 metric scaled-Jacobian target. Original and
//   explicitly rebound million-operation controls reject with a surface-remeshing diagnosis before
//   volume search. This is incompatibility classification, not positive convergence, surface
//   repair, native CAD acceptance or study qualification.
#[test]
fn retained_terminal_collar_requires_surface_remeshing() {
    compatibility::assert_incompatible(include_bytes!(
        "fixtures/terminal-noncommuting-collar.cbor"
    ));
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A retained96node/236cell diagnostic binds an explicit2048candidate-per-pass cap and
//   rejects substitution of a million-candidate policy before execution. Native governing requests
//   are independently projected and charged by the shared stage authority; this authority check
//   does not predict budget exhaustion or establish quality convergence, CAD acceptance or a
//   completed study.
#[test]
fn noncommuting_source_fixture_preserves_its_explicit_small_search_cap() {
    #[derive(Deserialize)]
    struct Controls {
        binding: VolumeQualityBinding,
        options: DelaunayVolumeMeshOptions,
    }
    let controls: Controls =
        serde_json::from_str(include_str!("fixtures/noncommuting-source-controls.json")).unwrap();
    assert_eq!(
        controls
            .options
            .sliver
            .maximum_candidate_evaluations_per_pass,
        2048
    );
    let decoded: DecodedEnvelope = runmat_canonical_codec::decode_contract_with_control(
        PREFIX,
        DOMAIN,
        include_bytes!("fixtures/noncommuting-source-quality.cbor"),
        LIMITS,
        &Control,
    )
    .unwrap();
    assert_eq!(decoded.binding, controls.binding);
    assert_eq!(decoded.input.options, controls.options);
    // A search cap is authority, not a prediction that one particular algorithm
    // must exhaust it rather than converge or report no admissible repair.
    // Rebinding controls must be explicit; a caller cannot silently upgrade it.
    let mut substituted = controls.options;
    substituted.sliver.maximum_candidate_evaluations_per_pass = 1_000_000;
    let error = replay_volume_quality(
        include_bytes!("fixtures/noncommuting-source-quality.cbor"),
        controls.binding,
        substituted,
        LIMITS,
        &Control,
        &NeverCancelled,
    )
    .err()
    .expect("a diagnostic cannot authorize substituted search controls");
    assert_eq!(error.kind, DelaunayVolumeMeshErrorKind::InvalidOptions);
    assert!(error.reason.contains("original controls differ"));
}

#[test]
fn diagnostic_replay_revalidates_the_original_coupled_state_without_cad() {
    for scoped in [false, true] {
        let (topology, provenance) = fixture(false, 0);
        let metric = metric(&provenance, scoped);
        let snapshot = VolumeQualitySnapshot {
            topology: &topology,
            provenance: &provenance,
            metric: &metric,
            fixed_nodes: &[],
            options: DelaunayVolumeMeshOptions::default(),
        };
        let bytes = encode_volume_quality_snapshot(&snapshot, binding(), LIMITS, &Control).unwrap();
        let result = replay_volume_quality(
            &bytes,
            binding(),
            snapshot.options,
            LIMITS,
            &Control,
            &NeverCancelled,
        )
        .unwrap();
        assert!(result.quality.worst_refinement_tetrahedron.is_none());
        assert!(result.quality.worst_optimization_tetrahedron.is_none());
        assert!(result.quality_mutations.iter().any(|step| matches!(
            step,
            super::super::DelaunayVolumeQualityMutation::Optimized(_)
        )));
        crate::cdt::validate_delaunay_volume_quality(
            &result.topology,
            &metric,
            &result.provenance,
            &result.quality,
            snapshot.options.quality,
            &NeverCancelled,
        )
        .unwrap();
    }
}

// witness: FEA-SHARED-RESOURCES-001/governed_path_equivalence @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Quality-stage diagnostics reject changed request/geometry/implementation identities,
//   changed original controls, damaged geometry, truncation and byte/item ceilings. Replay remains
//   diagnostic input and cannot manufacture complete CAD publication authority.
#[test]
fn diagnostics_reject_changed_identity_controls_topology_and_hostile_bounds() {
    let (mut topology, provenance) = fixture(false, 0);
    let metric = metric(&provenance, true);
    let options = DelaunayVolumeMeshOptions::default();
    let snapshot = VolumeQualitySnapshot {
        topology: &topology,
        provenance: &provenance,
        metric: &metric,
        fixed_nodes: &[],
        options,
    };
    let bytes = encode_volume_quality_snapshot(&snapshot, binding(), LIMITS, &Control).unwrap();
    let replay = |bytes: &[u8], expected, options, limits| {
        replay_volume_quality(bytes, expected, options, limits, &Control, &NeverCancelled).is_err()
    };
    for field in 0..3 {
        let mut expected = binding();
        match field {
            0 => expected.request = StableDigest::from_bytes([8; 32]),
            1 => expected.geometry = StableDigest::from_bytes([8; 32]),
            _ => expected.implementation = StableDigest::from_bytes([8; 32]),
        }
        assert!(replay(&bytes, expected, options, LIMITS));
    }
    let mut changed = options;
    changed.refinement.maximum_insertions += 1;
    assert!(replay(&bytes, binding(), changed, LIMITS));
    let mut limited = LIMITS;
    limited.maximum_encoded_bytes = bytes.len() - 1;
    assert!(replay(&bytes, binding(), options, limited));
    limited = LIMITS;
    limited.maximum_collection_items = 1;
    assert!(replay(&bytes, binding(), options, limited));
    assert!(replay(
        &bytes[..bytes.len() - 1],
        binding(),
        options,
        LIMITS
    ));
    topology.nodes[0].coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new(topology.nodes[1].coordinates_m.canonical_value());
    let corrupt = VolumeQualitySnapshot {
        topology: &topology,
        provenance: &provenance,
        metric: &metric,
        fixed_nodes: &[],
        options,
    };
    let bytes = encode_volume_quality_snapshot(&corrupt, binding(), LIMITS, &Control).unwrap();
    assert!(replay(&bytes, binding(), options, LIMITS));
}

#[test]
fn production_observer_runs_before_quality_and_validated_token_preserves_wire_bytes() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Observer(AtomicUsize);
    impl VolumeQualityObserver for Observer {
        fn observe(
            &self,
            snapshot: VolumeQualitySnapshot<'_>,
        ) -> Result<(), DelaunayVolumeMeshError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            let bytes = encode_volume_quality_snapshot(&snapshot, binding(), LIMITS, &Control)?;
            assert!(!bytes.is_empty());
            Ok(())
        }
    }
    let (topology, surface) = crate::cdt::constraints::tests::tetrahedron();
    let options = super::super::tests::volume_options();
    let metric = super::super::tests::metric_request();
    let observer = Observer(AtomicUsize::new(0));
    let admitted = construct_delaunay_volume_mesh_with_quality_observer(
        &topology,
        &surface,
        &metric,
        options,
        &NeverCancelled,
        &observer,
    )
    .unwrap();
    assert_eq!(observer.0.load(Ordering::Relaxed), 1);
    let bytes = admitted.encode(&Control).unwrap();
    assert!(
        bytes.len() as u64
            <= admitted
                .encoding_profile(&Control)
                .unwrap()
                .encoded_bytes_upper_bound
    );
    assert_eq!(
        bytes,
        encode_delaunay_volume_mesh(
            &admitted,
            &topology,
            &surface,
            &metric,
            options,
            &NeverCancelled,
        )
        .unwrap()
    );
    let decoded = decode_delaunay_volume_mesh(
        &bytes,
        &topology,
        &surface,
        &metric,
        options,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(*admitted, decoded);
}

#[test]
fn replay_preserves_a_one_shot_codec_cancellation() {
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Pulse(AtomicBool);
    impl CanonicalCodecControl for Pulse {
        fn is_cancelled(&self) -> bool {
            self.0.swap(false, Ordering::Relaxed)
        }
    }
    let (topology, provenance) = fixture(false, 0);
    let metric = metric(&provenance, false);
    let snapshot = VolumeQualitySnapshot {
        topology: &topology,
        provenance: &provenance,
        metric: &metric,
        fixed_nodes: &[],
        options: DelaunayVolumeMeshOptions::default(),
    };
    let bytes = encode_volume_quality_snapshot(&snapshot, binding(), LIMITS, &Control).unwrap();
    let failure = replay_volume_quality(
        &bytes,
        binding(),
        snapshot.options,
        LIMITS,
        &Pulse(AtomicBool::new(true)),
        &NeverCancelled,
    )
    .err()
    .unwrap();
    assert_eq!(failure.kind, DelaunayVolumeMeshErrorKind::Cancelled);
}

mod stalled;

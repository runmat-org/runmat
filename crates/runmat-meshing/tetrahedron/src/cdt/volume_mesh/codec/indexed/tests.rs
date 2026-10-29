use super::*;
use crate::cdt::test_fixtures::coupled_quality::{fixture, metric};
use runmat_meshing_core::NeverCancelled;
struct Control;
impl runmat_canonical_codec::CanonicalCodecControl for Control {
    fn is_cancelled(&self) -> bool {
        false
    }
}
fn mesh(permuted: bool, background: usize, scoped: bool) -> DelaunayVolumeMesh {
    let (topology, provenance) = fixture(permuted, background);
    let request = metric(&provenance, scoped);
    let quality = crate::cdt::evaluate_delaunay_volume_quality(
        &topology,
        &request,
        &provenance,
        crate::cdt::DelaunayVolumeQualityOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    if background == 0 {
        let result = crate::cdt::volume_mesh::quality_convergence::converge_volume_quality(
            topology,
            quality,
            &request,
            provenance,
            &[],
            DelaunayVolumeMeshOptions::default(),
            crate::cdt::volume_mesh::quality_convergence::QualityExecution::without_observer(
                &NeverCancelled,
            ),
        )
        .unwrap();
        assert!(result.quality_mutations.iter().any(|step| matches!(
            step,
            crate::cdt::DelaunayVolumeQualityMutation::Optimized(_)
        )));
        DelaunayVolumeMesh {
            topology: result.topology,
            provenance: result.provenance,
            quality: result.quality,
            facet_recovery_insertions: Vec::new(),
            quality_mutations: result.quality_mutations,
        }
    } else {
        DelaunayVolumeMesh {
            topology,
            provenance,
            quality,
            facet_recovery_insertions: Vec::new(),
            quality_mutations: Vec::new(),
        }
    }
}
fn decode(
    value: &serde_json::Value,
    budget: u64,
) -> Result<DelaunayVolumeMesh, super::super::super::DelaunayVolumeMeshError> {
    let decoded: DecodedEnvelope = serde_json::from_value(value.clone()).unwrap();
    decoded.mesh.into_mesh(
        DelaunayVolumeMeshOptions::default(),
        &NeverCancelled,
        budget,
        None,
    )
}
// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Compact artifact transport exactly preserves topology, provenance, complete accepted
//   histories, numeric quality and signed-zero bits under uniform/scoped metrics and identity
//   changes. This finite kernel round-trip does not establish native study acceptance or whole-task
//   memory.
#[test]
fn indexed_volume_preserves_all_topology_provenance_and_quality_values() {
    for permuted in [false, true] {
        for background in [0, 64] {
            for scoped in [false, true] {
                let mut original = mesh(permuted, background, scoped);
                original.quality.tetrahedra[0].resolved_metric.xy = -0.0;
                let prepared = VolumeMeshEncoding::new(&original, &Control).unwrap();
                let profile = prepared.encoding_profile(&Control).unwrap();
                let bytes = prepared.encode(&Control).unwrap();
                assert!(bytes.len() as u64 <= profile.encoded_bytes_upper_bound);
                let expansion = runmat_canonical_codec::canonical_chunked_payload_materialization_resident_bytes_estimate([bytes.as_slice()], &Control).unwrap();
                let decoded: DecodedEnvelope = runmat_canonical_codec::decode_contract(
                    CODEC_PREFIX,
                    CONTRACT_DOMAIN,
                    &bytes,
                    CONTRACT_LIMITS,
                )
                .unwrap();
                let recovered = decoded
                    .mesh
                    .into_mesh(
                        DelaunayVolumeMeshOptions::default(),
                        &NeverCancelled,
                        expansion,
                        None,
                    )
                    .unwrap();
                assert_eq!(recovered, original);
                assert_eq!(
                    VolumeMeshEncoding::new(&recovered, &Control)
                        .unwrap()
                        .encode(&Control)
                        .unwrap(),
                    bytes
                );
                for (left, right) in recovered
                    .quality
                    .tetrahedra
                    .iter()
                    .zip(&original.quality.tetrahedra)
                {
                    assert_eq!(
                        left.resolved_metric.xy.to_bits(),
                        right.resolved_metric.xy.to_bits()
                    );
                    assert_eq!(
                        left.optimization_violation_ratio.to_bits(),
                        right.optimization_violation_ratio.to_bits()
                    );
                }
            }
        }
    }
}
// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Foreign indices, duplicate tables/cells and unadmitted owned expansion reject in the
//   compact transport boundary. This finite hostile-input fixture does not inventory all validator
//   scratch or measure RSS.
#[test]
fn indexed_volume_rejects_foreign_indices_duplicate_tables_and_unadmitted_expansion() {
    let original = mesh(false, 0, true);
    let value =
        serde_json::to_value(VolumeMeshEncoding::new(&original, &Control).unwrap()).unwrap();
    assert_eq!(
        decode(&value, 0).unwrap_err().kind,
        super::super::super::DelaunayVolumeMeshErrorKind::ResourceLimit
    );
    for substitution in 0..5 {
        let mut changed = value.clone();
        match substitution {
            0 => changed["mesh"]["volume"]["cells"][0][0][0] = u32::MAX.into(),
            1 => changed["mesh"]["quality"]["values"][0][0] = u32::MAX.into(),
            2 => {
                let cell = changed["mesh"]["volume"]["cells"][0].clone();
                changed["mesh"]["volume"]["cells"]
                    .as_array_mut()
                    .unwrap()
                    .push(cell);
            }
            3 => {
                let entity = changed["mesh"]["volume"]["entities"][0].clone();
                changed["mesh"]["volume"]["entities"]
                    .as_array_mut()
                    .unwrap()
                    .insert(0, entity);
            }
            _ => changed["mesh"]["quality"]["worsts"][0] = u32::MAX.into(),
        }
        assert!(
            decode(&changed, u64::MAX).is_err(),
            "substitution {substitution}"
        );
    }
    // A valid dictionary string may be referenced many times. The expansion
    // budget rejects its owned copies rather than trusting the compact wire size.
    let mut amplified = value;
    amplified["mesh"]["volume"]["entities"]
        .as_array_mut()
        .unwrap()
        .last_mut()
        .unwrap()["source_topology_id"] = "z".repeat(8192).into();
    assert!(decode(&amplified, 4096).is_err());
}

mod scale;

// witness: FEA-SHARED-RESOURCES-001/phase_lifetimes @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: The public prepared encoder and its profiler retain a one-shot interruption as
//   cancellation, rather than re-querying the signal and relabeling it as invalid data. This
//   adapter regression does not certify all interruption points or native quiescence.
#[test]
fn public_encoding_keeps_one_shot_cancellation_distinct_from_invalid_artifacts() {
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Pulse(AtomicBool);
    impl runmat_canonical_codec::CanonicalCodecControl for Pulse {
        fn is_cancelled(&self) -> bool {
            self.0.swap(false, Ordering::Relaxed)
        }
    }
    let original = mesh(false, 0, false);
    let prepared = VolumeMeshEncoding::new(&original, &Control).unwrap();
    let cancelled = || Pulse(AtomicBool::new(true));
    assert_eq!(
        prepared.encoding_profile(&cancelled()).unwrap_err().kind,
        DelaunayVolumeMeshCodecErrorKind::Cancelled
    );
    assert_eq!(
        prepared.encode(&cancelled()).unwrap_err().kind,
        DelaunayVolumeMeshCodecErrorKind::Cancelled
    );
}

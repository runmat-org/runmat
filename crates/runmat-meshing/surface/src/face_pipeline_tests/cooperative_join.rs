//! Exercise the real decision, producer publication and cold consumer with the same fixtures.
use crate::*;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::{MeshingCancellationSignal, NeverCancelled};
use runmat_meshing_curve::SharedCurveMesh;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CancelAtJoin(AtomicUsize);
impl MeshingCancellationSignal for CancelAtJoin {
    #[track_caller]
    fn is_cancelled(&self) -> bool {
        let site = std::panic::Location::caller();
        if std::path::Path::new(site.file()).ends_with(std::path::Path::new("surface_mesh/join.rs"))
        {
            // Entry, batch validation and face entry must complete first.
            self.0.fetch_add(1, Ordering::SeqCst) >= 3
        } else {
            false
        }
    }
}

pub(super) fn verify(
    pass: &ExactSurfacePassResult,
    topology: &ExactBRepTopology,
    curves: &SharedCurveMesh,
    partitions: &[ExactFacePartitionResult],
) {
    super::publication_scale::verify_admitted_join(topology, curves, partitions, pass);
    let options = ExactSurfaceJoinOptions::default();
    assert_eq!(
        decide_exact_surface_pass_with_cancellation(
            curves,
            partitions,
            topology,
            options,
            &NeverCancelled
        )
        .unwrap(),
        *pass
    );
    let owned = decide_exact_surface_pass_owned(
        curves,
        partitions.to_vec(),
        topology,
        options,
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(
        owned.result(),
        pass,
        "move-based producer must preserve the original independently validated decision"
    );
    let bytes = encode_decided_exact_surface_pass_result(pass).unwrap();
    assert_eq!(
        encode_decided_exact_surface_pass_result_with_cancellation(pass, &NeverCancelled).unwrap(),
        bytes
    );
    assert_eq!(
        decode_exact_surface_pass_result_with_cancellation(
            &bytes,
            topology,
            curves,
            partitions,
            options,
            &NeverCancelled
        )
        .unwrap(),
        *pass
    );
    assert_eq!(
        decode_exact_surface_pass_result_with_cancellation(
            &bytes[..bytes.len() - 1],
            topology,
            curves,
            partitions,
            options,
            &NeverCancelled
        )
        .unwrap_err()
        .kind,
        ExactSurfaceMeshErrorKind::InvalidEncoding
    );
    let controlled = encode_exact_surface_mesh_from_pass_with_cancellation(
        pass,
        topology,
        curves,
        partitions,
        options,
        &NeverCancelled,
    );
    let original = encode_exact_surface_mesh_from_pass(pass, topology, curves, partitions, options);
    match (&controlled, &original) {
        (Ok(actual), Ok(expected)) => assert_bytes(actual, expected),
        _ => assert_eq!(
            controlled.as_ref().map(|_| ()),
            original.as_ref().map(|_| ())
        ),
    }
    let owned_bytes = owned.encode_with_cancellation(&NeverCancelled).unwrap();
    match &pass.outcome {
        ExactSurfacePassOutcome::Converged { .. } => {
            assert_bytes(&owned_bytes, controlled.as_ref().unwrap())
        }
        ExactSurfacePassOutcome::RequiresCurveSplits { .. } => assert_bytes(&owned_bytes, &bytes),
    }
    if let ExactSurfacePassOutcome::Converged { surface } = &pass.outcome {
        super::global_validation::verify(surface, topology);
        let signal = CancelAtJoin(AtomicUsize::new(0));
        let failure = decide_exact_surface_pass_with_cancellation(
            curves, partitions, topology, options, &signal,
        )
        .unwrap_err();
        assert!(signal.0.load(Ordering::SeqCst) >= 4);
        assert_eq!(
            failure.kind,
            ExactSurfaceConvergenceErrorKind::Surface(ExactSurfaceMeshErrorKind::Cancelled)
        );
        let bytes = controlled.unwrap();
        assert_eq!(
            decode_published_exact_surface_mesh_with_cancellation(
                &bytes,
                topology,
                options,
                &NeverCancelled
            )
            .unwrap(),
            *surface
        );
        for mutation in 0..7 {
            super::publication_scale::verify_global_admission(&bytes, surface, topology, options);
            let mut tampered = pass.clone();
            let ExactSurfacePassOutcome::Converged { surface } = &mut tampered.outcome else {
                unreachable!()
            };
            match mutation {
                0 => surface.triangles[0].acceptance_sample_count += 1,
                1 => surface.triangles[0].metric_edge_lengths[0] *= 0.5,
                2 => surface.triangles[0].unit_normal[0] += 0.125,
                3 => surface.nodes[0].uses[0].evaluator_uv[0] += 1e-7,
                4 => {
                    surface.nodes.pop();
                }
                5 => {
                    surface.face_ids.pop();
                }
                _ => {
                    surface.maximum_chordal_deviation_m = runmat_types::physical::Quantity::new(
                        surface.maximum_chordal_deviation_m.canonical_value() + (1e-7),
                    )
                }
            }
            let bytes = encode_decided_exact_surface_pass_result(&tampered).unwrap();
            assert!(
                decode_exact_surface_pass_result_with_cancellation(
                    &bytes,
                    topology,
                    curves,
                    partitions,
                    options,
                    &NeverCancelled
                )
                .is_err(),
                "cold borrowed prerequisite validation must reject mutation {mutation}"
            );
        }
        let mut tampered = pass.clone();
        let ExactSurfacePassOutcome::Converged { surface } = &mut tampered.outcome else {
            unreachable!()
        };
        surface.nodes[0].point_m[0] += 1.0;
        let bytes =
            encode_decided_exact_surface_pass_result_with_cancellation(&tampered, &NeverCancelled)
                .unwrap();
        assert_eq!(
            decode_exact_surface_pass_result_with_cancellation(
                &bytes,
                topology,
                curves,
                partitions,
                options,
                &NeverCancelled
            )
            .unwrap_err()
            .kind,
            ExactSurfaceMeshErrorKind::InvalidInput
        );
    }
}

fn assert_bytes(actual: &[u8], expected: &[u8]) {
    assert!(
        actual == expected,
        "canonical bytes differ: actual {} bytes, expected {} bytes, first differing offset {:?}",
        actual.len(),
        expected.len(),
        actual.iter().zip(expected).position(|(a, b)| a != b)
    );
}

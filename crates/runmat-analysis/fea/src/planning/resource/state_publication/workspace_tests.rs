//! Allocation-shape regressions for publication's extra source validation envelope.
//! These exercise shared profiler accounting, not numerical correctness of either assembly.
use runmat_canonical_codec::{profile_serialized, CanonicalLimits};

const LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 2_000_000, 65_536, 64);

#[test]
fn random_reassembly_scalar_workspace_fits_source_materialization_envelope() {
    for (intervals, values) in [(1, 1), (1, 16384), (64, 1024)] {
        // The source has I interval variance vectors and final variance + RMS vectors.
        // Metadata omitted here only reduces the shared envelope. Assembly simultaneously
        // owns cloned intervals, total/compensation (2N), reconstructed variance/RMS (2N)
        // and identity encoding of the reconstructed numerical result. The temporary flat N
        // vector is dropped before reconstruction, so the latter is the larger branch.
        let contributions = vec![vec![0.125_f64; values]; intervals];
        let numerical = (vec![0.125_f64; values], vec![0.353_f64; values]);
        let source = profile_serialized(&(&contributions, &numerical), LIMITS).unwrap();
        let reconstructed = profile_serialized(&numerical, LIMITS).unwrap();
        let live_scalar_copies = ((intervals + 4) * values * size_of::<f64>()) as u64;
        assert!(
            source.materialization_resident_bytes_estimate
                >= live_scalar_copies + reconstructed.materialization_resident_bytes_estimate
        );
    }
}

#[test]
fn spectrum_reassembly_scalar_workspace_fits_source_materialization_envelope() {
    for (directions, values) in [(1, 1), (1, 16384), (3, 8192)] {
        // Each direction has periodic, rigid and combined response. The source contains those
        // arrays in both assembly inputs and the numerical result, plus a final combined N.
        // Reassembly clones directions and then contributions (6DN), builds combined N and
        // validates it via a recombined N. The result identity encoding remains separate.
        let directions_data = vec![
            (
                vec![0.125_f64; values],
                vec![0.125_f64; values],
                vec![0.125_f64; values]
            );
            directions
        ];
        let combined = vec![0.125_f64; values];
        let numerical = (&directions_data, &combined);
        let source = profile_serialized(&(&directions_data, numerical), LIMITS).unwrap();
        let reconstructed = profile_serialized(&numerical, LIMITS).unwrap();
        let live_scalar_copies = ((6 * directions + 2) * values * size_of::<f64>()) as u64;
        assert!(
            source.materialization_resident_bytes_estimate
                >= live_scalar_copies + reconstructed.materialization_resident_bytes_estimate
        );
    }
}

#[test]
fn modal_governance_workspace_fits_basis_materialization_envelope() {
    for (modes, dofs) in [(1, 3), (128, 1024)] {
        let shapes = vec![vec![0.125_f64; dofs]; modes];
        let gram = vec![0.125_f64; modes * modes];
        let source = profile_serialized(&(&shapes, &gram), LIMITS).unwrap();
        // Frequency/residual vectors, temporary (start,end) ranges and evidence with two
        // usize plus four f64 fields per mode. Ranges and evidence overlap during collect.
        let governance =
            (modes * (2 * size_of::<f64>() + 4 * size_of::<usize>() + 4 * size_of::<f64>())) as u64;
        assert!(source.materialization_resident_bytes_estimate > governance);
    }
}

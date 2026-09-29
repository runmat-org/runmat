use super::*;

#[test]
fn dc_gain_distinguishes_small_finite_values_from_exact_singularities() {
    for (numerator, denominator, expected) in
        [(1.0, 1e-15, 1e15), (1e-15, 1e-15, 1.0), (1e-15, 1.0, 1e-15)]
    {
        let model = TfModel::new(
            vec![Complex64::new(numerator, 0.0)],
            vec![Complex64::new(denominator, 0.0)],
            TfOptions::default(),
        )
        .unwrap();
        let gain = model.dc_gain().unwrap();
        assert!(gain.re.is_finite());
        assert!((gain.re / expected - 1.0).abs() < 1e-14);
        assert_eq!(gain.im, 0.0);
    }
}

#[test]
fn dc_gain_retains_small_nonzero_components_at_exact_poles() {
    let model = TfModel::new(
        vec![Complex64::new(1e-15, -1e-30)],
        vec![Complex64::new(-1e-15, 0.0), Complex64::new(0.0, 0.0)],
        TfOptions::default(),
    )
    .unwrap();
    let gain = model.dc_gain().unwrap();
    assert!(gain.re.is_infinite() && gain.re.is_sign_negative());
    assert!(gain.im.is_infinite() && gain.im.is_sign_positive());
}

#[test]
fn response_real_admission_does_not_erase_tiny_imaginary_parts_or_delays() {
    let complex = TfModel::new(
        vec![Complex64::new(1.0, 1e-15)],
        vec![Complex64::new(1.0, 0.0)],
        TfOptions::default(),
    )
    .unwrap();
    for builtin in ["step", "impulse"] {
        assert_eq!(
            complex.to_real(builtin).unwrap_err().identifier(),
            Some(unsupported_model_identifier(builtin)),
        );
        for (input_delay, output_delay) in [(1e-15, 0.0), (0.0, 1e-15)] {
            let delayed = RealTfModel {
                numerator: vec![1.0],
                denominator: vec![1.0],
                sample_time: 0.0,
                input_delay,
                output_delay,
            };
            assert_eq!(
                delayed
                    .ensure_zero_delays(builtin)
                    .unwrap_err()
                    .identifier(),
                Some(unsupported_model_identifier(builtin))
            );
        }
    }
}

#[test]
fn coefficient_normalization_preserves_nonzero_scale_below_solver_tolerances() {
    let model = TfModel::new(
        vec![Complex64::new(2e-15, 0.0)],
        vec![Complex64::new(1e-15, 0.0), Complex64::new(2e-15, 0.0)],
        TfOptions::default(),
    )
    .unwrap();
    let normalized = model.normalized().unwrap();
    assert_eq!(normalized.numerator, vec![Complex64::new(2.0, 0.0)]);
    assert_eq!(
        normalized.denominator,
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]
    );
    for builtin in ["step", "impulse"] {
        let (num, den) = model.to_real(builtin).unwrap().normalized(builtin).unwrap();
        assert_eq!(num, vec![2.0]);
        assert_eq!(den, vec![1.0, 2.0]);
    }
}

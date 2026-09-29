use super::*;

#[test]
fn frequency_response_preserves_finite_extreme_scale_ratios_and_exact_poles() {
    for coefficient in [Complex64::new(1e-320, 0.0), Complex64::new(1e300, 1e300)] {
        let model =
            TfModel::new(vec![coefficient], vec![coefficient], TfOptions::default()).unwrap();
        let response = model.frequency_response(Complex64::new(0.0, 0.0));
        assert_eq!(response, Complex64::new(1.0, 0.0));
    }
    let model = TfModel::new(
        vec![Complex64::new(1.0, 0.0)],
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        TfOptions::default(),
    )
    .unwrap();
    let response = model.frequency_response(Complex64::new(0.0, 0.0));
    assert!(response.re.is_nan() && response.im.is_nan());
}

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

#[test]
fn coefficient_normalization_rejects_unrepresentable_finite_ratios() {
    let model = TfModel::new(
        vec![Complex64::new(1.0, 0.0)],
        vec![Complex64::new(1e-320, 0.0), Complex64::new(0.0, 0.0)],
        TfOptions::default(),
    )
    .unwrap();
    assert_eq!(
        model.normalized().unwrap_err().identifier(),
        Some("RunMat:tf:InvalidCoefficients")
    );
    for builtin in ["step", "impulse"] {
        assert_eq!(
            model
                .to_real(builtin)
                .unwrap()
                .normalized(builtin)
                .unwrap_err()
                .identifier(),
            Some(invalid_model_identifier(builtin))
        );
    }
}

#[test]
fn complex_normalization_avoids_intermediate_squared_norm_extremes() {
    for scale in [1e300, 1e-300] {
        let model = TfModel::new(
            vec![Complex64::new(2.0 * scale, -2.0 * scale)],
            vec![Complex64::new(scale, scale)],
            TfOptions::default(),
        )
        .unwrap();
        let normalized = model.normalized().expect("representable complex ratio");
        assert_eq!(normalized.denominator, vec![Complex64::new(1.0, 0.0)]);
        assert!((normalized.numerator[0] - Complex64::new(0.0, -2.0)).norm() < 1e-14);
    }
}

#[test]
fn dc_gain_exact_pole_direction_survives_extreme_scale_ratio() {
    let model = TfModel::new(
        vec![Complex64::new(1e-320, -1e-320)],
        vec![Complex64::new(1e300, 0.0), Complex64::new(0.0, 0.0)],
        TfOptions::default(),
    )
    .unwrap();
    let gain = model.dc_gain().expect("exact pole direction");
    assert!(gain.re.is_infinite() && gain.re.is_sign_positive());
    assert!(gain.im.is_infinite() && gain.im.is_sign_negative());
}

#[test]
fn dc_gain_nonzero_denominator_uses_representable_complex_ratio() {
    for scale in [1e300, 1e-300] {
        let model = TfModel::new(
            vec![Complex64::new(scale, -scale)],
            vec![Complex64::new(scale, scale)],
            TfOptions::default(),
        )
        .unwrap();
        let gain = model.dc_gain().expect("finite complex gain");
        assert!(gain.re.abs() < 1e-14);
        assert!((gain.im + 1.0).abs() < 1e-14);
    }
}

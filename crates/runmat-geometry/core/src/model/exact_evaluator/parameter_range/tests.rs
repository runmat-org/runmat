use super::*;

#[test]
fn closed_samples_preserve_endpoints_and_stay_monotone_across_scales() {
    for (start, end) in [
        (
            f64::from_bits(0xc02c0929e4f8b065),
            f64::from_bits(0x4035c37e673d6d73),
        ),
        (-3.0, 0.1),
        (-1.0, 0.123456789),
        (-0.0, 0.0),
        (1.0, 1.0_f64.next_up()),
        (-f64::MAX, f64::MAX),
        (f64::MAX / 2.0, f64::MAX),
        (-f64::MAX, -f64::MAX / 2.0),
        (-f64::MIN_POSITIVE, f64::MIN_POSITIVE),
        (0.0, f64::from_bits(1)),
    ] {
        let range = ParameterRange { start, end };
        let samples = range.uniform_samples::<1025>().unwrap();
        assert_eq!(samples[0].to_bits(), start.to_bits());
        assert_eq!(samples[1024].to_bits(), end.to_bits());
        assert!(samples
            .iter()
            .all(|p| p.is_finite() && *p >= start && *p <= end));
        assert!(samples.windows(2).all(|pair| pair[0] <= pair[1]));
    }
    let range = ParameterRange {
        start: -3.0,
        end: 0.1,
    };
    assert!(range.start + (range.end - range.start) > range.end);
    assert_eq!(
        range.uniform_samples::<9>().unwrap()[8].to_bits(),
        range.end.to_bits()
    );
}

#[test]
fn malformed_ranges_and_external_fractions_are_rejected() {
    let range = ParameterRange {
        start: -3.0,
        end: 0.1,
    };
    for fraction in [-f64::EPSILON, 1.0_f64.next_up(), f64::NAN, f64::INFINITY] {
        assert_eq!(
            range.parameter_at_fraction(fraction).unwrap_err().kind,
            GeometryEvaluationErrorKind::ParameterOutsideDomain
        );
    }
    for (start, end) in [(1.0, 0.0), (f64::NEG_INFINITY, 1.0), (0.0, f64::NAN)] {
        assert_eq!(
            ParameterRange { start, end }
                .parameter_at_fraction(0.5)
                .unwrap_err()
                .kind,
            GeometryEvaluationErrorKind::InvalidResult
        );
    }
    assert!(range.uniform_samples::<0>().is_err());
    assert!(range.uniform_samples::<1>().is_err());
}

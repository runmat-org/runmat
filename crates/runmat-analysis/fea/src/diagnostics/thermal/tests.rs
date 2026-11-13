use super::*;

#[test]
fn shared_borrowed_diagnostic_schema_preserves_legacy_json_and_canonical_bytes() {
    #[derive(Serialize)]
    #[serde(rename = "FeaDiagnostic")]
    struct LegacyDiagnostic {
        code: String,
        severity: FeaDiagnosticSeverity,
        message: String,
    }
    for severity in [
        FeaDiagnosticSeverity::Info,
        FeaDiagnosticSeverity::Warning,
        FeaDiagnosticSeverity::Error,
    ] {
        let actual = FeaDiagnostic {
            code: "FEA_TEST".into(),
            severity,
            message: "text with \"quotes\", \\ and 非ascii\n".into(),
        };
        let legacy = LegacyDiagnostic {
            code: actual.code.clone(),
            severity,
            message: actual.message.clone(),
        };
        let json = serde_json::to_vec(&actual).unwrap();
        assert_eq!(json, serde_json::to_vec(&legacy).unwrap());
        assert_eq!(
            serde_json::from_slice::<FeaDiagnostic>(&json).unwrap(),
            actual
        );
        let limits = CanonicalLimits::new(1 << 20, 100, 1 << 16, 32);
        assert_eq!(
            runmat_canonical_codec::encode_contract(b"test\0", "diagnostic", &actual, limits)
                .unwrap(),
            runmat_canonical_codec::encode_contract(b"test\0", "diagnostic", &legacy, limits)
                .unwrap()
        );
    }
}

#[test]
fn projection_uses_actual_diagnostic_wire_shape_and_longest_severities() {
    let projected = projection("step_\"\\\n非ascii");
    let actual: Vec<_> = projected
        .iter()
        .map(|record| record.message.diagnostic(record.severity))
        .collect();
    assert_eq!(thermal_diagnostic_record_count(), actual.len());
    assert_eq!(
        serde_json::to_value(&projected).unwrap(),
        serde_json::to_value(&actual).unwrap()
    );
    assert_eq!(
        thermal_diagnostic_profile("step_\"\\\n非ascii").unwrap(),
        profile_serialized(&actual, PROFILE_LIMITS).unwrap()
    );
    assert_eq!(actual[1].severity, FeaDiagnosticSeverity::Warning);
    assert_eq!(actual[3].severity, FeaDiagnosticSeverity::Error);
}

#[test]
fn scientific_float_width_and_roundtrip_cover_binary64_extremes() {
    let bound = format!("{:.16e}", -f64::MIN_POSITIVE).len();
    assert_eq!(bound, 24);
    let check = |value: f64| {
        let text = format!("{value:.16e}");
        assert!(text.len() <= bound, "{text}");
        assert_eq!(text.parse::<f64>().unwrap().to_bits(), value.to_bits());
    };
    for value in [
        0.0,
        -0.0,
        f64::MAX,
        -f64::MAX,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
    ] {
        check(value);
    }
    // Every finite exponent, both signs, with boundary and mixed significands.
    for exponent in 0..2047_u64 {
        for fraction in [0, 1, (1_u64 << 51) + 37, (1_u64 << 52) - 1] {
            for sign in [0, 1_u64 << 63] {
                check(f64::from_bits(sign | (exponent << 52) | fraction));
            }
        }
    }
}

#[test]
fn projected_profile_bounds_actual_finite_extreme_records_and_metadata_growth() {
    let bound = thermal_diagnostic_profile("step").unwrap();
    for value in [f64::MAX, -f64::MAX, -f64::from_bits(1), 0.0, 1.0] {
        let mut records = projection("step");
        records[0].message = ThermalDiagnosticMessage::Initial {
            step: "step",
            temperatures: [value; 3],
            counts: [usize::MAX; 5],
        };
        records[1].message = ThermalDiagnosticMessage::Stability([value; 4]);
        records[1].severity = FeaDiagnosticSeverity::Info;
        records[3].message = ThermalDiagnosticMessage::Energy([value; 9]);
        records[3].severity = FeaDiagnosticSeverity::Info;
        let actual: Vec<_> = records
            .iter()
            .map(|record| record.message.diagnostic(record.severity))
            .collect();
        let profile = profile_serialized(&actual, PROFILE_LIMITS).unwrap();
        assert!(profile.encoded_bytes_upper_bound <= bound.encoded_bytes_upper_bound);
        assert!(
            profile.materialization_resident_bytes_estimate
                <= bound.materialization_resident_bytes_estimate
        );
    }
    let longer = thermal_diagnostic_profile("step_with_more_authored_metadata").unwrap();
    assert!(longer.encoded_bytes_upper_bound > bound.encoded_bytes_upper_bound);
    assert!(
        longer.materialization_resident_bytes_estimate
            > bound.materialization_resident_bytes_estimate
    );
}

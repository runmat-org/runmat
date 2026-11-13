use super::*;

#[test]
fn cold_diagnostic_is_a_validated_claim_not_a_reconstructed_certificate() {
    let matrix = matrix(1.0, false);
    let SparseUniquenessVerdict::Verified {
        certificate,
        iterations,
    } = certify_sparse_uniqueness(
        &matrix,
        policy(),
        SparseUniquenessEnvelope::derive(&matrix, policy()).unwrap(),
    )
    .unwrap()
    else {
        panic!("independent invertible matrix")
    };
    let record =
        SparseUniquenessRecord::from_certificate(policy(), &certificate, iterations).unwrap();
    let value = serde_json::to_value(record).unwrap();
    let cold: SparseUniquenessRecord = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(cold, record);
    assert_eq!(
        cold.operator_identity(),
        matrix.numerical_identity().unwrap()
    );
    assert_eq!(
        cold.defect_infinity_upper(),
        Some(certificate.defect_infinity_upper())
    );
    assert_eq!(cold.dimension(), 2);
    // No API converts a transport record into RightInverseCertificate. A cold
    // claim is bound by the producing result/receipt's trusted object identity.
    for bits in [
        1.0f64.to_bits(),
        f64::INFINITY.to_bits(),
        f64::NAN.to_bits(),
        (-1.0f64).to_bits(),
    ] {
        let mut bad = value.clone();
        bad["claim"]["Verified"]["defect_infinity_upper_bits"] = bits.into();
        assert!(serde_json::from_value::<SparseUniquenessRecord>(bad).is_err());
    }
    for (key, replacement) in [("dimension", 0_u64), ("iterations", 17_u64)] {
        let mut bad = value.clone();
        bad[key] = replacement.into();
        assert!(serde_json::from_value::<SparseUniquenessRecord>(bad).is_err());
    }
    let mut bad = value;
    bad["policy"]["restart"] = 0.into();
    assert!(serde_json::from_value::<SparseUniquenessRecord>(bad).is_err());
}

#[test]
fn no_free_unknowns_is_explicit_and_never_a_missing_diagnostic() {
    let record =
        SparseUniquenessRecord::no_free_unknowns(runmat_execution::Digest::sha256([]), policy())
            .unwrap();
    assert_eq!(record.dimension(), 0);
    assert_eq!(record.iterations(), 0);
    assert_eq!(record.defect_infinity_upper(), None);
    let mut value = serde_json::to_value(record).unwrap();
    let cold: SparseUniquenessRecord = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(record, cold);
    value["dimension"] = 1.into();
    assert!(serde_json::from_value::<SparseUniquenessRecord>(value).is_err());
}

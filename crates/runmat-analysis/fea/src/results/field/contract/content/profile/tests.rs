use super::*;

fn definition(
    quantity: ObservationQuantity,
    shape: ObservationValueShape,
    representation: RecoveredFieldNumericRepresentation,
) -> RecoveredFieldDefinition {
    RecoveredFieldDefinition::new(
        "identity.profile",
        quantity,
        FieldLocation::Element,
        RecoveredFieldEntityDomain::SolidElements,
        shape,
        representation,
    )
    .unwrap()
}

#[test]
fn shape_profile_bounds_actual_identity_wire_for_advertised_storage_shapes() {
    for representation in [
        RecoveredFieldNumericRepresentation::Real,
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    ] {
        for (quantity, shape) in [
            (
                ObservationQuantity::Temperature,
                ObservationValueShape::Scalar,
            ),
            (
                ObservationQuantity::HeatFlux,
                ObservationValueShape::Vector3,
            ),
            (
                ObservationQuantity::Stress,
                ObservationValueShape::SymmetricTensor3,
            ),
        ] {
            let definition = definition(quantity, shape, representation);
            for count in [0, 1, 37] {
                let sample = RecoveredFieldSample::TimeSeconds(0.5);
                let values = (0..count * definition.storage_component_count().unwrap())
                    .map(|i| {
                        if i % 2 == 0 {
                            f64::MAX
                        } else {
                            -f64::MIN_POSITIVE
                        }
                    })
                    .collect();
                let storage = AnalysisField::host_f64(
                    definition.field_id(),
                    definition.storage_shape(count).unwrap(),
                    values,
                );
                let profile =
                    RecoveredField::identity_profile_shape(&definition, sample, count).unwrap();
                let actual =
                    profile_serialized(&(&definition, sample, &storage), FIELD_LIMITS).unwrap();
                assert_eq!(profile, actual);
                // Independent original tuple proves no domain, prefix, tuple-order or wire change.
                let original = encode_contract(
                    FIELD_PREFIX,
                    "recovered-field-identity",
                    &(&definition, sample, &storage),
                    FIELD_LIMITS,
                )
                .unwrap();
                assert!(original.len() as u64 <= profile.encoded_bytes_upper_bound);
                let identity = RecoveredField::identity_for(&definition, sample, &storage).unwrap();
                assert_eq!(identity.digest(), Digest::sha256(&original));
                let resident = runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(&original).unwrap();
                assert!(resident <= profile.materialization_resident_bytes_estimate);
            }
        }
    }
}

#[test]
fn identity_profile_rejects_invalid_metadata_overflow_and_codec_limits() {
    let valid = definition(
        ObservationQuantity::HeatFlux,
        ObservationValueShape::Vector3,
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    );
    assert!(RecoveredField::identity_profile_shape(
        &valid,
        RecoveredFieldSample::Final,
        usize::MAX
    )
    .is_err());
    assert!(
        RecoveredField::identity_profile_shape(&valid, RecoveredFieldSample::Final, 1_000_001)
            .is_err()
    );
    for sample in [
        RecoveredFieldSample::TimeSeconds(f64::NAN),
        RecoveredFieldSample::FrequencyHertz(f64::INFINITY),
        RecoveredFieldSample::LoadFactor(-0.0),
    ] {
        assert!(RecoveredField::identity_profile_shape(&valid, sample, 1).is_err());
    }
    let mut invalid = valid.clone();
    invalid.field_id.clear();
    assert!(
        RecoveredField::identity_profile_shape(&invalid, RecoveredFieldSample::Final, 1).is_err()
    );
    invalid = valid.clone();
    invalid.field_id = "a".repeat(4097);
    assert!(
        RecoveredField::identity_profile_shape(&invalid, RecoveredFieldSample::Final, 1).is_err()
    );
    invalid = valid;
    invalid.value_shape = ObservationValueShape::Scalar;
    assert!(
        RecoveredField::identity_profile_shape(&invalid, RecoveredFieldSample::Final, 1).is_err()
    );
}

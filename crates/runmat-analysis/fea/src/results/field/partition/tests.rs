use super::*;
use crate::results::field::{RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation};
use runmat_analysis_core::{FieldLocation, ObservationQuantity, ObservationValueShape};

#[test]
fn rejects_rehashed_partitions_with_incorrect_value_counts() {
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
        for representation in [
            RecoveredFieldNumericRepresentation::Real,
            RecoveredFieldNumericRepresentation::ComplexInterleaved,
        ] {
            let definition = RecoveredFieldDefinition::new(
                "partition.values",
                quantity,
                FieldLocation::Element,
                RecoveredFieldEntityDomain::VolumeElements,
                shape.clone(),
                representation,
            )
            .unwrap();
            let components = definition.storage_component_count().unwrap();
            let field = RecoveredField::new(
                definition.clone(),
                RecoveredFieldSample::Final,
                AnalysisField::host_f64(
                    definition.field_id(),
                    definition.layout(),
                    definition.storage_shape(3).unwrap(),
                    vec![1.0; 3 * components],
                ),
            )
            .unwrap();
            let valid = RecoveredFieldPartition::select(&field, vec![0, 2]).unwrap();
            assert_eq!(
                RecoveredFieldPartition::from_canonical_bytes(&valid.canonical_bytes().unwrap())
                    .unwrap(),
                valid
            );
            for count in [0, 2 * components - 1, 2 * components + 1] {
                let mut invalid = valid.clone();
                invalid.storage = AnalysisField::host_f64(
                    definition.field_id(),
                    valid.storage.layout.clone(),
                    valid.storage.shape.clone(),
                    vec![1.0; count],
                );
                // Rehash the malformed content so rejection cannot rely on a stale digest.
                invalid.identity = invalid.derive_identity().unwrap();
                assert!(
                    invalid.validate().is_err(),
                    "accepted {count} values for {components} components"
                );
                assert!(invalid.canonical_bytes().is_err());
                let bytes = encode_contract(
                    FIELD_PARTITION_PREFIX,
                    "recovered-field-partition",
                    &invalid,
                    FIELD_PARTITION_LIMITS,
                )
                .unwrap();
                assert!(RecoveredFieldPartition::from_canonical_bytes(&bytes).is_err());
                assert!(serde_json::from_value::<RecoveredFieldPartition>(
                    serde_json::to_value(&invalid).unwrap()
                )
                .is_err());
            }
        }
    }
}

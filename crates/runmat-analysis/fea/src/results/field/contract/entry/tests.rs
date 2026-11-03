use super::*;
use crate::results::field::RecoveredFieldView;
use runmat_analysis_core::{FieldLocation, ObservationQuantity};
use runmat_canonical_codec::profile_serialized_shape;

fn definitions() -> [RecoveredFieldDefinition; 3] {
    [
        (
            "temperature",
            ObservationQuantity::Temperature,
            ObservationValueShape::Scalar,
        ),
        (
            "flux",
            ObservationQuantity::HeatFlux,
            ObservationValueShape::Vector3,
        ),
        (
            "stress",
            ObservationQuantity::Stress,
            ObservationValueShape::SymmetricTensor3,
        ),
    ]
    .map(|(id, quantity, shape)| {
        RecoveredFieldDefinition::new(
            id,
            quantity,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            shape,
            RecoveredFieldNumericRepresentation::Real,
        )
        .unwrap()
    })
}

#[test]
fn projected_catalog_matches_actual_nested_entries_and_preserves_entry_wire() {
    let definitions = definitions();
    let counts = [1, RECOVERED_FIELD_PARTITION_ENTITY_COUNT + 3, 7];
    let mut rows: Vec<[RecoveredFieldArtifactEntry; 3]> = Vec::new();
    for index in 0..2 {
        let row = std::array::from_fn(|column| {
            let definition = &definitions[column];
            let storage = AnalysisField::host_f64(
                definition.field_id(),
                definition.layout(),
                definition.storage_shape(counts[column]).unwrap(),
                vec![1.5; counts[column] * definition.storage_component_count().unwrap()],
            );
            let sample = RecoveredFieldSample::TimeSeconds(index as f64);
            let view = RecoveredFieldView::new(definition, sample, &storage).unwrap();
            RecoveredFieldArtifactEntry {
                field: view.identity(),
                definition: definition.clone(),
                sample,
                partitions: view.canonical_partition_descriptors().unwrap(),
            }
        });
        rows.push(row);
    }
    let projected = RecoveredFieldArtifactEntry::profile_sampled_catalog_shape(
        &definitions,
        &counts,
        RecoveredFieldSample::TimeSeconds(0.0),
        rows.len(),
    )
    .unwrap();
    let actual = profile_serialized_shape(&rows, FIELD_LIMITS).unwrap();
    assert_eq!(projected, actual);
    let encoded = encode_contract(b"test\0", "catalog", &rows, FIELD_LIMITS).unwrap();
    assert!(encoded.len() as u64 <= projected.encoded_bytes_upper_bound);
    assert!(
        runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate(&encoded)
            .unwrap()
            <= projected.materialization_resident_bytes_estimate
    );

    #[derive(Serialize)]
    struct OriginalEntry<'a> {
        field: RecoveredFieldIdentity,
        definition: &'a RecoveredFieldDefinition,
        sample: RecoveredFieldSample,
        partitions: &'a [RecoveredFieldPartitionDescriptor],
    }
    let entry = &rows[0][1];
    let original = OriginalEntry {
        field: entry.field,
        definition: &entry.definition,
        sample: entry.sample,
        partitions: &entry.partitions,
    };
    assert_eq!(
        serde_json::to_value(entry).unwrap(),
        serde_json::to_value(&original).unwrap()
    );
    assert_eq!(
        encode_contract(b"test\0", "entry", entry, FIELD_LIMITS).unwrap(),
        encode_contract(b"test\0", "entry", &original, FIELD_LIMITS).unwrap()
    );
    let decoded: RecoveredFieldArtifactEntry =
        serde_json::from_value(serde_json::to_value(entry).unwrap()).unwrap();
    assert_eq!(&decoded, entry);
}

#[test]
fn catalog_projection_rejects_invalid_shape_arithmetic_metadata_and_limits() {
    let definitions = definitions();
    let profile = |definitions: &[RecoveredFieldDefinition], counts: &[usize], samples| {
        RecoveredFieldArtifactEntry::profile_sampled_catalog_shape(
            definitions,
            counts,
            RecoveredFieldSample::Final,
            samples,
        )
    };
    assert!(profile(&[], &[], 1).is_err());
    assert!(profile(&definitions, &[1], 1).is_err());
    assert!(profile(&definitions, &[1, 0, 1], 1).is_err());
    assert!(profile(&definitions, &[1, 1, 1], 0).is_err());
    assert!(profile(&definitions, &[1, usize::MAX, 1], 1).is_err());
    assert!(profile(&definitions, &[1, 1, 1], usize::MAX).is_err());
    // No fixed ceiling: a catalog above 1 GiB is profiled; its plan decides admission.
    assert!(
        profile(&definitions, &[1, 1, 1], 1_000_001)
            .unwrap()
            .encoded_bytes_upper_bound
            > 1 << 30
    );
    assert!(RecoveredFieldArtifactEntry::profile_sampled_catalog_shape(
        &definitions,
        &[1, 1, 1],
        RecoveredFieldSample::TimeSeconds(f64::NAN),
        1
    )
    .is_err());
    let mut invalid = definitions.clone();
    invalid[0].field_id.clear();
    assert!(profile(&invalid, &[1, 1, 1], 1).is_err());
}

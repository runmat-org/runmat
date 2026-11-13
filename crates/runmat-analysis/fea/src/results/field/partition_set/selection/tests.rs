use super::*;
use crate::results::field::{RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation};
use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
};

fn fixture() -> (RecoveredFieldArtifactEntry, Vec<RecoveredFieldPartition>) {
    let count = RECOVERED_FIELD_PARTITION_ENTITY_COUNT * 2 + 7;
    let definition = RecoveredFieldDefinition::new(
        "temperature",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let field = RecoveredField::new(
        definition.clone(),
        RecoveredFieldSample::Final,
        AnalysisField::host_f64(
            "temperature",
            definition.layout(),
            vec![count],
            vec![1.0; count],
        ),
    )
    .unwrap();
    let entry = RecoveredFieldArtifactEntry {
        field: field.identity(),
        definition,
        sample: RecoveredFieldSample::Final,
        partitions: canonical_field_partition_descriptors(&field).unwrap(),
    };
    (entry, canonical_field_partitions(&field).unwrap())
}

#[test]
fn metadata_selection_matches_nonadjacent_materialized_blocks_and_legacy_identity() {
    let (entry, partitions) = fixture();
    let selected = select_canonical_field_partitions(
        &entry,
        &[0, 3, RECOVERED_FIELD_PARTITION_ENTITY_COUNT * 2 + 6],
    )
    .unwrap();
    assert_eq!(
        selected.descriptors,
        vec![&entry.partitions[0], &entry.partitions[2]]
    );
    let set = RecoveredFieldPartitionSet::new(vec![partitions[0].clone(), partitions[2].clone()])
        .unwrap();
    assert_eq!(selected.identity, set.identity);
    let legacy_bytes = encode_contract(
        FIELD_PARTITION_SET_PREFIX,
        "recovered-field-partition-set-identity",
        &(
            set.schema_version,
            set.source_field,
            &set.definition,
            set.sample,
            set.partitions
                .iter()
                .map(|p| p.identity)
                .collect::<Vec<_>>(),
        ),
        FIELD_PARTITION_SET_LIMITS,
    )
    .unwrap();
    assert_eq!(selected.identity.digest(), Digest::sha256(legacy_bytes));
}

#[test]
fn selection_rejects_empty_unsorted_duplicate_and_out_of_range_indices() {
    let (entry, _) = fixture();
    for indices in [
        vec![],
        vec![1, 0],
        vec![1, 1],
        vec![RECOVERED_FIELD_PARTITION_ENTITY_COUNT * 2 + 7],
    ] {
        assert!(select_canonical_field_partitions(&entry, &indices).is_err());
    }
}

#[test]
fn selection_rejects_missing_reordered_overlapping_and_duplicate_storage_descriptors() {
    let (entry, _) = fixture();
    let mut missing = entry.clone();
    missing.partitions.remove(1);
    let mut reordered = entry.clone();
    reordered.partitions.swap(0, 1);
    let mut overlapping = entry.clone();
    overlapping.partitions[1].first_source_index -= 1;
    let mut duplicate = entry.clone();
    duplicate.partitions[1].partition = duplicate.partitions[0].partition;
    let mut short = entry.clone();
    short.partitions[0].entity_count -= 1;
    let mut zero = entry.clone();
    zero.partitions[2].entity_count = 0;
    let mut overflow = entry;
    overflow.partitions[2].entity_count = usize::MAX;
    for malformed in [
        missing,
        reordered,
        overlapping,
        duplicate,
        short,
        zero,
        overflow,
    ] {
        assert!(select_canonical_field_partitions(&malformed, &[0]).is_err());
    }
}

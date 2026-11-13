//! Isolated Rust-heap evidence for borrowed single-partition traversal, not full preparation.
#[path = "../support/peak_allocator.rs"]
mod peak_allocator;

use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
};
use runmat_analysis_fea::results::field::{
    RecoveredFieldDefinition, RecoveredFieldEntityDomain, RecoveredFieldNumericRepresentation,
    RecoveredFieldSample, RecoveredFieldView, RECOVERED_FIELD_PARTITION_ENTITY_COUNT,
};

fn measure(blocks: usize) -> (u64, u64) {
    let count = blocks * RECOVERED_FIELD_PARTITION_ENTITY_COUNT;
    let definition = RecoveredFieldDefinition::new(
        "temperature.residency",
        ObservationQuantity::Temperature,
        FieldLocation::Node,
        RecoveredFieldEntityDomain::Nodes,
        ObservationValueShape::Scalar,
        RecoveredFieldNumericRepresentation::Real,
    )
    .unwrap();
    let storage = AnalysisField::host_f64(
        definition.field_id(),
        definition.layout(),
        vec![count],
        vec![300.0; count],
    );
    // Identity encoding is measured separately: it still grows with the full field.
    let identity_profile =
        runmat_analysis_fea::results::field::RecoveredField::identity_profile_shape(
            &definition,
            RecoveredFieldSample::TimeSeconds(0.0),
            count,
        )
        .unwrap();
    let identity_baseline = peak_allocator::begin_measurement();
    let view = RecoveredFieldView::new(
        &definition,
        RecoveredFieldSample::TimeSeconds(0.0),
        &storage,
    )
    .unwrap();
    let identity_growth = peak_allocator::end_measurement().saturating_sub(identity_baseline);
    assert!(
        identity_growth <= identity_profile.materialization_resident_bytes_estimate,
        "identity peak={identity_growth}, estimate={}",
        identity_profile.materialization_resident_bytes_estimate
    );
    println!(
        "identity_peak_growth={identity_growth} identity_estimate={}",
        identity_profile.materialization_resident_bytes_estimate
    );
    assert!(std::ptr::eq(view.storage(), &storage));
    let baseline = peak_allocator::begin_measurement();
    let mut partitions = 0;
    for partition in view.canonical_partitions().unwrap() {
        let partition = partition.unwrap();
        assert_eq!(
            partition.source_indices.len(),
            RECOVERED_FIELD_PARTITION_ENTITY_COUNT
        );
        let bytes = partition.canonical_bytes().unwrap();
        std::hint::black_box(&bytes);
        partitions += 1;
        // Both encoded bytes and owned numeric partition drop before the next iteration.
    }
    let growth = peak_allocator::end_measurement().saturating_sub(baseline);
    assert_eq!(partitions, blocks);
    let source_bytes = (count * std::mem::size_of::<f64>()) as u64;
    println!("blocks={blocks} source_bytes={source_bytes} traversal_peak_growth={growth}");
    (growth, source_bytes)
}

#[test]
fn borrowed_partition_peak_does_not_scale_with_full_field_storage() {
    let (small, _) = measure(4);
    let (large, source_bytes) = measure(48);
    // Twelve times the source storage must not cause proportional live extraction storage.
    // A 2x regression envelope allows differing encoded source-index widths, not extra blocks.
    assert!(large <= small * 2, "small={small}, large={large}");
    assert!(
        large < source_bytes,
        "traversal retained a full numeric snapshot: {large} >= {source_bytes}"
    );
    println!(
        "process_peak_rss_bytes={:?}",
        peak_allocator::process_peak_rss_bytes()
    );
}

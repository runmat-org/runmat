//! The contribution ceiling bounds what partitions write, and the projected output construction
//! budget covers the measured encoding peak of a contribution at that ceiling.
use runmat_analysis_core::{
    ExtremaTiePolicy, ObservationId, ObservationRequest, ObservationTarget, ReductionPolicy,
    SeriesReduction, SpatialReduction,
};

use super::partition_tests::{prepared, spec};
use super::{
    evaluate_solid_observation_partition, prepare_solid_observation_field_partition,
    ObservationEntity, ObservationExecutionPolicy, ObservationPartitionContribution,
    ObservationPartitionPayload, ObservationPartitionPayloadCapacity, ObservationPartitionPlan,
};

fn capacity(
    extrema: bool,
    partition: &super::ObservationPartition,
    field: &crate::results::field::RecoveredFieldPartitionSet,
) -> ObservationPartitionPayloadCapacity {
    ObservationPartitionPayloadCapacity::Entities {
        extrema,
        entity_count: partition.target.entity_count(),
        values_per_entity: field.definition.storage_component_count().unwrap(),
    }
}

#[test]
fn contribution_ceiling_bounds_every_written_contribution() {
    let mut requests = vec![(
        false,
        ObservationRequest::Field(spec("observed", ObservationTarget::WholeModel)),
    )];
    for ties in [
        ExtremaTiePolicy::FirstCanonical,
        ExtremaTiePolicy::AllLocations,
    ] {
        let mut maximum = spec("observed", ObservationTarget::WholeModel);
        maximum.reduction = Some(ReductionPolicy {
            spatial: SpatialReduction::Maximum,
            series: SeriesReduction::None,
            extrema_ties: ties,
        });
        requests.push((true, ObservationRequest::Reduction(maximum)));
    }
    for (extrema, request) in requests {
        let (projection, plan, field) = prepared(request);
        let partition_plan = ObservationPartitionPlan::derive(
            &plan,
            &ObservationId("observed".into()),
            ObservationExecutionPolicy::canonical(std::num::NonZeroU32::new(3).unwrap()),
        )
        .unwrap();
        for partition in &partition_plan.partitions {
            let source = prepare_solid_observation_field_partition(
                &plan,
                &partition_plan,
                partition,
                &projection,
                &field,
            )
            .unwrap();
            let written = evaluate_solid_observation_partition(
                &plan,
                &partition_plan,
                partition,
                &projection,
                &source,
                None,
            )
            .unwrap()
            .canonical_bytes()
            .unwrap()
            .len() as u64;
            let ceiling = ObservationPartitionContribution::encoded_bytes_upper_bound(
                &partition_plan,
                partition,
                &source,
                None,
                capacity(extrema, partition, &source),
            )
            .unwrap();
            println!(
                "extrema={extrema} entities={} written={written} ceiling={ceiling}",
                partition.target.entity_count()
            );
            assert!(written <= ceiling, "written {written} exceeds {ceiling}");
            if !extrema {
                // Selected values fill the layout; the ceiling adds only scalar and header width.
                assert!(
                    ceiling < 2 * written,
                    "ceiling {ceiling} far above {written}"
                );
            }
        }
    }
}

#[test]
fn contribution_encoding_peak_fits_the_projected_construction_budget() {
    let (projection, plan, field) = prepared(ObservationRequest::Field(spec(
        "observed",
        ObservationTarget::WholeModel,
    )));
    let partition_plan = ObservationPartitionPlan::derive(
        &plan,
        &ObservationId("observed".into()),
        ObservationExecutionPolicy::serial(),
    )
    .unwrap();
    let partition = &partition_plan.partitions[0];
    let source = prepare_solid_observation_field_partition(
        &plan,
        &partition_plan,
        partition,
        &projection,
        &field,
    )
    .unwrap();
    // A contribution at the ceiling's item count: the encoder sees the widest entities (every
    // digest byte in its two-byte form) and full-width values, as the ceiling assumes.
    let identity = runmat_meshing_core::StableDigest::from_bytes([u8::MAX; 32]);
    let entity_count = 100_000;
    let contribution = ObservationPartitionContribution::new(
        &partition_plan,
        partition,
        &source,
        None,
        None,
        ObservationPartitionPayload::SelectedValues {
            entities: vec![ObservationEntity::Node(identity); entity_count],
            values: vec![f64::MAX; entity_count],
        },
    )
    .unwrap();
    let ceiling = ObservationPartitionContribution::encoded_bytes_upper_bound(
        &partition_plan,
        partition,
        &source,
        None,
        ObservationPartitionPayloadCapacity::Entities {
            extrema: false,
            entity_count,
            values_per_entity: 1,
        },
    )
    .unwrap();
    let construction =
        crate::planning::resource::projected_canonical_output_resource_facts(ceiling)
            .unwrap()
            .construction_resident_bytes_admission_budget
            .0;
    let (bytes, _, peak) =
        crate::topology::measure_test_allocations(|| contribution.canonical_bytes().unwrap());
    let encoded = bytes.len() as u64;
    println!(
        "contribution encoded={encoded} ceiling={ceiling} encoding_peak={peak} \
         construction_budget={construction} peak/encoded={:.2} budget/ceiling={:.2}",
        peak as f64 / encoded as f64,
        construction as f64 / ceiling as f64
    );
    assert!(encoded <= ceiling, "encoded {encoded} exceeds {ceiling}");
    assert!(
        ceiling - encoded <= 16,
        "ceiling {ceiling} exceeds the widest encoding {encoded} by more than header growth"
    );
    assert!(
        peak as u64 <= construction,
        "encoding peak {peak} exceeds construction budget {construction}"
    );
}

use super::*;

#[test]
fn static_transport_is_accounted_independently_of_numerical_storage() {
    let baseline = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let facts = PreparedInputResourceFacts {
        encoded_payload_bytes: ByteCount(1024),
        transport_closure_bytes: ByteCount(2048),
        decoded_resident_bytes_admission_budget: ByteCount(8192),
    };
    let mut planned = baseline.input.clone();
    let FeaResourceDemand::StructuralLinearPipeline(demand) = &mut planned.demand else {
        panic!("static demand")
    };
    demand.inputs = Some(facts.clone());
    let transported = build_resource_plan(planned.clone()).unwrap();
    assert_eq!(
        transported.breakdown.input_encoded_payload_bytes,
        facts.encoded_payload_bytes
    );
    assert_eq!(
        transported.breakdown.input_transport_closure_bytes,
        facts.transport_closure_bytes
    );
    assert_eq!(
        transported.breakdown.input_decoded_resident_bytes,
        facts.decoded_resident_bytes_admission_budget
    );
    assert_eq!(
        transported.request.memory_bytes,
        baseline.request.memory_bytes + 8192 + 2048
    );
    assert_eq!(
        transported.request.max_relay_bytes,
        baseline.request.max_relay_bytes + 2048
    );
    assert_eq!(
        transported.input.structural_operator(),
        baseline.input.structural_operator()
    );
    assert_ne!(transported.identity, baseline.identity);
    for changed in [
        PreparedInputResourceFacts {
            transport_closure_bytes: ByteCount(0),
            ..facts.clone()
        },
        PreparedInputResourceFacts {
            decoded_resident_bytes_admission_budget: ByteCount(1023),
            ..facts.clone()
        },
    ] {
        let mut invalid = planned.clone();
        let FeaResourceDemand::StructuralLinearPipeline(demand) = &mut invalid.demand else {
            unreachable!()
        };
        demand.inputs = Some(changed);
        assert!(build_resource_plan(invalid).is_err());
    }
}

#[test]
fn static_transport_deduplication_does_not_shrink_decoded_reservations() {
    let baseline = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let mut planned = baseline.input.clone();
    let FeaResourceDemand::StructuralLinearPipeline(demand) = &mut planned.demand else {
        panic!("static demand")
    };
    // Aggregate logical bytes may exceed the unique closure when content-addressed
    // chunks recur. Their decoded values still coexist and retain their full budget.
    demand.inputs = Some(PreparedInputResourceFacts {
        encoded_payload_bytes: ByteCount(8192),
        transport_closure_bytes: ByteCount(2048),
        decoded_resident_bytes_admission_budget: ByteCount(32768),
    });
    let deduplicated = build_resource_plan(planned).unwrap();
    assert_eq!(
        deduplicated.breakdown.input_encoded_payload_bytes,
        ByteCount(8192)
    );
    assert_eq!(
        deduplicated.breakdown.input_transport_closure_bytes,
        ByteCount(2048)
    );
    assert_eq!(
        deduplicated.request.memory_bytes,
        baseline.request.memory_bytes + 32768 + 2048
    );
    assert_eq!(
        deduplicated.request.max_relay_bytes,
        baseline.request.max_relay_bytes + 2048
    );
}

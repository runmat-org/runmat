use std::collections::BTreeSet;

use super::*;

fn request() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_500,
        memory_bytes: 2_048,
        scratch_bytes: 1_024,
        max_wall_millis: 1_000,
        max_artifact_bytes: 4_096,
        max_egress_bytes: 512,
        max_relay_bytes: 256,
        accelerators: Vec::new(),
        required_capabilities: BTreeSet::new(),
    }
}

#[test]
fn reserved_interval_accounting_is_exact_and_self_validating() {
    let accounting = ResourceAccounting::for_reserved_interval(request(), 20, 300, 40, 10).unwrap();
    assert_eq!(accounting.usage.cpu_core_millis, 30);
    assert_eq!(accounting.usage.memory_byte_millis, 40_960);
    assert_eq!(accounting.usage.retained_byte_millis, 6_000);
    accounting.validate().unwrap();

    let mut substituted = accounting;
    substituted.usage.cpu_core_millis += 1;
    assert!(substituted.validate().is_err());
}

#[test]
fn reserved_interval_accounting_rejects_invalid_or_unleased_allocations() {
    let mut invalid = request();
    invalid.cpu_millicores = 0;
    assert!(ResourceAccounting::for_reserved_interval(invalid, 1, 0, 0, 0).is_err());
    let mut unleased = request();
    unleased.accelerators = vec![crate::resource::AcceleratorRequest::generic_compute(1).unwrap()];
    assert!(ResourceAccounting::for_reserved_interval(unleased, 1, 0, 0, 0).is_err());
}

#[test]
fn accounting_summary_aggregates_heterogeneous_intervals_canonically() {
    let first = ResourceAccounting::for_reserved_interval(request(), 20, 300, 40, 10).unwrap();
    let mut smaller = request();
    smaller.cpu_millicores = 500;
    smaller.memory_bytes = 1_024;
    smaller.scratch_bytes = 512;
    let second = ResourceAccounting::for_reserved_interval(smaller, 10, 100, 20, 5).unwrap();
    let summary = ResourceAccountingSummary::aggregate(&[first, second], 25).unwrap();
    assert_eq!(summary.usage.cpu_core_millis, 35);
    assert_eq!(summary.usage.memory_byte_millis, 51_200);
    assert_eq!(summary.usage.wall_millis, 25);
    assert_eq!(summary.usage.egress_bytes, 60);
    assert_eq!(summary.usage.relay_bytes, 15);
    assert_eq!(summary.peak_cpu_millicores, 1_500);
    assert_eq!(summary.peak_memory_bytes, 2_048);
    assert_eq!(summary.peak_scratch_bytes, 1_024);
    assert_eq!(summary.artifact_bytes, 400);
}

#[test]
fn accounting_summary_rejects_empty_or_impossible_graph_intervals() {
    assert!(ResourceAccountingSummary::aggregate(&[], 0).is_err());
    let interval = ResourceAccounting::for_reserved_interval(request(), 20, 0, 0, 0).unwrap();
    assert!(ResourceAccountingSummary::aggregate(&[interval], 19).is_err());
}

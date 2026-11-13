use super::*;
use crate::resource::{AcceleratorClass, AcceleratorRequest, Capability};
fn fixture() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 2000,
        memory_bytes: 4096,
        scratch_bytes: 1024,
        max_wall_millis: 5000,
        max_artifact_bytes: 2048,
        max_egress_bytes: 8192,
        max_relay_bytes: 512,
        accelerators: vec![AcceleratorRequest {
            class: AcceleratorClass::new("gpu").unwrap(),
            count: 2,
            minimum_allocation_bytes: 16384,
            provider: None,
            required_features: Default::default(),
        }],
        required_capabilities: std::collections::BTreeSet::from([Capability::ProcessIsolation]),
    }
}
#[test]
fn each_independent_reservation_dimension_is_enforced_at_its_exact_boundary() {
    let required = fixture();
    required.require_within(&required).unwrap();
    for dimension in 0..7 {
        let mut available = required.clone();
        match dimension {
            0 => available.cpu_millicores -= 1,
            1 => available.memory_bytes -= 1,
            2 => available.scratch_bytes -= 1,
            3 => available.max_wall_millis -= 1,
            4 => available.max_artifact_bytes -= 1,
            5 => available.max_egress_bytes -= 1,
            6 => available.max_relay_bytes -= 1,
            _ => unreachable!(),
        }
        assert!(
            required.require_within(&available).is_err(),
            "dimension {dimension}"
        );
    }
}
#[test]
fn extra_capacity_cannot_substitute_an_accelerator_class_or_required_capability() {
    let required = fixture();
    for substitution in 0..4 {
        let mut available = required.clone();
        available.memory_bytes *= 2;
        match substitution {
            0 => available.accelerators[0].class = AcceleratorClass::new("other").unwrap(),
            1 => available.accelerators[0].count -= 1,
            2 => available.accelerators[0].minimum_allocation_bytes -= 1,
            3 => available.required_capabilities.clear(),
            _ => unreachable!(),
        }
        assert!(required.require_within(&available).is_err());
    }
}
#[test]
fn duplicate_accelerator_classes_cannot_pass_separate_capacity_checks_then_overreserve() {
    let mut request = fixture();
    request.accelerators.push(request.accelerators[0].clone());
    request
        .validate()
        .expect("overlapping requests are supported");
    assert!(request.require_within(&fixture()).is_err());
    fixture()
        .require_within(&request)
        .expect("four slots cover two");
    request
        .require_within(&request)
        .expect("four distinct reserved slots");
    let mut specialized = fixture();
    specialized.accelerators[0].provider = Some(crate::resource::AcceleratorProvider {
        id: crate::resource::AcceleratorProviderId::new("wgpu").unwrap(),
        version: crate::resource::AcceleratorProviderVersion::new("fixture-v1").unwrap(),
        abi_fingerprint: crate::Digest::sha256("reservation provider fixture"),
    });
    specialized.accelerators[0]
        .required_features
        .insert(crate::resource::AcceleratorFeature::UnifiedMemory);
    assert!(specialized.require_within(&fixture()).is_err());
    specialized.require_within(&specialized).unwrap();
}

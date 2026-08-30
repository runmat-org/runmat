use std::collections::BTreeSet;

use runmat_execution::identity::AttemptId;
use runmat_execution::resource::{
    AcceleratorAllocationDomainId, AcceleratorClass, AcceleratorDevice, AcceleratorDeviceId,
    AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId, AcceleratorProviderVersion,
    AcceleratorRequest, ResourceInventory, ResourceRequest,
};
use runmat_execution::Digest;
use runmat_execution_runner::scheduler::{
    assignment_for, fits, release, reserve, select_devices, ResourceAllocation,
};

fn provider(version: &str, abi: &[u8]) -> AcceleratorProvider {
    AcceleratorProvider {
        id: AcceleratorProviderId::new("wgpu").unwrap(),
        version: AcceleratorProviderVersion::new(version).unwrap(),
        abi_fingerprint: Digest::sha256(abi),
    }
}

fn device(id: &str, max_allocation_bytes: u64, provider: AcceleratorProvider) -> AcceleratorDevice {
    AcceleratorDevice {
        id: AcceleratorDeviceId::new(id).unwrap(),
        allocation_domain: runmat_execution::resource::AcceleratorAllocationDomainId::new(format!(
            "allocation-{id}"
        ))
        .unwrap(),
        class: AcceleratorClass::new("gpu").unwrap(),
        provider,
        max_allocation_bytes,
        features: BTreeSet::from([AcceleratorFeature::Compute]),
        inventory_epoch: 3,
    }
}

fn device_view(
    id: &str,
    allocation_domain: &str,
    max_allocation_bytes: u64,
    provider: AcceleratorProvider,
) -> AcceleratorDevice {
    let mut device = device(id, max_allocation_bytes, provider);
    device.allocation_domain = AcceleratorAllocationDomainId::new(allocation_domain).unwrap();
    device
}

fn request(provider: Option<AcceleratorProvider>) -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1_000,
        scratch_bytes: 1_000,
        max_wall_millis: 1_000,
        max_artifact_bytes: 0,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: vec![AcceleratorRequest {
            class: AcceleratorClass::new("gpu").unwrap(),
            count: 2,
            minimum_allocation_bytes: 8_000,
            provider,
            required_features: BTreeSet::from([AcceleratorFeature::Compute]),
        }],
        required_capabilities: BTreeSet::new(),
    }
}

#[test]
fn exact_devices_are_selected_leased_and_released() {
    let provider = provider("1.2.0", b"abi-a");
    let inventory = ResourceInventory {
        cpu_millicores: 4_000,
        memory_bytes: 8_000,
        scratch_bytes: 8_000,
        accelerators: vec![
            device("gpu-0", 8_000, provider.clone()),
            device("gpu-1", 16_000, provider.clone()),
        ],
        capabilities: BTreeSet::new(),
    };
    let request = request(Some(provider));
    let mut allocated = ResourceAllocation::default();
    let selected = select_devices(&inventory, &allocated, &request).unwrap();
    let assignment = assignment_for(AttemptId::derive(&[b"attempt"]), 7, &selected).unwrap();

    reserve(&mut allocated, &request, &assignment).unwrap();
    assert!(!fits(&inventory, &allocated, &request));
    assert_eq!(allocated.leased_accelerator_domains.len(), 2);

    release(&mut allocated, &request, &assignment);
    assert!(fits(&inventory, &allocated, &request));
    assert!(allocated.leased_accelerator_domains.is_empty());
}

#[test]
fn heterogeneous_provider_abi_is_rejected_before_assignment() {
    let requested_provider = provider("1.2.0", b"abi-a");
    let inventory = ResourceInventory {
        cpu_millicores: 4_000,
        memory_bytes: 8_000,
        scratch_bytes: 8_000,
        accelerators: vec![
            device("gpu-0", 16_000, requested_provider.clone()),
            device("gpu-1", 16_000, provider("1.2.0", b"abi-b")),
        ],
        capabilities: BTreeSet::new(),
    };
    let request = request(Some(requested_provider));
    assert!(!fits(&inventory, &ResourceAllocation::default(), &request));
}

#[test]
fn lease_identity_binds_every_admission_fact() {
    let provider = provider("1.2.0", b"abi-a");
    let selected = vec![
        device("gpu-0", 8_000, provider.clone()),
        device("gpu-1", 16_000, provider.clone()),
    ];
    let attempt = AttemptId::derive(&[b"attempt"]);
    let mut assignment = assignment_for(attempt, 7, &selected).unwrap();
    assignment.accelerator_leases[0]
        .features
        .insert(AcceleratorFeature::UnifiedMemory);
    assert!(assignment.accelerator_leases[0]
        .validate_for_attempt(attempt)
        .is_err());
}

#[test]
fn assignment_rejects_a_lease_that_does_not_satisfy_the_request() {
    let provider = provider("1.2.0", b"abi-a");
    let selected = vec![
        device("gpu-0", 8_000, provider.clone()),
        device("gpu-1", 16_000, provider.clone()),
    ];
    let attempt = AttemptId::derive(&[b"attempt"]);
    let mut assignment = assignment_for(attempt, 7, &selected).unwrap();
    assignment.accelerator_leases[0].features.clear();
    assert!(reserve(
        &mut ResourceAllocation::default(),
        &request(Some(provider)),
        &assignment,
    )
    .is_err());
}

#[test]
fn conflicting_device_reservation_is_transactional() {
    let provider = provider("1.2.0", b"abi-a");
    let selected = vec![
        device("gpu-0", 8_000, provider.clone()),
        device("gpu-1", 16_000, provider.clone()),
    ];
    let assignment = assignment_for(AttemptId::derive(&[b"attempt"]), 7, &selected).unwrap();
    let class = AcceleratorClass::new("gpu").unwrap();
    let mut allocated = ResourceAllocation {
        cpu_millicores: 250,
        memory_bytes: 500,
        scratch_bytes: 750,
        accelerator_counts: [(class, 1)].into_iter().collect(),
        leased_accelerator_domains: [
            runmat_execution::resource::AcceleratorAllocationDomainId::new("allocation-gpu-1")
                .unwrap(),
        ]
        .into_iter()
        .collect(),
    };
    let before = allocated.clone();

    assert!(reserve(&mut allocated, &request(Some(provider)), &assignment).is_err());
    assert_eq!(allocated, before);
}

#[test]
fn assignment_matching_handles_overlapping_requests_without_reusing_a_device() {
    let provider = provider("1.2.0", b"abi-a");
    let mut specialized = device("gpu-specialized", 16_000, provider.clone());
    specialized
        .features
        .insert(AcceleratorFeature::UnifiedMemory);
    let ordinary = device("gpu-ordinary", 16_000, provider.clone());
    let request = ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 1_000,
        scratch_bytes: 1_000,
        max_wall_millis: 1_000,
        max_artifact_bytes: 0,
        max_egress_bytes: 0,
        max_relay_bytes: 0,
        accelerators: vec![
            AcceleratorRequest {
                class: AcceleratorClass::new("gpu").unwrap(),
                count: 1,
                minimum_allocation_bytes: 8_000,
                provider: Some(provider.clone()),
                required_features: BTreeSet::new(),
            },
            AcceleratorRequest {
                class: AcceleratorClass::new("gpu").unwrap(),
                count: 1,
                minimum_allocation_bytes: 8_000,
                provider: Some(provider),
                required_features: BTreeSet::from([AcceleratorFeature::UnifiedMemory]),
            },
        ],
        required_capabilities: BTreeSet::new(),
    };
    let assignment = assignment_for(
        AttemptId::derive(&[b"overlapping"]),
        8,
        &[ordinary, specialized],
    )
    .unwrap();
    assert!(assignment.validate_for_request(&request).is_ok());
}

#[test]
fn provider_views_of_one_physical_device_cannot_fill_two_slots() {
    let cuda = provider("cuda-1", b"cuda-abi");
    let wgpu = provider("wgpu-1", b"wgpu-abi");
    let inventory = ResourceInventory {
        cpu_millicores: 4_000,
        memory_bytes: 8_000,
        scratch_bytes: 8_000,
        accelerators: vec![
            device_view("cuda-primary", "physical-primary", 16_000, cuda),
            device_view("wgpu-primary", "physical-primary", 16_000, wgpu),
        ],
        capabilities: BTreeSet::new(),
    };

    assert!(!fits(
        &inventory,
        &ResourceAllocation::default(),
        &request(None),
    ));
}

#[test]
fn provider_specific_requests_select_the_matching_view_of_a_shared_device() {
    let cuda = provider("cuda-1", b"cuda-abi");
    let wgpu = provider("wgpu-1", b"wgpu-abi");
    let inventory = ResourceInventory {
        cpu_millicores: 4_000,
        memory_bytes: 8_000,
        scratch_bytes: 8_000,
        accelerators: vec![
            device_view("cuda-primary", "physical-primary", 16_000, cuda.clone()),
            device_view("wgpu-primary", "physical-primary", 16_000, wgpu.clone()),
        ],
        capabilities: BTreeSet::new(),
    };
    let one_device_request = |provider| {
        let mut request = request(Some(provider));
        request.accelerators[0].count = 1;
        request
    };

    let selected_cuda = select_devices(
        &inventory,
        &ResourceAllocation::default(),
        &one_device_request(cuda),
    )
    .unwrap();
    assert_eq!(selected_cuda[0].id.as_str(), "cuda-primary");

    let selected_wgpu = select_devices(
        &inventory,
        &ResourceAllocation::default(),
        &one_device_request(wgpu),
    )
    .unwrap();
    assert_eq!(selected_wgpu[0].id.as_str(), "wgpu-primary");
}

#[test]
fn assignment_rejects_two_provider_views_of_one_physical_device() {
    let cuda = provider("cuda-1", b"cuda-abi");
    let wgpu = provider("wgpu-1", b"wgpu-abi");
    let views = vec![
        device_view("cuda-primary", "physical-primary", 16_000, cuda),
        device_view("wgpu-primary", "physical-primary", 16_000, wgpu),
    ];

    assert!(assignment_for(AttemptId::derive(&[b"aliased"]), 9, &views).is_err());
}

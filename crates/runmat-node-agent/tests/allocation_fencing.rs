use std::collections::BTreeSet;

use runmat_execution::resource::{
    AcceleratorAllocationDomainId, AcceleratorClass, AcceleratorDevice, AcceleratorDeviceId,
    AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId, AcceleratorProviderVersion,
    AcceleratorRequest,
};
use runmat_execution::Digest;
use runmat_execution_transport_native::control::{
    AllocationRole, NodeAllocation, NodeInventory, ResourceRequest,
};
use runmat_node_agent::allocation::{prepare, validate_allocation_set, validate_offer};

#[test]
fn stale_and_inventory_incompatible_offers_fail_before_sandbox_creation() {
    let directory = tempfile::tempdir().unwrap();
    let inventory = inventory();
    let mut allocation = allocation();
    allocation.expires_at_millis = 99;
    assert!(validate_offer(&allocation, &inventory, 100).is_err());
    assert!(!directory.path().join("allocations").exists());

    allocation.expires_at_millis = 200;
    allocation.resources.memory_bytes = inventory.memory_bytes + 1;
    assert!(validate_offer(&allocation, &inventory, 100).is_err());
    assert!(!directory.path().join("allocations").exists());

    allocation.resources.memory_bytes = inventory.memory_bytes;
    validate_offer(&allocation, &inventory, 100).unwrap();
    let sandbox = prepare(directory.path(), &allocation, &inventory).unwrap();
    assert!(sandbox.root.is_dir());
}

#[test]
fn overlapping_accelerator_requirements_cannot_reuse_one_matching_device() {
    let provider = AcceleratorProvider {
        id: AcceleratorProviderId::new("wgpu").unwrap(),
        version: AcceleratorProviderVersion::new("1").unwrap(),
        abi_fingerprint: Digest::sha256(b"abi"),
    };
    let specialized = AcceleratorDevice {
        id: AcceleratorDeviceId::new("gpu-specialized").unwrap(),
        allocation_domain: runmat_execution::resource::AcceleratorAllocationDomainId::new(
            "gpu-specialized-domain",
        )
        .unwrap(),
        class: AcceleratorClass::new("gpu").unwrap(),
        provider: provider.clone(),
        max_allocation_bytes: 16_000,
        features: BTreeSet::from([
            AcceleratorFeature::Compute,
            AcceleratorFeature::UnifiedMemory,
        ]),
        inventory_epoch: 1,
    };
    let incompatible = AcceleratorDevice {
        id: AcceleratorDeviceId::new("other-device").unwrap(),
        allocation_domain: runmat_execution::resource::AcceleratorAllocationDomainId::new(
            "other-device-domain",
        )
        .unwrap(),
        class: AcceleratorClass::new("other").unwrap(),
        provider: provider.clone(),
        max_allocation_bytes: 16_000,
        features: BTreeSet::from([AcceleratorFeature::Compute]),
        inventory_epoch: 1,
    };
    let mut inventory = inventory();
    inventory.accelerators = vec![specialized.clone(), incompatible.clone()];
    let mut allocation = allocation();
    allocation.resources.accelerators = vec![
        AcceleratorRequest {
            class: AcceleratorClass::new("gpu").unwrap(),
            count: 1,
            minimum_allocation_bytes: 1,
            provider: Some(provider.clone()),
            required_features: BTreeSet::new(),
        },
        AcceleratorRequest {
            class: AcceleratorClass::new("gpu").unwrap(),
            count: 1,
            minimum_allocation_bytes: 1,
            provider: Some(provider),
            required_features: BTreeSet::from([AcceleratorFeature::UnifiedMemory]),
        },
    ];
    allocation.accelerator_devices = vec![specialized, incompatible];

    assert!(validate_offer(&allocation, &inventory, 100).is_err());
}

#[test]
fn concurrent_allocations_cannot_use_two_views_of_one_physical_device() {
    let allocation_domain = AcceleratorAllocationDomainId::new("physical-gpu").unwrap();
    let view = |id: &str, provider: &str| AcceleratorDevice {
        id: AcceleratorDeviceId::new(id).unwrap(),
        allocation_domain: allocation_domain.clone(),
        class: AcceleratorClass::new("gpu").unwrap(),
        provider: AcceleratorProvider {
            id: AcceleratorProviderId::new(provider).unwrap(),
            version: AcceleratorProviderVersion::new("1").unwrap(),
            abi_fingerprint: Digest::sha256(provider.as_bytes()),
        },
        max_allocation_bytes: 16_000,
        features: BTreeSet::from([AcceleratorFeature::Compute]),
        inventory_epoch: 1,
    };
    let views = [view("cuda-view", "cuda"), view("wgpu-view", "wgpu")];
    let mut inventory = inventory();
    inventory.cpu_millicores = 2_000;
    inventory.memory_bytes = 2048;
    inventory.accelerators = views.to_vec();
    let allocations = views
        .into_iter()
        .enumerate()
        .map(|(index, device)| {
            let mut allocation = allocation();
            allocation.id = format!("lease-{index}");
            allocation.resources.cpu_millicores = 1_000;
            allocation.resources.memory_bytes = 1024;
            allocation.resources.accelerators = vec![AcceleratorRequest {
                class: device.class.clone(),
                count: 1,
                minimum_allocation_bytes: 1,
                provider: Some(device.provider.clone()),
                required_features: BTreeSet::from([AcceleratorFeature::Compute]),
            }];
            allocation.accelerator_devices = vec![device];
            allocation
        })
        .collect::<Vec<_>>();

    assert!(validate_allocation_set(&allocations, &inventory, 100).is_err());
}

fn inventory() -> NodeInventory {
    NodeInventory {
        cpu_millicores: 1_000,
        memory_bytes: 1024,
        scratch_bytes: 2048,
        accelerators: Vec::new(),
        host: runmat_node_agent::inventory::host(
            runmat_execution::security::ExecutionTrustTier::CustomerTrusted,
        )
        .unwrap(),
        capabilities: [("runmat.version".into(), env!("CARGO_PKG_VERSION").into())]
            .into_iter()
            .collect(),
    }
}

fn allocation() -> NodeAllocation {
    NodeAllocation {
        id: "lease-1".into(),
        run_id: "run-1".into(),
        project_id: "project-1".into(),
        queue: "default".into(),
        resources: ResourceRequest {
            cpu_millicores: 1_000,
            memory_bytes: 1024,
            scratch_bytes: 1024,
            accelerators: Vec::new(),
            maximum_wall_millis: 1_000,
        },
        accelerator_devices: Vec::new(),
        role: AllocationRole::Driver,
        state: "offered".into(),
        fencing_token: 1,
        expires_at_millis: 200,
    }
}

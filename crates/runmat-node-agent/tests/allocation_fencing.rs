use runmat_execution_transport_native::control::{
    AllocationRole, NodeAllocation, NodeInventory, ResourceRequest,
};
use runmat_node_agent::allocation::{prepare, validate_offer};

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

fn inventory() -> NodeInventory {
    NodeInventory {
        cpu_millicores: 1_000,
        memory_bytes: 1024,
        scratch_bytes: 2048,
        accelerator_count: 0,
        accelerator_class: None,
        accelerator_memory_bytes: 0,
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
            accelerator_count: 0,
            accelerator_class: None,
            accelerator_memory_bytes: 0,
            maximum_wall_millis: 1_000,
        },
        role: AllocationRole::Driver,
        state: "offered".into(),
        fencing_token: 1,
        expires_at_millis: 200,
    }
}

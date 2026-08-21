use std::collections::{BTreeMap, BTreeSet};

use runmat_execution::resource::{AcceleratorRequest, ResourceInventory, ResourceRequest};
use runmat_execution_runner::scheduler::{fits, ResourceAllocation};

#[test]
fn accelerator_count_and_memory_requirements_are_aggregated_conservatively() {
    let inventory = ResourceInventory {
        cpu_millicores: 4_000,
        memory_bytes: 8_000,
        scratch_bytes: 8_000,
        accelerators: vec![AcceleratorRequest {
            class: "gpu".into(),
            count: 2,
            memory_bytes_each: 16_000,
        }],
        capabilities: BTreeSet::new(),
    };
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
                class: "gpu".into(),
                count: 1,
                memory_bytes_each: 8_000,
            },
            AcceleratorRequest {
                class: "gpu".into(),
                count: 1,
                memory_bytes_each: 16_000,
            },
        ],
        required_capabilities: BTreeSet::new(),
    };
    assert!(fits(&inventory, &ResourceAllocation::default(), &request));
    assert!(!fits(
        &inventory,
        &ResourceAllocation {
            accelerator_counts: BTreeMap::from([("gpu".into(), 1)]),
            ..ResourceAllocation::default()
        },
        &request
    ));
}

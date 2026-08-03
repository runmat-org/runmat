#![cfg(unix)]

use std::os::unix::fs::PermissionsExt as _;

use runmat_execution_transport_native::control::{NodeAllocation, ResourceRequest};
use runmat_node_agent::allocation::{AllocationProcesses, Sandbox};

#[tokio::test]
async fn local_wall_limit_kills_the_owned_process_tree() {
    let directory = tempfile::tempdir().unwrap();
    let executable = directory.path().join("helper.sh");
    std::fs::write(&executable, "#!/bin/sh\nsleep 60\n").unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let sandbox = Sandbox {
        root: directory.path().to_path_buf(),
        stdout: directory.path().join("stdout.log"),
        stderr: directory.path().join("stderr.log"),
    };
    let allocation = NodeAllocation {
        id: "lease-limit".into(),
        run_id: "run".into(),
        project_id: "project".into(),
        queue: "default".into(),
        resources: ResourceRequest {
            cpu_millicores: 1,
            memory_bytes: 8 * 1024 * 1024 * 1024,
            scratch_bytes: 1024,
            accelerator_count: 0,
            accelerator_class: None,
            accelerator_memory_bytes: 0,
            maximum_wall_millis: 10,
        },
        state: "offered".into(),
        fencing_token: 1,
        expires_at_millis: 4_000_000_000_000,
    };
    let mut processes = AllocationProcesses::default();
    processes
        .launch_driver(&executable, &allocation, &sandbox)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let exceeded = processes
        .enforce_local_limits(chrono::Utc::now().timestamp_millis())
        .await
        .unwrap();
    assert_eq!(exceeded, vec!["lease-limit"]);
    assert_eq!(processes.active_count(), 0);
}

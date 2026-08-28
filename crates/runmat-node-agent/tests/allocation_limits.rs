#![cfg(unix)]

use std::os::unix::fs::PermissionsExt as _;

use runmat_execution_transport_native::control::{
    AllocationRole, DriverBootstrapCredential, NodeAllocation, ResourceRequest,
    WorkerBootstrapCredential,
};
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
        role: AllocationRole::Driver,
        state: "offered".into(),
        fencing_token: 1,
        expires_at_millis: 4_000_000_000_000,
    };
    let mut processes = AllocationProcesses::default();
    let bootstrap = DriverBootstrapCredential {
        run_id: allocation.run_id.clone(),
        org_id: "org".into(),
        project_id: allocation.project_id.clone(),
        allocation_lease_id: allocation.id.clone(),
        driver_lease_id: "driver-lease".into(),
        fencing_token: 1,
        credential: "credential".into(),
        expires_at_millis: allocation.expires_at_millis,
    };
    processes
        .launch_driver(
            &executable,
            &allocation,
            &sandbox,
            "https://server.test",
            &bootstrap,
        )
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

#[tokio::test]
async fn worker_launch_uses_only_scoped_bootstrap_material() {
    let directory = tempfile::tempdir().unwrap();
    let executable = directory.path().join("worker-env.sh");
    std::fs::write(
        &executable,
        "#!/bin/sh\nprintf '%s|%s|%s' \"$RUNMAT_EXECUTION_WORKER_RELAY_TICKET\" \"$RUNMAT_EXECUTION_DRIVER_FENCING_TOKEN\" \"${RUNMAT_NODE_CREDENTIAL-unset}\"\n",
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let sandbox = Sandbox {
        root: directory.path().join("allocation"),
        stdout: directory.path().join("worker.stdout"),
        stderr: directory.path().join("worker.stderr"),
    };
    std::fs::create_dir_all(&sandbox.root).unwrap();
    let allocation = NodeAllocation {
        id: "lease-worker".into(),
        run_id: "run-worker".into(),
        project_id: "project".into(),
        queue: "default".into(),
        resources: ResourceRequest {
            cpu_millicores: 1,
            memory_bytes: 64 * 1024 * 1024,
            scratch_bytes: 1024,
            accelerator_count: 0,
            accelerator_class: None,
            accelerator_memory_bytes: 0,
            maximum_wall_millis: 60_000,
        },
        role: AllocationRole::Worker,
        state: "active".into(),
        fencing_token: 3,
        expires_at_millis: 4_000_000_000_000,
    };
    let bootstrap = WorkerBootstrapCredential {
        run_id: allocation.run_id.clone(),
        org_id: "org".into(),
        project_id: allocation.project_id.clone(),
        allocation_lease_id: allocation.id.clone(),
        allocation_fencing_token: 3,
        driver_fencing_token: 9,
        endpoint_fingerprint: "f".repeat(64),
        run_key_envelope: vec![5; 32],
        expires_at_millis: allocation.expires_at_millis,
        relay_path: "/v1/execution/workers/lease-worker/relay".into(),
        relay_protocol: "runmat-worker-relay-v1".into(),
        relay_ticket: "scoped-ticket".into(),
    };
    let mut processes = AllocationProcesses::default();
    processes
        .launch_worker(
            &executable,
            &allocation,
            &sandbox,
            "https://server.test",
            &bootstrap,
        )
        .await
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if !processes.reap_finished().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("worker helper did not finish");
    assert_eq!(
        std::fs::read_to_string(&sandbox.stdout).unwrap(),
        "scoped-ticket|9|unset"
    );
}

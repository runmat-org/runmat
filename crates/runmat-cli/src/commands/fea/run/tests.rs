use runmat_analysis_core::{
    AcceleratorPreference, AuthoredExecutionPolicy, BackendPreference, CooperativePolicy,
    ExecutionResourceLimits, FanoutPolicy, PlacementPreference, ResiliencePolicy,
};
use runmat_config::project::{ProjectAcceleratorProfile, ProjectExecutionProfile};

use super::*;

fn remote_target() -> ExecutionTarget {
    ExecutionTarget::Remote(RemoteExecutionTarget {
        cluster: "gpu-cluster".into(),
        project: None,
        queue: "gpu".into(),
        trust_identity: "sha256:worker".into(),
        idempotency_key: None,
        workers: 2,
        detach: false,
    })
}

fn policy(accelerators: AcceleratorPreference) -> AuthoredExecutionPolicy {
    AuthoredExecutionPolicy {
        module_path: "project/main.fea".into(),
        policy_id: "scale".into(),
        placement: PlacementPreference::Cluster {
            profile: Some("gpu".into()),
        },
        backend: BackendPreference::Accelerator,
        resources: ExecutionResourceLimits {
            workers: None,
            maximum_memory_bytes_per_worker: None,
            accelerators,
        },
        fanout: FanoutPolicy::default(),
        cooperative: CooperativePolicy::default(),
        resilience: ResiliencePolicy::default(),
    }
}

#[test]
fn remote_accelerator_planning_requires_and_retains_exact_profile_allocation() {
    let profile = ProjectExecutionProfile {
        cluster: "gpu-cluster".into(),
        queue: Some("gpu".into()),
        trust_identity: Some("sha256:worker".into()),
        accelerator: Some(ProjectAcceleratorProfile {
            class: "metal".into(),
            memory_bytes: 16 << 30,
        }),
    };
    let allocation = resolve_compute_allocation(
        Some(&policy(AcceleratorPreference::Required)),
        &remote_target(),
        Some(&profile),
    )
    .unwrap();
    assert_eq!(allocation.backend, runmat_analysis_fea::ComputeBackend::Gpu);
    assert_eq!(allocation.accelerators.len(), 1);
    assert_eq!(allocation.accelerators[0].class, "metal");
    assert_eq!(allocation.accelerators[0].memory_bytes_each, 16 << 30);

    assert!(resolve_compute_allocation(
        Some(&policy(AcceleratorPreference::Required)),
        &remote_target(),
        None,
    )
    .is_err());
}

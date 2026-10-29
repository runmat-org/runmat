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
            cpu_threads_per_worker: Some(1),
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
        &thermal_steps(),
        &remote_target(),
        Some(&profile),
    )
    .unwrap();
    assert_eq!(allocation.backend, runmat_analysis_fea::ComputeBackend::Gpu);
    assert_eq!(allocation.accelerators.len(), 1);
    assert_eq!(
        allocation.accelerators[0].class,
        runmat_execution::resource::AcceleratorClass::new("metal").unwrap()
    );
    assert_eq!(
        allocation.accelerators[0].minimum_allocation_bytes,
        16 << 30
    );

    let error = resolve_compute_allocation(
        Some(&policy(AcceleratorPreference::Required)),
        &thermal_steps(),
        &remote_target(),
        None,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("FEA_ACCELERATOR_REQUIRED"));
    // Without the strict option a missing device falls back to the CPU reference.
    let fallback = resolve_compute_allocation(
        Some(&policy(AcceleratorPreference::Preferred)),
        &thermal_steps(),
        &remote_target(),
        None,
    )
    .unwrap();
    assert_eq!(fallback.backend, runmat_analysis_fea::ComputeBackend::Cpu);
    assert!(fallback.accelerators.is_empty());
}

fn thermal_steps() -> Vec<runmat_analysis_core::ProcedureStep> {
    vec![runmat_analysis_core::ProcedureStep::thermal_transient(
        "thermal_1",
    )]
}

#[test]
fn families_without_a_device_path_fall_back_unless_the_accelerator_is_required() {
    let profile = ProjectExecutionProfile {
        cluster: "gpu-cluster".into(),
        queue: Some("gpu".into()),
        trust_identity: Some("sha256:worker".into()),
        accelerator: Some(ProjectAcceleratorProfile {
            class: "metal".into(),
            memory_bytes: 16 << 30,
        }),
    };
    let steps = vec![runmat_analysis_core::ProcedureStep::structural_static(
        "static_1",
    )];
    let fallback = resolve_compute_allocation(
        Some(&policy(AcceleratorPreference::Preferred)),
        &steps,
        &remote_target(),
        Some(&profile),
    )
    .unwrap();
    assert_eq!(fallback.backend, runmat_analysis_fea::ComputeBackend::Cpu);
    assert!(fallback.accelerators.is_empty());
    let error = resolve_compute_allocation(
        Some(&policy(AcceleratorPreference::Required)),
        &steps,
        &remote_target(),
        Some(&profile),
    )
    .unwrap_err();
    let message = format!("{error:#}");
    assert!(message.contains("FEA_ACCELERATOR_REQUIRED"), "{message}");
    assert!(message.contains("structural static"), "{message}");
}

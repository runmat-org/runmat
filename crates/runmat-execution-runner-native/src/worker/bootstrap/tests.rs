use super::*;
use crate::protocol::{ReservedFeaLaunch, ReservedProgramExecutionRequest};
use runmat_execution::{
    Digest, OutputContract, ProgramCallable, ProgramFunctionId, ProgramInvocationContext,
    ProgramRevision,
};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest, ProgramTarget,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};

fn reserved(form: ExecutableForm) -> ReservedProgramExecutionRequest {
    let grant = crate::fea_delegation::test_grant();
    let recipe = ProgramBuildRecipe {
        schema_version: runmat_execution_artifact::PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: ProgramRevision::new(
            Digest::sha256(b"private-launch-graph"),
            Digest::sha256(b"private-launch-source"),
            runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        )
        .unwrap(),
        entrypoint: if form == ExecutableForm::MeshingWorkload {
            "meshing_workload"
        } else {
            "0"
        }
        .into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("private-launch-test"),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    // Only bootstrap admission is exercised here; actual meshing kernel and
    // artifact validation are covered by the real process conformance case.
    let artifact =
        ProgramArtifact::materialize(&recipe, form, b"bootstrap-fixture".to_vec()).unwrap();
    ReservedProgramExecutionRequest::new(
        ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe,
            artifact,
            callable: ProgramCallable::semantic(ProgramFunctionId(0), None),
            context: ProgramInvocationContext::Direct,
            assignment: Some(grant.assignment),
            job_id: None,
            arguments: Vec::new(),
            requested_outputs: 1,
            environment: Default::default(),
            console: None,
        },
        grant.resources,
    )
    .unwrap()
}

#[test]
fn meshing_bootstrap_accepts_canonical_reserved_paid_and_proxy_envelopes() {
    let request = reserved(ExecutableForm::MeshingWorkload);
    let grant = crate::fea_delegation::test_grant();
    let expected_assignment = request.request().assignment.clone();
    let expected_resources = request.resources().clone();
    for launch in [
        WorkerLaunchRequest::Reserved(Box::new(request.clone())),
        WorkerLaunchRequest::ReservedFea(Box::new(ReservedFeaLaunch::from_reserved(
            Box::new(request.clone()),
            None,
            grant,
        ))),
        WorkerLaunchRequest::ReservedFeaProxy {
            request: Box::new(request.clone()),
            console_budget: None,
            fea_parent_admission: true,
        },
    ] {
        let decoded =
            decode_launch(&serde_json::to_vec(&launch).unwrap(), WorkerRole::Meshing).unwrap();
        assert_eq!(decoded.assignment(), expected_assignment.as_ref());
        assert_eq!(decoded.resources(), Some(&expected_resources));
        assert_eq!(decoded.fea_proxy(), launch.fea_proxy());
        assert_eq!(decoded.fea_grant().is_some(), launch.fea_grant().is_some());
    }
}

#[test]
fn meshing_bootstrap_refuses_nonmeshing_and_invalid_console_controls_before_effects() {
    let launch =
        WorkerLaunchRequest::Reserved(Box::new(reserved(ExecutableForm::InterpreterBytecodeV2)));
    let payload = serde_json::to_vec(&launch).unwrap();
    assert!(decode_launch(&payload, WorkerRole::Program).is_ok());
    assert!(decode_launch(&payload, WorkerRole::Meshing).is_err());
    let launch = WorkerLaunchRequest::ReservedConsole {
        request: Box::new(reserved(ExecutableForm::MeshingWorkload)),
        console_budget: runmat_execution::console::ConsoleBudget::for_driver_allocation(1 << 30),
    };
    assert!(decode_launch(&serde_json::to_vec(&launch).unwrap(), WorkerRole::Meshing).is_err());
}

#[test]
fn meshing_trailing_bound_admits_renewals_and_denials_without_consuming_a_slot() {
    use crate::fea_delegation::{FeaLeaseUpdate, LeaseObserver};
    use crate::protocol::WorkerDriverMessage;
    let initial = FrameLimits {
        max_message_bytes: 64 * 1024 * 1024,
    };
    assert_eq!(WorkerRole::Program.trailing_limits(initial), initial);
    let limits = WorkerRole::Meshing.trailing_limits(initial);
    assert_eq!(
        u64::from(limits.max_message_bytes),
        crate::fea_delegation::MAX_GRANT_BYTES
    );
    let fixed = std::mem::size_of::<PrivateWorkerBootstrap>()
        + std::mem::size_of::<AdmittedWorkerLaunch>()
        + std::mem::size_of::<LeaseObserver>()
        + std::mem::size_of::<crate::fea_proxy::Replies>()
        + 128; // validated original run ID retained by the bound observer
    assert!(fixed as u64 <= crate::fea_delegation::MAX_GRANT_BYTES);
    let grant = crate::fea_delegation::test_grant();
    let scope = grant.install(&grant.assignment, &grant.resources).unwrap();
    let observer = LeaseObserver::default();
    observer.install(&grant, &scope).unwrap();
    let mut budget = grant.parent.budget;
    budget.granted_ops = budget
        .granted_ops
        .checked_add(runmat_simulation_usage::OperationCount::from_operations(
            1000,
        ))
        .unwrap();
    budget.expires_at_unix_millis += 1000;
    let update = |update| {
        serde_json::to_vec(&WorkerDriverMessage::FeaLeaseUpdate {
            run_id: grant.parent.request.run_id.clone(),
            attempt_id: grant.assignment.attempt_id,
            update,
        })
        .unwrap()
    };
    let renewed = update(FeaLeaseUpdate::Renewed { budget });
    assert!(renewed.len() <= limits.max_message_bytes as usize);
    assert!(observer.observe(&renewed));
    assert!(scope.check().is_ok());
    let denied = update(FeaLeaseUpdate::Denied {
        denied: runmat_runtime::analysis::fea_run_admission::FeaRunDenied::LeaseExpired,
    });
    assert!(observer.observe(&denied));
    assert!(scope.cancellation_token().is_cancelled());
    let mut oversized = std::io::Cursor::new((limits.max_message_bytes + 1).to_be_bytes());
    assert!(matches!(
        runmat_process_host::ipc::read_payload_blocking(&mut oversized, limits),
        Err(runmat_process_host::ProcessHostError::Protocol(_)),
    ));
}

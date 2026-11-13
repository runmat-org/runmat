//! Portable contract checks using the existing synthetic selected-pack fixture.
//! These do not execute a linked image or certify target memory behavior.
use super::*;
use crate::target_test::*;
use runmat_execution::{LanguageCompatibilityMode, ProgramEnvironment, ProgramRevision};
use runmat_test::identity::{SuiteId, TestId, TestIdentityInput};

#[test]
fn paired_entries_and_request_results_keep_exact_product_and_attempt_identity() {
    let (manifest, required, contents) = fixture();
    let selected = manifest.select(&required, None).unwrap();
    let retained = selected
        .components()
        .iter()
        .map(|id| (id.clone(), contents[id].clone()))
        .collect();
    let product = manifest
        .verify_selected_bytes(&selected, &retained)
        .unwrap()
        .selected_product_binding()
        .unwrap();
    let revision = ProgramRevision::new(
        Digest::sha256(b"program"),
        Digest::sha256(b"source"),
        ProgramEnvironment::new(
            1,
            1,
            product.runtime,
            product.catalog,
            LanguageCompatibilityMode::RunMat,
        )
        .unwrap(),
    )
    .unwrap();
    let suite = SuiteId::derive(&revision.canonical_identity(), "suite");
    let test_id = |parameter| {
        TestId::derive(&TestIdentityInput {
            owner_identity: "owner",
            relative_source_identity: "test.m",
            semantic_scheme: "function",
            semantic_item_path: "test",
            parameter_identity: parameter,
            fixture_identity: "none",
        })
    };
    let compiled = NativeCompiledEntry {
        function: runmat_types::ProgramFunctionId(1),
        compiled_function: runmat_types::FunctionId(11),
        body: Digest::sha256(b"test body"),
    };
    let mut tests = ["first", "second"]
        .into_iter()
        .map(|parameter| NativeTestEntry {
            suite: suite.clone(),
            test: test_id(parameter),
            compiled: compiled.clone(),
        })
        .collect::<Vec<_>>();
    tests.sort_by(|a, b| a.test.cmp(&b.test));
    let binding = NativePairedBuildBinding {
        schema_version: NATIVE_PAIRED_TEST_SCHEMA_VERSION,
        program_revision: revision,
        selected_product: product,
        application: NativeCompiledEntry {
            function: runmat_types::ProgramFunctionId(0),
            compiled_function: runmat_types::FunctionId(10),
            body: Digest::sha256(b"app body"),
        },
        tests,
    };
    let header = NativePairedProductHeader {
        schema_version: NATIVE_PAIRED_TEST_SCHEMA_VERSION,
        role: NativePairedProductRole::Test,
        binding: binding.clone(),
        linked_entries: vec![compiled],
    };
    assert_eq!(
        NativePairedProductHeader::from_canonical_bytes(&header.to_canonical_bytes().unwrap())
            .unwrap(),
        header
    );
    let mut wrong = binding.clone();
    wrong.selected_product.runtime = Digest::sha256(b"other runtime");
    assert!(wrong.validate().is_err());
    let mut wrong = binding.clone();
    wrong.tests[0].compiled = NativeCompiledEntry {
        function: wrong.application.function,
        compiled_function: wrong.application.compiled_function,
        body: Digest::sha256(b"different body"),
    };
    assert!(wrong.validate().is_err());

    let request = NativeTargetTestRequest {
        clock_inputs: None,
        local_service_inputs: None,
        schema_version: NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION,
        build_binding: binding.digest().unwrap(),
        suite,
        test: binding.tests[0].test.clone(),
        attempt: 0,
        inputs: crate::ProgramExecutionInputs {
            schema_version: crate::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: vec![],
            objects: vec![],
        },
        resources: runmat_execution::resource::ResourceRequest {
            cpu_millicores: 1000,
            memory_bytes: 4096,
            scratch_bytes: 0,
            max_wall_millis: 1000,
            max_artifact_bytes: 0,
            max_egress_bytes: 0,
            max_relay_bytes: 0,
            accelerators: vec![],
            required_capabilities: Default::default(),
        },
    };
    let mut next = request.clone();
    next.attempt += 1;
    assert_ne!(
        request.digest_for(&binding).unwrap(),
        next.digest_for(&binding).unwrap()
    );
    let mut result = NativeTargetTestResult {
        schema_version: NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION,
        request: request.digest_for(&binding).unwrap(),
        suite: request.suite.clone(),
        test: request.test.clone(),
        attempt: request.attempt,
        status: NativeTargetTestStatus::Passed,
        usage: None,
        diagnostic: String::new(),
    };
    assert!(result.validate_for(&request, &binding).is_err());
    result.usage = Some(NativeTargetTestUsage {
        configured_memory_bytes: 4096,
        peak_managed_reserved_bytes: 2048,
        peak_managed_claimed_bytes: 1024,
        elapsed_millis: 1,
    });
    assert_eq!(
        NativeTargetTestResult::from_canonical_bytes(
            &result.to_canonical_bytes(&request, &binding).unwrap(),
            &request,
            &binding
        )
        .unwrap(),
        result
    );
    assert!(result.validate_for(&next, &binding).is_err());
    result.status = NativeTargetTestStatus::Failed(NativeTargetTestFailure::Cancelled);
    assert_eq!(
        result.terminal_disposition(),
        runmat_test::result::TerminalDisposition::Cancelled
    );
    assert_eq!(
        NativeTargetTestResult::from_canonical_bytes(
            &result.to_canonical_bytes(&request, &binding).unwrap(),
            &request,
            &binding,
        )
        .unwrap(),
        result
    );
    result.usage = None;
    assert!(result.validate_for(&request, &binding).is_err());
    result.status = NativeTargetTestStatus::Failed(NativeTargetTestFailure::ProcessFailure);
    result.usage = None;
    result.validate_for(&request, &binding).unwrap();

    // Structural policy/request identity only: no executable/service issuer
    // is manufactured from this synthetic selected-product fixture.
    let mut clock_binding = binding.clone();
    clock_binding
        .selected_product
        .reached_effects
        .0
        .insert(runmat_types::EffectKind::Clock);
    clock_binding.selected_product.services.push(
        runmat_execution::host::NativeSelectedServiceBinding {
            kind: runmat_execution::host::NativeDeploymentServiceKind::Effect(
                runmat_types::EffectKind::Clock,
            ),
            abi: runmat_execution::host::native_clock_service_abi(),
            implementation: clock_binding.selected_product.runtime_component,
            behavior:
                runmat_execution::host::NativeDeploymentServiceBehavior::DeterministicDeclared,
        },
    );
    let mut clock_request = request.clone();
    clock_request.build_binding = clock_binding.digest().unwrap();
    assert!(
        clock_request.validate_for(&clock_binding).is_err(),
        "deterministic inputs cannot default to an arbitrary epoch"
    );
    clock_request.clock_inputs = Some(runmat_execution::host::NativeClockInputs {
        schema_version: runmat_execution::host::NATIVE_CLOCK_INPUTS_SCHEMA_VERSION,
        initial_monotonic_nanos: 17,
        initial_wall_unix_seconds: 1_700_000_000,
        initial_wall_nanosecond: 0,
        read_step_nanos: 1_000,
        local_offset_seconds: 0,
        process_cpu: None,
    });
    let digest = clock_request.digest_for(&clock_binding).unwrap();
    assert_eq!(
        NativeTargetTestRequest::from_canonical_bytes(
            &clock_request.to_canonical_bytes(&clock_binding).unwrap(),
            &clock_binding
        )
        .unwrap(),
        clock_request
    );
    let before = runmat_execution::host::NativeClockInvocationIdentity::new(
        digest,
        clock_request.clock_inputs.as_ref(),
    )
    .unwrap();
    clock_request.clock_inputs.as_mut().unwrap().read_step_nanos += 1;
    let changed = clock_request.digest_for(&clock_binding).unwrap();
    assert_ne!(
        digest, changed,
        "replay policy is part of the exact attempt identity"
    );
    assert_ne!(
        before,
        runmat_execution::host::NativeClockInvocationIdentity::new(
            changed,
            clock_request.clock_inputs.as_ref()
        )
        .unwrap()
    );
    clock_request.clock_inputs.as_mut().unwrap().read_step_nanos = 0;
    assert!(clock_request.validate_for(&clock_binding).is_err());
    clock_binding.selected_product.profile =
        runmat_execution::host::NativeDeploymentProfile::HostedNative;
    clock_binding.selected_product.services[0].behavior =
        runmat_execution::host::NativeDeploymentServiceBehavior::HostedAmbient;
    clock_request.build_binding = clock_binding.digest().unwrap();
    clock_request.clock_inputs.as_mut().unwrap().read_step_nanos = 1_000;
    assert!(
        clock_request.validate_for(&clock_binding).is_err(),
        "ambient policy must not accept even valid virtual inputs"
    );
    clock_request.clock_inputs = None;
    clock_request.validate_for(&clock_binding).unwrap();
}

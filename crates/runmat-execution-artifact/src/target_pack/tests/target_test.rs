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
        body: Digest::sha256(b"different body"),
    };
    assert!(wrong.validate().is_err());

    let request = NativeTargetTestRequest {
        schema_version: NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION,
        build_binding: binding.digest().unwrap(),
        suite,
        test: binding.tests[0].test.clone(),
        attempt: 0,
        inputs: crate::ProgramExecutionInputs {
            schema_version: crate::PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: vec![],
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
}

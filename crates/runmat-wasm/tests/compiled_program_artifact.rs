#![cfg(target_arch = "wasm32")]

use std::collections::BTreeSet;

use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution::{OutputContract, ProgramCallable, ProgramFunctionId};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
    ProgramExecutionResponse, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use wasm_bindgen_test::wasm_bindgen_test;

async fn request_with_contract(
    interop: runmat_types::InteropManifest,
    capabilities: BTreeSet<runmat_types::CapabilityRequirement>,
) -> ProgramExecutionRequest {
    request_for_source(interop, capabilities, "answer.m", "ans = 42").await
}

async fn request_for_source(
    interop: runmat_types::InteropManifest,
    capabilities: BTreeSet<runmat_types::CapabilityRequirement>,
    filename: &str,
    source: &str,
) -> ProgramExecutionRequest {
    // Source compilation uses the filesystem that the JavaScript host normally installs.
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::MemoryFsProvider::new(),
    ));
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new("root", filename, source),
            None,
        )
        .await
        .unwrap();
    let envelope = unit
        .portable_envelope_for_with_interop(None, interop.clone())
        .unwrap();
    assert!(capabilities.is_subset(&envelope.manifest.capabilities.0));
    let accelerators = runmat_execution::resource::accelerator_requirements_for_capabilities(
        &envelope.manifest.capabilities,
    )
    .unwrap();
    let function =
        usize::try_from(envelope.manifest.identity.entrypoint_function.unwrap().0).unwrap();
    let foreign_policy = if interop == runmat_types::InteropManifest::empty() {
        None
    } else {
        let mut policy_config = runmat_config::runtime::ForeignConfig::default();
        policy_config.mex.timeout_ms = Some(std::num::NonZeroU64::new(15_000).unwrap());
        Some(
            runmat_execution_artifact::foreign_policy_object_from_json_value(
                serde_json::to_value(&policy_config).unwrap(),
            )
            .unwrap()
            .descriptor,
        )
    };
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: envelope.manifest.identity.entrypoint.clone(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("portable-executable-unit-v3"),
        interop,
        foreign_policy,
        accelerators,
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::ExecutableUnitV4,
        envelope.canonical_bytes().unwrap(),
    )
    .unwrap();
    ProgramExecutionRequest {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: ProgramCallable::semantic(
            ProgramFunctionId(u32::try_from(function).expect("portable function id")),
            None,
        ),
        context: runmat_execution::ProgramInvocationContext::Direct,
        assignment: None,
        job_id: None,
        arguments: Vec::new(),
        requested_outputs: 1,
    }
}

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn browser_executes_the_exact_portable_artifact_without_a_project() {
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::MemoryFsProvider::new(),
    ));
    let request =
        request_with_contract(runmat_types::InteropManifest::empty(), BTreeSet::new()).await;

    let response =
        runmat_wasm::execute_program_artifact(serde_wasm_bindgen::to_value(&request).unwrap())
            .await
            .unwrap();
    let response: ProgramExecutionResponse = serde_wasm_bindgen::from_value(response).unwrap();
    let ProgramExecutionResponse::Success { outputs } = response else {
        panic!("browser rejected an exact portable program artifact: {response:?}");
    };
    let [ValuePayload::Inline(value)] = outputs.as_slice() else {
        panic!("browser returned an invalid output vector");
    };
    assert_eq!(**value, InlineValue::F64Bits(42.0_f64.to_bits()));
}

#[wasm_bindgen_test]
async fn browser_executes_a_named_function_portable_artifact() {
    let request = request_for_source(
        runmat_types::InteropManifest::empty(),
        BTreeSet::new(),
        "portableFunctionSmoke.m",
        "function result = portableFunctionSmoke()\nresult = 42;\nend\n",
    )
    .await;
    let response =
        runmat_wasm::execute_program_artifact(serde_wasm_bindgen::to_value(&request).unwrap())
            .await
            .expect("execute exact browser function artifact");
    let response: ProgramExecutionResponse = serde_wasm_bindgen::from_value(response).unwrap();
    let ProgramExecutionResponse::Success { outputs } = response else {
        panic!("browser rejected the named function artifact: {response:?}");
    };
    let [ValuePayload::Inline(value)] = outputs.as_slice() else {
        panic!("browser returned an invalid function output");
    };
    assert_eq!(**value, InlineValue::F64Bits(42.0_f64.to_bits()));
}

#[wasm_bindgen_test]
async fn browser_executes_deal_requested_discarded_and_zero_outputs_exactly() {
    let request = request_for_source(
        runmat_types::InteropManifest::empty(),
        BTreeSet::new(),
        "dealPortable.m",
        "function result = dealPortable()\nwide = uint64(9007199254740992) + uint64(1);\n[left, right] = deal(wide);\n[~, selected] = deal(uint8(3), wide);\ndeal(wide);\nassert(isa(left, 'uint64') && isa(right, 'uint64') && isa(selected, 'uint64'));\nassert(left == wide && right == wide && selected == wide);\nresult = selected;\nend\n",
    )
    .await;
    let response =
        runmat_wasm::execute_program_artifact(serde_wasm_bindgen::to_value(&request).unwrap())
            .await
            .expect("execute exact browser deal artifact");
    let response: ProgramExecutionResponse = serde_wasm_bindgen::from_value(response).unwrap();
    let ProgramExecutionResponse::Success { outputs } = response else {
        panic!("browser rejected the deal artifact: {response:?}");
    };
    let [ValuePayload::Inline(value)] = outputs.as_slice() else {
        panic!("browser returned an invalid deal output");
    };
    assert_eq!(**value, InlineValue::U64(9_007_199_254_740_993));
}

#[wasm_bindgen_test]
async fn browser_deal_count_error_keeps_identifier_and_evaluates_input_once() {
    let _provider = runmat_filesystem::replace_provider(std::sync::Arc::new(
        runmat_filesystem::MemoryFsProvider::new(),
    ));
    let request = request_for_source(
        runmat_types::InteropManifest::empty(),
        BTreeSet::new(),
        "dealCountProbe.m",
        "function result = dealCountProbe()\nglobal dealCalls; dealCalls = 0; caught = false;\ntry\n operation = @deal;\n [a,b,c] = operation(nextDealInput(), uint16(8));\ncatch failure\n caught = strcmp(failure.identifier, 'RunMat:deal:InputOutputCountMismatch');\n assert(caught, failure.identifier);\nend\nassert(caught);\nresult = dealCalls;\nend\nfunction value = nextDealInput()\nglobal dealCalls; dealCalls = dealCalls + 1;\nvalue = uint64(9007199254740992) + uint64(1);\nend\n",
    )
    .await;
    let response =
        runmat_wasm::execute_program_artifact(serde_wasm_bindgen::to_value(&request).unwrap())
            .await
            .expect("execute exact browser count-error artifact");
    let response: ProgramExecutionResponse = serde_wasm_bindgen::from_value(response).unwrap();
    let ProgramExecutionResponse::Success { outputs } = response else {
        panic!("browser failed to catch the typed deal error: {response:?}");
    };
    let [ValuePayload::Inline(value)] = outputs.as_slice() else {
        panic!("browser returned an invalid effect-count output");
    };
    assert_eq!(**value, InlineValue::F64Bits(1.0_f64.to_bits()));
}

#[wasm_bindgen_test]
async fn browser_filesystem_admission_tracks_the_installed_read_provider() {
    use std::sync::Arc;

    use runmat_filesystem::{MemoryFsProvider, PlaceholderFsProvider};

    let request = request_for_source(
        runmat_types::InteropManifest::empty(),
        BTreeSet::from([runmat_types::CapabilityRequirement::Filesystem]),
        "read-provider.m",
        "ans = fileread('fixture.txt');",
    )
    .await;
    {
        let _provider = runmat_filesystem::replace_provider(Arc::new(PlaceholderFsProvider));
        let response =
            runmat_wasm::execute_program_artifact(serde_wasm_bindgen::to_value(&request).unwrap())
                .await
                .unwrap();
        let response: ProgramExecutionResponse = serde_wasm_bindgen::from_value(response).unwrap();
        let ProgramExecutionResponse::Failure { message } = response else {
            panic!("browser admitted a filesystem requirement without a read provider");
        };
        assert!(message.contains("browser host rejected unavailable executable capabilities"));
    }
    {
        let provider = MemoryFsProvider::new();
        provider
            .write_project_path("fixture.txt", b"original browser file")
            .unwrap();
        let _provider = runmat_filesystem::replace_provider(Arc::new(provider));
        let response =
            runmat_wasm::execute_program_artifact(serde_wasm_bindgen::to_value(&request).unwrap())
                .await
                .unwrap();
        let response: ProgramExecutionResponse = serde_wasm_bindgen::from_value(response).unwrap();
        assert!(matches!(response, ProgramExecutionResponse::Success { .. }));
    }
}

#[wasm_bindgen_test]
async fn browser_rejects_native_mex_artifact_before_program_execution() {
    let interop = runmat_types::InteropManifest {
        schema_version: runmat_types::INTEROP_MANIFEST_SCHEMA_VERSION,
        foreign_types: Vec::new(),
        adapters: vec![runmat_types::ForeignAdapterRequirement {
            adapter: runmat_types::ForeignAdapterId::new("mex-c").unwrap(),
            minimum_version: 1,
            capabilities: runmat_types::CapabilitySet(BTreeSet::from([
                runmat_types::CapabilityRequirement::NativeCode,
                runmat_types::CapabilityRequirement::ForeignRuntime,
            ])),
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
            artifact_identities: vec![runmat_types::ForeignArtifactIdentity::new(
                "mex:v1:sha256:fixture",
            )
            .unwrap()],
        }],
        adapter_contracts: Vec::new(),
    };
    let request = request_with_contract(interop, BTreeSet::new()).await;
    let response =
        runmat_wasm::execute_program_artifact(serde_wasm_bindgen::to_value(&request).unwrap())
            .await
            .unwrap();
    let response: ProgramExecutionResponse = serde_wasm_bindgen::from_value(response).unwrap();
    let ProgramExecutionResponse::Failure { message } = response else {
        panic!("browser admitted a native MEX artifact");
    };
    assert!(message.contains("browser host rejected unavailable executable capabilities"));
    assert!(message.contains("mex-c [mex:v1:sha256:fixture]"));
}

#[wasm_bindgen_test]
async fn browser_rejects_an_unavailable_accelerator_recipe_before_execution() {
    use runmat_execution::resource::{
        AcceleratorClass, AcceleratorFeature, AcceleratorProvider, AcceleratorProviderId,
        AcceleratorProviderVersion, AcceleratorRequest,
    };

    let mut request =
        request_with_contract(runmat_types::InteropManifest::empty(), BTreeSet::new()).await;
    request.recipe.accelerators = vec![AcceleratorRequest {
        class: AcceleratorClass::new("gpu").unwrap(),
        count: 1,
        minimum_allocation_bytes: 1 << 20,
        provider: Some(AcceleratorProvider {
            id: AcceleratorProviderId::new("runmat.cuda").unwrap(),
            version: AcceleratorProviderVersion::new("1.0.0").unwrap(),
            abi_fingerprint: runmat_execution::Digest::sha256(b"cuda-native-v1"),
        }),
        required_features: [AcceleratorFeature::Compute].into_iter().collect(),
    }];
    request.artifact = ProgramArtifact::materialize(
        &request.recipe,
        ExecutableForm::ExecutableUnitV4,
        request.artifact.executable_bytes.clone(),
    )
    .unwrap();

    let response =
        runmat_wasm::execute_program_artifact(serde_wasm_bindgen::to_value(&request).unwrap())
            .await
            .unwrap();
    let response: ProgramExecutionResponse = serde_wasm_bindgen::from_value(response).unwrap();
    let ProgramExecutionResponse::Failure { message } = response else {
        panic!("browser admitted an unavailable accelerator recipe");
    };
    assert!(message.contains("unavailable execution resources"));
}

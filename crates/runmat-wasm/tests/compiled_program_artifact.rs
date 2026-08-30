#![cfg(target_arch = "wasm32")]

use std::collections::BTreeSet;

use runmat_execution::value::{InlineValue, ValuePayload};
use runmat_execution::{OutputContract, ProgramCallable, ProgramFunctionId};
use runmat_execution_artifact::{
    ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
    ProgramExecutionResponse, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
};
use wasm_bindgen_test::wasm_bindgen_test;

async fn request_with_contract(
    interop: runmat_types::InteropManifest,
    capabilities: BTreeSet<runmat_types::CapabilityRequirement>,
) -> ProgramExecutionRequest {
    let mut session = runmat_core::RunMatSession::with_options(false, false).unwrap();
    let unit = session
        .compile_executable_unit(
            runmat_core::ExecutableSource::new("root", "answer.m", "ans = 42"),
            None,
        )
        .await
        .unwrap();
    let mut envelope = unit
        .portable_envelope_for_with_interop(None, interop.clone())
        .unwrap();
    envelope.manifest.capabilities.0.extend(capabilities);
    let accelerators = runmat_execution::resource::accelerator_requirements_for_capabilities(
        &envelope.manifest.capabilities,
    )
    .unwrap();
    let function = usize::try_from(envelope.manifest.identity.entrypoint_function.0).unwrap();
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: envelope.manifest.identity.program.clone(),
        entrypoint: function.to_string(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: "interpreter".into(),
        target: ProgramTarget::portable("portable-executable-unit-v3"),
        interop,
        accelerators,
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::ExecutableUnitV3,
        envelope.canonical_bytes().unwrap(),
    )
    .unwrap();
    ProgramExecutionRequest {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V5,
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
    let request = request_with_contract(
        runmat_types::InteropManifest::empty(),
        BTreeSet::from([
            runmat_types::CapabilityRequirement::HostRuntime,
            runmat_types::CapabilityRequirement::Filesystem,
            runmat_types::CapabilityRequirement::UserInterface,
            runmat_types::CapabilityRequirement::ParallelRuntime,
        ]),
    )
    .await;

    let response =
        runmat_wasm::execute_program_artifact(serde_wasm_bindgen::to_value(&request).unwrap())
            .await
            .unwrap();
    let response: ProgramExecutionResponse = serde_wasm_bindgen::from_value(response).unwrap();
    let ProgramExecutionResponse::Success {
        value: ValuePayload::Inline(value),
    } = response
    else {
        panic!("browser rejected an exact portable program artifact");
    };
    assert_eq!(*value, InlineValue::F64Bits(42.0_f64.to_bits()));
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
        ExecutableForm::ExecutableUnitV3,
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

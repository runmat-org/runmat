//! Host witness: real selected pack -> one frozen source -> both links ->
//! application execution and framed test execution. No qualification is minted.

use std::{path::PathBuf, time::Duration};

use anyhow::{bail, ensure, Context, Result};
use runmat_aot::{
    link::{LinkRequest, PairedLinkRequests},
    selected_admission::{self, SelectedHostAdmission, SelectedPaths},
    NativeObjectOptions, NativePairArtifactRecipes,
};
use runmat_core::{ExecutableSource, RunMatSession};
use runmat_execution::{Digest, OutputContract, ProgramInvocationContext};
use runmat_execution_artifact::target_test::{
    NativeTargetTestFailure, NativeTargetTestRequest, NativeTargetTestStatus,
    NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION,
};
use runmat_execution_artifact::{
    ProgramBuildRecipe, ProgramExecutionInputs, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_test::descriptor::TestSelector;
use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource};
use runmat_test_runner_native::target_test::{
    run_compiled_native_application, run_compiled_native_target_test, NativeTargetTestCompletion,
    NativeTargetTestRunLimits,
};

// The application and passing test consume the same nontrivial arithmetic;
// negative controls distinguish typed assertion and language failures.
const SOURCE: &str = "function tests = pairedWitnessTest()\n assert(sharedAnswer() == 42);\n tests = functiontests(localfunctions);\nend\nfunction y = sharedAnswer()\n y = (2 + 5) * 6;\nend\nfunction testPass()\n assert(sharedAnswer() == 42);\nend\nfunction testAssertion()\n assert(sharedAnswer() == 41);\nend\nfunction testLanguage()\n error('paired language failure');\nend\n";

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let arguments = std::env::args_os()
        .skip(1)
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let [manifest, components, host_admission, output] = arguments.as_slice() else {
        bail!("usage: native_paired PACK_MANIFEST COMPONENT_ROOT INDEPENDENT_HOST_ADMISSION EXISTING_OUTPUT_DIRECTORY");
    };
    ensure!(output.is_dir(), "output directory must already exist");
    let paths = SelectedPaths {
        manifest: manifest.clone(),
        components: components.clone(),
        host_admission: host_admission.clone(),
    };
    let mut session = RunMatSession::with_options(false, false)?;
    let snapshot = FrozenTestRunSnapshot::freeze(
        Digest::sha256(SOURCE).to_string(),
        Digest::sha256(SOURCE).to_string(),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        Digest::sha256(b"native-paired-semantic-witness-v1").to_string(),
        vec![SavedRunSource {
            owner_identity: "path:native-paired-witness".into(),
            relative_path: "pairedWitnessTest.m".into(),
            content: SOURCE.into(),
        }],
        Vec::new(),
    )?;
    let prepared = session.prepare_tests(&snapshot, &TestSelector::default())?;
    ensure!(
        prepared.discovery.diagnostics.is_empty(),
        "discovery diagnostics: {:?}",
        prepared.discovery.diagnostics
    );
    let control_count = prepared.plan.tests().count();
    ensure!(
        control_count == 3,
        "expected three semantic controls, found {control_count}"
    );
    let descriptor = &prepared
        .plan
        .tests()
        .next()
        .context("missing test")?
        .procedure
        .source;
    let unit = session
        .compile_test_executable_unit(
            ExecutableSource::new(
                descriptor.owner_identity.clone(),
                descriptor.relative_path.clone(),
                SOURCE,
            ),
            descriptor,
            &snapshot,
            false,
        )
        .await?;
    let options = NativeObjectOptions::default();
    // Report the same lowered demands before selection can fail. The target
    // comes from operator admission and exact pack probe bytes, not names in
    // the witness or an ambient target guess. Compilation rechecks these inputs.
    let host = SelectedHostAdmission::load(&paths.host_admission)?;
    let manifest = selected_admission::load_manifest(&paths.manifest)?;
    let probe = selected_admission::read_component(
        &paths.components,
        &manifest,
        &manifest.abi_probe_component,
        &host.link.storage_grant(),
    )?;
    let target = runmat_native_codegen::NativeTarget::Native {
        target: Box::new(host.expected_target.clone()),
        probe: Box::new(runmat_execution::host::NativeAbiProbe::from_canonical_bytes(&probe)?),
    };
    target.validate()?;
    let native = unit.prepare_native_compilation_for_with_interop(None, options.interop.clone())?;
    let entrypoint = native.entrypoint();
    let assembly = native.lower(target)?;
    println!(
        "lowered native requirements: capabilities {:?}; effects {:?}; services {:?}",
        assembly.requirements.capabilities,
        assembly.required_effects(),
        assembly.required_runtime_services()
    );
    drop(assembly);
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: unit.revision().program_revision.clone(),
        entrypoint: entrypoint.0.to_string(),
        outputs: OutputContract {
            requested_outputs: 0,
            owned_work: None,
        },
        execution_mode: "aot".into(),
        // Materialization replaces this template with the verified native target.
        target: ProgramTarget::portable("paired-recipe-template"),
        interop: options.interop.clone(),
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let link = |name: &str| LinkRequest {
        output: output.join(if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.into()
        }),
        linker: None,
        keep_temps: false,
        overwrite: false,
    };
    let application = link("paired-application");
    let tests = link("paired-tests");
    let product = runmat_aot::compile_operator_selected_pair(
        &unit,
        &prepared.plan,
        &paths,
        options,
        NativePairArtifactRecipes {
            application: recipe.clone(),
            tests: recipe,
        },
        PairedLinkRequests {
            application: &application,
            tests: &tests,
        },
    )?;
    let limits = NativeTargetTestRunLimits::default();
    let app = run_compiled_native_application(&product, limits, 64 * 1024, Duration::from_secs(30))
        .await?;
    ensure!(
        app.exit.success,
        "application failed: {:?}: {}",
        app.exit,
        app.stderr
    );
    println!(
        "application passed; paired binding {}",
        product.linked().binding().digest()?
    );
    for descriptor in prepared.plan.tests() {
        let expected = match descriptor.display_name.rsplit('/').next() {
            Some("testPass") => NativeTargetTestStatus::Passed,
            Some("testAssertion") => {
                NativeTargetTestStatus::Failed(NativeTargetTestFailure::Assertion)
            }
            Some("testLanguage") => {
                NativeTargetTestStatus::Failed(NativeTargetTestFailure::LanguageException)
            }
            _ => bail!("unexpected semantic witness {}", descriptor.display_name),
        };
        let request = NativeTargetTestRequest {
            schema_version: NATIVE_TARGET_TEST_PROTOCOL_SCHEMA_VERSION,
            build_binding: product.linked().binding().digest()?,
            suite: descriptor.suite_id.clone(),
            test: descriptor.id.clone(),
            attempt: 1,
            inputs: ProgramExecutionInputs {
                schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
                context: ProgramInvocationContext::Direct,
                arguments: Vec::new(),
                objects: Vec::new(),
            },
            resources: runmat_execution::resource::ResourceRequest {
                cpu_millicores: 1000,
                memory_bytes: 64 * 1024 * 1024,
                scratch_bytes: 0,
                max_wall_millis: 30_000,
                max_artifact_bytes: 0,
                max_egress_bytes: 0,
                max_relay_bytes: 0,
                accelerators: Vec::new(),
                required_capabilities: Default::default(),
            },
        };
        let report = run_compiled_native_target_test(&product, &request, limits).await?;
        let NativeTargetTestCompletion::Receiver(result) = &report.completion else {
            bail!("no receiver result: {report:?}");
        };
        ensure!(
            result.status == expected,
            "{}: expected {expected:?}, got {result:?}; stderr {}",
            descriptor.display_name,
            report.stderr
        );
        let usage = result.usage.context("receiver omitted managed usage")?;
        ensure!(
            usage.configured_memory_bytes == request.resources.memory_bytes,
            "wrong configured memory scope"
        );
        ensure!(
            usage.peak_managed_claimed_bytes <= usage.peak_managed_reserved_bytes
                && usage.peak_managed_reserved_bytes <= usage.configured_memory_bytes,
            "invalid managed ledger peaks"
        );
        println!(
            "{}: {:?}; managed usage {:?}; {}",
            descriptor.display_name, result.status, usage, result.diagnostic
        );
    }
    Ok(())
}

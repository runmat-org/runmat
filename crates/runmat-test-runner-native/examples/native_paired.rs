//! Host witness: real selected pack -> one frozen source -> both links ->
//! application execution and framed test execution. The final receipt records
//! completed outcomes but grants no pack, link, reopen, or execution authority.

use std::{io::Write, path::PathBuf, time::Duration};

#[path = "native_paired/callback.rs"]
mod callback;
#[path = "native_paired/loopback.rs"]
mod loopback;
#[path = "native_paired/receipt.rs"]
mod receipt;
#[path = "native_paired/review.rs"]
mod review;
#[path = "native_paired/witness.rs"]
mod witness;

use anyhow::{bail, ensure, Context, Result};
use runmat_aot::{
    installation::{issue_completed_installation_admission, ExternalNativeInstallationIntent},
    link::{LinkRequest, PairedLinkRequests},
    selected_admission::{self, SelectedHostAdmission, SelectedPaths},
    NativeObjectOptions, NativePairArtifactRecipes, OperatorPairReopenRequest, PersistedNativePair,
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
    run_compiled_native_application, run_compiled_native_target_test,
    NativePairedQualificationReceipt, NativeTargetTestCompletion, NativeTargetTestRunLimits,
    QualifiedNativeTest,
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
    if arguments
        .first()
        .is_some_and(|value| value.as_os_str() == std::ffi::OsStr::new("--validate-receipt"))
    {
        ensure!(
            arguments.len() == 2,
            "usage: native_paired --validate-receipt PATH"
        );
        return receipt::validate(&arguments[1]);
    }
    let reopening = arguments
        .first()
        .is_some_and(|value| value.as_os_str() == std::ffi::OsStr::new("--reopen"));
    let awaiting_review = arguments.first().is_some_and(|value| {
        value.as_os_str() == std::ffi::OsStr::new("--await-installation-review")
    });
    let issuing = awaiting_review
        || arguments
            .first()
            .is_some_and(|value| value.as_os_str() == std::ffi::OsStr::new("--issue-installation"));
    ensure!(
        if reopening || issuing {
            matches!(arguments.len(), 9 | 10)
        } else {
            matches!(arguments.len(), 4 | 5)
        },
        "usage: native_paired PACK_MANIFEST COMPONENT_ROOT INDEPENDENT_HOST_ADMISSION EXISTING_OUTPUT_DIRECTORY [WITNESS_JSON]\n       native_paired --reopen PACK_MANIFEST COMPONENT_ROOT INDEPENDENT_HOST_ADMISSION EXISTING_OUTPUT_DIRECTORY PERSISTED_PAIR APPLICATION_INSTALLATION TEST_INSTALLATION QUALIFICATION_OUTPUT [WITNESS_JSON]\n       native_paired --issue-installation PACK_MANIFEST COMPONENT_ROOT INDEPENDENT_HOST_ADMISSION EXISTING_OUTPUT_DIRECTORY APPLICATION_INTENT TEST_INTENT APPLICATION_ADMISSION_OUTPUT TEST_ADMISSION_OUTPUT [WITNESS_JSON]"
    );
    let offset = usize::from(reopening || issuing);
    let manifest = &arguments[offset];
    let components = &arguments[offset + 1];
    let host_admission = &arguments[offset + 2];
    let output = &arguments[offset + 3];
    let witness_index = if reopening || issuing { 9 } else { 4 };
    ensure!(output.is_dir(), "output directory must already exist");
    let paths = SelectedPaths {
        manifest: manifest.clone(),
        components: components.clone(),
        host_admission: host_admission.clone(),
    };
    let host = SelectedHostAdmission::load(&paths.host_admission)?;
    let witness = witness::WitnessMode::load(arguments.get(witness_index).map(PathBuf::as_path))?;
    witness.validate(&host)?;
    let clock_inputs = witness.clock_inputs();
    let source = witness.source(SOURCE)?;
    let mut session = RunMatSession::with_options(false, false)?;
    let snapshot = FrozenTestRunSnapshot::freeze(
        Digest::sha256(&source).to_string(),
        Digest::sha256(&source).to_string(),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
        Digest::sha256(b"native-paired-semantic-witness-v1").to_string(),
        vec![SavedRunSource {
            owner_identity: "path:native-paired-witness".into(),
            relative_path: "pairedWitnessTest.m".into(),
            content: source.clone(),
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
        control_count == witness.control_count(),
        "expected {} semantic controls, found {control_count}",
        witness.control_count()
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
                source,
            ),
            descriptor,
            &snapshot,
            false,
        )
        .await?;
    let callback_artifact = witness.callback_artifact()?;
    let foreign_policy = callback_artifact
        .as_ref()
        .map(|artifact| artifact.policy.clone());
    let foreign_policy_descriptor = foreign_policy
        .as_ref()
        .map(|policy| policy.descriptor.clone());
    let options = NativeObjectOptions {
        clock_inputs: clock_inputs.clone(),
        local_service_inputs: host.local_service_inputs.clone(),
        interop: callback_artifact
            .as_ref()
            .map_or_else(runmat_types::InteropManifest::empty, |artifact| {
                artifact.interop.clone()
            }),
        foreign_policy,
        native_interfaces: callback_artifact.map_or_else(
            runmat_native_ffi::NativeInterfaceArtifactBundle::empty,
            |artifact| artifact.bundle,
        ),
        ..Default::default()
    };
    // Report the same lowered demands before selection can fail. The target
    // comes from operator admission and exact pack probe bytes, not names in
    // the witness or an ambient target guess. Compilation rechecks these inputs.
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
    witness.assert_lowered_effects(&assembly.required_effects())?;
    witness.assert_lowered_services(&assembly.required_deployment_services(), &host)?;
    println!(
        "lowered native requirements: capabilities {:?}; effects {:?}; services {:?}",
        assembly.requirements.capabilities,
        assembly.required_effects(),
        assembly.required_deployment_services()
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
        foreign_policy: foreign_policy_descriptor,
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
    let recipes = NativePairArtifactRecipes {
        application: recipe.clone(),
        tests: recipe,
    };
    let (product, persisted, qualification_output) = if reopening {
        let persisted_path = &arguments[5];
        let persisted = PersistedNativePair::from_canonical_bytes(
            &selected_admission::read_bounded(
                persisted_path,
                PersistedNativePair::MAX_ENCODED_BYTES as u64,
            )
            .with_context(|| format!("read persisted pair `{}`", persisted_path.display()))?,
        )?;
        let product = runmat_aot::reopen_operator_selected_pair(
            &unit,
            &prepared.plan,
            &paths,
            options,
            recipes,
            &persisted,
            OperatorPairReopenRequest {
                application_installation: &arguments[6],
                test_installation: &arguments[7],
                application_image: application.output.clone(),
                test_image: tests.output.clone(),
            },
        )?;
        ensure!(
            product.persisted()? == persisted,
            "reopened product differs from persisted pair"
        );
        (product, persisted, arguments[8].clone())
    } else {
        let product = runmat_aot::compile_operator_selected_pair(
            &unit,
            &prepared.plan,
            &paths,
            options,
            recipes,
            PairedLinkRequests {
                application: &application,
                tests: &tests,
            },
        )?;
        let persisted = product.persisted()?;
        write_new(
            &output.join("native-paired-product-v1.json"),
            &persisted.to_canonical_bytes()?,
        )?;
        (
            product,
            persisted,
            output.join("native-paired-qualification-v1.json"),
        )
    };
    let limits = NativeTargetTestRunLimits::default();
    // Fixed operator-selected ports are part of the witness source identity.
    // Bind them exactly once across fresh execution/issuance/reopen and fail
    // cleanly rather than silently rebuilding a different artifact.
    let loopback = witness
        .loopback_ports()
        .map(|(http, tcp)| loopback::LoopbackWitness::start(http, tcp, 3))
        .transpose()?;
    let app = run_compiled_native_application(&product, limits, 64 * 1024, Duration::from_secs(30))
        .await?;
    ensure!(
        app.exit.success,
        "application failed: {:?}: {}",
        app.exit,
        app.stderr
    );
    witness.assert_application_output(&app.stdout)?;
    println!(
        "application passed; paired binding {}",
        product.linked().binding().digest()?
    );
    let mut qualified_tests = Vec::new();
    for descriptor in prepared.plan.tests() {
        let expected = match descriptor.display_name.rsplit('/').next() {
            Some("testPass") => NativeTargetTestStatus::Passed,
            Some("testAssertion") => {
                NativeTargetTestStatus::Failed(NativeTargetTestFailure::Assertion)
            }
            Some("testLanguage") => {
                NativeTargetTestStatus::Failed(NativeTargetTestFailure::LanguageException)
            }
            Some("testCallbackError") if witness.has_callback_error_control() => {
                NativeTargetTestStatus::Failed(NativeTargetTestFailure::LanguageException)
            }
            _ => bail!("unexpected semantic witness {}", descriptor.display_name),
        };
        let request = NativeTargetTestRequest {
            clock_inputs: clock_inputs.clone(),
            local_service_inputs: host.local_service_inputs.clone(),
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
        ensure!(
            [&report.stdout, &report.stderr].iter().all(|stream| {
                stream.completion == runmat_process_host::ProcessOutputCompletion::EndOfStream
                    && !stream.truncated
            }),
            "paired witness did not retain complete application output: stdout {}; stderr {}",
            report.stdout,
            report.stderr
        );
        let NativeTargetTestCompletion::Receiver(result) = &report.completion else {
            bail!("no receiver result: {report:?}");
        };
        ensure!(
            result.status == expected,
            "{}: expected {expected:?}, got {result:?}; stderr {}",
            descriptor.display_name,
            report.stderr
        );
        witness.assert_test_result(descriptor.display_name.as_str(), result)?;
        qualified_tests.push(QualifiedNativeTest {
            request: request.clone(),
            result: result.clone(),
            expected,
        });
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
    let qualification = NativePairedQualificationReceipt::new(persisted, &app, qualified_tests)?;
    if let Some(loopback) = loopback {
        loopback.finish()?;
    }
    write_new(&qualification_output, &qualification.to_canonical_bytes()?)?;
    if issuing {
        if awaiting_review {
            review::wait(&arguments[5], &arguments[6]).await?;
        }
        issue_installation_admissions(
            &product,
            &arguments[5],
            &arguments[6],
            &arguments[7],
            &arguments[8],
        )?;
    }
    Ok(())
}

fn issue_installation_admissions(
    product: &runmat_aot::CompiledSelectedNativePair,
    application_intent: &std::path::Path,
    test_intent: &std::path::Path,
    application_output: &std::path::Path,
    test_output: &std::path::Path,
) -> Result<()> {
    ensure!(
        application_output != test_output,
        "application and test admission outputs must be distinct"
    );
    let application_intent = ExternalNativeInstallationIntent::from_operator_bytes(
        &selected_admission::read_bounded(application_intent, 1024 * 1024)?,
    )?;
    let test_intent = ExternalNativeInstallationIntent::from_operator_bytes(
        &selected_admission::read_bounded(test_intent, 1024 * 1024)?,
    )?;
    // Issuance occurs after the application and all three semantic controls
    // completed above. Each record still requires its independently supplied
    // expected image, product, deployment qualification, layout, loader
    // inventory, and storage grant; completed image bytes cannot mint them.
    let application = issue_completed_installation_admission(
        product.linked().application(),
        &application_intent,
    )?;
    let tests = issue_completed_installation_admission(product.linked().tests(), &test_intent)?;
    let application = application.to_canonical_bytes()?;
    let tests = tests.to_canonical_bytes()?;
    ensure!(
        !application_output.exists() && !test_output.exists(),
        "installation admission output already exists"
    );
    write_new(application_output, &application)?;
    write_new(test_output, &tests)
}

fn write_new(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("create `{}`", path.display()))?;
    output
        .write_all(bytes)
        .with_context(|| format!("write `{}`", path.display()))?;
    output
        .sync_all()
        .with_context(|| format!("synchronize `{}`", path.display()))
}

#[path = "analysis_process_conformance/fluid_fourier.rs"]
mod fluid_fourier;
#[path = "analysis_process_conformance/fluid_mass_flow.rs"]
mod fluid_mass_flow;
#[path = "analysis_process_conformance/fluid_study_fixture.rs"]
mod fluid_study_fixture;
#[path = "support/fluid.rs"]
mod fluid_support;
#[path = "support/phase_execution.rs"]
mod phase_execution_support;
#[path = "support/fea_admission.rs"]
mod fea_admission_support;
#[path = "analysis_process_conformance/static_recovery.rs"]
mod static_recovery;
use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use runmat_analysis_execution::{
    build_analysis_task_submission, build_resource_planned_task_submission,
    import_analysis_objects, import_analysis_output_set, import_structural_linear_completion,
    prepare_structural_linear_program, prepare_structural_resource_usage, AnalysisExecutionContext,
    AnalysisHostWorkload, PreparedAnalysisProgram, PreparedResourceUsage,
    PreparedStructuralLinearProgram, StructuralLinearProgramPreparation,
    StructuralSequencePreparation,
};
use runmat_analysis_fea::constraints::{
    CanonicalStructuralConstraintPlan, CanonicalStructuralTopologyIdentity, ConstraintEnforcement,
};
use runmat_analysis_fea::contracts::{
    FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE, FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
    FEA_FIELD_STRUCTURAL_STRESS,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StructuralFrequencySolverPlan, StructuralFrequencyWorkload,
    StructuralFrequencyWorkloadInput, StructuralLinearWorkload, StructuralLinearWorkloadResult,
    DEFAULT_ANALYSIS_CHUNK_BYTES,
};
use runmat_analysis_fea::fixtures::{fixture_model, mixed_solid_beam_shell_model, FixtureId};
use runmat_analysis_fea::frequency::FrequencyResponsePlan;
use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
use runmat_analysis_fea::ComputeBackend;
use runmat_execution::identity::ArtifactId;
use runmat_execution::resource::{Capability, ResourceInventory, ResourceRequest};
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValuePayload;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{
    archive::{write_bundle, ArchiveLimits},
    ArtifactAccess, ExecutableForm, ExecutionBundleBuilder, ProgramExecutionResponse,
};
use runmat_execution_runner::{PoolSpec, WorkerSpec};
use runmat_execution_runner_native::{
    execute_analysis_program_request, execute_structural_frequency_locally,
    execute_structural_frequency_remotely, execute_structural_modal_transient_locally,
    execute_structural_modal_transient_remotely, execute_structural_sequence_locally,
    execute_structural_sequence_remotely, execute_structural_transient_remotely,
    run_remote_worker_quic, NativeAnalysisHostLimits, NativeAnalysisKernel, NativeExecutionConfig,
    NativeExecutionError, NativeProgramSession, NativeProgramTask, QuicRemoteWorkerChannel,
    RemotePoolDriver, RemoteWorkerChannel, RemoteWorkerChannelConfig, RemoteWorkerQuicRequest,
    StructuralFrequencyExecutionPreparation, StructuralSequenceExecutionPolicy,
};
use runmat_execution_transport_native::frame::FrameLimits;
use runmat_execution_transport_native::overlay::{PinnedQuicEndpoint, QuicOverlayListener};
use runmat_meshing_core::{ElementOrder, PersistentEntityId, PersistentEntityKind, StableDigest};

#[path = "analysis_process_conformance/assessment.rs"]
mod assessment;
#[cfg(feature = "occt-native")]
#[path = "analysis_process_conformance/remote_families.rs"]
mod remote_families;
#[cfg(feature = "occt-native")]
#[path = "analysis_process_conformance/assessment_publication.rs"]
mod assessment_publication;
#[path = "analysis_process_conformance/electrothermal.rs"]
mod electrothermal;
#[cfg(feature = "occt-native")]
#[path = "analysis_process_conformance/fluid_authored.rs"]
mod fluid_authored;
#[path = "analysis_process_conformance/fluid_transient.rs"]
mod fluid_transient;
#[path = "analysis_process_conformance/magnetic.rs"]
mod magnetic;
#[cfg(feature = "occt-native")]
#[path = "analysis_process_conformance/magnetic_authored.rs"]
mod magnetic_authored;
#[cfg(feature = "occt-native")]
#[path = "analysis_process_conformance/maxwell_authored.rs"]
mod maxwell_authored;
#[path = "analysis_process_conformance/modal_transient.rs"]
mod modal_transient;
#[path = "analysis_process_conformance/nonlinear.rs"]
mod nonlinear;
#[cfg(feature = "occt-native")]
#[path = "analysis_process_conformance/nonlinear_authored.rs"]
mod nonlinear_authored;
#[path = "analysis_process_conformance/publication_chain.rs"]
mod publication_chain;
#[path = "analysis_process_conformance/signal.rs"]
mod signal;
#[path = "analysis_process_conformance/thermal.rs"]
mod thermal;
#[path = "analysis_process_conformance/thermal_prestress.rs"]
mod thermal_prestress;
#[path = "analysis_process_conformance/thermal_radiation.rs"]
mod thermal_radiation;
#[path = "analysis_process_conformance/transient.rs"]
mod transient;
use publication_chain::{success_response, PublicationChainDriver};

fn main() {
    let mode = std::env::args_os().nth(1);
    if mode.as_deref()
        == Some(std::ffi::OsStr::new(
            runmat_process_host::HiddenMode::MeshingWorker.marker(),
        ))
    {
        let root = std::env::var_os(runmat_execution_runner_native::NATIVE_OBJECT_STORE_ROOT_ENV)
            .expect("meshing worker object-store root");
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(runmat_execution_runner_native::run_meshing_worker_stdio(
                Arc::new(runmat_execution_runner_native::native_meshing_kernel_dispatcher()),
                std::path::Path::new(&root),
                runmat_execution_runner_native::NativeMeshingHostLimits::default(),
            ))
            .unwrap();
        return;
    }
    if mode.as_deref() == Some(std::ffi::OsStr::new("--child"))
        || mode.as_deref()
            == Some(std::ffi::OsStr::new(
                runmat_process_host::HiddenMode::ExecutionWorker.marker(),
            ))
    {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(runmat_execution_runner_native::run_worker_stdio())
            .unwrap();
        return;
    }
    if mode.as_deref()
        == Some(std::ffi::OsStr::new(
            runmat_process_host::HiddenMode::ExecutionDriver.marker(),
        ))
    {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(run_durable_driver());
        return;
    }
    let mut runtime = if std::env::args_os().any(|argument| {
        argument == "--fluid-authored-only"
            || argument == "--nonlinear-authored-only"
            || argument == "--remote-families-only"
    }) {
        tokio::runtime::Builder::new_multi_thread()
    } else {
        tokio::runtime::Builder::new_current_thread()
    };
    let admission = fea_admission_support::context();
    runtime
        .enable_all()
        .build()
        .unwrap()
        .block_on(admission.run_future(parent()));
}

async fn parent() {
    // This standalone process is the invocation host, as CLI bootstrap is in
    // the product. Capture the same finite host publication allowance rather
    // than accidentally testing numerical completion under an embedded default.
    runmat_runtime::analysis::configure_fea_runtime(runmat_runtime::analysis::FeaRuntimeConfig {
        publication_staging_bytes: runmat_execution_runner_native::publication_staging_bytes(),
        ..Default::default()
    })
    .unwrap();
    if std::env::args_os().any(|argument| argument == "--remote-families-only") {
        #[cfg(feature = "occt-native")]
        remote_families::run().await;
        #[cfg(not(feature = "occt-native"))]
        panic!("remote family conformance requires occt-native");
        #[cfg(feature = "occt-native")]
        return;
    }
    if std::env::args_os().any(|argument| argument == "--fluid-massflow-only") {
        fluid_mass_flow::run().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--fluid-fourier-only") {
        fluid_fourier::run().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--fluid-transient-only") {
        fluid_transient::run().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--nonlinear-authored-only") {
        #[cfg(feature = "occt-native")]
        nonlinear_authored::run().await;
        #[cfg(not(feature = "occt-native"))]
        panic!("authored J2 conformance requires occt-native");
        #[cfg(feature = "occt-native")]
        return;
    }
    if std::env::args_os().any(|argument| argument == "--static-recovery-only") {
        static_recovery::run().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--nonlinear-only") {
        nonlinear::run().await;
        return;
    }
    #[cfg(feature = "occt-native")]
    if std::env::args_os().any(|argument| argument == "--assessment-publication-only") {
        assessment_publication::run().await;
        return;
    }
    #[cfg(feature = "occt-native")]
    if std::env::args_os().any(|argument| argument == "--fluid-authored-only") {
        fluid_authored::run().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--assessment-only") {
        assessment::run().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--maxwell-authored-only") {
        #[cfg(feature = "occt-native")]
        maxwell_authored::run().await;
        #[cfg(not(feature = "occt-native"))]
        panic!("Maxwell authored conformance requires occt-native");
        #[cfg(feature = "occt-native")]
        return;
    }
    if std::env::args_os().any(|argument| argument == "--thermal-radiation-fine-only") {
        thermal_radiation::run_fine().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--thermal-radiation-only") {
        thermal_radiation::run().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--electrothermal-only") {
        electrothermal::run().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--magnetic-authored-only") {
        #[cfg(feature = "occt-native")]
        magnetic_authored::run().await;
        #[cfg(not(feature = "occt-native"))]
        panic!("magnetic authored conformance requires occt-native");
        #[cfg(feature = "occt-native")]
        return;
    }
    if std::env::args_os().any(|argument| argument == "--magnetic-only") {
        magnetic::run().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--signal-layout-only") {
        signal::run().await;
        return;
    }
    if std::env::args_os().any(|argument| argument == "--acoustic-signal-only") {
        signal::run_acoustic().await;
        return;
    }
    #[cfg(feature = "occt-native")]
    magnetic_authored::run().await;
    #[cfg(feature = "occt-native")]
    maxwell_authored::run().await;
    magnetic::run().await;
    let directory = tempfile::tempdir().unwrap();
    let (host, program, input_objects, resource_plan) =
        fixture(revision(), "analysis-process-conformance");
    let request = program.request.clone();
    let (second_host, second_program, _second_inputs, second_resource_plan) = fixture_for_study(
        revision(),
        "analysis-process-conformance",
        digest("authored-study-second"),
    );
    let second_request = second_program.request.clone();
    let (mixed_host, mixed_program, _mixed_inputs, mixed_resource_plan) =
        mixed_fixture(revision(), "analysis-process-conformance");
    let mixed_request = mixed_program.request.clone();
    let (explicit_host, explicit_program, explicit_inputs, explicit_resource_plan) =
        explicit_fixture(revision(), "analysis-process-conformance");
    let explicit_request = explicit_program.request.clone();
    let mut config = NativeExecutionConfig::for_current_executable().unwrap();
    config.executable = std::env::current_exe().unwrap();
    config.worker_arguments = vec!["--child".into()];
    config.max_workers = 2;
    config.store_root = directory.path().join("native-analysis-session");
    config.worker_capabilities = capabilities();
    let session = NativeProgramSession::new(config).unwrap();
    thermal::run(&session).await;
    thermal_prestress::run_local().await;
    let mut store = session.object_store();
    program.write_to(&mut store).unwrap();
    second_program.write_to(&mut store).unwrap();
    mixed_program.write_to(&mut store).unwrap();
    explicit_program.write_to(&mut store).unwrap();
    let submissions = vec![
        (
            request.clone(),
            analysis_submission(
                session.scope_id(),
                session.pool_id(),
                &host,
                &request,
                Some(&resource_plan),
                resource_plan.request.clone(),
            ),
        ),
        (
            second_request.clone(),
            analysis_submission(
                session.scope_id(),
                session.pool_id(),
                &second_host,
                &second_request,
                Some(&second_resource_plan),
                second_resource_plan.request.clone(),
            ),
        ),
        (
            mixed_request.clone(),
            analysis_submission(
                session.scope_id(),
                session.pool_id(),
                &mixed_host,
                &mixed_request,
                Some(&mixed_resource_plan),
                mixed_resource_plan.request.clone(),
            ),
        ),
        (
            explicit_request.clone(),
            analysis_submission(
                session.scope_id(),
                session.pool_id(),
                &explicit_host,
                &explicit_request,
                Some(&explicit_resource_plan),
                explicit_resource_plan.request.clone(),
            ),
        ),
    ];
    let mut tasks = session.submit_graph(submissions).unwrap();
    let explicit_task = tasks.pop().unwrap();
    let mixed_task = tasks.pop().unwrap();
    let second_task = tasks.pop().unwrap();
    let task = tasks.pop().unwrap();
    let process_result = wait_for_native_task(&task).await;
    let second_result = wait_for_native_task(&second_task).await;
    let mixed_result = wait_for_native_task(&mixed_task).await;
    let explicit_result = wait_for_native_task(&explicit_task).await;
    let process_payload = import_result(&store, success_response(&process_result), &host);
    assert_eq!(
        import_result(&store, success_response(&second_result), &second_host),
        process_payload
    );

    let mut inline_store = store.clone();
    let inline = execute_analysis_program_request(
        &request,
        &mut inline_store,
        &NativeAnalysisKernel,
        NativeAnalysisHostLimits::default(),
    );
    if !matches!(inline, ProgramExecutionResponse::ExternalizedSuccess { .. }) {
        panic!("inline analysis execution failed: {inline:?}");
    }
    let inline_payload = import_result(&inline_store, inline, &host);
    assert_eq!(process_payload, inline_payload);

    let decoded = StructuralLinearWorkloadResult::from_canonical_bytes(&process_payload).unwrap();
    assert!(!decoded.run.fields_are_empty());
    assert!(decoded
        .run
        .diagnostics
        .iter()
        .all(|diagnostic| !diagnostic.message.contains("artifact_path")));
    let mixed_payload = import_result(&store, success_response(&mixed_result), &mixed_host);
    let mixed_decoded =
        StructuralLinearWorkloadResult::from_canonical_bytes(&mixed_payload).unwrap();
    assert!(matches!(
        mixed_decoded.constraint_result.topology,
        CanonicalStructuralTopologyIdentity::ConformalMixed { .. }
    ));
    assert_eq!(
        mixed_decoded
            .run
            .field(FEA_FIELD_STRUCTURAL_STRESS)
            .expect("mixed result retains solid stress")
            .shape,
        vec![1, 6]
    );
    assert_eq!(
        mixed_decoded
            .run
            .field(FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE)
            .expect("mixed result retains beam axial force")
            .shape,
        vec![1]
    );
    assert_eq!(
        mixed_decoded
            .run
            .field(FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT)
            .expect("mixed result retains shell bending moment")
            .shape,
        vec![1, 3]
    );
    let mut mixed_inline_store = store.clone();
    let mixed_inline = execute_analysis_program_request(
        &mixed_request,
        &mut mixed_inline_store,
        &NativeAnalysisKernel,
        NativeAnalysisHostLimits::default(),
    );
    assert_eq!(
        import_result(&mixed_inline_store, mixed_inline, &mixed_host),
        mixed_payload
    );
    let explicit_payload =
        import_result(&store, success_response(&explicit_result), &explicit_host);
    let explicit_decoded =
        StructuralLinearWorkloadResult::from_canonical_bytes(&explicit_payload).unwrap();
    assert!(matches!(
        explicit_decoded.constraint_result.topology,
        CanonicalStructuralTopologyIdentity::Explicit { .. }
    ));
    let mut explicit_inline_store = store.clone();
    let explicit_inline = execute_analysis_program_request(
        &explicit_request,
        &mut explicit_inline_store,
        &NativeAnalysisKernel,
        NativeAnalysisHostLimits::default(),
    );
    assert_eq!(
        import_result(&explicit_inline_store, explicit_inline, &explicit_host),
        explicit_payload
    );

    let prepared_solve = PreparedStructuralLinearProgram {
        host: host.clone(),
        program: program.clone(),
        inputs: input_objects,
        resource_plan,
    };
    let solve_resource_usage = resource_usage(&store, &prepared_solve, &process_result);
    let prepared_explicit = PreparedStructuralLinearProgram {
        host: explicit_host.clone(),
        program: explicit_program,
        inputs: explicit_inputs,
        resource_plan: explicit_resource_plan,
    };
    let explicit_resource_usage = resource_usage(&store, &prepared_explicit, &explicit_result);
    let mut publication_driver = publication_chain::ProcessDriver {
        session: &session,
        store: &mut store,
    };
    let published = publication_chain::execute_publication_chain(
        &mut publication_driver,
        &prepared_solve,
        success_response(&process_result),
        &solve_resource_usage,
        revision(),
    )
    .await;
    assert_eq!(
        published.source_result,
        decoded_result_identity(&host, &process_payload)
    );
    let explicit_published = publication_chain::execute_publication_chain(
        &mut publication_driver,
        &prepared_explicit,
        success_response(&explicit_result),
        &explicit_resource_usage,
        revision(),
    )
    .await;
    assert_eq!(
        explicit_published.source_result,
        decoded_result_identity(&explicit_host, &explicit_payload)
    );

    let mut sequence_config = NativeExecutionConfig::for_current_executable().unwrap();
    sequence_config.executable = std::env::current_exe().unwrap();
    sequence_config.worker_arguments = vec!["--child".into()];
    sequence_config.max_workers = 2;
    sequence_config.store_root = directory.path().join("native-analysis-sequence");
    let sequence = execute_structural_sequence_locally(
        sequence_config,
        sequence_preparation(revision(), "local-analysis-sequence"),
        sequence_policy(),
    )
    .await
    .unwrap();
    assert_sequence(&sequence);

    let mut frequency_config = NativeExecutionConfig::for_current_executable().unwrap();
    frequency_config.executable = std::env::current_exe().unwrap();
    frequency_config.worker_arguments = vec!["--child".into()];
    frequency_config.max_workers = 2;
    frequency_config.store_root = directory.path().join("native-analysis-frequency");
    let frequency = execute_structural_frequency_locally(
        frequency_config,
        frequency_preparation(revision(), "local-analysis-frequency"),
        sequence_policy(),
    )
    .await
    .unwrap();
    assert_frequency(&frequency, "local-analysis-frequency");
    let expected_frequency_publication = frequency.publication.clone();
    let expected_frequency_result = frequency.result.result.clone();

    let mut cancelled_sequence_config = NativeExecutionConfig::for_current_executable().unwrap();
    cancelled_sequence_config.executable = std::env::current_exe().unwrap();
    cancelled_sequence_config.worker_arguments = vec!["--child".into()];
    cancelled_sequence_config.max_workers = 2;
    cancelled_sequence_config.store_root = directory.path().join("cancelled-analysis-sequence");
    // Every task now has an authoritative plan: fallback resources no longer control
    // state publication. Set the authored solve budget to exercise deadline cancellation.
    let mut cancellation_preparation =
        sequence_preparation(revision(), "cancelled-analysis-sequence");
    for workload in cancellation_preparation.workloads.values_mut() {
        for step in &mut workload.model.sequence.steps {
            step.accuracy.resources.maximum_wall_time_milliseconds = 1;
        }
    }
    let cancellation = execute_structural_sequence_locally(
        cancelled_sequence_config,
        cancellation_preparation,
        sequence_policy(),
    )
    .await;
    let Err(cancellation) = cancellation else {
        panic!("one-millisecond structural sequence unexpectedly completed");
    };
    assert!(
        matches!(
            cancellation,
            NativeExecutionError::TaskCancelled {
                cancellation: runmat_execution_runner::TaskCancellation {
                    cause: runmat_execution_runner::TaskCancellationCause::WallTime { .. },
                    ..
                },
                quiescence: runmat_execution_runner_native::NativeTaskQuiescence::Confirmed,
                ..
            }
        ),
        "unexpected structural sequence cancellation error: {cancellation}"
    );

    let mut user_cancelled_config = NativeExecutionConfig::for_current_executable().unwrap();
    user_cancelled_config.executable = std::env::current_exe().unwrap();
    user_cancelled_config.worker_arguments = vec!["--child".into()];
    user_cancelled_config.max_workers = 2;
    user_cancelled_config.store_root = directory.path().join("user-cancelled-analysis-sequence");
    let interrupt = runmat_execution::cancellation::CancellationSource::new();
    interrupt.cancel(runmat_execution::CancellationReason::User);
    let interrupt_guard = runmat_runtime::interrupt::replace_interrupt(Some(interrupt.token()));
    let cancellation = execute_structural_sequence_locally(
        user_cancelled_config,
        sequence_preparation(revision(), "user-cancelled-analysis-sequence"),
        sequence_policy(),
    )
    .await;
    assert!(matches!(cancellation, Err(NativeExecutionError::Cancelled)));
    drop(interrupt_guard);

    remote_conformance(
        process_payload,
        mixed_payload.clone(),
        explicit_payload.clone(),
        published.clone(),
        explicit_published.clone(),
        expected_frequency_publication,
        expected_frequency_result,
    )
    .await;
    durable_conformance(
        inline_payload,
        mixed_payload,
        explicit_payload,
        published,
        explicit_published,
        solve_resource_usage,
        explicit_resource_usage,
    )
    .await;
    signal::run().await;
}

async fn wait_for_native_task(task: &NativeProgramTask) -> runmat_execution_runner::AttemptSuccess {
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            if let Some(result) = task.try_result() {
                break result.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap()
}

fn decoded_result_identity(
    host: &AnalysisHostWorkload,
    payload: &[u8],
) -> runmat_analysis_fea::execution::SolverResultIdentity {
    runmat_analysis_fea::execution::SolverResultIdentity::new(
        host.workload.identity,
        Digest::sha256(payload),
    )
    .unwrap()
}

fn resource_usage(
    store: &impl CacheImport,
    solve: &PreparedStructuralLinearProgram,
    success: &runmat_execution_runner::AttemptSuccess,
) -> PreparedResourceUsage {
    let result = import_structural_linear_completion(
        store,
        success_response(success),
        solve,
        ObjectInventoryLimits::default(),
    )
    .unwrap();
    prepare_structural_resource_usage(
        solve,
        &result,
        success,
        DEFAULT_ANALYSIS_CHUNK_BYTES,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
}

fn assert_publication_semantics(
    actual: &runmat_analysis_fea::results::publication::PublishedAnalysisResult,
    expected: &runmat_analysis_fea::results::publication::PublishedAnalysisResult,
) {
    // Publication identity intentionally binds measured attempt accounting. Equivalent work run
    // through another execution driver must retain its analysis sources and gate decision, while
    // naming the resource-usage evidence produced by that execution rather than an earlier run.
    assert_eq!(actual.schema_version, expected.schema_version);
    assert_eq!(actual.producer, expected.producer);
    assert_eq!(actual.source_result, expected.source_result);
    assert_eq!(actual.source_observations, expected.source_observations);
    assert_eq!(actual.source_accuracy, expected.source_accuracy);
    assert_eq!(actual.source_resource_plan, expected.source_resource_plan);
    assert_eq!(actual.accuracy_status, expected.accuracy_status);
    assert_eq!(actual.gate, expected.gate);
}

async fn run_durable_driver() {
    use runmat_execution_runner_native::supervisor::{self, BatchDriverInvocation};

    let is_success = |response: &ProgramExecutionResponse| {
        matches!(
            response,
            ProgramExecutionResponse::Success { .. }
                | ProgramExecutionResponse::ExternalizedSuccess { .. }
                | ProgramExecutionResponse::GraphSuccess { .. }
        )
    };
    let invocation = runmat_execution_runner_native::supervisor::prepare_batch_driver().unwrap();
    let succeeded = match invocation {
        BatchDriverInvocation::Program {
            attempt,
            submission,
            lease,
            ..
        } => is_success(
            &supervisor::execute_and_complete_program_batch(*attempt, *submission, *lease)
                .await
                .unwrap()
                .response,
        ),
        BatchDriverInvocation::GraphTask {
            attempt,
            submission,
            lease,
            ..
        } => is_success(
            &supervisor::execute_and_complete_graph_batch(*attempt, *submission, *lease)
                .await
                .unwrap()
                .response,
        ),
        BatchDriverInvocation::Graph {
            job_directory,
            job_id,
            submission,
        } => {
            let response = supervisor::execute_graph_batch(*submission, job_id).await;
            let succeeded = is_success(&response);
            supervisor::complete_batch_driver_with_response(&job_directory, response).unwrap();
            succeeded
        }
        BatchDriverInvocation::Script { .. } => {
            panic!("analysis conformance driver does not execute script jobs")
        }
    };
    if !succeeded {
        std::process::exit(1);
    }
}

async fn durable_conformance(
    expected_payload: Vec<u8>,
    expected_mixed_payload: Vec<u8>,
    expected_explicit_payload: Vec<u8>,
    expected_publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
    expected_explicit_publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
    resource_usage: PreparedResourceUsage,
    explicit_resource_usage: PreparedResourceUsage,
) {
    let directory = tempfile::tempdir().unwrap();
    let mut driver = publication_chain::DurableDriver::open({
        let mut execution = NativeExecutionConfig::for_current_executable().unwrap();
        execution.worker_arguments = vec!["--child".into()];
        execution.max_workers = 2;
        execution.worker_capabilities.extend(capabilities());
        execution.max_object_bytes = 64 * 1024 * 1024;
        runmat_execution_runner_native::supervisor::LocalSupervisorConfig::new(
            execution,
            runmat_execution_runner_native::supervisor::SupervisorPaths::new(
                directory.path().join("durable-analysis"),
            )
            .unwrap(),
        )
        .unwrap()
    });
    thermal::run_driver(&mut driver, "durable-analysis", revision()).await;
    let (host, program, input_objects, resource_plan) = fixture(revision(), "durable-analysis");
    let solve = PreparedStructuralLinearProgram {
        host,
        program,
        inputs: input_objects,
        resource_plan,
    };
    let solve_host = solve.host.clone();
    let solve_program = solve.program.clone();
    let solve_success = driver
        .execute(&solve_host, &solve_program, Some(&solve.resource_plan))
        .await;
    assert_eq!(
        import_result(driver.store(), solve_success.response.clone(), &solve.host,),
        expected_payload
    );
    let publication = publication_chain::execute_publication_chain(
        &mut driver,
        &solve,
        solve_success.response,
        &resource_usage,
        revision(),
    )
    .await;
    assert_publication_semantics(&publication, &expected_publication);
    assert_ne!(
        publication.source_resource_usage, expected_publication.source_resource_usage,
        "process and durable publications must retain their own measured execution evidence"
    );

    let (mixed_host, mixed_program, _, mixed_resource_plan) =
        mixed_fixture(revision(), "durable-analysis");
    let mixed_success = driver
        .execute(&mixed_host, &mixed_program, Some(&mixed_resource_plan))
        .await;
    assert_eq!(
        import_result(driver.store(), mixed_success.response, &mixed_host),
        expected_mixed_payload
    );

    let (explicit_host, explicit_program, explicit_inputs, explicit_resource_plan) =
        explicit_fixture(revision(), "durable-analysis");
    let explicit = PreparedStructuralLinearProgram {
        host: explicit_host,
        program: explicit_program,
        inputs: explicit_inputs,
        resource_plan: explicit_resource_plan,
    };
    let explicit_success = driver
        .execute(
            &explicit.host,
            &explicit.program,
            Some(&explicit.resource_plan),
        )
        .await;
    assert_eq!(
        import_result(
            driver.store(),
            explicit_success.response.clone(),
            &explicit.host,
        ),
        expected_explicit_payload
    );
    let publication = publication_chain::execute_publication_chain(
        &mut driver,
        &explicit,
        explicit_success.response,
        &explicit_resource_usage,
        revision(),
    )
    .await;
    assert_publication_semantics(&publication, &expected_explicit_publication);
    assert_ne!(
        publication.source_resource_usage, expected_explicit_publication.source_resource_usage,
        "process and durable publications must retain their own measured execution evidence"
    );
}

async fn remote_conformance(
    expected_payload: Vec<u8>,
    expected_mixed_payload: Vec<u8>,
    expected_explicit_payload: Vec<u8>,
    expected_publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
    expected_explicit_publication: runmat_analysis_fea::results::publication::PublishedAnalysisResult,
    expected_frequency_publication: runmat_analysis_fea::execution::StructuralFrequencyPublicationManifest,
    expected_frequency_result: runmat_analysis_fea::execution::StructuralFrequencySolverResult,
) {
    let project_root = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(project_root.path().join("src")).unwrap();
    let project_document = "[package]\nname = \"remote-analysis\"\n[sources]\nroots = [\"src\"]\n[runtime.language]\ncompat = \"matlab\"\n";
    std::fs::write(project_root.path().join("runmat.toml"), project_document).unwrap();
    let project = runmat_package::build_frozen_project(
        &project_root.path().join("runmat.toml"),
        BTreeSet::new(),
    )
    .unwrap();
    let project_identity = project.revision();
    let revision = ProgramRevision::new(
        Digest::from_bytes(*project_identity.graph_digest.bytes()),
        Digest::from_bytes(*project_identity.source_revision.bytes()),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap();
    let (host, program, input_objects, resource_plan) =
        fixture(revision.clone(), "remote-analysis");
    let request = program.request.clone();
    let (second_host, second_program, _second_inputs, second_resource_plan) = fixture_for_study(
        revision.clone(),
        "remote-analysis",
        digest("remote-authored-study-second"),
    );
    let second_request = second_program.request.clone();
    let (mixed_host, mixed_program, _mixed_inputs, mixed_resource_plan) =
        mixed_fixture(revision.clone(), "remote-analysis");
    let mixed_request = mixed_program.request.clone();
    let (explicit_host, explicit_program, explicit_inputs, explicit_resource_plan) =
        explicit_fixture(revision.clone(), "remote-analysis");
    let explicit_request = explicit_program.request.clone();
    let bundle = ExecutionBundleBuilder::native(&project, revision.clone())
        .unwrap()
        .with_project_runtime_policy_object(
            runmat_core::cluster_settings::capture_project_policy(
                &runmat_config::runtime::ClusterRuntimePolicy::from_project_document(
                    project_document,
                    runmat_config::document::RunmatConfigFormat::Toml,
                )
                .unwrap(),
            )
            .unwrap(),
        )
        .unwrap()
        .with_compiled_package_closure()
        .with_materialized_program(
            request.recipe.clone(),
            ExecutableForm::AnalysisWorkload,
            request.artifact.executable_bytes.clone(),
        )
        .build()
        .unwrap();
    let mut bundle_bytes = Vec::new();
    write_bundle(&bundle, &mut bundle_bytes, ArchiveLimits::default()).unwrap();

    let mut modal_local_config = NativeExecutionConfig::for_current_executable().unwrap();
    modal_local_config.executable = std::env::current_exe().unwrap();
    modal_local_config.worker_arguments = vec!["--child".into()];
    modal_local_config.max_workers = 2;
    modal_local_config.store_root = project_root.path().join("local-modal-transient");
    let mut modal_local_publisher = transient::RecordingPublisher {
        closures: Vec::new(),
    };
    let modal_local = execute_structural_modal_transient_locally(
        modal_local_config,
        modal_transient::preparation(
            revision.clone(),
            "remote-analysis",
            "modal-transient-local-encryption",
        ),
        sequence_policy(),
        None,
        &mut modal_local_publisher,
    )
    .await
    .unwrap();
    assert_eq!(modal_local_publisher.closures.len(), 2);

    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(vec!["runmat.execution".into()]).unwrap();
    let certificate_der = cert.der().to_vec();
    let signing_key_der = signing_key.serialize_der();
    let first_listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der.clone()],
        signing_key_der.clone(),
        FrameLimits::default(),
    )
    .unwrap();
    let second_listener = QuicOverlayListener::bind(
        SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        vec![certificate_der.clone()],
        signing_key_der,
        FrameLimits::default(),
    )
    .unwrap();
    let first_authority = first_listener.local_addr().unwrap();
    let second_authority = second_listener.local_addr().unwrap();
    let scope_id = runmat_execution::ExecutionScopeId::derive(&[b"remote-analysis-scope"]);
    let pool_id = runmat_execution::PoolId::derive(&[b"remote-analysis-pool"]);
    let first_worker = WorkerSpec {
        host: phase_execution_support::worker_host_inventory(),
        id: runmat_execution::identity::WorkerId::derive(&[b"remote-analysis-worker-a"]),
        pool_id,
        resources: inventory(),
    };
    let second_worker = WorkerSpec {
        host: phase_execution_support::worker_host_inventory(),
        id: runmat_execution::identity::WorkerId::derive(&[b"remote-analysis-worker-b"]),
        pool_id,
        resources: inventory(),
    };
    let run_key =
        runmat_execution_artifact::encryption::RunKeyMaterial::from_entropy([29; 32]).unwrap();
    let first_server = run_remote_worker_quic(RemoteWorkerQuicRequest {
        listener: first_listener,
        run_identity: "remote-analysis".into(),
        worker: first_worker.clone(),
        driver_fence: 41,
        session_id: [31; 16],
        run_key: run_key.clone(),
        limits: FrameLimits::default(),
    });
    let second_server = run_remote_worker_quic(RemoteWorkerQuicRequest {
        listener: second_listener,
        run_identity: "remote-analysis".into(),
        worker: second_worker.clone(),
        driver_fence: 41,
        session_id: [32; 16],
        run_key: run_key.clone(),
        limits: FrameLimits::default(),
    });
    let client = async {
        let first_channel = QuicRemoteWorkerChannel::connect(
            RemoteWorkerChannelConfig {
                run_identity: "remote-analysis".into(),
                node_identity: "remote-analysis-node-a".into(),
                worker: first_worker,
                driver_fence: 41,
                session_id: [31; 16],
                run_key: run_key.clone(),
                limits: FrameLimits::default(),
            },
            &PinnedQuicEndpoint {
                authority: first_authority,
                server_name: "runmat.execution".into(),
                certificate_der: certificate_der.clone(),
            },
        )
        .await
        .unwrap();
        let second_channel = QuicRemoteWorkerChannel::connect(
            RemoteWorkerChannelConfig {
                run_identity: "remote-analysis".into(),
                node_identity: "remote-analysis-node-b".into(),
                worker: second_worker,
                driver_fence: 41,
                session_id: [32; 16],
                run_key,
                limits: FrameLimits::default(),
            },
            &PinnedQuicEndpoint {
                authority: second_authority,
                server_name: "runmat.execution".into(),
                certificate_der,
            },
        )
        .await
        .unwrap();
        let pool = RemotePoolDriver::new_with_value_scope(
            scope_id,
            PoolSpec {
                id: pool_id,
                min_workers: 2,
                max_workers: 2,
                max_in_flight: 2,
                resource_limit: scaled_inventory(2),
            },
            41,
            bundle_bytes.clone(),
            "remote-analysis",
        )
        .unwrap();
        pool.add_worker(first_channel.clone()).await.unwrap();
        pool.add_worker(second_channel.clone()).await.unwrap();
        for (reference, bytes) in program.objects() {
            pool.register_execution_object(reference.clone(), bytes.to_vec())
                .unwrap();
        }
        for (reference, bytes) in second_program.objects() {
            pool.register_execution_object(reference.clone(), bytes.to_vec())
                .unwrap();
        }
        for (reference, bytes) in mixed_program.objects() {
            pool.register_execution_object(reference.clone(), bytes.to_vec())
                .unwrap();
        }
        for (reference, bytes) in explicit_program.objects() {
            pool.register_execution_object(reference.clone(), bytes.to_vec())
                .unwrap();
        }
        let mut completions = pool
            .submit_graph(vec![
                (
                    request.clone(),
                    analysis_submission(
                        scope_id,
                        pool_id,
                        &host,
                        &request,
                        Some(&resource_plan),
                        resource_plan.request.clone(),
                    ),
                ),
                (
                    second_request.clone(),
                    analysis_submission(
                        scope_id,
                        pool_id,
                        &second_host,
                        &second_request,
                        Some(&second_resource_plan),
                        second_resource_plan.request.clone(),
                    ),
                ),
                (
                    mixed_request.clone(),
                    analysis_submission(
                        scope_id,
                        pool_id,
                        &mixed_host,
                        &mixed_request,
                        Some(&mixed_resource_plan),
                        mixed_resource_plan.request.clone(),
                    ),
                ),
                (
                    explicit_request.clone(),
                    analysis_submission(
                        scope_id,
                        pool_id,
                        &explicit_host,
                        &explicit_request,
                        Some(&explicit_resource_plan),
                        explicit_resource_plan.request.clone(),
                    ),
                ),
            ])
            .unwrap();
        let explicit_completion = completions.pop().unwrap();
        let mixed_completion = completions.pop().unwrap();
        let second_completion = completions.pop().unwrap();
        let completion = completions.pop().unwrap();
        let success = tokio::time::timeout(Duration::from_secs(15), completion.wait())
            .await
            .expect("remote analysis completion timeout")
            .unwrap();
        let second_success =
            tokio::time::timeout(Duration::from_secs(15), second_completion.wait())
                .await
                .expect("second remote analysis completion timeout")
                .unwrap();
        let mixed_success = tokio::time::timeout(Duration::from_secs(15), mixed_completion.wait())
            .await
            .expect("remote mixed analysis completion timeout")
            .unwrap();
        let explicit_success =
            tokio::time::timeout(Duration::from_secs(15), explicit_completion.wait())
                .await
                .expect("remote explicit analysis completion timeout")
                .unwrap();
        let mut result_store = publication_chain::MemoryStore::default();
        for reference in success
            .result_objects()
            .iter()
            .chain(second_success.result_objects())
            .chain(mixed_success.result_objects())
            .chain(explicit_success.result_objects())
        {
            result_store.0.insert(
                reference.logical_digest,
                pool.execution_object(reference)
                    .unwrap()
                    .expect("verified remote analysis result")
                    .to_vec(),
            );
        }
        let payload = import_result(&result_store, success_response(&success), &host);
        assert_eq!(payload, expected_payload);
        assert_eq!(
            import_result(
                &result_store,
                success_response(&second_success),
                &second_host,
            ),
            expected_payload
        );
        assert_eq!(
            import_result(&result_store, success_response(&mixed_success), &mixed_host,),
            expected_mixed_payload
        );
        assert_eq!(
            import_result(
                &result_store,
                success_response(&explicit_success),
                &explicit_host,
            ),
            expected_explicit_payload
        );
        let solve = PreparedStructuralLinearProgram {
            host: host.clone(),
            program: program.clone(),
            inputs: input_objects,
            resource_plan,
        };
        let solve_resource_usage = resource_usage(&result_store, &solve, &success);
        let explicit = PreparedStructuralLinearProgram {
            host: explicit_host.clone(),
            program: explicit_program.clone(),
            inputs: explicit_inputs,
            resource_plan: explicit_resource_plan,
        };
        let explicit_resource_usage = resource_usage(&result_store, &explicit, &explicit_success);
        let mut publication_driver = publication_chain::RemoteDriver {
            pool: Arc::clone(&pool),
            store: result_store,
            scope_id,
            pool_id,
        };
        thermal::run_driver(&mut publication_driver, "remote-analysis", revision.clone()).await;
        thermal_prestress::run_remote(Arc::clone(&pool), revision.clone()).await;
        let publication = publication_chain::execute_publication_chain(
            &mut publication_driver,
            &solve,
            success_response(&success),
            &solve_resource_usage,
            revision.clone(),
        )
        .await;
        assert_publication_semantics(&publication, &expected_publication);
        assert_ne!(
            publication.source_resource_usage, expected_publication.source_resource_usage,
            "local and remote publications must retain their own measured execution evidence"
        );
        let publication = publication_chain::execute_publication_chain(
            &mut publication_driver,
            &explicit,
            success_response(&explicit_success),
            &explicit_resource_usage,
            revision.clone(),
        )
        .await;
        assert_publication_semantics(&publication, &expected_explicit_publication);
        assert_ne!(
            publication.source_resource_usage, expected_explicit_publication.source_resource_usage,
            "local and remote publications must retain their own measured execution evidence"
        );
        let sequence = execute_structural_sequence_remotely(
            Arc::clone(&pool),
            sequence_preparation(revision.clone(), "remote-analysis"),
            sequence_policy(),
        )
        .await
        .unwrap();
        assert_sequence(&sequence);
        let frequency = execute_structural_frequency_remotely(
            Arc::clone(&pool),
            frequency_preparation(revision.clone(), "remote-analysis"),
            sequence_policy(),
        )
        .await
        .unwrap();
        assert_frequency(&frequency, "remote-analysis");
        assert_eq!(
            frequency.publication.workload(),
            expected_frequency_publication.workload()
        );
        assert_eq!(
            frequency.publication.result(),
            expected_frequency_publication.result()
        );
        assert_eq!(
            frequency.publication.receipt(),
            expected_frequency_publication.receipt()
        );
        assert_eq!(
            frequency.publication.samples(),
            expected_frequency_publication.samples()
        );
        assert_eq!(
            frequency.publication.histories(),
            expected_frequency_publication.histories()
        );
        assert_ne!(
            frequency.publication.resource_usage(),
            expected_frequency_publication.resource_usage(),
            "local and remote publications must retain their own measured execution evidence"
        );
        assert_eq!(frequency.result.result, expected_frequency_result);

        let mut modal_remote_publisher = transient::RecordingPublisher {
            closures: Vec::new(),
        };
        let modal_remote = execute_structural_modal_transient_remotely(
            Arc::clone(&pool),
            modal_transient::preparation(
                revision.clone(),
                "remote-analysis",
                "modal-transient-remote-encryption",
            ),
            sequence_policy(),
            None,
            &mut modal_remote_publisher,
        )
        .await
        .unwrap();
        assert_eq!(modal_remote_publisher.closures.len(), 2);
        modal_transient::assert_numerical_equality(&modal_remote, &modal_local);

        let modal_replacement_pool = RemotePoolDriver::new_with_value_scope(
            runmat_execution::ExecutionScopeId::derive(&[
                b"remote-modal-transient-replacement-scope",
            ]),
            PoolSpec {
                id: pool_id,
                min_workers: 1,
                max_workers: 2,
                max_in_flight: 1,
                resource_limit: scaled_inventory(2),
            },
            41,
            bundle_bytes.clone(),
            "remote-analysis",
        )
        .unwrap();
        modal_replacement_pool
            .add_worker(first_channel.clone())
            .await
            .unwrap();
        let replacement_preparation = modal_transient::preparation(
            revision.clone(),
            "remote-analysis",
            "modal-transient-replacement-encryption",
        );
        let mut modal_interrupted_publisher = transient::FailAfterFirstPublisher {
            closures: Vec::new(),
        };
        execute_structural_modal_transient_remotely(
            Arc::clone(&modal_replacement_pool),
            replacement_preparation,
            sequence_policy(),
            None,
            &mut modal_interrupted_publisher,
        )
        .await
        .expect_err("the injected loss must interrupt modal-transient execution");
        assert_eq!(modal_interrupted_publisher.closures.len(), 1);
        let modal_recovery = modal_interrupted_publisher.closures.pop().unwrap();
        let modal_first_worker_id =
            runmat_execution::identity::WorkerId::derive(&[b"remote-analysis-worker-a"]);
        modal_replacement_pool
            .remove_worker(modal_first_worker_id, true)
            .await
            .unwrap();
        modal_replacement_pool
            .add_worker(second_channel.clone())
            .await
            .unwrap();
        let mut modal_replacement_publisher = transient::RecordingPublisher {
            closures: Vec::new(),
        };
        let modal_restarted = execute_structural_modal_transient_remotely(
            Arc::clone(&modal_replacement_pool),
            modal_transient::preparation(
                revision.clone(),
                "remote-analysis",
                "modal-transient-replacement-encryption",
            ),
            sequence_policy(),
            Some(&modal_recovery),
            &mut modal_replacement_publisher,
        )
        .await
        .unwrap();
        assert_eq!(modal_replacement_publisher.closures.len(), 1);
        modal_transient::assert_numerical_equality(&modal_restarted, &modal_remote);
        assert_eq!(
            modal_replacement_pool
                .snapshot()
                .attempts
                .values()
                .map(|attempt| attempt.request.worker_id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                modal_first_worker_id,
                runmat_execution::identity::WorkerId::derive(&[b"remote-analysis-worker-b"]),
            ])
        );
        modal_replacement_pool
            .remove_worker(
                runmat_execution::identity::WorkerId::derive(&[b"remote-analysis-worker-b"]),
                true,
            )
            .await
            .unwrap();

        let mut transient_publisher = transient::RecordingPublisher {
            closures: Vec::new(),
        };
        let transient_result = execute_structural_transient_remotely(
            Arc::clone(&pool),
            transient::preparation(revision.clone(), "remote-analysis"),
            sequence_policy(),
            None,
            &mut transient_publisher,
        )
        .await
        .unwrap();
        assert_eq!(transient_result.result.result.samples().len(), 5);
        assert_eq!(transient_result.result.numerical_receipt.blocks.len(), 2);
        assert_eq!(transient_result.resource_attempts.len(), 4);
        assert!(transient_result.resource_attempts[1..3].iter().all(|attempt| {
            matches!(
                attempt.plan.input.phase,
                runmat_analysis_fea::planning::resource::FeaResourcePhase::StructuralTransientBlock(_)
            )
        }));
        assert!(transient_result
            .resource_attempts
            .iter()
            .all(|attempt| attempt.plan.validate_identity().is_ok()));
        assert_eq!(transient_publisher.closures.len(), 2);
        let final_closure = transient_publisher.closures.last().unwrap();
        assert_eq!(final_closure.checkpoint.completed_block_count(), 2);

        let replacement_preparation = transient::preparation_with_encryption(
            revision.clone(),
            "remote-analysis",
            "remote-transient-replacement-encryption",
        );
        let replacement_pool = RemotePoolDriver::new_with_value_scope(
            runmat_execution::ExecutionScopeId::derive(&[b"remote-analysis-replacement-scope"]),
            PoolSpec {
                id: pool_id,
                min_workers: 1,
                max_workers: 2,
                max_in_flight: 2,
                resource_limit: scaled_inventory(2),
            },
            41,
            bundle_bytes.clone(),
            "remote-analysis",
        )
        .unwrap();
        replacement_pool
            .add_worker(first_channel.clone())
            .await
            .unwrap();
        replacement_pool
            .add_worker(second_channel.clone())
            .await
            .unwrap();
        let mut interrupted_publisher = transient::FailAfterFirstPublisher {
            closures: Vec::new(),
        };
        let interruption = execute_structural_transient_remotely(
            Arc::clone(&replacement_pool),
            replacement_preparation.clone(),
            sequence_policy(),
            None,
            &mut interrupted_publisher,
        )
        .await
        .expect_err("the injected driver loss must interrupt transient execution");
        assert_eq!(
            interrupted_publisher.closures.len(),
            1,
            "interruption occurred before recovery publication: {interruption}"
        );
        let accepted_before_replacement = interrupted_publisher.closures.pop().unwrap();
        assert_eq!(
            accepted_before_replacement
                .checkpoint
                .completed_block_count(),
            1
        );
        let first_block_worker = replacement_pool
            .snapshot()
            .attempts
            .values()
            .next()
            .expect("first transient block must have one worker attempt")
            .request
            .worker_id;
        replacement_pool
            .remove_worker(first_block_worker, false)
            .await
            .unwrap();
        let mut replacement_publisher = transient::RecordingPublisher {
            closures: Vec::new(),
        };
        let replacement_result = execute_structural_transient_remotely(
            Arc::clone(&replacement_pool),
            replacement_preparation,
            sequence_policy(),
            Some(&accepted_before_replacement),
            &mut replacement_publisher,
        )
        .await
        .unwrap();
        assert_eq!(replacement_publisher.closures.len(), 1);
        assert_eq!(
            replacement_result.result.result,
            transient_result.result.result
        );
        assert_eq!(
            replacement_result.result.numerical_receipt,
            transient_result.result.numerical_receipt
        );
        let replacement_workers = replacement_pool
            .snapshot()
            .attempts
            .values()
            .map(|attempt| attempt.request.worker_id)
            .collect::<BTreeSet<_>>();
        assert_eq!(replacement_workers.len(), 2);
        let first_worker_id =
            runmat_execution::identity::WorkerId::derive(&[b"remote-analysis-worker-a"]);
        if first_block_worker == first_worker_id {
            second_channel.drain().await.unwrap();
        } else {
            first_channel.drain().await.unwrap();
        }
    };
    let (first_server, second_server, ()) = tokio::join!(first_server, second_server, client);
    first_server.unwrap();
    second_server.unwrap();
}

fn import_result(
    store: &impl CacheImport,
    response: ProgramExecutionResponse,
    host: &AnalysisHostWorkload,
) -> Vec<u8> {
    let output_set = import_analysis_output_set(store, response, host).unwrap();
    let [result, _receipt, profiles] = output_set.manifest.outputs.as_slice() else {
        panic!("analysis result set must contain result, numerical receipt and state profiles")
    };
    assert_eq!(
        profiles.kind,
        AnalysisNumericalObjectKind::StatePublicationProfiles
    );
    let result_identity = runmat_analysis_fea::execution::SolverResultIdentity::new(
        host.workload.identity,
        result.payload_digest,
    )
    .unwrap();
    import_analysis_objects(
        store,
        &result.root,
        AnalysisNumericalObjectKind::SolverResult,
        result_identity.digest(),
        result.payload_digest,
        &host.artifact_access,
        ObjectInventoryLimits::default(),
    )
    .unwrap()
    .payload()
}

fn fixture(
    revision: ProgramRevision,
    authorization_scope: &str,
) -> (
    AnalysisHostWorkload,
    PreparedAnalysisProgram,
    Vec<runmat_analysis_execution::PreparedAnalysisObjects>,
    runmat_analysis_fea::planning::resource::FeaResourcePlan,
) {
    fixture_for_study(revision, authorization_scope, digest("authored-study"))
}

fn fixture_for_study(
    revision: ProgramRevision,
    authorization_scope: &str,
    authored_study_digest: Digest,
) -> (
    AnalysisHostWorkload,
    PreparedAnalysisProgram,
    Vec<runmat_analysis_execution::PreparedAnalysisObjects>,
    runmat_analysis_fea::planning::resource::FeaResourcePlan,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.sequence.steps[0].observations = vec![structural_displacement_observation()];
    let step_id = model.sequence.steps[0].step_id.clone();
    let solve = StructuralLinearWorkload {
        schema_version: 2,
        model,
        step_id,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::canonical(
                std::num::NonZeroU32::new(2).unwrap(),
            ),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let mesh = solver_mesh(&solve.model);
    let access = ArtifactAccess {
        authorization_scope: authorization_scope.into(),
        encryption_context: digest("encryption"),
    };
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest,
        frozen_project_digest: None,
        program_revision: revision,
        artifact_access: access,
        workload: solve,
        mesh: Some(mesh),
        state_input: None,
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap();
    (
        prepared.host,
        prepared.program,
        prepared.inputs,
        prepared.resource_plan,
    )
}

fn explicit_fixture(
    revision: ProgramRevision,
    authorization_scope: &str,
) -> (
    AnalysisHostWorkload,
    PreparedAnalysisProgram,
    Vec<runmat_analysis_execution::PreparedAnalysisObjects>,
    runmat_analysis_fea::planning::resource::FeaResourcePlan,
) {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    model.sequence.steps[0].observations = vec![structural_displacement_observation()];
    let solve = StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::canonical(
                std::num::NonZeroU32::new(2).unwrap(),
            ),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("explicit-authored-study"),
        frozen_project_digest: None,
        program_revision: revision,
        artifact_access: ArtifactAccess {
            authorization_scope: authorization_scope.into(),
            encryption_context: digest("explicit-encryption"),
        },
        workload: solve,
        mesh: None,
        state_input: None,
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap();
    assert!(prepared
        .inputs
        .iter()
        .all(|input| input.manifest.kind != AnalysisNumericalObjectKind::Mesh));
    (
        prepared.host,
        prepared.program,
        prepared.inputs,
        prepared.resource_plan,
    )
}

fn mixed_fixture(
    revision: ProgramRevision,
    authorization_scope: &str,
) -> (
    AnalysisHostWorkload,
    PreparedAnalysisProgram,
    Vec<runmat_analysis_execution::PreparedAnalysisObjects>,
    runmat_analysis_fea::planning::resource::FeaResourcePlan,
) {
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let attached_coordinates = mesh.topology.nodes[0].coordinates_m.canonical_value();
    mesh.topology.nodes[0]
        .provenance
        .push(PersistentEntityId::new(
            PersistentEntityKind::Vertex,
            "beam_root".into(),
            vec!["root".into()],
        ));
    mesh.topology.nodes[0].provenance.sort();
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            "root".into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    mesh.topology.boundary_faces[1]
        .provenance
        .push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            "solid_load".into(),
            vec!["root".into()],
        ));
    mesh.topology.boundary_faces[1].provenance.sort();

    let mut model = mixed_solid_beam_shell_model(attached_coordinates);
    let mut beam_material = model.materials[0].clone();
    beam_material.material_id = "mat_beam_alloy".into();
    beam_material.name = "Beam alloy".into();
    beam_material.mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(4_430.0));
    beam_material
        .require_mechanical_mut()
        .unwrap()
        .youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(113.8e9);
    model.materials.push(beam_material);
    let beam_assignment = model
        .material_assignments
        .iter_mut()
        .find(|assignment| {
            matches!(
                assignment.target,
                runmat_analysis_core::PropertyRegionTarget::Beam(_)
            )
        })
        .expect("mixed fixture declares a beam material assignment");
    beam_assignment.expected_material_id = "mat_beam_alloy".into();
    beam_assignment.assigned_material_id = "mat_beam_alloy".into();
    let mut shell_material = model.materials[0].clone();
    shell_material.material_id = "mat_shell_alloy".into();
    shell_material.name = "Shell alloy".into();
    shell_material.mass_density_kg_per_m3 = Some(runmat_types::physical::MassDensity::new(2_700.0));
    shell_material
        .require_mechanical_mut()
        .unwrap()
        .youngs_modulus_pa = runmat_types::physical::YoungsModulus::new(69.0e9);
    model.materials.push(shell_material);
    let shell_assignment = model
        .material_assignments
        .iter_mut()
        .find(|assignment| {
            matches!(
                assignment.target,
                runmat_analysis_core::PropertyRegionTarget::Shell(_)
            )
        })
        .expect("mixed fixture declares a shell material assignment");
    shell_assignment.expected_material_id = "mat_shell_alloy".into();
    shell_assignment.assigned_material_id = "mat_shell_alloy".into();
    model
        .constraints
        .push(runmat_analysis_core::AnalysisConstraint::Structural(
            runmat_analysis_core::StructuralConstraint {
                constraint_id: "solid_beam_ux_relation".into(),
                frame: runmat_analysis_core::CoordinateFrame::Global,
                kind: runmat_analysis_core::StructuralConstraintKind::Equation(
                    runmat_analysis_core::StructuralConstraintEquation {
                        terms: vec![
                            runmat_analysis_core::StructuralConstraintEquationTerm {
                                target: runmat_analysis_core::RegionTarget::point("node:1"),
                                dof: runmat_analysis_core::StructuralTranslationDof::Ux,
                                coefficient: 1.0,
                            },
                            runmat_analysis_core::StructuralConstraintEquationTerm {
                                target: runmat_analysis_core::RegionTarget::point("node:102"),
                                dof: runmat_analysis_core::StructuralTranslationDof::Ux,
                                coefficient: -1.0,
                            },
                        ],
                        right_hand_side: runmat_analysis_core::Displacement::new(0.0),
                    },
                ),
            },
        ));
    let assignment = model.material_assignments.first().unwrap();
    let region = runmat_meshing_core::fixtures::fixture_entity_for_selection(
        assignment
            .selected_region_id()
            .expect("fixture uses selected material region"),
    )
    .expect("fixture material target is a fixture selector");
    mesh.topology.volume_elements[0].region_id = region.clone();
    mesh.topology.regions[0].region_id = region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let compiled_properties =
        runmat_analysis_fea::properties::compile_property_map(&model, Some(&mesh)).unwrap();
    assert_eq!(compiled_properties.solid_materials().len(), 1);
    assert_eq!(compiled_properties.beam_properties().len(), 1);
    assert_eq!(compiled_properties.shell_properties().len(), 1);
    assert_eq!(
        compiled_properties
            .solid_materials()
            .next()
            .unwrap()
            .material
            .material_id,
        "mat_steel"
    );
    assert_eq!(
        compiled_properties
            .beam_properties()
            .next()
            .unwrap()
            .material
            .material_id,
        "mat_beam_alloy"
    );
    let shell = compiled_properties.shell_properties().next().unwrap();
    let runmat_analysis_fea::properties::CompiledShellPropertyOwner::Homogeneous {
        material, ..
    } = &shell.owner
    else {
        panic!("mixed fixture uses a homogeneous shell");
    };
    assert_eq!(material.material_id, "mat_shell_alloy");
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    let solve = StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::canonical(
                std::num::NonZeroU32::new(2).unwrap(),
            ),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let prepared = prepare_structural_linear_program(StructuralLinearProgramPreparation {
        authored_study_digest: digest("mixed-authored-study"),
        frozen_project_digest: None,
        program_revision: revision,
        artifact_access: ArtifactAccess {
            authorization_scope: authorization_scope.into(),
            encryption_context: digest("mixed-encryption"),
        },
        workload: solve,
        mesh: Some(mesh),
        state_input: None,
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_fea::planning::resource::StaticSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    })
    .unwrap();
    let constraint_plan =
        CanonicalStructuralConstraintPlan::from_canonical_bytes(&prepared.inputs[2].payload())
            .unwrap();
    assert_eq!(
        constraint_plan.enforcement,
        ConstraintEnforcement::AffineReduction
    );
    assert!(constraint_plan
        .rows
        .iter()
        .any(|row| row.constraint_id == "solid_beam_ux_relation"));
    (
        prepared.host,
        prepared.program,
        prepared.inputs,
        prepared.resource_plan,
    )
}

fn structural_displacement_observation() -> runmat_analysis_core::ObservationRequest {
    runmat_analysis_core::ObservationRequest::Field(runmat_analysis_core::ObservationSpec {
        observation_id: runmat_analysis_core::ObservationId("whole_displacement".into()),
        quantity: runmat_analysis_core::ObservationQuantity::Displacement,
        value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
        target: runmat_analysis_core::ObservationTarget::WholeModel,
        source_location: runmat_analysis_core::FieldLocation::Node,
        output_location: runmat_analysis_core::FieldLocation::Node,
        recovery: runmat_analysis_core::RecoveryPolicy::Native,
        component: runmat_analysis_core::ObservationComponent::All,
        frame: runmat_analysis_core::CoordinateFrame::Global,
        sign_convention: runmat_analysis_core::SignConvention::Native,
        complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
        sampling: runmat_analysis_core::SamplingPolicy::Final,
        reduction: None,
        storage: runmat_analysis_core::ObservationStoragePolicy {
            precision: runmat_analysis_core::SignalPrecision::Real64,
            compression: runmat_analysis_core::CompressionPolicy::Lossless,
            retention: runmat_analysis_core::RetentionPolicy::Canonical,
            partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
        },
        purpose: runmat_analysis_core::ObservationPurpose::Publication,
        cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(
            runmat_analysis_core::ObservationCostEstimate {
                operation_count: 1_000,
                peak_resident_bytes: 1_000_000,
                stored_bytes: 1_000_000,
                transfer_bytes: 1_000_000,
                collective_count: 0,
            },
        ),
    })
}

fn sequence_preparation(
    revision: ProgramRevision,
    authorization_scope: &str,
) -> StructuralSequencePreparation {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut root = model.sequence.steps[0].clone();
    root.step_id = runmat_analysis_core::ProcedureStepId("preload".into());
    root.state_outputs[0].persistence = runmat_analysis_core::StatePersistence::Canonical;
    let state_id = root.state_outputs[0].state_id.clone();
    let branch = |step_id: &str| {
        let mut step = root.clone();
        step.step_id = runmat_analysis_core::ProcedureStepId(step_id.into());
        step.state_inputs = vec![runmat_analysis_core::StateInput {
            input_id: format!("{step_id}-initial-guess"),
            required_kind: runmat_analysis_core::StateKind::StructuralPrimary,
            source: runmat_analysis_core::StateSource::Upstream {
                producer_step: root.step_id.clone(),
                state_id: state_id.clone(),
            },
            use_as: runmat_analysis_core::StateUse::InitialGuess,
            conflict: runmat_analysis_core::InitialStateConflictPolicy::Reject,
        }];
        step
    };
    let right = branch("right");
    let left = branch("left");
    let transition =
        |consumer: &runmat_analysis_core::ProcedureStepId| runmat_analysis_core::StateTransition {
            transition_id: runmat_analysis_core::TransitionId(format!(
                "preload-to-{}",
                consumer.as_str()
            )),
            producer_step: root.step_id.clone(),
            consumer_step: consumer.clone(),
            state_id: state_id.clone(),
            method: runmat_analysis_core::StateTransitionMethod::InitialGuessOnly,
            topology: runmat_analysis_core::TopologyTransitionPolicy::RequireIdentical,
            reference: runmat_analysis_core::ReferenceStatePolicy::Preserve,
        };
    model.sequence = runmat_analysis_core::AnalysisSequence {
        sequence_id: "process-preload-branches".into(),
        steps: vec![root.clone(), right.clone(), left.clone()],
        dependencies: vec![
            runmat_analysis_core::SequenceDependency {
                predecessor: root.step_id.clone(),
                successor: right.step_id.clone(),
                reason: runmat_analysis_core::SequenceDependencyReason::StateTransfer,
            },
            runmat_analysis_core::SequenceDependency {
                predecessor: root.step_id.clone(),
                successor: left.step_id.clone(),
                reason: runmat_analysis_core::SequenceDependencyReason::StateTransfer,
            },
        ],
        transitions: vec![transition(&right.step_id), transition(&left.step_id)],
        branches: vec![runmat_analysis_core::SequenceBranch {
            branch_id: runmat_analysis_core::SequenceBranchId("post-preload".into()),
            producer_step: root.step_id.clone(),
            state_ids: vec![state_id],
            consumer_steps: vec![right.step_id.clone(), left.step_id.clone()],
        }],
    };
    let workload = StructuralLinearWorkload {
        schema_version: 2,
        model,
        step_id: root.step_id,
        backend: ComputeBackend::Cpu,
        preconditioner: SpdPreconditionerKind::Jacobi,
        algebra_backend: LinearAlgebraBackendKind::CpuReference,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::canonical(
                std::num::NonZeroU32::new(2).unwrap(),
            ),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let mesh = solver_mesh(&workload.model);
    let workloads = workload
        .model
        .sequence
        .steps
        .iter()
        .map(|step| {
            let mut step_workload = workload.clone();
            step_workload.step_id = step.step_id.clone();
            (step.step_id.clone(), step_workload)
        })
        .collect();
    StructuralSequencePreparation {
        authored_study_digest: digest("process-sequence-study"),
        frozen_project_digest: None,
        program_revision: revision,
        artifact_access: ArtifactAccess {
            authorization_scope: authorization_scope.into(),
            encryption_context: digest("process-sequence-encryption"),
        },
        workloads,
        mesh: Some(mesh),
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        solve_selection:
            runmat_analysis_execution::StructuralSequenceSolveSelection::WithinCeiling {
                machine_memory_bytes: None,
            },
    }
}

fn sequence_policy() -> StructuralSequenceExecutionPolicy {
    StructuralSequenceExecutionPolicy {
        resources: ResourceRequest {
            cpu_millicores: 1_000,
            memory_bytes: 256 * 1024 * 1024,
            scratch_bytes: 256 * 1024 * 1024,
            max_wall_millis: 60_000,
            max_artifact_bytes: 64 * 1024 * 1024,
            max_egress_bytes: 64 * 1024 * 1024,
            max_relay_bytes: 64 * 1024 * 1024,
            accelerators: Vec::new(),
            required_capabilities: capabilities(),
        },
        retry: RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    }
}

fn frequency_preparation(
    revision: ProgramRevision,
    authorization_scope: &str,
) -> StructuralFrequencyExecutionPreparation {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = runmat_analysis_core::ProcedureStep::structural_harmonic("frequency");
    step.independent_domain =
        runmat_analysis_core::IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    let runmat_analysis_core::ObservationRequest::Field(mut field) =
        structural_displacement_observation()
    else {
        unreachable!("the structural observation fixture is a field request")
    };
    field.sampling = runmat_analysis_core::SamplingPolicy::FrequencyHertz(vec![20.0, 40.0]);
    field.complex_representation = runmat_analysis_core::ComplexRepresentation::Real;
    let mut history = field.clone();
    history.observation_id = runmat_analysis_core::ObservationId("displacement_history".into());
    step.observations = vec![runmat_analysis_core::ObservationRequest::History(history)];
    model.sequence = runmat_analysis_core::AnalysisSequence::from_steps(
        "process-structural-frequency",
        vec![step.clone()],
    );
    let mesh = solver_mesh(&model);
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: step.step_id,
        backend: ComputeBackend::Cpu,
        observation_execution:
            runmat_analysis_fea::results::observation::ObservationExecutionPolicy::canonical(
                std::num::NonZeroU32::new(2).unwrap(),
            ),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct {
            policy: runmat_analysis_fea::solve::dynamics::DirectHarmonicPolicy::default(),
        },
        prestress_transition: None,
    })
    .unwrap();
    StructuralFrequencyExecutionPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        workload,
        mesh,
        prestress_state: None,
        modal_state: None,
        program_revision: revision,
        artifact_access: ArtifactAccess {
            authorization_scope: authorization_scope.into(),
            encryption_context: digest("process-frequency-encryption"),
        },
        chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
        inventory_limits: ObjectInventoryLimits::default(),
        observation_limits: runmat_analysis_fea::results::observation::ObservationLimits::default(),
    }
}

fn assert_frequency(
    frequency: &runmat_execution_runner_native::CompletedStructuralFrequencyStudy,
    authorization_scope: &str,
) {
    assert_eq!(frequency.result.result.samples().len(), 2);
    assert_eq!(frequency.entries.len(), 2);
    assert_eq!(frequency.observation_samples.len(), 2);
    assert!(frequency.observation_samples.iter().all(|sample| sample
        .observations
        .as_ref()
        .is_some_and(|observations| observations.manifest.results.len() == 1)));
    assert_eq!(frequency.histories.len(), 1);
    assert_eq!(
        frequency.histories[0]
            .checkpoint
            .checkpoint
            .manifest()
            .observed_sample_count,
        2
    );
    assert_eq!(
        frequency.resource_usage.evidence.result_digest,
        frequency.result.result.manifest().identity().digest()
    );
    assert!(frequency.resource_usage.evidence.overruns.is_empty());
    let access = ArtifactAccess {
        authorization_scope: authorization_scope.into(),
        encryption_context: digest("process-frequency-encryption"),
    };
    let (catalog, roots) = frequency
        .published_artifacts(&access)
        .expect("frequency publication must expose its exact typed object closure");
    catalog
        .validate()
        .expect("frequency object catalog must be canonical");
    roots
        .validate(&frequency.publication, &catalog)
        .expect("frequency root roles must match the semantic publication");
    assert_eq!(
        roots.samples().len(),
        frequency.result.result.samples().len()
    );
    assert_eq!(roots.histories().len(), frequency.histories.len());
}

fn assert_sequence(sequence: &runmat_analysis_execution::CompletedStructuralSequence) {
    assert_eq!(
        sequence
            .steps
            .iter()
            .map(|step| step.step_id.as_str())
            .collect::<Vec<_>>(),
        vec!["preload", "right", "left"]
    );
    assert_eq!(sequence.states.len(), 3);
    for state in &sequence.states {
        let runmat_analysis_execution::StructuralSequenceStateResourceAttempt::Metered(attempt) =
            &state.resource_attempt
        else {
            panic!("native state publication must retain its admitted scheduler attempt");
        };
        assert!(matches!(
            attempt.plan.input.phase,
            runmat_analysis_fea::planning::resource::FeaResourcePhase::SemanticStatePublication(_)
        ));
        assert!(
            runmat_analysis_fea::planning::resource::allocation_matches_plan(
                &attempt.accounting.resources.allocation,
                &attempt.plan.request,
            )
        );
    }
    assert_eq!(
        sequence.steps[1].result.run.fields,
        sequence.steps[2].result.run.fields
    );
    assert!(sequence.steps[1..].iter().all(|step| {
        step.result.run.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "FEA_CONVERGENCE" && diagnostic.message.contains("iterations=0 ")
        })
    }));
    assert_eq!(
        sequence
            .steps
            .iter()
            .map(|step| step.objects.manifest.logical_identity)
            .collect::<BTreeSet<_>>()
            .len(),
        3
    );
    for step in &sequence.steps {
        let usage = step
            .resource_usage
            .as_ref()
            .expect("native sequence solves must retain scheduler resource accounting");
        assert_eq!(usage.evidence.result_digest, step.result_identity.digest());
        assert!(usage.evidence.overruns.is_empty());
        let ProgramExecutionResponse::ExternalizedSuccess { result_objects, .. } = &step.response
        else {
            panic!("completed native sequence step must retain its successful response");
        };
        assert_eq!(
            usage.evidence.actual.artifact_bytes.0,
            result_objects
                .iter()
                .map(|reference| reference.encoded_length)
                .sum::<u64>()
        );
    }
}

fn solver_mesh(
    model: &runmat_analysis_core::AnalysisModel,
) -> runmat_meshing_core::SolverMeshArtifact {
    let mut mesh =
        runmat_meshing_core::fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for (index, face) in mesh.topology.boundary_faces.iter_mut().enumerate() {
        face.provenance.push(PersistentEntityId::new(
            PersistentEntityKind::Face,
            if index == 0 { "root" } else { "tip" }.into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    let assignment = model.material_assignments.first().unwrap();
    let region = runmat_meshing_core::fixtures::fixture_entity_for_selection(
        assignment
            .selected_region_id()
            .expect("fixture uses selected material region"),
    )
    .expect("fixture material target is a fixture selector");
    mesh.topology.volume_elements[0].region_id = region.clone();
    mesh.topology.regions[0].region_id = region;
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    mesh
}

fn capabilities() -> BTreeSet<Capability> {
    BTreeSet::from([
        Capability::ProcessIsolation,
        Capability::Custom(format!(
            "runmat.analysis.host:{}",
            runmat_analysis_execution::ANALYSIS_HOST_ABI
        )),
    ])
}

fn inventory() -> ResourceInventory {
    ResourceInventory {
        cpu_millicores: 1_000,
        // Tet10 thermal history finalization reserves about 625 MiB, including
        // canonical decoding and output construction. Keep admission bounds intact.
        memory_bytes: 1024 * 1024 * 1024,
        scratch_bytes: 256 * 1024 * 1024,
        accelerators: Vec::new(),
        capabilities: capabilities(),
    }
}

fn scaled_inventory(worker_count: u32) -> ResourceInventory {
    let mut inventory = inventory();
    inventory.cpu_millicores *= worker_count;
    inventory.memory_bytes *= u64::from(worker_count);
    inventory.scratch_bytes *= u64::from(worker_count);
    inventory
}

fn analysis_submission(
    scope_id: runmat_execution::ExecutionScopeId,
    pool_id: runmat_execution::PoolId,
    host: &AnalysisHostWorkload,
    request: &runmat_execution_artifact::ProgramExecutionRequest,
    resource_plan: Option<&runmat_analysis_fea::planning::resource::FeaResourcePlan>,
    resources: ResourceRequest,
) -> runmat_execution_runner::TaskSubmission {
    let mut resources = resources;
    resources.required_capabilities.extend(capabilities());
    let roots = request
        .arguments
        .iter()
        .map(|value| match value {
            ValuePayload::Object(root) => (**root).clone(),
            ValuePayload::Inline(_) => panic!("analysis inputs must remain externalized"),
            ValuePayload::Distributed(_) | ValuePayload::Composite(_) => {
                panic!("analysis inputs require external object roots, not service handles")
            }
        })
        .collect();
    let context = AnalysisExecutionContext {
        host: phase_execution_support::host_requirement(request),
        scope_id,
        pool_id,
        program_artifact_id: ArtifactId::derive(&[request.artifact.id.0.bytes()]),
        resources,
        retry: RetryPolicy::IdempotentInfrastructure,
        deadline_unix_millis: None,
        priority: 0,
    };
    match resource_plan {
        Some(resource_plan) => build_resource_planned_task_submission(
            host,
            resource_plan,
            roots,
            Vec::new(),
            BTreeSet::new(),
            context,
        ),
        None => build_analysis_task_submission(host, roots, Vec::new(), BTreeSet::new(), context),
    }
    .unwrap()
}

fn analysis_resources() -> ResourceRequest {
    ResourceRequest {
        cpu_millicores: 1_000,
        memory_bytes: 256 * 1024 * 1024,
        scratch_bytes: 256 * 1024 * 1024,
        max_wall_millis: 60_000,
        max_artifact_bytes: 64 * 1024 * 1024,
        max_egress_bytes: 64 * 1024 * 1024,
        max_relay_bytes: 64 * 1024 * 1024,
        accelerators: Vec::new(),
        required_capabilities: capabilities(),
    }
}

fn revision() -> ProgramRevision {
    ProgramRevision::new(
        digest("graph"),
        digest("source"),
        runmat_core::program_environment(runmat_core::CompatMode::Matlab),
    )
    .unwrap()
}

fn digest(label: &str) -> Digest {
    Digest::sha256(label)
}

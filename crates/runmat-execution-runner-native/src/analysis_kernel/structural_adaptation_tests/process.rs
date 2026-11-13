//! Real process boundaries: original solve producer exits before independent cold consumers.
use super::*;
use runmat_analysis_execution::{AnalysisBridgeError, AnalysisBridgeResult};
use std::{path::Path, process::Stdio, time::Duration};
mod source;
mod source_store;
mod wire;
use wire::{Action, Completion, Evidence, Request, SourceAdmissionCase};

const CHILD_REQUEST: &str = "RUNMAT_FEA_ADAPTIVE_PROCESS_REQUEST";
const TEST_NAME: &str = "analysis_kernel::structural_adaptation_tests::process::producer_termination_and_cold_consumer_replay_preserve_original_evidence";

fn execute_child(directory: &Path, request: &Request) -> AnalysisBridgeResult<()> {
    let mut store = FilesystemObjectStore::open(
        directory.join("objects"),
        ObjectInventoryLimits::default().max_object_bytes,
    )?;
    match &request.action {
        Action::Produce => {
            let mut responses = Vec::new();
            for original in &request.requests {
                let completed = execute_serial_analysis(
                    original,
                    &mut store,
                    &NativeAnalysisKernel,
                    1024,
                    ObjectInventoryLimits::default(),
                )?;
                let host = runmat_analysis_execution::AnalysisHostWorkload::from_program_request(
                    original, &store,
                )?;
                responses
                    .push(AnalysisHostResponse::completed(&host, &completed)?.program_response());
            }
            wire::write(
                &directory.join(&request.completion_file),
                "completion",
                &Completion {
                    producer_pid: std::process::id(),
                    responses: responses.try_into().unwrap(),
                },
            );
        }
        Action::Consume {
            admission,
            evidence_file,
        } => {
            let completion: Completion =
                wire::read(&directory.join(&request.completion_file), "completion");
            assert_ne!(completion.producer_pid, std::process::id());
            let source_store = source_store::original_store(
                &store,
                &request.requests[0],
                &completion.responses[0],
                *admission,
            )?;
            let source = source::OriginalSolve::import(
                &source_store,
                &request.requests[0],
                &completion.responses[0],
            )?;
            let target = source::OriginalSolve::import(
                &source_store,
                &request.requests[1],
                &completion.responses[1],
            )?;
            let originals = [source.source(), target.source()]
                .into_iter()
                .flat_map(|s| [s.invocation, s.mesh, s.result, s.receipt]);
            let required =
                runmat_analysis_execution::prepared_input_resource_facts_from_refs(originals)?
                    .decoded_resident_bytes_admission_budget
                    .0;
            let maximum = match admission {
                SourceAdmissionCase::Exact
                | SourceAdmissionCase::Cancelled
                | SourceAdmissionCase::MissingSourceChunk
                | SourceAdmissionCase::CorruptSourceChunk => required,
                SourceAdmissionCase::RequiredMinusOne => required.checked_sub(1).unwrap(),
            };
            let _cancellation = if matches!(admission, SourceAdmissionCase::Cancelled) {
                let token = runmat_analysis_fea::progress::FeaCancellationToken::new();
                token.cancel();
                Some(runmat_analysis_fea::progress::replace_fea_progress_context(
                    runmat_analysis_fea::progress::FeaProgressContext {
                        cancellation: token,
                        ..Default::default()
                    },
                ))
            } else {
                None
            };
            let observation = ObservationId("tip_displacement".into());
            let input = StructuralAdaptationNativeIterationInput {
                source: source.source(),
                target: target.source(),
                transfer_map: &request.map,
                lineage: &request.lineage,
                observation_id: &observation,
                accuracy: &request.accuracy,
                source_limits: fixture::source_limits(),
                maximum_source_decode_bytes: maximum,
                estimator_options: StructuralRecoveryEstimatorOptions::default(),
                previous: None,
            };
            let iteration = evaluate_original_structural_adaptation(input)?;
            source.verify_affine_original();
            target.verify_affine_original();
            if iteration.solved.iteration.decision.status
                != runmat_analysis_fea::adaptation::StructuralAdaptationDecisionStatus::Converged
            {
                return Err(AnalysisBridgeError::Identity(
                    "cold native patch did not meet shared accuracy",
                ));
            }
            let cold: StructuralAdaptationNativeIteration =
                serde_json::from_slice(&serde_json::to_vec(&iteration).unwrap()).unwrap();
            cold.validate_against(input)?;
            wire::write(
                &directory.join(evidence_file),
                "evidence",
                &Evidence {
                    consumer_pid: std::process::id(),
                    producer_pid: completion.producer_pid,
                    iteration,
                },
            );
        }
    }
    Ok(())
}

fn run_process(
    directory: &Path,
    ordinal: usize,
    request: &Request,
) -> (u32, std::process::ExitStatus, String) {
    use std::io::Read;
    let path = directory.join(format!("child-{ordinal}.cbor"));
    wire::write(&path, "request", request);
    let log_path = directory.join(format!("child-{ordinal}.log"));
    let log = std::fs::File::create(&log_path).unwrap();
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", TEST_NAME, "--nocapture"])
        .env(CHILD_REQUEST, &path)
        .stdout(Stdio::from(log.try_clone().unwrap()))
        .stderr(Stdio::from(log))
        .spawn()
        .unwrap();
    let pid = child.id();
    let started = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > Duration::from_secs(45) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("native adaptive child {pid} exceeded its 45-second witness limit");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let mut log = String::new();
    std::fs::File::open(log_path)
        .unwrap()
        .take(1024 * 1024)
        .read_to_string(&mut log)
        .unwrap();
    (pid, status, log)
}

#[test]
fn producer_termination_and_cold_consumer_replay_preserve_original_evidence() {
    if let Some(path) = std::env::var_os(CHILD_REQUEST) {
        let path = std::path::PathBuf::from(path);
        let request: Request = wire::read(&path, "request");
        execute_child(path.parent().unwrap(), &request)
            .expect("cold native adaptive child rejected original sources");
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let coarse = fixtures::canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let fine = fixtures::connected_tetrahedron_solver_mesh();
    let mut model = fixture::model(&coarse);
    let accuracy = fixture::plan();
    model.sequence.steps[0].accuracy = accuracy.clone();
    let (map, lineage) = fixture::lineage(&coarse, &fine);
    // Preparation does not execute a solve. Only the child receives NativeAnalysisKernel.
    let source = prepare(&model, &coarse);
    let target = prepare(&model, &fine);
    let mut store = FilesystemObjectStore::open(
        directory.path().join("objects"),
        ObjectInventoryLimits::default().max_object_bytes,
    )
    .unwrap();
    source.program.write_to(&mut store).unwrap();
    target.program.write_to(&mut store).unwrap();
    drop(store);
    let mut request = Request {
        action: Action::Produce,
        requests: [source.program.request, target.program.request],
        map,
        lineage,
        accuracy,
        completion_file: "completion.cbor".into(),
    };
    let (producer, status, log) = run_process(directory.path(), 0, &request);
    assert!(status.success(), "{log}");
    let completion: Completion =
        wire::read(&directory.path().join("completion.cbor"), "completion");
    assert_eq!(completion.producer_pid, producer);
    // try_wait above observed and reaped this exact producer before either consumer starts.
    let mut consumers = Vec::new();
    let mut evidence = Vec::new();
    for (ordinal, file) in [(1, "evidence.cbor"), (2, "replay.cbor")] {
        request.action = Action::Consume {
            admission: SourceAdmissionCase::Exact,
            evidence_file: file.into(),
        };
        let (pid, status, log) = run_process(directory.path(), ordinal, &request);
        assert!(status.success(), "{log}");
        let cold: Evidence = wire::read(&directory.path().join(file), "evidence");
        assert_eq!(cold.producer_pid, producer);
        assert_eq!(cold.consumer_pid, pid);
        assert_ne!(pid, producer);
        consumers.push(pid);
        evidence.push(cold.iteration);
    }
    assert_ne!(consumers[0], consumers[1]);
    assert_eq!(evidence[0], evidence[1]);
    for (ordinal, admission, reason) in [
        (
            3,
            SourceAdmissionCase::RequiredMinusOne,
            "decoder inventory exceeds",
        ),
        (
            4,
            SourceAdmissionCase::Cancelled,
            "adaptive source decoding cancelled",
        ),
        (
            6,
            SourceAdmissionCase::MissingSourceChunk,
            "analysis chunk is absent",
        ),
        (
            7,
            SourceAdmissionCase::CorruptSourceChunk,
            "chunk bytes differ",
        ),
    ] {
        let file = format!("rejected-{ordinal}.cbor");
        request.action = Action::Consume {
            admission,
            evidence_file: file.clone(),
        };
        let (_, status, log) = run_process(directory.path(), ordinal, &request);
        assert!(!status.success(), "rejected source unexpectedly completed");
        assert!(log.contains(reason), "{log}");
        assert!(!directory.path().join(file).exists());
    }
    request.requests.swap(0, 1);
    request.action = Action::Consume {
        admission: SourceAdmissionCase::Exact,
        evidence_file: "wrong-order.cbor".into(),
    };
    let (_, status, log) = run_process(directory.path(), 5, &request);
    assert!(!status.success());
    assert!(log.contains("analysis output set differs"), "{log}");
    assert!(!directory.path().join("wrong-order.cbor").exists());
    println!("original producer {producer} exited; cold consumers {consumers:?} retain identical evidence; budget/cancellation/order/loss/corruption reject without an evidence file");
}

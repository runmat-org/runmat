use super::*;
use crate::supervisor::{
    tests::{original_program_submission, test_config},
    DurableExecutionContext,
};
use runmat_execution_runner::{DriverAction, DriverCommand};

fn read_page(
    job: &Path,
    request: &AttemptRequest,
    cursor: ConsoleCursor,
    budget: ConsoleReadBudget,
) -> NativeExecutionResult<Option<ConsolePage>> {
    ConsoleReadPlan::bind(job, request)?.read_page(cursor, budget)
}

async fn assigned(
    temp: &tempfile::TempDir,
) -> (
    std::sync::Arc<super::super::LocalSupervisor>,
    super::super::model::DurableProgramAttempt,
) {
    let config = test_config(temp);
    let supervisor = super::super::LocalSupervisor::open(config.clone()).unwrap();
    let context: DurableExecutionContext = supervisor
        .execution_context(Some("original-prefix"))
        .await
        .unwrap();
    let mut submission = original_program_submission(context);
    // This fixture uses a legacy program; test storage against the original task
    // assignment directly, rather than claiming a V4 scientific execution.
    submission.task.request.console = Some(ConsoleLimits {
        max_records: 4,
        max_text_bytes: 100,
    });
    let request = {
        let mut scheduler = supervisor.scheduler.lock().await;
        scheduler.tick().unwrap();
        scheduler
            .admit(
                runmat_execution::JobId::derive(&[b"original-prefix"]),
                &submission.task,
            )
            .unwrap()
            .into_iter()
            .find_map(|action| {
                if let DriverAction::Launch(r) = action {
                    Some(r)
                } else {
                    None
                }
            })
            .unwrap()
    };
    let attempt = super::super::model::DurableProgramAttempt {
        schema_version: 1,
        request,
        config: config.execution,
        job_directory: temp.path().join("original-job"),
    };
    super::super::filesystem::secure_directory(&attempt.job_directory).unwrap();
    (supervisor, attempt)
}

#[tokio::test]
async fn original_live_prefix_pages_and_missing_footer_survive_publisher_loss() {
    let temp = tempfile::tempdir().unwrap();
    let (supervisor, attempt) = assigned(&temp).await;
    let lease =
        super::super::DurableAttemptLease::acquire(&attempt.job_directory, &attempt.request)
            .unwrap();
    let mut publisher = ConsoleJournal::create(&attempt, &lease).unwrap().unwrap();
    let records = [
        ConsoleRecord {
            sequence: 1,
            stream: ConsoleStream::Stdout,
            text: "λ🙂water".into(),
        },
        ConsoleRecord {
            sequence: 2,
            stream: ConsoleStream::ClearScreen,
            text: "".into(),
        },
        ConsoleRecord {
            sequence: 3,
            stream: ConsoleStream::Stderr,
            text: "original failure".into(),
        },
    ];
    for record in &records {
        publisher.push(record).unwrap();
    }
    assert!(publisher.push(&records[0]).is_err());
    drop(publisher);
    drop(lease);
    let mut cursor = ConsoleCursor::default();
    let mut text = String::new();
    let mut ordinals = vec![];
    while cursor.sequence <= 3 {
        let page = read_page(
            &attempt.job_directory,
            &attempt.request,
            cursor,
            ConsoleReadBudget {
                max_fragments: 1,
                max_text_bytes: 4,
            },
        )
        .unwrap()
        .unwrap();
        assert_eq!(page.completion, ConsoleCompletion::Interrupted);
        assert!(page.source_receipt.is_none());
        assert_ne!(page.next_cursor, cursor);
        page.validate().unwrap();
        for f in &page.fragments {
            text.push_str(&f.text);
            ordinals.push(f.sequence);
        }
        cursor = page.next_cursor;
    }
    assert_eq!(text, "λ🙂wateroriginal failure");
    assert!(ordinals.contains(&2));
    let prefix = read_transcript(&attempt.job_directory, &attempt.request)
        .unwrap()
        .unwrap();
    assert_eq!(prefix.records(), records);
    assert_eq!(prefix.completion(), ConsoleCompletion::Interrupted);
    let mut report = runmat_execution_runner::port::BackendReport::for_request(
        &attempt.request,
        runmat_execution_runner::AttemptReport::Lost {
            message: "original publisher ended".into(),
        },
    );
    report.console = Some(Box::new(prefix));
    supervisor
        .scheduler
        .lock()
        .await
        .command(DriverCommand::WorkerLostWithReport {
            worker_id: attempt.request.worker_id,
            report: Box::new(report),
        })
        .unwrap();
}

#[tokio::test]
async fn receipt_identity_tamper_missing_write_and_reservation_reject() {
    let temp = tempfile::tempdir().unwrap();
    let (_, attempt) = assigned(&temp).await;
    let lease =
        super::super::DurableAttemptLease::acquire(&attempt.job_directory, &attempt.request)
            .unwrap();
    let mut publisher = ConsoleJournal::create(&attempt, &lease).unwrap().unwrap();
    let record = ConsoleRecord {
        sequence: 1,
        stream: ConsoleStream::Stdout,
        text: "original".into(),
    };
    publisher.push(&record).unwrap();
    let mut inventory = ConsoleInventory::new(attempt.request.task.console.unwrap()).unwrap();
    inventory.push(&record).unwrap();
    let receipt = inventory.receipt(ConsoleCompletion::Complete);
    publisher.finish(&receipt).unwrap();
    assert!(publisher
        .push(&ConsoleRecord {
            sequence: 2,
            ..record.clone()
        })
        .is_err());
    let original = read_transcript(&attempt.job_directory, &attempt.request)
        .unwrap()
        .unwrap();
    assert_eq!(original.source_receipt(), Some(&receipt));
    let mut foreign = attempt.request.clone();
    foreign.driver_fence += 1;
    assert!(read_transcript(&attempt.job_directory, &foreign).is_err());
    let path = record_path(&publisher.root, 1);
    let mut wire: serde_json::Value = super::super::filesystem::read_json(&path).unwrap();
    wire["record"]["text"] = "tampered".into();
    atomic_json(&path, &wire).unwrap();
    assert!(read_transcript(&attempt.job_directory, &attempt.request).is_err());
    assert!(read_page(
        &attempt.job_directory,
        &attempt.request,
        ConsoleCursor::default(),
        ConsoleReadBudget {
            max_fragments: 1,
            max_text_bytes: 4
        }
    )
    .is_err());
    std::fs::remove_file(path).unwrap();
    assert!(read_transcript(&attempt.job_directory, &attempt.request).is_err());
    let mut task = attempt.request.task;
    task.resources.scratch_bytes = 0;
    assert!(validate_reservation(&task).is_err());
}

#[tokio::test]
async fn failed_initial_head_publication_does_not_expose_an_assignment_marker() {
    let temp = tempfile::tempdir().unwrap();
    let (_, attempt) = assigned(&temp).await;
    let lease =
        super::super::DurableAttemptLease::acquire(&attempt.job_directory, &attempt.request)
            .unwrap();
    let root = directory(&attempt.job_directory, &attempt.request);
    // A directory at the head destination forces the existing filesystem owner
    // to reject publication. The journal must not publish its discovery marker.
    super::super::filesystem::secure_directory(&root.join("head.json")).unwrap();
    assert!(ConsoleJournal::create(&attempt, &lease).is_err());
    assert!(!root.join("original.json").exists());
    assert!(read_page(
        &attempt.job_directory,
        &attempt.request,
        ConsoleCursor::default(),
        ConsoleReadBudget {
            max_fragments: 1,
            max_text_bytes: 4
        }
    )
    .unwrap()
    .is_none());
}

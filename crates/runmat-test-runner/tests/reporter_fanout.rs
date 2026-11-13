mod common;

use futures::executor::block_on;
use runmat_test_runner::host::NeverCancelled;
use runmat_test_runner::reporter::{
    EventObserver, HumanReporter, JsonReporter, JunitReporter, ReporterFanout, TapReporter,
};
use runmat_test_runner::telemetry::NoopTelemetry;
use runmat_test_runner::{Coordinator, CoordinatorConfig};

use common::{crashed, plan, FakeBackend, PendingClock, Step};

struct CountingObserver(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl EventObserver for CountingObserver {
    fn event(
        &mut self,
        _event: &runmat_test::event::TestEvent,
    ) -> runmat_test_runner::RunnerResult<()> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn every_report_is_well_formed_after_an_infrastructure_failure() {
    let plan = plan(&["crashes"]);
    let backend = FakeBackend::new([Step::Result(Err(crashed("boom")))]);
    let mut reporters = ReporterFanout::default();
    let observed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    reporters.push_observer(CountingObserver(observed.clone()));
    reporters.push(HumanReporter::default());
    reporters.push(JsonReporter::default());
    reporters.push(JunitReporter::default());
    reporters.push(TapReporter::default());

    let run = block_on(Coordinator::new(CoordinatorConfig::default()).unwrap().run(
        plan,
        &backend,
        &PendingClock,
        &NeverCancelled,
        &NoopTelemetry,
        &mut reporters,
    ))
    .unwrap();

    assert_eq!(run.reports.len(), 4);
    let human = String::from_utf8(run.reports[0].bytes.clone()).unwrap();
    assert!(human.contains("tests/sample.m: crashes"));
    let json: serde_json::Value = serde_json::from_slice(&run.reports[1].bytes).unwrap();
    assert_eq!(json["schema_version"], 2);
    assert_eq!(
        json["result"]["tests"][0]["report_identity"]["display_name"],
        "crashes"
    );
    let junit = String::from_utf8(run.reports[2].bytes.clone()).unwrap();
    assert!(junit.starts_with("<?xml"));
    assert!(junit.contains("<testcase name=\"crashes\" classname=\"suite\""));
    assert!(junit.ends_with("</testsuite>\n"));
    let tap = String::from_utf8(run.reports[3].bytes.clone()).unwrap();
    assert!(tap.starts_with("TAP version 13\n"));
    assert!(tap.contains("\n1..1\n"));
    assert!(tap.contains("tests/sample.m: crashes"));
    assert_eq!(
        observed.load(std::sync::atomic::Ordering::Relaxed),
        run.events.len()
    );
}

#[test]
fn raw_worker_output_is_process_scoped_bounded_and_shared_across_reports() {
    use runmat_execution::process_output::{
        ProcessOutputBytes, ProcessStreamCompletion, ProcessStreamOutput,
    };
    use runmat_test::event::{
        TestEvent, TestEventPayload, WorkerProcessOutput, WorkerProcessRetirement,
    };
    use runmat_test::identity::WorkerSessionId;
    use runmat_test_runner::reporter::Reporter;

    let bytes = ProcessOutputBytes::new(b"not ok 99\r<xml>\0\xff".to_vec());
    let shared = bytes.clone();
    assert_eq!(bytes.as_slice().as_ptr(), shared.as_slice().as_ptr());
    let capture = ProcessStreamOutput {
        observed_bytes: Some(bytes.as_slice().len() as u64),
        bytes,
        truncated: false,
        completion: ProcessStreamCompletion::EndOfStream,
    };
    let output = WorkerProcessOutput {
        worker: WorkerSessionId("process:42:0".into()),
        retirement: WorkerProcessRetirement::Verified,
        stdout: capture,
        stderr: ProcessStreamOutput {
            bytes: ProcessOutputBytes::new(Vec::new()),
            observed_bytes: Some(0),
            truncated: false,
            completion: ProcessStreamCompletion::EndOfStream,
        },
    };
    output.validate(1024).unwrap();
    assert!(output.validate(1).is_err());
    let event = TestEvent {
        sequence: 1,
        run_id: plan(&["sample"]).plan.run_id,
        payload: TestEventPayload::ProcessOutput { output },
    };
    let encoded = serde_json::to_vec(&event).unwrap();
    assert_eq!(
        serde_json::from_slice::<TestEvent>(&encoded).unwrap(),
        event
    );
    let mut human = HumanReporter::default();
    let mut tap = TapReporter::default();
    let mut junit = JunitReporter::default();
    human.event(&event).unwrap();
    tap.event(&event).unwrap();
    junit.event(&event).unwrap();
    let result = runmat_test::result::RunResult {
        run_id: event.run_id.clone(),
        state: runmat_test::result::ResultState::PASSED,
        tests: Vec::new(),
    };
    let human = String::from_utf8(human.finish(&result).unwrap().bytes).unwrap();
    assert!(human.contains("Worker process process:42:0 stdout"));
    let tap = String::from_utf8(tap.finish(&result).unwrap().bytes).unwrap();
    assert!(tap.contains("# not ok 99\n# <xml>"));
    assert!(!tap.contains("\nnot ok 99"));
    let xml = String::from_utf8(junit.finish(&result).unwrap().bytes).unwrap();
    assert!(xml.contains("<system-out>"));
    assert!(xml.contains("&lt;xml&gt;"));
    assert!(!xml.contains('\0'));
}

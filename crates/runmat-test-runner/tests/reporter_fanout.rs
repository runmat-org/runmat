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
    reporters.push(HumanReporter);
    reporters.push(JsonReporter::default());
    reporters.push(JunitReporter);
    reporters.push(TapReporter);

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

mod common;

use futures::executor::block_on;
use runmat_test_runner::host::NeverCancelled;
use runmat_test_runner::reporter::{
    HumanReporter, JsonReporter, JunitReporter, ReporterFanout, TapReporter,
};
use runmat_test_runner::telemetry::NoopTelemetry;
use runmat_test_runner::{Coordinator, CoordinatorConfig};

use common::{crashed, plan, FakeBackend, PendingClock, Step};

#[test]
fn every_report_is_well_formed_after_an_infrastructure_failure() {
    let plan = plan(&["crashes"]);
    let backend = FakeBackend::new([Step::Result(Err(crashed("boom")))]);
    let mut reporters = ReporterFanout::default();
    reporters.push(HumanReporter::default());
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
    serde_json::from_slice::<serde_json::Value>(&run.reports[1].bytes).unwrap();
    let junit = String::from_utf8(run.reports[2].bytes.clone()).unwrap();
    assert!(junit.starts_with("<?xml"));
    assert!(junit.ends_with("</testsuite>\n"));
    let tap = String::from_utf8(run.reports[3].bytes.clone()).unwrap();
    assert!(tap.starts_with("TAP version 13\n"));
    assert!(tap.contains("\n1..1\n"));
}

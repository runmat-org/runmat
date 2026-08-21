use runmat_test::coverage::CoverageAggregate;
use runmat_test::event::{PluginStatus, TestEvent, TestEventPayload};
use runmat_test::result::RunResult;

use crate::coordinator::CoordinatedRun;
use crate::reporter::RenderedReport;

use super::{PluginError, PluginOutput};

pub trait TestPlugin {
    fn name(&self) -> &str;

    fn event(&mut self, _event: &TestEvent) -> Result<Option<PluginOutput>, PluginError> {
        Ok(None)
    }

    fn finish(
        &mut self,
        _result: &RunResult,
        _coverage: &CoverageAggregate,
    ) -> Result<Option<PluginOutput>, PluginError> {
        Ok(None)
    }
}

struct PluginSlot {
    plugin: Box<dyn TestPlugin>,
    failed: bool,
}

#[derive(Default)]
pub struct PluginFanout {
    plugins: Vec<PluginSlot>,
}

impl PluginFanout {
    pub fn push(&mut self, plugin: impl TestPlugin + 'static) {
        self.plugins.push(PluginSlot {
            plugin: Box::new(plugin),
            failed: false,
        });
    }

    /// Project the canonical event stream through plugins. A plugin cannot
    /// mutate result state; failures disable only that plugin and are appended
    /// to the same event stream.
    pub fn apply(&mut self, run: &mut CoordinatedRun) {
        let source_events = run.events.clone();
        let mut plugin_events = Vec::new();
        let mut reports = Vec::new();
        for event in &source_events {
            self.dispatch_event(event, &mut plugin_events, &mut reports);
        }
        self.dispatch_finish(&run.result, &run.coverage, &mut plugin_events, &mut reports);
        run.plugin_failures += plugin_events
            .iter()
            .filter(|payload| {
                matches!(
                    payload,
                    TestEventPayload::Plugin {
                        status: PluginStatus::Failed,
                        ..
                    }
                )
            })
            .count();
        let terminal = run
            .events
            .last()
            .is_some_and(|event| matches!(event.payload, TestEventPayload::RunFinished { .. }))
            .then(|| run.events.pop())
            .flatten();
        let mut sequence = run.events.last().map_or(0, |event| event.sequence + 1);
        for payload in plugin_events {
            run.events.push(TestEvent {
                sequence,
                run_id: run.result.run_id.clone(),
                payload,
            });
            sequence += 1;
        }
        if let Some(mut terminal) = terminal {
            terminal.sequence = sequence;
            run.events.push(terminal);
        }
        run.reports.extend(reports);
    }

    fn dispatch_event(
        &mut self,
        event: &TestEvent,
        observations: &mut Vec<TestEventPayload>,
        reports: &mut Vec<RenderedReport>,
    ) {
        let hook = hook_name(&event.payload);
        for slot in &mut self.plugins {
            if slot.failed {
                continue;
            }
            let name = slot.plugin.name().to_owned();
            match slot.plugin.event(event) {
                Ok(Some(output)) => record_success(name, hook, output, observations, reports),
                Ok(None) => {}
                Err(error) => record_failure(slot, name, hook, error, observations),
            }
        }
    }

    fn dispatch_finish(
        &mut self,
        result: &RunResult,
        coverage: &CoverageAggregate,
        observations: &mut Vec<TestEventPayload>,
        reports: &mut Vec<RenderedReport>,
    ) {
        for slot in &mut self.plugins {
            if slot.failed {
                continue;
            }
            let name = slot.plugin.name().to_owned();
            match slot.plugin.finish(result, coverage) {
                Ok(Some(output)) => record_success(name, "finish", output, observations, reports),
                Ok(None) => {}
                Err(error) => record_failure(slot, name, "finish", error, observations),
            }
        }
    }
}

fn record_success(
    plugin: String,
    hook: &str,
    output: PluginOutput,
    observations: &mut Vec<TestEventPayload>,
    reports: &mut Vec<RenderedReport>,
) {
    reports.extend(output.reports);
    observations.push(TestEventPayload::Plugin {
        plugin,
        hook: hook.into(),
        status: PluginStatus::Completed,
        message: output.message,
    });
}

fn record_failure(
    slot: &mut PluginSlot,
    plugin: String,
    hook: &str,
    error: PluginError,
    observations: &mut Vec<TestEventPayload>,
) {
    slot.failed = true;
    observations.push(TestEventPayload::Plugin {
        plugin,
        hook: hook.into(),
        status: PluginStatus::Failed,
        message: Some(error.message),
    });
}

fn hook_name(payload: &TestEventPayload) -> &'static str {
    match payload {
        TestEventPayload::RunStarted => "run_started",
        TestEventPayload::TestStarted { .. } => "test_started",
        TestEventPayload::PhaseStarted { .. } => "phase_started",
        TestEventPayload::PhaseFinished { .. } => "phase_finished",
        TestEventPayload::Qualification { .. } => "qualification",
        TestEventPayload::Diagnostic { .. } => "diagnostic",
        TestEventPayload::Output { .. } => "output",
        TestEventPayload::Artifact { .. } => "artifact",
        TestEventPayload::Plugin { .. } => "plugin",
        TestEventPayload::TestFinished { .. } => "test_finished",
        TestEventPayload::RunFinished { .. } => "run_finished",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_test::identity::RunId;
    use runmat_test::result::{ResultState, RunResult};

    struct FailingPlugin;

    impl TestPlugin for FailingPlugin {
        fn name(&self) -> &str {
            "failure"
        }

        fn event(&mut self, _event: &TestEvent) -> Result<Option<PluginOutput>, PluginError> {
            Err(PluginError::new("isolated failure"))
        }
    }

    #[test]
    fn failure_is_isolated_and_recorded_without_changing_results() {
        let run_id = RunId::derive("program", "plugin-test");
        let result = RunResult {
            run_id: run_id.clone(),
            state: ResultState::PASSED,
            tests: Vec::new(),
        };
        let mut run = CoordinatedRun {
            result: result.clone(),
            events: vec![TestEvent {
                sequence: 0,
                run_id,
                payload: TestEventPayload::RunStarted,
            }],
            reports: Vec::new(),
            infrastructure_failures: 0,
            plugin_failures: 0,
            isolation: crate::host::IsolationMode::None,
            coverage: CoverageAggregate::default(),
        };
        let mut plugins = PluginFanout::default();
        plugins.push(FailingPlugin);
        plugins.apply(&mut run);
        assert_eq!(run.result, result);
        assert!(run.events.iter().any(|event| matches!(
            event.payload,
            TestEventPayload::Plugin {
                status: PluginStatus::Failed,
                ..
            }
        )));
    }
}

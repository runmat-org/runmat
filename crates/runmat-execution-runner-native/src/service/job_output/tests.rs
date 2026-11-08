use super::*;
use crate::supervisor::{LocalJobState, LocalJobWorkloadKind};
use runmat_execution::{
    identity::AttemptId, process_output::ProcessOutputBytes, JobHandle, JobId, OutputContract,
    RunId, TaskId,
};
use runmat_execution_runner::task::AttemptDisposition;
use std::{cell::RefCell, collections::VecDeque, rc::Rc};

enum Observation {
    Lookup(LocalJobState),
    MissingPage,
    Page { retired: bool },
}

struct OrderedSource {
    handle: SharedJobHandle,
    attempt: AttemptId,
    observations: RefCell<VecDeque<Observation>>,
}

impl OrderedSource {
    fn new(observations: impl IntoIterator<Item = Observation>) -> Self {
        Self {
            handle: SharedJobHandle::new(JobHandle {
                id: JobId::derive(&[b"retirement-snapshot-job"]),
                run_id: RunId::derive(&[b"retirement-snapshot-run"]),
                generation: 1,
                outputs: OutputContract {
                    requested_outputs: 0,
                    owned_work: None,
                },
            }),
            attempt: AttemptId::derive(&[b"retirement-snapshot-attempt"]),
            observations: RefCell::new(observations.into_iter().collect()),
        }
    }
}

impl OriginalOutputSource for OrderedSource {
    fn raw_output_attempt(
        &self,
        handle: SharedJobHandle,
        ordinal: u32,
    ) -> Result<RawProcessOutputLookup, ExecutionServiceError> {
        assert_eq!(handle, self.handle);
        assert_eq!(ordinal, 1);
        let Some(Observation::Lookup(state)) = self.observations.borrow_mut().pop_front() else {
            panic!("output lookup occurred outside its observation order");
        };
        Ok(RawProcessOutputLookup {
            handle: handle.into_wire_unbounded(),
            task: Some(TaskId::derive(&[b"retirement-snapshot-task"])),
            ordinal,
            state,
            workload_kind: LocalJobWorkloadKind::Program,
            retain_until_unix_millis: u64::MAX,
            attempt: Some(RawProcessOutputAttempt {
                attempt_id: self.attempt,
                ordinal: 1,
                terminal: state.is_terminal().then_some(AttemptDisposition::Cancelled),
                not_started: false,
            }),
        })
    }

    fn raw_output_page(
        &self,
        handle: SharedJobHandle,
        attempt_id: AttemptId,
        cursor: RawProcessOutputCursor,
    ) -> Result<Option<RawProcessOutputPage>, ExecutionServiceError> {
        assert_eq!(handle, self.handle);
        assert_eq!(attempt_id, self.attempt);
        assert_eq!(cursor, RawProcessOutputCursor::default());
        let retired = match self.observations.borrow_mut().pop_front() {
            Some(Observation::Page { retired }) => retired,
            Some(Observation::MissingPage) => return Ok(None),
            _ => panic!("output page occurred outside its observation order"),
        };
        let state = if retired {
            RawOutputState::EndOfStream
        } else {
            RawOutputState::Reading
        };
        Ok(Some(RawProcessOutputPage {
            attempt_id,
            stdout: ProcessOutputBytes::new(if retired {
                b"late".to_vec()
            } else {
                Vec::new()
            }),
            stderr: ProcessOutputBytes::new(Vec::new()),
            next: RawProcessOutputCursor {
                stdout: if retired { 4 } else { 0 },
                stderr: 0,
            },
            stdout_state: state,
            stderr_state: state,
            retired,
        }))
    }
}

#[test]
fn terminal_lookup_refreshes_an_older_page_and_keeps_unverified_retirement_closed() {
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    for retired in [true, false] {
        let source = OrderedSource::new([
            Observation::Lookup(LocalJobState::Running),
            Observation::Page { retired: false },
            Observation::Lookup(LocalJobState::Cancelled),
            Observation::Page { retired },
        ]);
        let mut output = DurableCallerOutput::default();
        let result = output.pump_open(&source, &source.handle, &runtime);
        if retired {
            assert_eq!(result, Ok(true));
            // The refreshed page still needs delivery through the normal pump.
            assert!(!output.complete());
            assert_eq!(output.cursor, RawProcessOutputCursor::default());
            assert_eq!(output.eof, [false; 2]);
        } else {
            assert!(matches!(result, Err(ExecutionServiceError::WorkerLost(_))));
        }
        assert!(source.observations.borrow().is_empty());
    }
}

#[test]
fn terminal_lookup_refreshes_header_discovery_and_requires_an_original_journal() {
    let runtime = RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    for visible in [true, false] {
        let source = OrderedSource::new([
            Observation::Lookup(LocalJobState::Running),
            Observation::MissingPage,
            Observation::Lookup(LocalJobState::Cancelled),
            if visible {
                Observation::Page { retired: true }
            } else {
                Observation::MissingPage
            },
        ]);
        let mut output = DurableCallerOutput::default();
        let result = output.pump_open(&source, &source.handle, &runtime);
        if visible {
            assert_eq!(result, Ok(true));
            assert!(!output.complete());
            assert_eq!(output.cursor, RawProcessOutputCursor::default());
        } else {
            assert!(matches!(
                result,
                Err(ExecutionServiceError::Infrastructure(_))
            ));
        }
        assert!(source.observations.borrow().is_empty());
    }
}

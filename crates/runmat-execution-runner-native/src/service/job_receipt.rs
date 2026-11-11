//! Caller-owned durable results, with distinct portable and selected-graph authority.
use std::{cell::RefCell, rc::Rc};

use runmat_execution::JobHandle;
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_runtime::{
    context::RuntimeContext,
    execution::{ExecutionServiceError, GraphResultReceipt, RootedValueSequence},
};
use runmat_value::ValueSequence;

use super::job_output::DurableCallerOutput;

impl super::NativeExecutionService {
    pub(super) fn retain_durable_receipt(
        &self,
        submitted: crate::durable::SubmittedJob,
        authority: JobResultAuthority,
    ) -> Result<Rc<DurableJobReceipt>, ExecutionServiceError> {
        loop {
            let existing = self
                .state
                .lock()
                .expect("native service poisoned")
                .durable_jobs
                .get(&submitted.handle.id)
                .map(Rc::clone);
            if let Some(existing) = existing {
                if existing.handle != submitted.handle
                    || existing.retain_until_unix_millis != submitted.retain_until_unix_millis
                    || !existing.authority.same_admission(&authority)
                {
                    return Err(ExecutionServiceError::ForeignScope);
                }
                let state = self.state.lock().expect("native service poisoned");
                if state
                    .durable_jobs
                    .get(&submitted.handle.id)
                    .is_some_and(|current| Rc::ptr_eq(current, &existing))
                {
                    return Ok(existing);
                }
                continue;
            }
            if matches!(&authority, JobResultAuthority::SelectedGraph(_))
                && submitted.disposition == crate::supervisor::JobSubmissionDisposition::Duplicate
            {
                // Persistent bytes cannot recreate an issuing session's selected leases.
                return Err(ExecutionServiceError::UnknownHandle);
            }
            let mut state = self.state.lock().expect("native service poisoned");
            if state.durable_jobs.contains_key(&submitted.handle.id) {
                continue;
            }
            let record = Rc::new(DurableJobReceipt::new(
                submitted.handle,
                authority,
                submitted.retain_until_unix_millis,
            ));
            state
                .durable_jobs
                .insert(record.handle.id, Rc::clone(&record));
            return Ok(record);
        }
    }

    pub(super) fn durable_receipt(
        &self,
        handle: &runmat_value::SharedJobHandle,
    ) -> Result<Rc<DurableJobReceipt>, ExecutionServiceError> {
        let existing = self
            .state
            .lock()
            .expect("native service poisoned")
            .durable_jobs
            .get(&handle.id)
            .map(Rc::clone);
        if let Some(record) = existing {
            return if &record.handle == handle.as_ref() {
                Ok(record)
            } else {
                Err(ExecutionServiceError::UnknownHandle)
            };
        }
        // Only portable programs can be imported without a local selected receipt.
        // The supervisor validates the complete original handle before returning metadata.
        let lookup = self.durable.raw_output_attempt(handle.clone(), 1)?;
        if lookup.workload_kind != crate::supervisor::LocalJobWorkloadKind::Program {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        self.retain_durable_receipt(
            crate::durable::SubmittedJob {
                handle: lookup.handle,
                retain_until_unix_millis: lookup.retain_until_unix_millis,
                disposition: crate::supervisor::JobSubmissionDisposition::Duplicate,
            },
            JobResultAuthority::Direct,
        )
    }

    /// Release retained V3/source leases only when the durable supervisor's
    /// terminal/retention authority has expired. Never hold the service lock
    /// across the supervisor query, which may reenter a callback or block.
    pub(super) fn prune_expired_job_receipts(&self) -> Result<(), ExecutionServiceError> {
        self.prune_expired_job_receipts_matching(None)
    }

    /// Awaiting one job must not synchronously query retention for unrelated jobs.
    /// Submission still reclaims all expired receipts; an awaited expired receipt
    /// keeps the same supervisor-authorized retirement check.
    pub(super) fn prune_expired_job_receipt(
        &self,
        id: runmat_execution::JobId,
    ) -> Result<(), ExecutionServiceError> {
        self.prune_expired_job_receipts_matching(Some(id))
    }

    fn prune_expired_job_receipts_matching(
        &self,
        selected: Option<runmat_execution::JobId>,
    ) -> Result<(), ExecutionServiceError> {
        let now = crate::supervisor::unix_millis();
        let candidates = {
            let state = self.state.lock().expect("native service poisoned");
            state
                .durable_jobs
                .iter()
                .filter(|(id, record)| {
                    selected.is_none_or(|selected| selected == **id)
                        && record.retain_until_unix_millis <= now
                })
                .map(|(id, record)| (*id, Rc::clone(record)))
                .collect::<Vec<_>>()
        };
        if candidates.is_empty() {
            return Ok(());
        }
        let expired = self
            .durable
            .expired_job_receipts(candidates.iter().map(|(id, _)| *id).collect())?;
        let mut state = self.state.lock().expect("native service poisoned");
        let mut retired = Vec::new();
        for id in expired {
            if let Some((_, original)) = candidates.iter().find(|(candidate, _)| *candidate == id) {
                if state
                    .durable_jobs
                    .get(&id)
                    .is_some_and(|record| Rc::ptr_eq(record, original))
                {
                    retired.extend(state.durable_jobs.remove(&id));
                }
            }
        }
        drop(state);
        drop(retired);
        Ok(())
    }
}

pub(super) enum JobResultAuthority {
    Direct,
    SelectedGraph(Rc<dyn GraphResultReceipt>),
}

impl JobResultAuthority {
    pub(super) fn check_context(
        &self,
        runtime: &RuntimeContext,
    ) -> Result<(), ExecutionServiceError> {
        match self {
            Self::Direct => Ok(()),
            Self::SelectedGraph(receipt) => receipt.check_issuing_context(runtime),
        }
    }

    pub(super) fn same_admission(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Direct, Self::Direct) => true,
            (Self::SelectedGraph(left), Self::SelectedGraph(right)) => {
                left.same_selected_admission(right.as_ref())
            }
            _ => false,
        }
    }

    pub(super) fn decode_failure(
        &self,
        runtime: &RuntimeContext,
        failure: &runmat_execution::ProgramRuntimeFailure,
    ) -> Result<runmat_runtime::RuntimeError, ExecutionServiceError> {
        match self {
            Self::Direct => Ok(
                ExecutionServiceError::RuntimeFailure(Box::new(failure.clone()))
                    .into_runtime_error_in_context(Some(runtime)),
            ),
            Self::SelectedGraph(receipt) => receipt.decode_failure_in_context(runtime, failure),
        }
    }

    pub(super) fn decode(
        &self,
        runtime: &RuntimeContext,
        response: ProgramExecutionResponse,
        requested: usize,
    ) -> Result<RootedValueSequence, ExecutionServiceError> {
        self.check_context(runtime)?;
        match self {
            Self::Direct => {
                let value = crate::durable::decode_outputs(response, requested)?;
                ValueSequence::single(value)
                    .map_err(|error| ExecutionServiceError::Failed(error.to_string()))
                    .and_then(RootedValueSequence::new)
            }
            Self::SelectedGraph(receipt) => match response {
                ProgramExecutionResponse::GraphSuccess { graph } => {
                    let result = receipt.decode_in_context(runtime, &graph)?;
                    if result.as_slice().len() != requested {
                        return Err(ExecutionServiceError::InvalidOutputContract);
                    }
                    Ok(result)
                }
                ProgramExecutionResponse::Failure { message } => {
                    Err(ExecutionServiceError::Failed(message))
                }
                ProgramExecutionResponse::RuntimeFailure { failure } => {
                    Err(self.decode_failure(runtime, &failure)?.into())
                }
                _ => Err(ExecutionServiceError::Infrastructure(
                    "graph job returned an incompatible durable response".into(),
                )),
            },
        }
    }
}

pub(super) struct DurableJobReceipt {
    pub(super) handle: JobHandle,
    pub(super) authority: JobResultAuthority,
    // One GC-rooted result owner, shared by repeated fetches until retention expires.
    pub(super) terminal: RefCell<Option<Result<RootedValueSequence, ExecutionServiceError>>>,
    pub(super) retain_until_unix_millis: u64,
    pub(super) output: RefCell<DurableCallerOutput>,
}

impl DurableJobReceipt {
    pub(super) fn new(
        handle: JobHandle,
        authority: JobResultAuthority,
        retain_until_unix_millis: u64,
    ) -> Self {
        Self {
            handle,
            authority,
            terminal: RefCell::new(None),
            retain_until_unix_millis,
            output: RefCell::new(Default::default()),
        }
    }
}

//! Native adapter state retains the original minted handle, never caller projections.
use super::*;

pub(super) enum FutureState {
    Deferred(DeferredCall),
    ExecutingInCaller,
    Running(Arc<TaskCompletion>),
    Completed(Result<Value, ExecutionServiceError>),
    Cancelled,
}

pub(super) struct FutureRecord {
    pub handle: FutureHandle,
    pub state: FutureState,
}

pub(super) struct TaskRecord {
    pub handle: TaskHandle,
    pub future_id: FutureId,
}

pub(super) struct State {
    pub next_future: u64,
    pub next_task: u64,
    pub futures: HashMap<FutureId, FutureRecord>,
    pub tasks: HashMap<TaskId, TaskRecord>,
}

impl State {
    pub fn future(&self, handle: &FutureHandle) -> Result<&FutureRecord, ExecutionServiceError> {
        let record = self
            .futures
            .get(&handle.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if record.handle != *handle {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        Ok(record)
    }

    pub fn future_mut(
        &mut self,
        handle: &FutureHandle,
    ) -> Result<&mut FutureRecord, ExecutionServiceError> {
        let record = self
            .futures
            .get_mut(&handle.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if record.handle != *handle {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        Ok(record)
    }

    pub fn task_future(&self, handle: &TaskHandle) -> Result<&FutureHandle, ExecutionServiceError> {
        let task = self
            .tasks
            .get(&handle.id)
            .ok_or(ExecutionServiceError::UnknownHandle)?;
        if task.handle != *handle {
            return Err(ExecutionServiceError::UnknownHandle);
        }
        self.futures
            .get(&task.future_id)
            .map(|record| &record.handle)
            .ok_or(ExecutionServiceError::UnknownHandle)
    }
}

//! Linear native worker preparation outside the service state mutex.

use super::{
    materialize_call, FutureState, NativeExecutionService, RunningCompletion, State, TaskRecord,
};
use crate::driver::{LocalDriver, LocalGraphSubmission, LocalProgramSubmission};
use runmat_execution::{CancellationReason, FutureHandle, TaskHandle, TaskId};
use runmat_runtime::execution::{
    DeferredCall, DeferredProgram, ExecutionServiceError, GraphResultReceipt,
};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub(super) struct Reservation {
    pub(super) id: u64,
    pub(super) call: Rc<DeferredCall>,
    pub(super) cancelled: bool,
}

enum PreparedSubmission {
    Direct(Box<PreparedDirectSubmission>),
    Graph(Box<PreparedGraphSubmission>),
}

struct PreparedDirectSubmission {
    callable: runmat_execution::ProgramCallable,
    invocation_context: runmat_execution::ProgramInvocationContext,
    recipe: runmat_execution_artifact::ProgramBuildRecipe,
    artifact: runmat_execution_artifact::ProgramArtifact,
    inputs: Vec<runmat_execution::value::ValuePayload>,
}

struct PreparedGraphSubmission {
    request: runmat_execution_artifact::GraphExecutionRequest,
    receipt: Rc<dyn GraphResultReceipt>,
}

fn prepare_submission(
    call: &DeferredCall,
    outputs: runmat_execution::OutputContract,
) -> Result<PreparedSubmission, ExecutionServiceError> {
    let target = runmat_execution_artifact::ProgramTarget::portable(format!(
        "{}-{}-interpreter-bytecode-v2",
        std::env::consts::ARCH,
        std::env::consts::OS
    ));
    if matches!(&call.program, DeferredProgram::SelectedGraph(_)) {
        let (_, request, receipt) = runmat_vm::materialize_deferred_graph_call(
            call,
            outputs,
            target,
            runmat_execution::value::ValueLimits::default(),
        )?;
        return Ok(PreparedSubmission::Graph(Box::new(
            PreparedGraphSubmission { request, receipt },
        )));
    }
    let (callable, invocation_context, recipe, artifact, inputs) = materialize_call(call, outputs)?;
    Ok(PreparedSubmission::Direct(Box::new(
        PreparedDirectSubmission {
            callable,
            invocation_context,
            recipe,
            artifact,
            inputs,
        },
    )))
}

fn same_reservation(state: &FutureState, id: u64) -> Option<bool> {
    match state {
        FutureState::Materializing(reservation) if reservation.id == id => {
            Some(reservation.cancelled)
        }
        _ => None,
    }
}

fn reserve(
    state: &Mutex<State>,
    future: &FutureHandle,
) -> Result<(u64, Rc<DeferredCall>), ExecutionServiceError> {
    let mut state = state.lock().expect("native service poisoned");
    let call = match state.futures.validate(&future.id, &future.outputs)? {
        FutureState::Deferred(call) => Rc::clone(call),
        _ => {
            return Err(ExecutionServiceError::Failed(
                "future has already been scheduled".into(),
            ));
        }
    };
    call.require_spawn_admission(&future.outputs)?;
    let id = state.next_materialization;
    state.next_materialization = id.checked_add(1).ok_or_else(|| {
        ExecutionServiceError::Infrastructure("native materialization identity exhausted".into())
    })?;
    let retired = state.futures.insert(
        future.id,
        FutureState::Materializing(Reservation {
            id,
            call: Rc::clone(&call),
            cancelled: false,
        }),
    );
    drop(state);
    drop(retired);
    Ok((id, call))
}

fn settle_failed(
    state: &Mutex<State>,
    future: &FutureHandle,
    id: u64,
) -> Result<(), ExecutionServiceError> {
    let mut state = state.lock().expect("native service poisoned");
    let current = state
        .futures
        .get(&future.id)
        .and_then(|future| same_reservation(future, id))
        .ok_or(ExecutionServiceError::Cancelled)?;
    let next = if current {
        FutureState::Cancelled
    } else {
        let FutureState::Materializing(reservation) = state
            .futures
            .get(&future.id)
            .expect("reservation was validated")
        else {
            unreachable!("validated materialization changed under mutex")
        };
        FutureState::Deferred(Rc::clone(&reservation.call))
    };
    let retired = state
        .futures
        .insert(future.id, next)
        .expect("reservation was validated");
    drop(state);
    drop(retired);
    if current {
        Err(ExecutionServiceError::Cancelled)
    } else {
        Ok(())
    }
}

fn settle_submission_failed(
    state: &Mutex<State>,
    future: &FutureHandle,
    id: u64,
    failure: ExecutionServiceError,
) -> Result<(), ExecutionServiceError> {
    let mut state = state.lock().expect("native service poisoned");
    let cancelled = state
        .futures
        .get(&future.id)
        .and_then(|future| same_reservation(future, id))
        .ok_or(ExecutionServiceError::Cancelled)?;
    let terminal = if cancelled {
        FutureState::Cancelled
    } else {
        FutureState::Completed(Err(failure))
    };
    let retired = state.futures.insert(future.id, terminal);
    drop(state);
    drop(retired);
    if cancelled {
        Err(ExecutionServiceError::Cancelled)
    } else {
        Ok(())
    }
}

struct MaterializationGuard<'a> {
    state: &'a Mutex<State>,
    driver: &'a Arc<LocalDriver>,
    future: &'a FutureHandle,
    id: u64,
    task_id: Option<TaskId>,
    armed: bool,
    submitting: bool,
}

impl Drop for MaterializationGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            if self.submitting {
                if let Some(task_id) = self.task_id {
                    // A driver panic may poison its own lock. Preserve the
                    // original unwind and never make this future schedulable
                    // again, even if task-level cancellation also fails.
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.driver.cancel_task(task_id, CancellationReason::User)
                    }));
                }
                let _ = settle_submission_failed(
                    self.state,
                    self.future,
                    self.id,
                    ExecutionServiceError::Infrastructure(
                        "native submission unwound after entering the driver".into(),
                    ),
                );
            } else {
                let _ = settle_failed(self.state, self.future, self.id);
            }
        }
    }
}

mod batch;
pub(super) use batch::{release as release_callback_batch, submit as submit_callback_batch};

pub(super) fn spawn(
    service: &NativeExecutionService,
    future: &FutureHandle,
    owner: Option<runmat_runtime::context::host_allocation::HostControlAllocationOwner>,
) -> Result<TaskHandle, ExecutionServiceError> {
    service.validate_scope(future.scope_id)?;
    let (reservation_id, call) = reserve(&service.state, future)?;
    let mut guard = MaterializationGuard {
        state: &service.state,
        driver: &service.driver,
        future,
        id: reservation_id,
        task_id: None,
        armed: true,
        submitting: false,
    };

    let prepared = match prepare_submission(&call, future.outputs.clone()) {
        Ok(prepared) => prepared,
        Err(error) => {
            drop(call);
            let outcome = settle_failed(&service.state, future, reservation_id);
            guard.armed = false;
            outcome?;
            return Err(error);
        }
    };

    let task_id = {
        let mut state = service.state.lock().expect("native service poisoned");
        let cancelled = state
            .futures
            .get(&future.id)
            .and_then(|future| same_reservation(future, reservation_id))
            .ok_or(ExecutionServiceError::Cancelled)?;
        if cancelled {
            drop(state);
            drop(call);
            let outcome = settle_failed(&service.state, future, reservation_id);
            guard.armed = false;
            outcome?;
            return Err(ExecutionServiceError::Cancelled);
        }
        let sequence = state.next_task;
        state.next_task = sequence.wrapping_add(1);
        TaskId::derive(&[service.scope_id.bytes(), &sequence.to_be_bytes()])
    };

    guard.task_id = Some(task_id);
    guard.submitting = true;
    let submitted = match prepared {
        PreparedSubmission::Direct(prepared) => {
            let PreparedDirectSubmission {
                callable,
                invocation_context,
                recipe,
                artifact,
                inputs,
            } = *prepared;
            service
                .driver
                .submit_with_control(
                    LocalProgramSubmission {
                        task_id,
                        callable,
                        invocation_context,
                        recipe,
                        artifact,
                        inputs,
                        outputs: future.outputs.clone(),
                        retry: call.retry,
                        environment: call.environment.clone(),
                        console: call.console,
                    },
                    owner,
                )
                .map(|completion| (completion, None))
        }
        PreparedSubmission::Graph(prepared) => {
            let PreparedGraphSubmission { request, receipt } = *prepared;
            service
                .driver
                .submit_graph_with_control(
                    LocalGraphSubmission {
                        task_id,
                        request,
                        retry: call.retry,
                        task: None,
                    },
                    owner,
                )
                .map(|completion| (completion, Some(receipt)))
        }
    };
    let (completion, graph_receipt) = match submitted {
        Ok(accepted) => accepted,
        Err(error) => {
            drop(call);
            let failure = match service.driver.cancel_task(task_id, CancellationReason::User) {
                Ok(_) => ExecutionServiceError::Failed(error.to_string()),
                Err(cancel_error) => ExecutionServiceError::Infrastructure(format!(
                    "native submission failed ({error}) and accepted-task cancellation failed ({cancel_error})"
                )),
            };
            let outcome =
                settle_submission_failed(&service.state, future, reservation_id, failure.clone());
            guard.armed = false;
            outcome?;
            return Err(failure);
        }
    };

    drop(call);

    let (committed, retired) = {
        let mut state = service.state.lock().expect("native service poisoned");
        let cancelled = state
            .futures
            .get(&future.id)
            .and_then(|future| same_reservation(future, reservation_id))
            .unwrap_or(true);
        let current = state
            .futures
            .get(&future.id)
            .and_then(|future| same_reservation(future, reservation_id));
        if current.is_none() {
            (false, None)
        } else if cancelled {
            let retired = state.futures.insert(future.id, FutureState::Cancelled);
            (false, retired)
        } else {
            let running = RunningCompletion::new(Arc::clone(&completion), graph_receipt);
            let retired = state
                .futures
                .insert(future.id, FutureState::Running(running));
            state.tasks.insert(
                task_id,
                TaskRecord {
                    future_id: future.id,
                    generation: 1,
                    read: false,
                    claimed: false,
                    completion_order: None,
                    retention: runmat_runtime::execution::TaskRetention::Refetchable,
                    callback_completion: None,
                },
            );
            (true, retired)
        }
    };
    drop(retired);
    guard.armed = false;
    if !committed {
        completion.cancel();
        match service
            .driver
            .cancel_task(task_id, CancellationReason::User)
        {
            Ok(true) => return Err(ExecutionServiceError::Cancelled),
            Ok(false) => {
                return Err(ExecutionServiceError::Infrastructure(
                    "accepted native submission disappeared before task cancellation".into(),
                ));
            }
            Err(error) => {
                return Err(ExecutionServiceError::Infrastructure(format!(
                    "cancelled materialization could not retire its accepted task: {error}"
                )));
            }
        }
    }
    Ok(TaskHandle {
        id: task_id,
        scope_id: service.scope_id,
        generation: 1,
        outputs: future.outputs.clone(),
    })
}

#[cfg(test)]
mod tests;

use super::*;
use futures::stream::FuturesUnordered;
use runmat_execution::cancellation::CancellationSource;
use std::collections::HashSet;
use std::pin::Pin;
use std::rc::Rc;

type CallbackKey = (u64, u64);
type CallbackFuture = Pin<Box<dyn Future<Output = MexCallbackResult> + 'static>>;

#[derive(Default)]
pub(super) struct ClientCallbackTasks {
    pending: FuturesUnordered<CallbackFuture>,
    active: HashMap<CallbackKey, CancellationSource>,
    completed: HashSet<CallbackKey>,
}

impl ClientCallbackTasks {
    pub(super) fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    pub(super) async fn next(&mut self) -> Option<MexCallbackResult> {
        self.pending.next().await
    }

    pub(super) fn start(
        &mut self,
        callback: MexCallbackRequest,
        origin: RuntimeContext,
        snapshots: Rc<SharedSnapshotStore>,
    ) -> Result<(), MexWireError> {
        callback
            .validate()
            .map_err(|error| wire_error("RunMat:MEX:CallbackProtocol", error.to_string(), None))?;
        let child_depth = origin.mex_callback_ancestry().child().ok_or_else(|| {
            wire_error(
                "RunMat:MEX:CallbackDepth",
                "callback depth limit exceeded",
                None,
            )
        })?;
        if callback.depth != child_depth.wire() {
            return Err(wire_error(
                "RunMat:MEX:CallbackProtocol",
                "callback depth does not match the originating invocation",
                None,
            ));
        }
        let key = (callback.request_id, callback.callback_id);
        if self.active.contains_key(&key) || self.completed.contains(&key) {
            return Err(wire_error(
                "RunMat:MEX:CallbackProtocol",
                "duplicate callback identity",
                None,
            ));
        }
        let cancellation = CancellationSource::new();
        let origin = callback_runtime(
            origin.with_mex_callback_ancestry(child_depth),
            &cancellation,
        );
        self.active.insert(key, cancellation);
        self.pending
            .push(Box::pin(handle_callback(callback, origin, snapshots)));
        Ok(())
    }

    pub(super) fn cancel(
        &mut self,
        cancellation: MexCallbackCancellation,
    ) -> Result<(), MexWireError> {
        let key = (cancellation.request_id, cancellation.callback_id);
        if let Some(source) = self.active.get(&key) {
            source.cancel(runmat_execution::CancellationReason::User);
            return Ok(());
        }
        if self.completed.contains(&key) {
            return Ok(());
        }
        Err(wire_error(
            "RunMat:MEX:CallbackProtocol",
            "cancellation refers to an unknown callback",
            None,
        ))
    }

    pub(super) fn finish(&mut self, response: &MexCallbackResult) -> Result<(), MexWireError> {
        let key = (response.request_id, response.callback_id);
        if self.active.remove(&key).is_none() {
            return Err(wire_error(
                "RunMat:MEX:CallbackProtocol",
                "completed callback identity is not active",
                None,
            ));
        }
        self.completed.insert(key);
        Ok(())
    }

    pub(super) fn release_request(&mut self, request_id: u64) {
        // OriginReleased is terminal for the host-side request. Completed IDs
        // no longer need tombstones; still-running callbacks keep their exact
        // reply route, but must stop executing effects for an orphaned host.
        for ((request, _), source) in &self.active {
            if *request == request_id {
                source.cancel(runmat_execution::CancellationReason::User);
            }
        }
        self.completed.retain(|(request, _)| *request != request_id);
    }
}

fn callback_runtime(origin: RuntimeContext, cancellation: &CancellationSource) -> RuntimeContext {
    let combined = origin.cancellation().or(&cancellation.token());
    origin.with_cancellation_token(combined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foreign::MexCallbackAncestry;
    use runmat_execution::CancellationReason;

    #[test]
    fn exact_running_cancellation_reaches_runtime_without_cancelling_sibling() {
        let parent_cancel = CancellationSource::new();
        let parent = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_cancellation_token(parent_cancel.token());
        let mut tasks = ClientCallbackTasks::default();
        let snapshots = Rc::new(SharedSnapshotStore::create().expect("snapshot store"));
        for callback_id in [1, 2] {
            tasks
                .start(
                    MexCallbackRequest {
                        request_id: 7,
                        callback_id,
                        depth: 1,
                        operation: MexCallbackOperation::Eval {
                            command: "pending".into(),
                        },
                    },
                    parent.clone(),
                    Rc::clone(&snapshots),
                )
                .unwrap();
        }
        let first_runtime = callback_runtime(parent.clone(), tasks.active.get(&(7, 1)).unwrap());
        let second_runtime = callback_runtime(parent.clone(), tasks.active.get(&(7, 2)).unwrap());
        assert!(tasks.has_pending());

        tasks
            .cancel(MexCallbackCancellation {
                request_id: 7,
                callback_id: 1,
            })
            .unwrap();
        assert!(first_runtime.is_cancelled());
        assert!(!second_runtime.is_cancelled());
        assert!(!parent.is_cancelled());

        tasks
            .finish(&MexCallbackResult {
                request_id: 7,
                callback_id: 1,
                outcome: Ok(MexCallbackOutput::Unit),
            })
            .unwrap();
        tasks
            .cancel(MexCallbackCancellation {
                request_id: 7,
                callback_id: 1,
            })
            .expect("late cancellation of an exact completed callback is harmless");
        assert!(tasks
            .cancel(MexCallbackCancellation {
                request_id: 7,
                callback_id: 3,
            })
            .is_err());
        assert!(!second_runtime.is_cancelled());

        tasks.release_request(7);
        assert!(second_runtime.is_cancelled());
        assert!(tasks.completed.is_empty());

        let unrelated = CancellationSource::new();
        let unrelated_runtime = callback_runtime(parent.clone(), &unrelated);
        tasks.active.insert((8, 1), unrelated);
        tasks.release_request(7);
        assert!(!unrelated_runtime.is_cancelled());
        parent_cancel.cancel(CancellationReason::User);
        assert!(unrelated_runtime.is_cancelled());
    }

    #[test]
    fn callback_admission_uses_exact_parent_ancestry_before_starting_work() {
        let parent = RuntimeContext::new(Rc::new(crate::execution::RuntimeExecutionService::new()))
            .with_mex_callback_ancestry(MexCallbackAncestry::from_wire(1).unwrap());
        let snapshots = Rc::new(SharedSnapshotStore::create().expect("snapshot store"));
        let mut tasks = ClientCallbackTasks::default();
        let callback = |callback_id, depth| MexCallbackRequest {
            request_id: 19,
            callback_id,
            depth,
            operation: MexCallbackOperation::Eval {
                command: "nested".into(),
            },
        };
        let mismatch = tasks
            .start(callback(1, 1), parent.clone(), Rc::clone(&snapshots))
            .expect_err("a valid wire depth must still match the originating invocation");
        assert_eq!(mismatch.identifier, "RunMat:MEX:CallbackProtocol");
        assert!(!tasks.has_pending());
        assert!(tasks.active.is_empty());

        tasks
            .start(callback(1, 2), parent.clone(), Rc::clone(&snapshots))
            .expect("the exact child depth is admitted");
        assert_eq!(tasks.active.len(), 1);
        assert_eq!(parent.mex_callback_ancestry().wire(), 1);

        let limit = parent.with_mex_callback_ancestry(
            MexCallbackAncestry::from_wire(64).expect("protocol maximum"),
        );
        let exceeded = tasks
            .start(callback(2, 64), limit.clone(), snapshots)
            .expect_err("a 65th callback must be rejected before work starts");
        assert_eq!(exceeded.identifier, "RunMat:MEX:CallbackDepth");
        assert_eq!(tasks.active.len(), 1);
        assert_eq!(limit.mex_callback_ancestry().wire(), 64);
    }
}

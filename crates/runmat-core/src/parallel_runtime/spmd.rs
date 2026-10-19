use super::collective::{LabCollectiveService, SharedCollectives};
use runmat_execution::{CompositeHandle, CompositeId, GangHandle, GangRequest, SpmdTaskContext};
use runmat_execution_runner::{DistributedStore, GangCoordinator};
use runmat_runtime::context::{
    RuntimeCollectiveService, RuntimeServiceFuture, RuntimeSpmdAdmission, RuntimeSpmdGangLease,
    RuntimeSpmdOutput, RuntimeSpmdRetainedOutput, RuntimeSpmdService,
};
use runmat_runtime::RuntimeError;
use runmat_types::{LabCount, ParallelRegionId};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Default)]
pub(crate) struct CoreSpmdService {
    gangs: Rc<RefCell<GangCoordinator>>,
    collectives: Rc<SharedCollectives>,
    values: Rc<RefCell<DistributedStore>>,
}

struct CoreGangLease {
    handle: GangHandle,
    gangs: Rc<RefCell<GangCoordinator>>,
    collectives: Rc<SharedCollectives>,
    retired: Cell<bool>,
}

impl RuntimeSpmdGangLease for CoreGangLease {
    fn gang_handle(&self) -> &GangHandle {
        &self.handle
    }

    fn retire(&self) -> Result<(), RuntimeError> {
        if self.retired.get() {
            return Ok(());
        }
        // No coordinator borrow is held across a service callback or await.
        // The same exact generation is checked by GangCoordinator::retire.
        self.gangs
            .borrow_mut()
            .retire(&self.handle)
            .map_err(runtime_error)?;
        // Collective completion can wake external execution. Publish the
        // retired state after releasing the coordinator borrow and before
        // those callbacks can observe or reenter this lease.
        self.retired.set(true);
        self.collectives.retire_gang(&self.handle);
        Ok(())
    }
}

impl Drop for CoreGangLease {
    fn drop(&mut self) {
        let _ = self.retire();
    }
}

impl CoreSpmdService {
    pub(super) fn new(values: Rc<RefCell<DistributedStore>>) -> Self {
        Self {
            gangs: Rc::default(),
            collectives: Rc::default(),
            values,
        }
    }
}

impl RuntimeSpmdService for CoreSpmdService {
    fn admit(
        &self,
        request: GangRequest,
        available_labs: LabCount,
        region: ParallelRegionId,
    ) -> RuntimeServiceFuture<Result<RuntimeSpmdAdmission, RuntimeError>> {
        let admitted = self
            .gangs
            .borrow_mut()
            .admit(request, available_labs)
            .map_err(runtime_error)
            .map(|gang| {
                let lifetime: Rc<dyn RuntimeSpmdGangLease> = Rc::new(CoreGangLease {
                    handle: gang.handle.clone(),
                    gangs: Rc::clone(&self.gangs),
                    collectives: Rc::clone(&self.collectives),
                    retired: Cell::new(false),
                });
                (gang, lifetime)
            });
        let collectives = Rc::clone(&self.collectives);
        Box::pin(async move {
            let (gang, lifetime) = admitted?;
            let labs = gang
                .ranks
                .iter()
                .copied()
                .map(|rank| {
                    Rc::new(LabCollectiveService::new(
                        SpmdTaskContext {
                            gang: gang.handle.clone(),
                            region,
                            rank,
                        },
                        Rc::clone(&collectives),
                    )) as Rc<dyn RuntimeCollectiveService>
                })
                .collect();
            let admission = RuntimeSpmdAdmission {
                gang,
                labs,
                lifetime,
            };
            admission.validate()?;
            Ok(admission)
        })
    }

    fn rank_finished(
        &self,
        gang: &GangHandle,
        rank: runmat_types::LabRank,
    ) -> Result<(), RuntimeError> {
        if rank.0 == 0 || rank.0 > gang.labs.0 {
            return Err(runtime_error(
                "completed SPMD rank is outside the admitted gang",
            ));
        }
        self.collectives.rank_finished(gang, rank);
        Ok(())
    }

    fn rank_failed(
        &self,
        gang: &GangHandle,
        rank: runmat_types::LabRank,
    ) -> Result<(), RuntimeError> {
        if rank.0 == 0 || rank.0 > gang.labs.0 {
            return Err(runtime_error(
                "failed SPMD rank is outside the admitted gang",
            ));
        }
        self.collectives.fail_gang(gang, "an SPMD peer failed");
        Ok(())
    }

    fn retain_outputs(
        &self,
        gang: GangHandle,
        region: ParallelRegionId,
        outputs: Vec<RuntimeSpmdOutput>,
    ) -> RuntimeServiceFuture<Result<Vec<RuntimeSpmdRetainedOutput>, RuntimeError>> {
        let result = outputs
            .into_iter()
            .map(|output| {
                if output.value.function != region.0.function {
                    return Err(runtime_error(
                        "SPMD output identity does not belong to the executing function",
                    ));
                }
                let distributed = output
                    .entries
                    .iter()
                    .filter_map(|entry| match entry {
                        Some(runmat_execution::SpmdOutputValue::Distributed(snapshot)) => {
                            Some((**snapshot).clone())
                        }
                        Some(runmat_execution::SpmdOutputValue::Value(_)) | None => None,
                    })
                    .collect::<Vec<_>>();
                if !distributed.is_empty() {
                    if distributed.len() != output.entries.len()
                        || distributed
                            .iter()
                            .any(|snapshot| snapshot.handle != distributed[0].handle)
                        || distributed.iter().enumerate().any(|(index, snapshot)| {
                            usize::try_from(snapshot.owned.layout.rank.0).ok() != Some(index + 1)
                        })
                    {
                        return Err(runtime_error(
                            "SPMD output must contain one ordered shard from every rank of one distributed value",
                        ));
                    }
                    for snapshot in &distributed {
                        self.values
                            .borrow_mut()
                            .import_shard(snapshot.clone())
                            .map_err(runtime_error)?;
                    }
                    return Ok(RuntimeSpmdRetainedOutput::Distributed(
                        distributed[0].handle.clone(),
                    ));
                }
                let entries = output
                    .entries
                    .into_iter()
                    .map(|entry| match entry {
                        Some(runmat_execution::SpmdOutputValue::Value(value)) => Ok(Some(value)),
                        Some(runmat_execution::SpmdOutputValue::Distributed(_)) => {
                            unreachable!("distributed entries were handled above")
                        }
                        None => Ok(None),
                    })
                    .collect::<Result<Vec<_>, RuntimeError>>()?;
                let handle = CompositeHandle {
                    id: CompositeId::derive(&[
                        gang.id.bytes(),
                        &gang.generation.to_be_bytes(),
                        &region.0.ordinal.to_be_bytes(),
                        &output.value.local.to_be_bytes(),
                    ]),
                    owner_region: region,
                    scope_id: gang.scope_id,
                    generation: gang.generation,
                    gang: gang.clone(),
                    value: output.fact,
                };
                self.values
                    .borrow_mut()
                    .insert_composite(handle.clone(), entries)
                    .map_err(runtime_error)?;
                Ok(RuntimeSpmdRetainedOutput::Composite(handle))
            })
            .collect();
        Box::pin(async move { result })
    }
}

fn runtime_error(error: impl std::fmt::Display) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:parallel:GangAdmission",
        error.to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{
        CollectiveInvocation, CollectiveRequest, CollectiveSequence, ExecutionScopeId, GangId,
        PoolHandle, PoolId,
    };
    use runmat_types::{CollectiveId, ProgramFunctionId, RegionId, SpmdLabRequirement};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct Reentry {
        lease: Rc<dyn RuntimeSpmdGangLease>,
        second_lab: Rc<dyn RuntimeCollectiveService>,
        gangs: Rc<RefCell<GangCoordinator>>,
        handle: GangHandle,
        region: ParallelRegionId,
    }

    thread_local! {
        static REENTRY: RefCell<Option<Reentry>> = const { RefCell::new(None) };
    }

    struct InspectWake(AtomicUsize);

    impl Wake for InspectWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }

        fn wake_by_ref(self: &Arc<Self>) {
            REENTRY.with(|slot| {
                let borrowed = slot.borrow();
                let state = borrowed.as_ref().expect("reentrant test state");
                assert!(
                    state.gangs.borrow().snapshot(&state.handle).is_err(),
                    "the exact gang must retire before waking a collective waiter"
                );
                state
                    .lease
                    .retire()
                    .expect("reentrant retirement is idempotent");
                // execute() submits before returning its future. That submit
                // borrows the same pending-sender map; a live RefMut held
                // across the first sender's wake would panic here.
                let request = CollectiveRequest {
                    context: state.second_lab.context().clone(),
                    id: CollectiveId {
                        region: state.region,
                        ordinal: 2,
                    },
                    sequence: CollectiveSequence(1),
                    invocation: CollectiveInvocation::Barrier,
                };
                drop(state.second_lab.execute(request));
            });
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn admission_future_and_last_lease_drop_retire_exact_gang_generation() {
        let values = Rc::new(RefCell::new(DistributedStore::default()));
        let service = CoreSpmdService::new(values);
        let scope_id = ExecutionScopeId::derive(&[b"spmd-lifetime"]);
        let pool = PoolHandle {
            id: PoolId::derive(&[b"spmd-lifetime-pool"]),
            scope_id,
            generation: 1,
        };
        let request = GangRequest {
            pool: pool.clone(),
            labs: SpmdLabRequirement::Exact { labs: LabCount(2) },
        };
        let region = ParallelRegionId(RegionId {
            function: ProgramFunctionId(1),
            ordinal: 1,
        });

        // Core reserves the gang before returning the admission future. Its
        // own future must therefore hold the lifetime even when never polled.
        let pending = service.admit(request.clone(), LabCount(2), region);
        let first = GangHandle {
            id: GangId::derive(&[pool.id.bytes(), pool.scope_id.bytes(), &1_u64.to_be_bytes()]),
            scope_id,
            generation: 1,
            pool: pool.clone(),
            labs: LabCount(2),
        };
        assert!(service.gangs.borrow().snapshot(&first).is_ok());
        drop(pending);
        assert!(service.gangs.borrow().snapshot(&first).is_err());

        let admission = futures::executor::block_on(service.admit(request, LabCount(2), region))
            .expect("second admitted generation");
        admission.validate().expect("exact lease identity");
        let second = admission.gang.handle.clone();
        let retained = Rc::clone(&admission.lifetime);
        let first_lab = Rc::clone(&admission.labs[0]);
        let second_lab = Rc::clone(&admission.labs[1]);
        drop(admission);
        assert!(service.gangs.borrow().snapshot(&second).is_ok());

        let request = CollectiveRequest {
            context: first_lab.context().clone(),
            id: CollectiveId { region, ordinal: 1 },
            sequence: CollectiveSequence(1),
            invocation: CollectiveInvocation::Barrier,
        };
        let mut pending = first_lab.execute(request);
        let waking = Arc::new(InspectWake(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&waking));
        let mut task = Context::from_waker(&waker);
        assert!(matches!(pending.as_mut().poll(&mut task), Poll::Pending));
        REENTRY.with(|slot| {
            *slot.borrow_mut() = Some(Reentry {
                lease: Rc::clone(&retained),
                second_lab,
                gangs: Rc::clone(&service.gangs),
                handle: second.clone(),
                region,
            });
        });
        retained.retire().expect("normal completion retirement");
        assert_eq!(waking.0.load(Ordering::SeqCst), 1);
        assert!(matches!(
            pending.as_mut().poll(&mut task),
            Poll::Ready(Err(_))
        ));
        REENTRY.with(|slot| {
            slot.borrow_mut().take();
        });
        retained
            .retire()
            .expect("same lease retirement is idempotent");
        assert!(service.gangs.borrow().snapshot(&second).is_err());
        drop(retained);
        assert!(service.gangs.borrow().snapshot(&second).is_err());
    }
}

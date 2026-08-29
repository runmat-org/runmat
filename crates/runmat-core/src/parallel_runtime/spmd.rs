use super::collective::{LabCollectiveService, SharedCollectives};
use runmat_execution::{CompositeHandle, CompositeId, GangHandle, GangRequest, SpmdTaskContext};
use runmat_execution_runner::{DistributedStore, GangCoordinator};
use runmat_runtime::context::{
    RuntimeCollectiveService, RuntimeServiceFuture, RuntimeSpmdAdmission, RuntimeSpmdOutput,
    RuntimeSpmdRetainedOutput, RuntimeSpmdService,
};
use runmat_runtime::RuntimeError;
use runmat_types::{LabCount, ParallelRegionId};
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Default)]
pub(crate) struct CoreSpmdService {
    gangs: RefCell<GangCoordinator>,
    collectives: Rc<SharedCollectives>,
    values: Rc<RefCell<DistributedStore>>,
}

impl CoreSpmdService {
    pub(super) fn new(values: Rc<RefCell<DistributedStore>>) -> Self {
        Self {
            gangs: RefCell::default(),
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
            .map_err(runtime_error);
        let collectives = Rc::clone(&self.collectives);
        Box::pin(async move {
            let gang = admitted?;
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
            let admission = RuntimeSpmdAdmission { gang, labs };
            admission.validate()?;
            Ok(admission)
        })
    }

    fn retire(&self, gang: GangHandle) -> RuntimeServiceFuture<Result<(), RuntimeError>> {
        self.collectives.retire_gang(&gang);
        let retired = self
            .gangs
            .borrow_mut()
            .retire(&gang)
            .map(|_| ())
            .map_err(runtime_error);
        Box::pin(async move { retired })
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

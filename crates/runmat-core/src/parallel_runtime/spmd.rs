use super::collective::{LabCollectiveService, SharedCollectives};
use runmat_execution::{CompositeHandle, CompositeId, GangHandle, GangRequest, SpmdTaskContext};
use runmat_execution_runner::{DistributedStore, GangCoordinator};
use runmat_runtime::context::{
    RuntimeCollectiveService, RuntimeServiceFuture, RuntimeSpmdAdmission, RuntimeSpmdOutput,
    RuntimeSpmdService,
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
        self.collectives.fail_gang(&gang, "gang retired");
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

    fn retain_outputs(
        &self,
        gang: GangHandle,
        region: ParallelRegionId,
        outputs: Vec<RuntimeSpmdOutput>,
    ) -> RuntimeServiceFuture<Result<Vec<CompositeHandle>, RuntimeError>> {
        let result = outputs
            .into_iter()
            .map(|output| {
                if output.value.function != region.0.function {
                    return Err(runtime_error(
                        "SPMD output identity does not belong to the executing function",
                    ));
                }
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
                    .insert_composite(handle.clone(), output.entries)
                    .map_err(runtime_error)?;
                Ok(handle)
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

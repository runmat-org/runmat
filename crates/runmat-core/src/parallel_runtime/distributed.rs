use runmat_execution::{DistributedObjectId, DistributedValueHandle, PoolSnapshot};
use runmat_execution_runner::{DistributedStore, OwnedPartition};
use runmat_runtime::context::{
    RuntimeDistributedCallRequest, RuntimeDistributedService, RuntimeDistributedSnapshot,
    RuntimeServiceFuture,
};
use runmat_runtime::parallel::distribution::DistributedPartitionValue;
use runmat_runtime::RuntimeError;
use runmat_types::{DistributedValueContract, DistributionScheme, LabRank};
use runmat_value::Value;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

pub(super) struct CoreDistributedService {
    store: Rc<RefCell<DistributedStore>>,
    next_generation: Cell<u64>,
}

impl CoreDistributedService {
    pub(super) fn new(store: Rc<RefCell<DistributedStore>>) -> Self {
        Self {
            store,
            next_generation: Cell::new(1),
        }
    }

    fn reserve_generation(&self) -> Result<u64, RuntimeError> {
        let generation = self.next_generation.get();
        self.next_generation.set(
            generation
                .checked_add(1)
                .ok_or_else(|| error("distributed value generation overflowed"))?,
        );
        Ok(generation)
    }
}

impl RuntimeDistributedService for CoreDistributedService {
    fn create(
        &self,
        contract: DistributedValueContract,
        input: Value,
        pool: PoolSnapshot,
    ) -> RuntimeServiceFuture<Result<DistributedValueHandle, RuntimeError>> {
        if pool.state != runmat_execution::PoolState::Ready
            || pool.workers == 0
            || pool.handle.generation == 0
        {
            return Box::pin(async {
                Err(error(
                    "distributed values require a ready, nonempty, live pool",
                ))
            });
        }
        let generation = match self.reserve_generation() {
            Ok(generation) => generation,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        Box::pin(create_value(
            Rc::clone(&self.store),
            generation,
            contract,
            input,
            pool.handle,
            pool.workers,
        ))
    }

    fn inspect(
        &self,
        handle: DistributedValueHandle,
    ) -> RuntimeServiceFuture<Result<RuntimeDistributedSnapshot, RuntimeError>> {
        let result = self
            .store
            .borrow()
            .layouts(&handle)
            .map(|partitions| RuntimeDistributedSnapshot { handle, partitions })
            .map_err(error);
        Box::pin(async move { result })
    }

    fn local_part(
        &self,
        handle: DistributedValueHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>> {
        let result = self
            .store
            .borrow()
            .local_part(&handle, rank)
            .map_err(error)
            .and_then(|part| {
                runmat_runtime::execution::value_codec::decode_inline_value(&part.value)
                    .map_err(error)
            });
        Box::pin(async move { result })
    }

    fn materialize(
        &self,
        handle: DistributedValueHandle,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>> {
        let parts = self.store.borrow().cloned_parts(&handle).map_err(error);
        Box::pin(materialize(handle, parts))
    }

    fn redistribute(
        &self,
        handle: DistributedValueHandle,
        scheme: DistributionScheme,
    ) -> RuntimeServiceFuture<Result<DistributedValueHandle, RuntimeError>> {
        let parts = self.store.borrow().cloned_parts(&handle).map_err(error);
        let generation = match self.reserve_generation() {
            Ok(generation) => generation,
            Err(error) => return Box::pin(async move { Err(error) }),
        };
        let store = Rc::clone(&self.store);
        Box::pin(async move {
            let input = materialize(handle.clone(), parts).await?;
            create_value(
                store,
                generation,
                DistributedValueContract {
                    id: handle.contract,
                    value: handle.value,
                    scheme,
                    owner: handle.owner,
                    materializable: handle.materializable,
                },
                input,
                handle.pool,
                handle.partition_count.0,
            )
            .await
        })
    }

    fn invoke(
        &self,
        _request: RuntimeDistributedCallRequest,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>> {
        Box::pin(async {
            Err(error(
                "distributed builtin routing requires an admitted locality contract",
            ))
        })
    }

    fn composite_entry(
        &self,
        handle: runmat_execution::CompositeHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<Option<Value>, RuntimeError>> {
        let result = self
            .store
            .borrow()
            .composite_entry(&handle, rank)
            .map_err(error)
            .and_then(|entry| {
                entry
                    .map(runmat_runtime::execution::value_codec::decode_inline_value)
                    .transpose()
                    .map_err(error)
            });
        Box::pin(async move { result })
    }
}

async fn create_value(
    store: Rc<RefCell<DistributedStore>>,
    generation: u64,
    contract: DistributedValueContract,
    input: Value,
    pool: runmat_execution::PoolHandle,
    workers: u32,
) -> Result<DistributedValueHandle, RuntimeError> {
    let partition_count = runmat_types::LabCount(workers);
    let (shape, parts) = runmat_runtime::parallel::distribution::partition_value(
        &input,
        &contract.scheme,
        partition_count,
    )
    .await?;
    let handle = DistributedValueHandle {
        id: DistributedObjectId::derive(&[
            pool.scope_id.bytes(),
            &contract.id.function.0.to_be_bytes(),
            &contract.id.ordinal.to_be_bytes(),
            &generation.to_be_bytes(),
        ]),
        contract: contract.id,
        owner: contract.owner,
        scope_id: pool.scope_id,
        generation,
        pool,
        partition_count,
        value: contract.value,
        global_shape: shape
            .iter()
            .map(|dimension| {
                u64::try_from(*dimension).map_err(|_| error("distributed shape exceeds u64"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        scheme: contract.scheme,
        materializable: contract.materializable,
    };
    store
        .borrow_mut()
        .insert(handle.clone(), encode_parts(parts)?)
        .map_err(error)?;
    Ok(handle)
}

async fn materialize(
    handle: DistributedValueHandle,
    parts: Result<Vec<OwnedPartition>, RuntimeError>,
) -> Result<Value, RuntimeError> {
    if !handle.materializable {
        return Err(error(
            "this distributed value contract forbids driver materialization",
        ));
    }
    let shape = handle
        .global_shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| error("distributed shape exceeds this host"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let parts = decode_parts(parts?)?;
    runmat_runtime::parallel::distribution::materialize_partitions(&shape, &handle.scheme, &parts)
        .await
}

fn decode_parts(
    parts: Vec<OwnedPartition>,
) -> Result<Vec<DistributedPartitionValue>, RuntimeError> {
    parts
        .into_iter()
        .map(|part| {
            Ok(DistributedPartitionValue {
                layout: part.layout,
                value: runmat_runtime::execution::value_codec::decode_inline_value(&part.value)
                    .map_err(error)?,
            })
        })
        .collect()
}

fn encode_parts(
    parts: Vec<DistributedPartitionValue>,
) -> Result<Vec<OwnedPartition>, RuntimeError> {
    parts
        .into_iter()
        .map(|part| {
            Ok(OwnedPartition {
                layout: part.layout,
                value: runmat_runtime::execution::value_codec::encode_inline_value(&part.value)
                    .map_err(error)?,
            })
        })
        .collect()
}

fn error(message: impl std::fmt::Display) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:parallel:DistributedValue",
        message.to_string(),
    )
}

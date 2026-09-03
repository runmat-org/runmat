use runmat_execution::{DistributedObjectId, DistributedValueHandle, PoolSnapshot};
use runmat_execution_runner::{DistributedStore, OwnedPartition};
use runmat_runtime::context::{
    RuntimeDistributedCallRequest, RuntimeDistributedInvocation, RuntimeDistributedRetirement,
    RuntimeDistributedService, RuntimeDistributedSnapshot, RuntimeServiceFuture,
};
use runmat_runtime::parallel::distribution::DistributedPartitionValue;
use runmat_runtime::RuntimeError;
use runmat_types::{DistributedValueContract, DistributionScheme, FactSatisfaction, LabRank};
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
    fn retire_pool(
        &self,
        pool: runmat_execution::PoolHandle,
    ) -> Result<RuntimeDistributedRetirement, RuntimeError> {
        let retired = self.store.borrow_mut().retire_pool(&pool);
        Ok(RuntimeDistributedRetirement {
            distributed: retired.distributed,
            composites: retired.composites,
        })
    }

    fn create(
        &self,
        contract: DistributedValueContract,
        input: Value,
        scheme: DistributionScheme,
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
            scheme,
            pool.handle,
            pool.workers,
        ))
    }

    fn create_worker(
        &self,
        contract: DistributedValueContract,
        input: Value,
        scheme: DistributionScheme,
        context: runmat_execution::SpmdTaskContext,
    ) -> RuntimeServiceFuture<Result<DistributedValueHandle, RuntimeError>> {
        if !matches!(contract.owner, runmat_types::DistributedOwner::Region(region) if region == context.region)
            || contract.coordination.is_none()
            || context.gang.labs.0 == 0
        {
            return Box::pin(async {
                Err(error(
                    "worker distributed construction disagrees with its compiler-owned region contract",
                ))
            });
        }
        Box::pin(create_worker_value(
            Rc::clone(&self.store),
            contract,
            input,
            scheme,
            context,
        ))
    }

    fn build_worker(
        &self,
        contract: DistributedValueContract,
        local_part: Value,
        global_shape: Vec<u64>,
        scheme: DistributionScheme,
        layouts: Vec<runmat_execution::DistributedPartitionLayout>,
        context: runmat_execution::SpmdTaskContext,
    ) -> RuntimeServiceFuture<Result<DistributedValueHandle, RuntimeError>> {
        if !matches!(contract.owner, runmat_types::DistributedOwner::Region(region) if region == context.region)
            || contract.coordination.is_none()
            || context.gang.labs.0 == 0
        {
            return Box::pin(async {
                Err(error(
                    "worker local-part construction disagrees with its compiler-owned region contract",
                ))
            });
        }
        Box::pin(build_worker_value(
            Rc::clone(&self.store),
            contract,
            local_part,
            global_shape,
            scheme,
            layouts,
            context,
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

    fn export_local(
        &self,
        handle: DistributedValueHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<runmat_execution::DistributedShardSnapshot, RuntimeError>>
    {
        let result = self
            .store
            .borrow()
            .export_local(&handle, rank)
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
                    construction: runmat_types::DistributedConstruction::Fixed {
                        scheme: scheme.clone(),
                    },
                    owner: handle.owner,
                    coordination: None,
                    materializable: handle.materializable,
                },
                input,
                scheme,
                handle.pool,
                handle.partition_count.0,
            )
            .await
        })
    }

    fn invoke(
        &self,
        request: RuntimeDistributedCallRequest,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>> {
        let Some(entry) = runmat_builtins::builtin_catalog_entry_by_name(&request.builtin.0) else {
            return Box::pin(async {
                Err(error(
                    "distributed builtin request is not admitted for partition-local execution",
                ))
            });
        };
        if entry.placement.distributed
            == runmat_builtins::BuiltinDistributedPolicy::ScalarLikePrototype
        {
            return invoke_scalar_like_prototype(Rc::clone(&self.store), request, entry);
        }
        if !matches!(
            entry.placement.distributed,
            runmat_builtins::BuiltinDistributedPolicy::MapUnary
                | runmat_builtins::BuiltinDistributedPolicy::MapUnaryConstrained(_)
        ) {
            return Box::pin(async {
                Err(error(
                    "distributed builtin request is not admitted for partition-local execution",
                ))
            });
        }
        if request.outputs.len() != request.requested_outputs || request.arguments.len() != 1 {
            return Box::pin(async {
                Err(error(
                    "partition-local unary execution requires one input and one fact per requested output",
                ))
            });
        }
        let Value::Distributed(handle) = &request.arguments[0] else {
            return Box::pin(async {
                Err(error(
                    "partition-local unary execution requires a distributed input",
                ))
            });
        };
        let inference = runmat_builtins::infer_partition_local_call(
            entry,
            &runmat_types::CallRequest {
                arguments: request
                    .arguments
                    .iter()
                    .map(runmat_runtime::value_fact::value_fact)
                    .collect(),
                literals: request.literals.clone(),
                outputs: runmat_types::OutputSelection::new(match request.requested_outputs {
                    0 => runmat_types::RequestedOutputCount::Zero,
                    1 => runmat_types::RequestedOutputCount::One,
                    count => runmat_types::RequestedOutputCount::Exactly(count),
                }),
            },
        );
        if !inference.diagnostics.is_empty() || inference.outputs != request.outputs {
            return Box::pin(async {
                Err(error(
                    "distributed builtin request disagrees with canonical output inference",
                ))
            });
        }
        let source = (**handle).clone();
        let layouts = self.store.borrow().layouts(&source).map_err(error);
        let parts = match &request.invocation {
            RuntimeDistributedInvocation::Client => {
                self.store.borrow().cloned_parts(&source).map_err(error)
            }
            RuntimeDistributedInvocation::Worker(context) => {
                if context.gang.pool != source.pool
                    || context.gang.scope_id != source.scope_id
                    || context.gang.labs != source.partition_count
                {
                    Err(error(
                        "partition-local invocation disagrees with its admitted worker context",
                    ))
                } else {
                    self.store
                        .borrow()
                        .local_part(&source, context.rank)
                        .cloned()
                        .map(|part| vec![part])
                        .map_err(error)
                }
            }
        };
        let generation = match source.generation.checked_add(1) {
            Some(generation) => generation,
            None => return Box::pin(async { Err(error("distributed map generation overflowed")) }),
        };
        let store = Rc::clone(&self.store);
        Box::pin(async move {
            let mut partition_outputs = (0..request.requested_outputs)
                .map(|_| Vec::new())
                .collect::<Vec<Vec<OwnedPartition>>>();
            for part in parts? {
                let input =
                    runmat_runtime::execution::value_codec::decode_inline_value(&part.value)
                        .map_err(error)?;
                let value = runmat_runtime::call_builtin_async_with_outputs(
                    &request.builtin.0,
                    &[input],
                    request.requested_outputs,
                )
                .await?;
                let values = split_partition_outputs(value, request.requested_outputs)?;
                for (index, value) in values.into_iter().enumerate() {
                    partition_outputs[index].push(OwnedPartition {
                        layout: part.layout.clone(),
                        value: runmat_runtime::execution::value_codec::encode_inline_value(&value)
                            .map_err(error)?,
                    });
                }
            }
            let layouts = layouts?;
            let mut values = Vec::with_capacity(request.requested_outputs);
            for (index, (fact, outputs)) in request
                .outputs
                .into_iter()
                .zip(partition_outputs)
                .enumerate()
            {
                let output_index = u64::try_from(index)
                    .map_err(|_| error("distributed output index exceeds portable range"))?
                    .to_le_bytes();
                let output_count = u64::try_from(request.requested_outputs)
                    .map_err(|_| error("distributed output count exceeds portable range"))?
                    .to_le_bytes();
                let handle = DistributedValueHandle {
                    id: DistributedObjectId::derive(&[
                        b"partition-local-map-v2",
                        source.scope_id.bytes(),
                        source.id.bytes(),
                        request.builtin.0.as_bytes(),
                        &output_count,
                        &output_index,
                    ]),
                    contract: source.contract,
                    owner: source.owner,
                    scope_id: source.scope_id,
                    generation,
                    pool: source.pool.clone(),
                    partition_count: source.partition_count,
                    value: fact,
                    global_shape: source.global_shape.clone(),
                    scheme: source.scheme.clone(),
                    materializable: source.materializable,
                };
                for output in outputs {
                    store
                        .borrow_mut()
                        .insert_local_coordinated(handle.clone(), layouts.clone(), output)
                        .map_err(error)?;
                }
                values.push(Value::Distributed(Box::new(handle)));
            }
            Ok(match values.len() {
                1 => values.pop().expect("one distributed output was created"),
                _ => Value::OutputList(values),
            })
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

fn invoke_scalar_like_prototype(
    store: Rc<RefCell<DistributedStore>>,
    request: RuntimeDistributedCallRequest,
    entry: &'static runmat_builtins::BuiltinCatalogEntry,
) -> RuntimeServiceFuture<Result<Value, RuntimeError>> {
    if request.requested_outputs != 1 || request.outputs.len() != 1 || request.arguments.len() != 2
    {
        return Box::pin(async {
            Err(error(
                "distributed scalar-like execution requires a like option, one prototype, and one output",
            ))
        });
    }
    let Value::Distributed(handle) = &request.arguments[1] else {
        return Box::pin(async {
            Err(error(
                "distributed scalar-like execution requires a distributed prototype",
            ))
        });
    };
    let inference = runmat_builtins::infer_partition_local_call(
        entry,
        &runmat_types::CallRequest {
            arguments: request
                .arguments
                .iter()
                .map(runmat_runtime::value_fact::value_fact)
                .collect(),
            literals: request.literals.clone(),
            outputs: runmat_types::OutputSelection::new(runmat_types::RequestedOutputCount::One),
        },
    );
    if !inference.diagnostics.is_empty() || inference.outputs != request.outputs {
        return Box::pin(async {
            Err(error(
                "distributed scalar-like request disagrees with canonical output inference",
            ))
        });
    }

    let source = (**handle).clone();
    let generation = match source.generation.checked_add(1) {
        Some(generation) => generation,
        None => {
            return Box::pin(async { Err(error("distributed scalar-like generation overflowed")) })
        }
    };
    Box::pin(async move {
        let keyword = request.arguments[0].clone();
        let scalar = match &request.invocation {
            RuntimeDistributedInvocation::Client => {
                let part = store
                    .borrow()
                    .cloned_parts(&source)
                    .map_err(error)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| error("distributed prototype has no partitions"))?;
                let prototype =
                    runmat_runtime::execution::value_codec::decode_inline_value(&part.value)
                        .map_err(error)?;
                runmat_runtime::call_builtin_async_with_outputs(
                    &request.builtin.0,
                    &[keyword, prototype],
                    1,
                )
                .await?
            }
            RuntimeDistributedInvocation::Worker(context) => {
                if context.gang.pool != source.pool
                    || context.gang.scope_id != source.scope_id
                    || context.gang.labs != source.partition_count
                {
                    return Err(error(
                        "distributed scalar-like invocation disagrees with its admitted worker context",
                    ));
                }
                let part = store
                    .borrow()
                    .local_part(&source, context.rank)
                    .cloned()
                    .map_err(error)?;
                let prototype =
                    runmat_runtime::execution::value_codec::decode_inline_value(&part.value)
                        .map_err(error)?;
                runmat_runtime::call_builtin_async_with_outputs(
                    &request.builtin.0,
                    &[keyword, prototype],
                    1,
                )
                .await?
            }
        };

        let value = request
            .outputs
            .into_iter()
            .next()
            .expect("one scalar-like output fact was validated");
        let handle = DistributedValueHandle {
            id: DistributedObjectId::derive(&[
                b"partition-local-scalar-like-v1",
                source.scope_id.bytes(),
                source.id.bytes(),
                request.builtin.0.as_bytes(),
            ]),
            contract: source.contract,
            owner: source.owner,
            scope_id: source.scope_id,
            generation,
            pool: source.pool.clone(),
            partition_count: source.partition_count,
            value,
            global_shape: vec![1, 1],
            scheme: source.scheme.clone(),
            materializable: source.materializable,
        };

        match request.invocation {
            RuntimeDistributedInvocation::Client => {
                let (_, parts) = runmat_runtime::parallel::distribution::partition_value(
                    &scalar,
                    &source.scheme,
                    source.partition_count,
                )
                .await?;
                store
                    .borrow_mut()
                    .insert(handle.clone(), encode_parts(parts)?)
                    .map_err(error)?;
            }
            RuntimeDistributedInvocation::Worker(context) => {
                let (_, layouts, local) =
                    runmat_runtime::parallel::distribution::partition_local_value(
                        &scalar,
                        &source.scheme,
                        source.partition_count,
                        context.rank,
                    )
                    .await?;
                let payload =
                    runmat_runtime::execution::value_codec::encode_inline_value(&local.value)
                        .map_err(error)?;
                store
                    .borrow_mut()
                    .insert_local_coordinated(
                        handle.clone(),
                        layouts,
                        OwnedPartition {
                            layout: local.layout,
                            value: payload,
                        },
                    )
                    .map_err(error)?;
            }
        }
        Ok(Value::Distributed(Box::new(handle)))
    })
}

async fn create_value(
    store: Rc<RefCell<DistributedStore>>,
    generation: u64,
    contract: DistributedValueContract,
    input: Value,
    scheme: DistributionScheme,
    pool: runmat_execution::PoolHandle,
    workers: u32,
) -> Result<DistributedValueHandle, RuntimeError> {
    let partition_count = runmat_types::LabCount(workers);
    // The compiler contract supplies the stable creation identity. The live
    // handle records the admitted runtime representation so later partition
    // calls retain its exact class, shape, storage, and residency facts.
    let value = runmat_runtime::value_fact::value_fact(&input);
    validate_value_contract(&value, &contract)?;
    let (shape, parts) =
        runmat_runtime::parallel::distribution::partition_value(&input, &scheme, partition_count)
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
        value,
        global_shape: shape
            .iter()
            .map(|dimension| {
                u64::try_from(*dimension).map_err(|_| error("distributed shape exceeds u64"))
            })
            .collect::<Result<Vec<_>, _>>()?,
        scheme,
        materializable: contract.materializable,
    };
    store
        .borrow_mut()
        .insert(handle.clone(), encode_parts(parts)?)
        .map_err(error)?;
    Ok(handle)
}

async fn create_worker_value(
    store: Rc<RefCell<DistributedStore>>,
    contract: DistributedValueContract,
    input: Value,
    scheme: DistributionScheme,
    context: runmat_execution::SpmdTaskContext,
) -> Result<DistributedValueHandle, RuntimeError> {
    let value = runmat_runtime::value_fact::value_fact(&input);
    validate_value_contract(&value, &contract)?;
    let (global_shape, layouts, local) =
        runmat_runtime::parallel::distribution::partition_local_value(
            &input,
            &scheme,
            context.gang.labs,
            context.rank,
        )
        .await?;
    let handle = DistributedValueHandle {
        id: DistributedObjectId::derive(&[
            context.gang.id.bytes(),
            &context.gang.generation.to_be_bytes(),
            &contract.id.function.0.to_be_bytes(),
            &contract.id.ordinal.to_be_bytes(),
        ]),
        contract: contract.id,
        owner: contract.owner,
        scope_id: context.gang.scope_id,
        generation: context.gang.generation,
        pool: context.gang.pool,
        partition_count: context.gang.labs,
        value,
        global_shape,
        scheme,
        materializable: contract.materializable,
    };
    let payload =
        runmat_runtime::execution::value_codec::encode_inline_value(&local.value).map_err(error)?;
    store
        .borrow_mut()
        .insert_local_coordinated(
            handle.clone(),
            layouts,
            OwnedPartition {
                layout: local.layout,
                value: payload,
            },
        )
        .map_err(error)?;
    Ok(handle)
}

async fn build_worker_value(
    store: Rc<RefCell<DistributedStore>>,
    contract: DistributedValueContract,
    local_part: Value,
    global_shape: Vec<u64>,
    scheme: DistributionScheme,
    layouts: Vec<runmat_execution::DistributedPartitionLayout>,
    context: runmat_execution::SpmdTaskContext,
) -> Result<DistributedValueHandle, RuntimeError> {
    let layout = layouts
        .iter()
        .find(|layout| layout.rank == context.rank)
        .cloned()
        .ok_or_else(|| error("local worker has no authoritative distributed partition layout"))?;
    let local_shape = runmat_runtime::parallel::distribution::value_shape(&local_part).await?;
    if local_shape != layout.local_shape {
        return Err(error(
            "local-part shape disagrees with this worker's authoritative partition layout",
        ));
    }
    let mut value = runmat_runtime::value_fact::value_fact(&local_part);
    value.shape = runmat_types::ShapeFact::from(
        global_shape
            .iter()
            .map(|extent| usize::try_from(*extent).ok())
            .collect::<Vec<_>>(),
    );
    let global_element_count = global_shape
        .iter()
        .try_fold(1_u64, |count, extent| count.checked_mul(*extent));
    if value.storage == runmat_types::StorageFact::Scalar && global_element_count != Some(1) {
        value.storage = runmat_types::StorageFact::Dense;
    }
    validate_value_contract(&value, &contract)?;
    let handle = DistributedValueHandle {
        id: DistributedObjectId::derive(&[
            context.gang.id.bytes(),
            &context.gang.generation.to_be_bytes(),
            &contract.id.function.0.to_be_bytes(),
            &contract.id.ordinal.to_be_bytes(),
        ]),
        contract: contract.id,
        owner: contract.owner,
        scope_id: context.gang.scope_id,
        generation: context.gang.generation,
        pool: context.gang.pool,
        partition_count: context.gang.labs,
        value,
        global_shape,
        scheme,
        materializable: contract.materializable,
    };
    let payload =
        runmat_runtime::execution::value_codec::encode_inline_value(&local_part).map_err(error)?;
    store
        .borrow_mut()
        .insert_local_coordinated(
            handle.clone(),
            layouts,
            OwnedPartition {
                layout,
                value: payload,
            },
        )
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

fn split_partition_outputs(
    value: Value,
    requested_outputs: usize,
) -> Result<Vec<Value>, RuntimeError> {
    match (requested_outputs, value) {
        (0, _) => Ok(Vec::new()),
        (1, Value::OutputList(mut values)) if values.len() == 1 => Ok(vec![values.remove(0)]),
        (1, value) => Ok(vec![value]),
        (count, Value::OutputList(values)) if values.len() == count => Ok(values),
        (count, Value::OutputList(values)) => Err(error(format!(
            "partition-local builtin returned {} outputs for {count} requested outputs",
            values.len()
        ))),
        (count, _) => Err(error(format!(
            "partition-local builtin returned one value for {count} requested outputs"
        ))),
    }
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

fn validate_value_contract(
    value: &runmat_types::ValueFact,
    contract: &DistributedValueContract,
) -> Result<(), RuntimeError> {
    let expected = &contract.value;
    let mismatch = if !value.kind.satisfies(&expected.kind) {
        Some("class")
    } else if !value.shape.satisfies(&expected.shape) {
        Some("global shape")
    } else if expected.storage != runmat_types::StorageFact::Unknown
        && value.storage != expected.storage
    {
        Some("storage")
    } else if expected.layout != runmat_types::LayoutFact::Unknown
        && value.layout != expected.layout
    {
        Some("layout")
    } else if !value.residency.satisfies(&expected.residency) {
        Some("residency")
    } else {
        None
    };
    mismatch.map_or(Ok(()), |field| {
        Err(error(format!(
            "distributed runtime value disagrees with its compiler-owned {field} contract"
        )))
    })
}

fn error(message: impl std::fmt::Display) -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:parallel:DistributedValue",
        message.to_string(),
    )
}

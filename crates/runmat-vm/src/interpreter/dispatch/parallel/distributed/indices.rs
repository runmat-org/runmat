use crate::bytecode::program::ExecutionContext;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub(super) async fn execute(
    service: &dyn runmat_runtime::context::RuntimeDistributedService,
    has_lab: bool,
    requested_outputs: u8,
    mut arguments: Vec<Value>,
    execution: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    let lab = if has_lab {
        Some(runmat_runtime::parallel::codistributor::designated_worker(
            &arguments.pop().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "InvalidDistributedInstruction",
                    "globalIndices is missing its lab operand",
                )
            })?,
        )?)
    } else {
        None
    };
    let [value, dimension] = super::arguments::decode(arguments)?;
    let dimension = runmat_runtime::parallel::codistributor::distribution_dimension(&dimension)?;
    let current_context = execution
        .runtime
        .service_ports()
        .collective()
        .map(|collective| collective.context().clone());
    let rank = lab
        .or_else(|| current_context.as_ref().map(|context| context.rank))
        .ok_or_else(|| {
            crate::interpreter::errors::mex(
                "GlobalIndicesLabRequired",
                "globalIndices requires a lab argument outside an SPMD worker context",
            )
        })?;
    let layouts = match value {
        Value::Distributed(handle) => {
            runmat_runtime::parallel::lease::validate_distributed(&execution.runtime, &handle)?;
            service.inspect(*handle).await?.partitions
        }
        codistributor
            if runmat_runtime::parallel::codistributor::is_codistributor(&codistributor) =>
        {
            let global_shape =
                runmat_runtime::parallel::codistributor::declared_global_shape(&codistributor)?;
            let labs = if let Some(context) = &current_context {
                context.gang.labs
            } else {
                let workers = execution
                    .runtime
                    .execution()
                    .current_pool()
                    .map_err(super::super::execution_error)?
                    .ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "GlobalIndicesPoolRequired",
                            "globalIndices requires a live pool to resolve a codistributor",
                        )
                    })?
                    .workers;
                runmat_types::LabCount(workers)
            };
            let scheme = runmat_runtime::parallel::codistributor::resolve(
                &codistributor,
                &global_shape,
                labs,
            )?;
            runmat_runtime::parallel::distribution::partition_layouts(&global_shape, &scheme, labs)?
        }
        _ => {
            return Err(crate::interpreter::errors::mex(
                "GlobalIndicesValueRequired",
                "globalIndices requires a distributed value or complete codistributor",
            ));
        }
    };
    let selection = layouts
        .iter()
        .find(|layout| layout.rank == rank)
        .and_then(|layout| {
            layout
                .selections
                .iter()
                .find(|selection| selection.dimension() == dimension)
        })
        .ok_or_else(|| {
            crate::interpreter::errors::mex(
                "GlobalIndicesSelection",
                "requested lab or dimension lies outside the distributed layout",
            )
        })?;
    result(selection, requested_outputs)
}

fn result(
    selection: &runmat_execution::PartitionSelection,
    requested_outputs: u8,
) -> Result<Value, RuntimeError> {
    let indices = match selection {
        runmat_execution::PartitionSelection::Range(range) => (range.start..range.end)
            .map(|index| index + 1)
            .collect::<Vec<_>>(),
        runmat_execution::PartitionSelection::Strided {
            start, step, count, ..
        } => (0..*count)
            .map(|offset| {
                start
                    .checked_add(step.checked_mul(offset).ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "GlobalIndicesOverflow",
                            "global index multiplication overflowed",
                        )
                    })?)
                    .and_then(|index| index.checked_add(1))
                    .ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "GlobalIndicesOverflow",
                            "global index addition overflowed",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?,
        runmat_execution::PartitionSelection::Indices { indices, .. } => indices
            .iter()
            .map(|index| {
                index.checked_add(1).ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "GlobalIndicesOverflow",
                        "global index addition overflowed",
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    if requested_outputs == 1 {
        let length = indices.len();
        return runmat_value::Tensor::new_integer(
            runmat_value::IntegerStorage::U64(indices),
            vec![1, length],
        )
        .map(Value::Tensor)
        .map_err(|error| {
            crate::interpreter::errors::mex("GlobalIndicesShape", &error.to_string())
        });
    }
    let (first, last) = indices
        .first()
        .copied()
        .zip(indices.last().copied())
        .unwrap_or((1, 0));
    Ok(Value::OutputList(vec![
        Value::Int(runmat_value::IntValue::U64(first)),
        Value::Int(runmat_value::IntValue::U64(last)),
    ]))
}

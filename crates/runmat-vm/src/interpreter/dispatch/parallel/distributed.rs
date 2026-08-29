use crate::bytecode::program::ExecutionContext;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub(super) async fn execute(
    bytecode: &crate::Bytecode,
    operation: &crate::BytecodeDistributedOp,
    arguments: Vec<Value>,
    execution: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    use crate::BytecodeDistributedOp as Op;

    let service = execution
        .runtime
        .service_ports()
        .require_distributed("distributed value operation")
        .map_err(super::capability_error)?
        .clone();
    match operation {
        Op::Create { id, owner, scheme } => {
            let [input] = decode_arguments(arguments)?;
            let contract = bytecode
                .distributed_values
                .iter()
                .find(|contract| contract.id == *id)
                .filter(|contract| contract.owner == *owner && contract.scheme == *scheme)
                .cloned()
                .ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "DistributedContractMissing",
                        "distributed instruction has no matching compiler-owned semantic contract",
                    )
                })?;
            let pool = execution
                .runtime
                .execution()
                .ensure_pool(runmat_execution::PoolRequest::automatic())
                .map_err(super::execution_error)?;
            service
                .create(contract, input, pool)
                .await
                .map(|handle| Value::Distributed(Box::new(handle)))
        }
        Op::LocalPart => {
            let [input] = decode_arguments(arguments)?;
            let Value::Distributed(handle) = input else {
                return Err(crate::interpreter::errors::mex(
                    "DistributedValueRequired",
                    "getLocalPart requires a distributed value",
                ));
            };
            runmat_runtime::parallel::lease::validate_distributed(&execution.runtime, &handle)?;
            let rank = if let Some(collective) = execution.runtime.service_ports().collective() {
                collective.context().rank
            } else if handle.partition_count.0 == 1 {
                runmat_types::LabRank(1)
            } else {
                return Err(crate::interpreter::errors::mex(
                    "DistributedRankContextRequired",
                    "getLocalPart requires an SPMD rank context for a multi-partition value",
                ));
            };
            service.local_part(*handle, rank).await
        }
        Op::Materialize => {
            let [input] = decode_arguments(arguments)?;
            let Value::Distributed(handle) = input else {
                return Err(crate::interpreter::errors::mex(
                    "DistributedValueRequired",
                    "gather requires a distributed value on this execution path",
                ));
            };
            runmat_runtime::parallel::lease::validate_distributed(&execution.runtime, &handle)?;
            service.materialize(*handle).await
        }
        Op::Codistributor => {
            let [input] = decode_arguments(arguments)?;
            let Value::Distributed(handle) = input else {
                return Err(crate::interpreter::errors::mex(
                    "DistributedValueRequired",
                    "getCodistributor requires a distributed value",
                ));
            };
            runmat_runtime::parallel::lease::validate_distributed(&execution.runtime, &handle)?;
            runmat_runtime::parallel::codistributor::from_scheme(
                &handle.scheme,
                &handle.global_shape,
            )
        }
        Op::Redistribute => {
            let [input, codistributor] = decode_arguments(arguments)?;
            let Value::Distributed(handle) = input else {
                return Err(crate::interpreter::errors::mex(
                    "DistributedValueRequired",
                    "redistribute requires a distributed value",
                ));
            };
            runmat_runtime::parallel::lease::validate_distributed(&execution.runtime, &handle)?;
            let scheme = runmat_runtime::parallel::codistributor::resolve(
                &codistributor,
                &handle.global_shape,
                handle.partition_count,
            )?;
            service
                .redistribute(*handle, scheme)
                .await
                .map(|handle| Value::Distributed(Box::new(handle)))
        }
    }
}

fn decode_arguments<const N: usize>(arguments: Vec<Value>) -> Result<[Value; N], RuntimeError> {
    arguments.try_into().map_err(|_| {
        crate::interpreter::errors::mex(
            "InvalidDistributedInstruction",
            "distributed instruction operand count does not match its bytecode contract",
        )
    })
}

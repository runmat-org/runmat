use crate::bytecode::program::ExecutionContext;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

mod arguments;
mod construction;
mod coordination;
mod indices;

pub(crate) async fn encode_spmd_output(
    runtime: &runmat_runtime::context::RuntimeContext,
    value: &Value,
) -> Result<runmat_execution::SpmdOutputValue, RuntimeError> {
    if let Value::Distributed(handle) = value {
        runmat_runtime::parallel::lease::validate_distributed(runtime, handle)?;
        let collective = runtime
            .service_ports()
            .require_collective("distributed SPMD output")
            .map_err(super::capability_error)?;
        let service = runtime
            .service_ports()
            .require_distributed("distributed SPMD output")
            .map_err(super::capability_error)?;
        return service
            .export_local((**handle).clone(), collective.context().rank)
            .await
            .map(Box::new)
            .map(runmat_execution::SpmdOutputValue::Distributed);
    }
    runmat_runtime::execution::value_codec::encode_inline_value(value)
        .map(runmat_execution::SpmdOutputValue::Value)
        .map_err(super::value_codec_error)
}

pub(super) async fn execute(
    bytecode: &crate::Bytecode,
    operation: &crate::BytecodeDistributedOp,
    arguments: Vec<Value>,
    execution: &ExecutionContext,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    use crate::BytecodeDistributedOp as Op;

    let service = execution
        .runtime
        .service_ports()
        .require_distributed("distributed value operation")
        .map_err(super::capability_error)?
        .clone();
    let constructor = construction::Executor::new(&*service, bytecode, execution);
    if let Op::GlobalIndices {
        has_lab,
        requested_outputs,
    } = operation
    {
        return indices::execute(
            &*service,
            *has_lab,
            *requested_outputs,
            arguments,
            execution,
        )
        .await;
    }
    let value = match operation {
        Op::Create { id, owner, scheme } => {
            constructor
                .create(*id, *owner, scheme.clone(), arguments)
                .await
        }
        Op::Codistributed {
            id,
            owner,
            overload,
            coordination,
        } => {
            constructor
                .codistributed(*id, *owner, *overload, *coordination, arguments)
                .await
        }
        Op::Build {
            id,
            owner,
            has_codistributor,
            validation,
            coordination,
        } => {
            constructor
                .build(
                    *id,
                    *owner,
                    *has_codistributor,
                    *validation,
                    *coordination,
                    arguments,
                )
                .await
        }
        Op::LocalPart => {
            let [input] = arguments::decode(arguments)?;
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
            let [input] = arguments::decode(arguments)?;
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
            let [input] = arguments::decode(arguments)?;
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
        Op::GlobalIndices { .. } => unreachable!("handled before scalar distributed operations"),
        Op::Redistribute => {
            let [input, codistributor] = arguments::decode(arguments)?;
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
    }?;
    runmat_value::ValueSequence::single(value)
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
}

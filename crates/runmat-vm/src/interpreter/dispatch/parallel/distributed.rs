use crate::bytecode::program::ExecutionContext;
use runmat_runtime::{execution::RootedValueSequence, RuntimeError};
use runmat_value::Value;

mod arguments;
mod construction;
mod coordination;
mod indices;

/// Validate and retain export metadata before releasing a guarded slot. The
/// returned future does not borrow `handle` and invokes the service only when
/// polled, after the caller has released that slot guard.
fn prepare_spmd_distributed_export<'runtime>(
    runtime: &'runtime runmat_runtime::context::RuntimeContext,
    handle: &runmat_value::SharedDistributedHandle,
) -> Result<
    impl std::future::Future<Output = Result<runmat_execution::SpmdOutputValue, RuntimeError>>
        + use<'runtime>,
    RuntimeError,
> {
    runmat_runtime::parallel::lease::validate_distributed(runtime, handle)?;
    let rank = runtime
        .service_ports()
        .require_collective("distributed SPMD output")
        .map_err(super::capability_error)?
        .context()
        .rank;
    let service = runtime
        .service_ports()
        .require_distributed("distributed SPMD output")
        .map_err(super::capability_error)?;
    let handle = handle.clone();
    Ok(async move {
        service
            .export_local(handle, rank)
            .await
            .map(Box::new)
            .map(runmat_execution::SpmdOutputValue::Distributed)
    })
}

/// Encode a completed interpreter slot without retaining its shared-control
/// read guard across a distributed export suspension. Distributed export
/// retains its existing handle metadata once, without cloning array payloads
/// or adding a boxed Value; inline values are encoded synchronously.
pub(crate) async fn encode_spmd_interpreter_output(
    runtime: &runmat_runtime::context::RuntimeContext,
    value: crate::interpreter::vars::InterpreterValueRef<'_>,
) -> Result<runmat_execution::SpmdOutputValue, RuntimeError> {
    if let Value::Distributed(handle) = &*value {
        let export = prepare_spmd_distributed_export(runtime, handle)?;
        drop(value);
        return export.await;
    }
    runmat_runtime::execution::value_codec::encode_inline_value_in_context(&value, runtime)
        .map(runmat_execution::SpmdOutputValue::Value)
        .map_err(super::value_codec_error)
}

pub(super) async fn execute(
    bytecode: &crate::Bytecode,
    operation: &crate::BytecodeDistributedOp,
    arguments: Vec<Value>,
    execution: &ExecutionContext,
) -> Result<RootedValueSequence, RuntimeError> {
    use crate::BytecodeDistributedOp as Op;

    let _argument_roots = runmat_runtime::execution::RootedValueLease::for_values(arguments.iter())
        .map_err(|error| error.into_runtime_error())?;

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
        let sequence = indices::execute(
            &*service,
            *has_lab,
            *requested_outputs,
            arguments,
            execution,
        )
        .await?;
        return RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error());
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
            return service.local_part(handle, rank).await?.try_map_owned(
                |value| {
                    runmat_value::ValueSequence::single(value)
                        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
                },
                |error| error.into_runtime_error(),
            );
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
            return service.materialize(handle).await?.try_map_owned(
                |value| {
                    runmat_value::ValueSequence::single(value)
                        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
                },
                |error| error.into_runtime_error(),
            );
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
                .redistribute(handle, scheme)
                .await
                .map(|handle| Value::Distributed(handle.into()))
        }
    }?;
    let sequence = runmat_value::ValueSequence::single(value)
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)?;
    RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())
}

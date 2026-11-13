use crate::bytecode::program::ExecutionContext;
use runmat_runtime::execution::value_codec::encode_inline_value_in_context;
use runmat_runtime::execution::{
    RootedValue, RootedValueLease, RootedValueSequence, RootedValues, RootedValuesBuilder,
};
use runmat_runtime::sequence::ResolveValueSequence;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

use super::{capability_error, value_codec_error};

pub(super) async fn execute(
    bytecode: &crate::Bytecode,
    id: runmat_types::CollectiveId,
    operation: crate::BytecodeCollectiveOp,
    arguments: Vec<Value>,
    execution: &ExecutionContext,
) -> Result<RootedValueSequence, RuntimeError> {
    let contract = bytecode
        .collective_contracts
        .iter()
        .find(|contract| contract.id == id)
        .ok_or_else(|| {
            crate::interpreter::errors::mex(
                "CollectiveContractMissing",
                "collective instruction has no compiler-owned semantic contract",
            )
        })?;
    if !collective_operation_matches(&contract.operation, operation) {
        return Err(crate::interpreter::errors::mex(
            "CollectiveContractMismatch",
            "collective instruction disagrees with its compiler-owned semantic contract",
        ));
    }
    let collective = execution
        .runtime
        .service_ports()
        .require_collective("SPMD collective")
        .map_err(capability_error)?
        .clone();
    let sequence = collective.next_sequence(id)?;
    let invocation = collective_invocation(
        operation,
        arguments,
        collective.context(),
        &execution.runtime,
    )?;
    let response = collective
        .execute(runmat_execution::CollectiveRequest {
            context: collective.context().clone(),
            id,
            sequence,
            invocation,
        })
        .await?;
    collective_response(
        &execution.runtime,
        response,
        operation,
        &bytecode.function_registry,
    )
    .await
}

fn collective_operation_matches(
    contract: &runmat_types::CollectiveOperation,
    instruction: crate::BytecodeCollectiveOp,
) -> bool {
    use crate::BytecodeCollectiveOp as Bytecode;
    use runmat_types::CollectiveOperation as Contract;

    match (contract, instruction) {
        (Contract::Barrier, Bytecode::Barrier)
        | (Contract::Broadcast, Bytecode::Broadcast { .. })
        | (Contract::Gather, Bytecode::Gather)
        | (Contract::Scatter, Bytecode::Scatter)
        | (Contract::AllGather, Bytecode::AllGather)
        | (Contract::Cat, Bytecode::Cat { .. })
        | (Contract::FunctionalReduce, Bytecode::FunctionalReduce { .. })
        | (Contract::Send, Bytecode::Send { .. })
        | (Contract::SendReceive, Bytecode::SendReceive { .. })
        | (Contract::Probe, Bytecode::Probe { .. }) => true,
        (
            Contract::Receive {
                requested_outputs: expected,
            },
            Bytecode::Receive {
                requested_outputs, ..
            },
        ) => *expected == requested_outputs,
        (Contract::Reduce { operator: expected }, Bytecode::Reduce { operator })
        | (Contract::AllReduce { operator: expected }, Bytecode::AllReduce { operator }) => {
            *expected == operator
        }
        _ => false,
    }
}

fn collective_invocation(
    operation: crate::BytecodeCollectiveOp,
    mut arguments: Vec<Value>,
    context: &runmat_execution::SpmdTaskContext,
    runtime: &runmat_runtime::context::RuntimeContext,
) -> Result<runmat_execution::CollectiveInvocation, RuntimeError> {
    use crate::BytecodeCollectiveOp as Op;
    use runmat_execution::{CollectiveInvocation, CollectiveMessageTag, ReceiveSelection};

    let encode =
        |value: &Value| encode_inline_value_in_context(value, runtime).map_err(value_codec_error);
    Ok(match operation {
        Op::Barrier => CollectiveInvocation::Barrier,
        Op::Broadcast { has_input } => {
            let input = has_input.then(|| arguments.remove(0));
            let root = lab_rank(arguments.remove(0), context)?;
            CollectiveInvocation::Broadcast {
                root,
                value: if context.rank == root {
                    input.as_ref().map(encode).transpose()?
                } else {
                    None
                },
            }
        }
        Op::Gather => {
            let value = encode(&arguments.remove(0))?;
            let root = lab_rank(arguments.remove(0), context)?;
            CollectiveInvocation::Gather { root, value }
        }
        Op::Scatter => {
            let value = arguments.remove(0);
            let root = lab_rank(arguments.remove(0), context)?;
            let values = if context.rank == root {
                let Value::Cell(values) = value else {
                    return Err(crate::interpreter::errors::mex(
                        "CollectiveScatterInput",
                        "scatter root must provide one cell entry per lab",
                    ));
                };
                Some(
                    values
                        .data()
                        .iter()
                        .map(encode)
                        .collect::<Result<Vec<_>, _>>()?,
                )
            } else {
                None
            };
            CollectiveInvocation::Scatter { root, values }
        }
        Op::AllGather => CollectiveInvocation::AllGather {
            value: encode(&arguments.remove(0))?,
        },
        Op::Reduce { operator } => {
            let value = encode(&arguments.remove(0))?;
            let root = lab_rank(arguments.remove(0), context)?;
            CollectiveInvocation::Reduce {
                root: Some(root),
                operator,
                value,
            }
        }
        Op::AllReduce { operator } => CollectiveInvocation::Reduce {
            root: None,
            operator,
            value: encode(&arguments.remove(0))?,
        },
        Op::Cat { has_root } => {
            let value = encode(&arguments.remove(0))?;
            let dimension = positive_u32(arguments.remove(0), "concatenation dimension")?;
            let root = has_root
                .then(|| lab_rank(arguments.remove(0), context))
                .transpose()?;
            CollectiveInvocation::Cat {
                root,
                dimension,
                value,
            }
        }
        Op::FunctionalReduce { has_root } => {
            let reducer = encode(&arguments.remove(0))?;
            let value = encode(&arguments.remove(0))?;
            let root = has_root
                .then(|| optional_lab_rank(arguments.remove(0), context))
                .transpose()?
                .flatten();
            CollectiveInvocation::FunctionalReduce {
                root,
                reducer,
                value,
            }
        }
        Op::Send { has_tag } => {
            let value = encode(&arguments.remove(0))?;
            let destination = lab_rank(arguments.remove(0), context)?;
            let tag = if has_tag {
                message_tag(arguments.remove(0))?
            } else {
                CollectiveMessageTag(0)
            };
            CollectiveInvocation::Send {
                destination,
                tag,
                value,
            }
        }
        Op::Receive {
            has_source,
            has_tag,
            ..
        }
        | Op::Probe {
            has_source,
            has_tag,
        } => {
            let source = if has_source {
                receive_source(arguments.remove(0), context)?
            } else {
                None
            };
            let tag = has_tag
                .then(|| message_tag(arguments.remove(0)))
                .transpose()?;
            let selection = ReceiveSelection { source, tag };
            if matches!(operation, Op::Receive { .. }) {
                CollectiveInvocation::Receive { selection }
            } else {
                CollectiveInvocation::Probe { selection }
            }
        }
        Op::SendReceive { has_tag } => {
            let destination = optional_lab_rank(arguments.remove(0), context)?;
            let source = optional_lab_rank(arguments.remove(0), context)?;
            let value = encode(&arguments.remove(0))?;
            let tag = if has_tag {
                message_tag(arguments.remove(0))?
            } else {
                CollectiveMessageTag(0)
            };
            CollectiveInvocation::SendReceive {
                destination,
                source,
                tag,
                value,
            }
        }
    })
}

async fn collective_response(
    runtime: &runmat_runtime::context::RuntimeContext,
    response: runmat_execution::CollectiveResponse,
    operation: crate::BytecodeCollectiveOp,
    function_registry: &crate::FunctionRegistry,
) -> Result<RootedValueSequence, RuntimeError> {
    use runmat_execution::CollectiveResponse;

    match response {
        CollectiveResponse::Complete => single_output(empty_value()),
        CollectiveResponse::Value { value } => single_rooted_output(decode_value(&value)?),
        CollectiveResponse::Received { value, source, tag } => {
            let value = decode_value(&value)?;
            let requested_outputs = match operation {
                crate::BytecodeCollectiveOp::Receive {
                    requested_outputs, ..
                } => requested_outputs,
                crate::BytecodeCollectiveOp::SendReceive { .. } => 1,
                _ => 1,
            };
            if requested_outputs == 1 {
                single_rooted_output(value)
            } else {
                value.try_map_owned(
                    |value| {
                        let mut outputs = vec![value, Value::Num(f64::from(source.0))];
                        if requested_outputs == 3 {
                            outputs.push(Value::Int(runmat_value::IntValue::U64(tag.0)));
                        }
                        runmat_value::ValueSequence::comma_separated(outputs)
                            .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
                    },
                    |error| error.into_runtime_error(),
                )
            }
        }
        CollectiveResponse::Values { values } => {
            let count = values.len();
            decode_values(&values)?.with_unbounded_values(|values| {
                runmat_value::CellArray::new_with_shape(values, vec![count, 1])
                    .map(Value::Cell)
                    .map_err(|error| {
                        runmat_runtime::runtime_error::semantic_error(
                            "RunMat:CollectiveResult",
                            error,
                        )
                    })
                    .and_then(single_output)
            })?
        }
        CollectiveResponse::ReductionInputs { operator, values } => {
            decode_values(&values)?
                .with_unbounded_values_async(|values| async move {
                    let mut values = values.into_iter();
                    let accumulator = values.next().ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "CollectiveReductionEmpty",
                            "collective reduction received no lab contributions",
                        )
                    })?;
                    let mut accumulator = RootedValue::new(accumulator)
                        .map_err(|error| error.into_runtime_error())?;
                    for contribution in values {
                        accumulator = runmat_runtime::parallel::reduction::combine_in_context(
                            Some(runtime),
                            operator,
                            accumulator,
                            contribution,
                        )
                        .await?;
                    }
                    single_rooted_output(accumulator)
                })
                .await?
        }
        CollectiveResponse::ConcatenationInputs { dimension, values } => {
            if values.len() == 1 {
                return single_rooted_output(decode_value(&values[0])?);
            }
            decode_values(&values)?
                .with_unbounded_values_async(|values| async move {
                    let mut arguments = Vec::with_capacity(values.len() + 1);
                    arguments.push(Value::Int(runmat_value::IntValue::U32(dimension)));
                    arguments.extend(values);
                    crate::call::execute_callable_descriptor(
                        runtime,
                        runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                            runmat_hir::CallableIdentity::Builtin(runmat_types::BuiltinId(
                                "cat".into(),
                            )),
                            arguments,
                            1,
                            runmat_hir::CallableFallbackPolicy::None,
                            runmat_runtime::call::descriptor::CallableCallKind::Direct,
                        ),
                    )
                    .await
                })
                .await?
        }
        CollectiveResponse::FunctionalReductionInputs { reducer, values } => {
            let reducer = decode_value(&reducer)?;
            decode_values(&values)?
                .with_unbounded_values_async(|values| async move {
                    let mut values = values.into_iter();
                    let accumulator = values.next().ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "CollectiveReductionEmpty",
                            "functional reduction received no lab contributions",
                        )
                    })?;
                    let mut accumulator = RootedValue::new(accumulator)
                        .map_err(|error| error.into_runtime_error())?;
                    for contribution in values {
                        let (arguments, argument_roots) = accumulator
                            .with_owned(|accumulator| {
                                let arguments = vec![accumulator, contribution];
                                RootedValueLease::for_values(arguments.iter())
                                    .map(|roots| (arguments, roots))
                            })
                            .map_err(|error| error.into_runtime_error())?;
                        let descriptor =
                            runmat_runtime::call::descriptor::CallableDescriptor::from_feval_value(
                                (*reducer).clone(),
                                arguments,
                                1,
                                function_registry,
                            );
                        let outputs =
                            crate::call::execute_callable_descriptor(runtime, descriptor).await?;
                        drop(argument_roots);
                        accumulator = outputs.try_map_owned(
                            |sequence| {
                                let outputs = sequence.resolve(
                                    runmat_types::SequenceUse::RequireSingle,
                                    runmat_runtime::sequence::SequenceResolutionContext::default(),
                                )?;
                                outputs.try_into_first().map_err(|_| {
                                    runmat_runtime::sequence::sequence_error_to_runtime(
                                        runmat_value::ValueSequenceError::SharedOutputStorage,
                                    )
                                })?.ok_or_else(|| crate::interpreter::errors::mex(
                                    "CollectiveResult", "collective reducer returned no output",
                                ))
                            },
                            |error| error.into_runtime_error(),
                        )?;
                    }
                    single_rooted_output(accumulator)
                })
                .await?
        }
        CollectiveResponse::Probe { available } => single_output(Value::Bool(available)),
        CollectiveResponse::Agreement { .. } | CollectiveResponse::DistributedBuild { .. } => {
            Err(crate::interpreter::errors::mex(
                "CollectiveContract",
                "internal distributed construction response reached a public collective instruction",
            ))
        }
    }
}

fn decode_value(
    payload: &runmat_execution::value::ValuePayload,
) -> Result<RootedValue, RuntimeError> {
    runmat_runtime::execution::value_codec::decode_inline_value_owned(payload).map_err(|error| {
        match error {
            runmat_runtime::execution::value_codec::CooperativeValueCodecError::Value(error) => {
                super::value_codec_error(error)
            }
            runmat_runtime::execution::value_codec::CooperativeValueCodecError::Execution(
                error,
            ) => error.into_runtime_error(),
        }
    })
}

fn decode_values(
    payloads: &[runmat_execution::value::ValuePayload],
) -> Result<RootedValues, RuntimeError> {
    let mut decoded = RootedValuesBuilder::with_capacity(payloads.len());
    for payload in payloads {
        decoded.push_owned(decode_value(payload)?);
    }
    decoded.finish().map_err(|error| error.into_runtime_error())
}

fn single_output(value: Value) -> Result<RootedValueSequence, RuntimeError> {
    runmat_value::ValueSequence::single(value)
        .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
        .and_then(root_sequence)
}

fn single_rooted_output(value: RootedValue) -> Result<RootedValueSequence, RuntimeError> {
    value.try_map_owned(
        |value| {
            runmat_value::ValueSequence::single(value)
                .map_err(runmat_runtime::sequence::sequence_error_to_runtime)
        },
        |error| error.into_runtime_error(),
    )
}

fn root_sequence(
    sequence: runmat_value::ValueSequence,
) -> Result<RootedValueSequence, RuntimeError> {
    RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())
}

fn receive_source(
    value: Value,
    context: &runmat_execution::SpmdTaskContext,
) -> Result<Option<runmat_types::LabRank>, RuntimeError> {
    match value {
        Value::String(value)
            if value
                .as_str()
                .is_some_and(|text| text.eq_ignore_ascii_case("any")) =>
        {
            Ok(None)
        }
        Value::CharArray(value)
            if value
                .row_string()
                .is_some_and(|value| value.eq_ignore_ascii_case("any")) =>
        {
            Ok(None)
        }
        value => lab_rank(value, context).map(Some),
    }
}

fn optional_lab_rank(
    value: Value,
    context: &runmat_execution::SpmdTaskContext,
) -> Result<Option<runmat_types::LabRank>, RuntimeError> {
    if is_empty_value(&value) {
        Ok(None)
    } else {
        lab_rank(value, context).map(Some)
    }
}

fn is_empty_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(value) if value.is_empty()
    ) || matches!(value, Value::Cell(value) if value.data().is_empty())
        || matches!(value, Value::LogicalArray(value) if value.is_empty())
}

fn empty_value() -> Value {
    Value::Tensor(runmat_value::Tensor::zeros(vec![0, 0]))
}

fn lab_rank(
    value: Value,
    context: &runmat_execution::SpmdTaskContext,
) -> Result<runmat_types::LabRank, RuntimeError> {
    let rank = positive_u32(value, "lab rank")?;
    if rank > context.gang.labs.0 {
        return Err(crate::interpreter::errors::mex(
            "CollectiveRankOutsideGang",
            "collective lab rank lies outside the admitted gang",
        ));
    }
    Ok(runmat_types::LabRank(rank))
}

fn message_tag(value: Value) -> Result<runmat_execution::CollectiveMessageTag, RuntimeError> {
    let tag = match value {
        Value::Int(value) => value.try_to_u64(),
        Value::Num(value)
            if value.is_finite()
                && value.fract() == 0.0
                && value >= 0.0
                && value <= u64::MAX as f64 =>
        {
            Some(value as u64)
        }
        _ => None,
    }
    .ok_or_else(|| {
        crate::interpreter::errors::mex(
            "CollectiveTagInvalid",
            "collective message tag must be a nonnegative integer scalar",
        )
    })?;
    Ok(runmat_execution::CollectiveMessageTag(tag))
}

fn positive_u32(value: Value, label: &str) -> Result<u32, RuntimeError> {
    let value = match value {
        Value::Int(value) => value.try_to_u64(),
        Value::Num(value)
            if value.is_finite()
                && value.fract() == 0.0
                && value >= 1.0
                && value <= u32::MAX as f64 =>
        {
            Some(value as u64)
        }
        _ => None,
    };
    value
        .filter(|value| *value > 0)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            runmat_runtime::runtime_error::semantic_error(
                "RunMat:CollectiveIntegerInvalid",
                format!("{label} must be a positive integer scalar"),
            )
        })
}

use super::super::PreparedOutputTarget;
use super::support::{empty_cell_root, root, stack_underflow};
use super::{admit, PreparedDispatchFuture, PreparedResult, Request};
use crate::bytecode::Instr;
use futures::FutureExt;
use runmat_runtime::RuntimeError;
use runmat_value::IntValue;
use runmat_value::Value;

#[inline(never)]
pub(super) fn begin_output_assignment<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction, state, ..
            } = request;
            match instruction {
                Instr::BeginOutputAssignment { target_count } => {
                    state.begin_output_assignment(*target_count)?
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn begin_fixed_assignment<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction, state, ..
            } = request;
            match instruction {
                Instr::BeginFixedAssignment {
                    source_statement,
                    root_slot,
                } => {
                    state.begin_output_assignment(1)?;
                    state.fixed_assignment_owner = Some((*source_statement, *root_slot));
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn prepare_fixed_output_target<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction, state, ..
            } = request;
            match instruction {
                Instr::PrepareFixedOutputTarget => {
                    state.push_output_target(PreparedOutputTarget::Fixed)?
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn prepare_discard_output_target<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction, state, ..
            } = request;
            match instruction {
                Instr::PrepareDiscardOutputTarget => {
                    state.push_output_target(PreparedOutputTarget::Discard)?
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn begin_sequence_output_target<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                vars,
                state,
                context,
                ..
            } = request;
            match instruction {
                Instr::BeginSequenceOutputTarget {
                    root_slot,
                    root_creation,
                } => {
                    match state.fixed_assignment_owner {
                        Some((_, expected))
                            if expected == *root_slot
                                && root_creation.retains_original_receiver() => {}
                        None if !root_creation.retains_original_receiver() => {}
                        _ => {
                            return Err(crate::interpreter::errors::mex(
                                "FixedAssignmentOwner",
                                "destination root disagrees with its fixed preparation",
                            ))
                        }
                    }
                    let root_presence = context.authority.refresh(vars, *root_slot)?;
                    let retained_seed = match root_creation {
                        runmat_mir::SequenceRootCreation::OrdinaryContainer(intent)
                            if root_presence.is_absent() =>
                        {
                            Some(runmat_runtime::sequence::assignment_container_in_context(
                                Some(&context.authority.context.runtime),
                                *intent,
                            )?)
                        }
                        _ => None,
                    };
                    let absent_root = if root_presence.is_absent() {
                        use runmat_mir::SequenceRootCreation as Creation;
                        use runmat_types::AssignmentCreationPolicy as Policy;
                        match root_creation {
                            Creation::EmptyCell | Creation::OrdinaryCell => Some(empty_cell_root()),
                            Creation::Ordinary(Policy::CreateArrayByIndex) => Some(Value::Tensor(
                                runmat_value::Tensor::new(Vec::new(), vec![0, 0]).map_err(
                                    |error| {
                                        crate::interpreter::errors::mex(
                                            "IndexAssignmentBaseInit",
                                            &error,
                                        )
                                    },
                                )?,
                            )),
                            Creation::Ordinary(Policy::CreateStructFieldPath) => {
                                Some(Value::Struct(runmat_value::StructValue::new()))
                            }
                            Creation::Ordinary(Policy::ExistingOnly) => {
                                return Err(crate::interpreter::errors::mex(
                                    "UndefinedVariable",
                                    "deletion requires an existing assignment root",
                                ))
                            }
                            Creation::Ordinary(_) => {
                                return Err(crate::interpreter::errors::mex(
                                    "InvalidAssignmentCreationPolicy",
                                    "unsupported ordinary assignment root creation policy",
                                ))
                            }
                            Creation::OrdinaryContainer(_) => {
                                retained_seed.as_ref().map(|seed| seed.value().clone())
                            }
                            Creation::Existing => None,
                        }
                    } else {
                        None
                    };
                    let original = root(vars, *root_slot)?;
                    match absent_root {
                        Some(seed) => state.begin_destination(
                            &context.authority.context.runtime,
                            *root_slot,
                            &seed,
                            Some(seed.clone()),
                            *root_creation,
                            context.bytecode,
                        )?,
                        None => state.begin_destination(
                            &context.authority.context.runtime,
                            *root_slot,
                            &original,
                            None,
                            *root_creation,
                            context.bytecode,
                        )?,
                    }
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn begin_prepared_index_selectors<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction, state, ..
            } = request;
            match instruction {
                Instr::BeginPreparedIndexSelectors { component_count } => {
                    state.begin_index_selectors(*component_count)?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn load_prepared_index_end<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                stack,
                state,
                context,
                ..
            } = request;
            match instruction {
                Instr::LoadPreparedIndexEnd { component } => {
                    let component_count = state.index_selector_count()?;
                    if state.prepared_index_receiver.is_none() {
                        let prepared = state
                            .destination_builder_mut()?
                            .prepare_current_receiver_for_end_in_context(
                                &context.authority.context.runtime,
                                Some(context.authority.current_function_name),
                            )
                            .await?;
                        state.prepared_index_receiver = Some(prepared);
                    }
                    let receiver = state
                        .prepared_index_receiver
                        .as_ref()
                        .ok_or_else(stack_underflow)?;
                    let value = runmat_runtime::object::protocol::resolve_subscript_end_in_context(
                        &context.authority.context.runtime,
                        receiver,
                        *component,
                        component_count,
                    )
                    .await?;
                    value.with_owned(|value| stack.try_push(value))?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn begin_contextual_index_selectors<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                stack,
                state,
                ..
            } = request;
            match instruction {
                Instr::BeginContextualIndexSelectors { component_count } => {
                    let base = stack.last().ok_or_else(stack_underflow)?;
                    let shape =
                        runmat_runtime::builtins::common::shape::value_dimensions_borrowed(base)?;
                    state.begin_contextual_index(*component_count, shape.as_slice())?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn load_contextual_index_end<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                stack,
                state,
                ..
            } = request;
            match instruction {
                Instr::LoadContextualIndexEnd { component } => {
                    let extent = state.contextual_index_extent(*component)?;
                    stack.try_push(Value::Num(extent as f64))?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn finish_contextual_index_selectors<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction, state, ..
            } = request;
            match instruction {
                Instr::FinishContextualIndexSelectors { component_count } => {
                    state.finish_contextual_index(*component_count)?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

#[inline(never)]
pub(super) fn load_prepared_output_cardinality<'a>(
    request: Request<'a>,
    map: impl FnOnce(Result<bool, RuntimeError>) -> PreparedResult + 'a,
) -> PreparedDispatchFuture<'a> {
    let context = request.context.authority.context;
    admit(&context.runtime, move || {
        async move {
            let Request {
                instruction,
                stack,
                state,
                ..
            } = request;
            match instruction {
                Instr::LoadPreparedOutputCardinality => {
                    let total = u64::try_from(state.output_cardinality()?).map_err(|_| {
                        crate::interpreter::errors::mex(
                            "DestinationCardinalityOverflow",
                            "output destination cardinality exceeds the supported limit",
                        )
                    })?;
                    stack.try_push(Value::Int(IntValue::U64(total)))?;
                }
                _ => return Ok(false),
            }
            Ok(true)
        }
        .map(map)
    })
}

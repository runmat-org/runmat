use runmat_mir::{MirConstant, MirOperand, MirRvalue};
use runmat_runtime::native::NativeValueRef;
use runmat_value::Value;

use crate::{NativeExecutorError, NativeExecutorResult};

use super::state::HostState;

pub(super) fn evaluate_rvalue(
    state: &mut HostState,
    value: &MirRvalue,
    requested_outputs: usize,
    output_local: Option<runmat_native_codegen::NativeLocalId>,
) -> NativeExecutorResult<Vec<NativeValueRef>> {
    if output_local.is_some_and(|local| state.function.index_expression(local).is_some()) {
        // Context-dependent selector temporaries are recipes, not ordinary
        // eager values. Evaluating them here would execute `end` calls with no
        // base shape and replay their effects during the actual index site.
        return Ok(vec![state.arena.insert(Value::Num(0.0))]);
    }
    match value {
        MirRvalue::Use(operand) => evaluate_operand(state, operand).map(|value| vec![value]),
        MirRvalue::Unary(operator, operand) => {
            let argument = materialize_operand(state, operand)?;
            super::operator::evaluate(state, *operator, vec![argument]).map(|value| vec![value])
        }
        MirRvalue::Binary(left, operator, right) => {
            let left = materialize_operand(state, left)?;
            let right = materialize_operand(state, right)?;
            super::operator::evaluate(state, *operator, vec![left, right]).map(|value| vec![value])
        }
        MirRvalue::ShortCircuit {
            left,
            op,
            right_temps,
            right,
        } => {
            let left = materialize_operand(state, left)?;
            let left_truth = logical_truth(state, &left, "short-circuit left operand")?;
            let short_circuited = match op {
                runmat_mir::MirShortCircuitOp::And => !left_truth,
                runmat_mir::MirShortCircuitOp::Or => left_truth,
            };
            let result = if short_circuited {
                left_truth
            } else {
                for statement in right_temps {
                    execute_embedded_statement(state, &statement.kind)?;
                }
                let right = materialize_operand(state, right)?;
                logical_truth(state, &right, "short-circuit right operand")?
            };
            Ok(vec![state.arena.insert(Value::Bool(result))])
        }
        MirRvalue::Range { start, step, end } => {
            let mut arguments = vec![materialize_operand(state, start)?];
            if let Some(step) = step {
                arguments.push(materialize_operand(state, step)?);
            }
            arguments.push(materialize_operand(state, end)?);
            super::call::builtin(state, "colon", arguments, 1).map(|values| {
                values
                    .into_iter()
                    .map(|value| state.arena.insert(value))
                    .collect()
            })
        }
        MirRvalue::Call(call) => super::call::evaluate(state, call).map(|values| {
            values
                .into_iter()
                .map(|value| state.arena.insert(value))
                .collect()
        }),
        MirRvalue::Aggregate {
            kind,
            rows,
            cols,
            elements,
        } => super::aggregate::evaluate(state, kind, *rows, *cols, elements)
            .map(|value| vec![state.arena.insert(value)]),
        MirRvalue::StructLiteral { fields } => {
            super::aggregate::structure(state, fields).map(|value| vec![state.arena.insert(value)])
        }
        MirRvalue::ObjectLiteral { class_name, fields } => {
            super::aggregate::object(state, class_name, fields)
                .map(|value| vec![state.arena.insert(value)])
        }
        MirRvalue::Index { base, indexing } => {
            super::indexing::read(state, base, indexing, requested_outputs).map(|values| {
                values
                    .into_iter()
                    .map(|value| state.arena.insert(value))
                    .collect()
            })
        }
        MirRvalue::Member { base, member } => {
            let base = materialize_operand(state, base)?;
            let value = super::sync::complete(
                &state.runtime,
                runmat_runtime::object::resolve::load_member(
                    base,
                    member.0.clone(),
                    false,
                    Some(&state.function.name),
                ),
                "member read",
            )?;
            Ok(vec![state.arena.insert(value)])
        }
        MirRvalue::DynamicMember { base, member } => {
            let base = materialize_operand(state, base)?;
            let member = materialize_operand(state, member)?;
            let member = String::try_from(&member).map_err(|error| {
                NativeExecutorError::from(runmat_runtime::runtime_error::semantic_error(
                    "DynamicFieldName",
                    error,
                ))
            })?;
            let value = super::sync::complete(
                &state.runtime,
                runmat_runtime::object::resolve::load_member_dynamic(
                    base,
                    member,
                    false,
                    Some(&state.function.name),
                ),
                "dynamic member read",
            )?;
            Ok(vec![state.arena.insert(value)])
        }
        MirRvalue::WorkspaceFirstStaticProperty {
            workspace_name,
            class_name,
            property,
        } => {
            let value = runmat_runtime::workspace::lookup(&workspace_name.0)
                .or_else(|| {
                    state
                        .function
                        .locals
                        .iter()
                        .find(|local| local.name.as_deref() == Some(workspace_name.0.as_str()))
                        .and_then(|local| state.locals.get(local.id.0 as usize))
                        .copied()
                        .filter(|value| !value.is_null())
                        .and_then(|value| state.arena.get(value).ok().cloned())
                })
                .map(Ok)
                .unwrap_or_else(|| {
                    runmat_runtime::object::resolve::load_static_member(
                        class_name,
                        &property.0,
                        Some(&state.function.name),
                    )
                    .map_err(NativeExecutorError::from)
                })?;
            Ok(vec![state.arena.insert(value)])
        }
        MirRvalue::MetaClass(name) => Ok(vec![state.arena.insert(Value::String(
            name.0
                .iter()
                .map(|segment| segment.0.as_str())
                .collect::<Vec<_>>()
                .join("."),
        ))]),
        MirRvalue::Colon => Ok(vec![state.arena.insert(Value::Num(0.0))]),
        MirRvalue::End => Ok(vec![state.arena.insert(Value::Num(-0.0))]),
        MirRvalue::Future {
            function,
            args,
            requested_outputs,
            ..
        } => {
            let arguments = super::call::materialize_arguments(state, args)?;
            runmat_runtime::execution::validate_spawn_capture(&Value::OutputList(
                arguments.clone(),
            ))?;
            let program = if state.runtime.execution().requires_program_capture() {
                Some(state.program_capture.clone().ok_or_else(|| {
                    NativeExecutorError::Host(
                        "native async execution is missing its exact program".into(),
                    )
                })?)
            } else {
                None
            };
            let future = state
                .runtime
                .execution()
                .create_future(runmat_runtime::execution::DeferredCall {
                    function: function.0,
                    arguments,
                    requested_outputs: requested_outputs.fixed_count(),
                    program_revision: state.runtime.program_revision().cloned(),
                    program,
                })
                .map_err(execution_service_error)?;
            Ok(vec![state.arena.insert(Value::Future(future))])
        }
        MirRvalue::Spawn(operand) => {
            let value = materialize_operand(state, operand)?;
            let Value::Future(future) = value else {
                return Err(runmat_runtime::runtime_error::semantic_error(
                    "SpawnOperandInvalid",
                    "spawn expects a lazy future produced by an async call",
                )
                .into());
            };
            let task = state
                .runtime
                .execution()
                .spawn(&future)
                .map_err(execution_service_error)?;
            Ok(vec![state.arena.insert(Value::Task(task))])
        }
        MirRvalue::Distributed(_) | MirRvalue::Collective(_) => Err(NativeExecutorError::Host(
            "predeclared distributed capability rejection reached native execution".into(),
        )),
    }
}

fn execution_service_error(
    error: runmat_runtime::execution::ExecutionServiceError,
) -> NativeExecutorError {
    runmat_runtime::runtime_error::semantic_error("ExecutionService", error.to_string()).into()
}

fn execute_embedded_statement(
    state: &mut HostState,
    statement: &runmat_mir::MirStmtKind,
) -> NativeExecutorResult<()> {
    match statement {
        runmat_mir::MirStmtKind::Assign {
            place: runmat_mir::MirPlace::Local(local),
            value,
        } => {
            let native_local = u32::try_from(local.0)
                .map(runmat_native_codegen::NativeLocalId)
                .map_err(|_| {
                    NativeExecutorError::Host("embedded local exceeds native schema".into())
                })?;
            let mut values = evaluate_rvalue(state, value, 1, Some(native_local))?;
            if values.len() != 1 {
                return Err(NativeExecutorError::Host(
                    "embedded short-circuit assignment did not produce one value".into(),
                ));
            }
            state.set_local(local.0, values.remove(0))
        }
        runmat_mir::MirStmtKind::Expr(value) => {
            let _ = evaluate_rvalue(state, value, 0, None)?;
            Ok(())
        }
        other => Err(NativeExecutorError::Host(format!(
            "verified short-circuit payload contains invalid statement {other:?}"
        ))),
    }
}

fn logical_truth(state: &HostState, value: &Value, label: &str) -> NativeExecutorResult<bool> {
    super::sync::complete(
        &state.runtime,
        runmat_runtime::condition::logical_truth_from_value(value, label),
        "logical truth evaluation",
    )
}

pub(super) fn materialize_operand(
    state: &mut HostState,
    operand: &MirOperand,
) -> NativeExecutorResult<Value> {
    let reference = evaluate_operand(state, operand)?;
    state.arena.get(reference).cloned()
}

pub(super) fn evaluate_operand(
    state: &mut HostState,
    operand: &MirOperand,
) -> NativeExecutorResult<NativeValueRef> {
    match operand {
        MirOperand::Local(local) => {
            let value = state.locals.get(local.0).copied().ok_or_else(|| {
                NativeExecutorError::Host(format!("local {} is out of bounds", local.0))
            })?;
            if value.is_null()
                && u32::try_from(local.0)
                    .ok()
                    .map(runmat_native_codegen::NativeLocalId)
                    .is_some_and(|local| state.function.abi.fixed_inputs.contains(&local))
            {
                return Err(NativeExecutorError::from(
                    runmat_runtime::runtime_error::semantic_error(
                        "NotEnoughInputs",
                        "Not enough input arguments.",
                    ),
                ));
            }
            Ok(value)
        }
        MirOperand::Constant(constant) => {
            let value = match constant {
                MirConstant::Number(text) => Value::Num(text.parse::<f64>().map_err(|error| {
                    NativeExecutorError::Host(format!(
                        "invalid MIR numeric constant {text:?}: {error}"
                    ))
                })?),
                MirConstant::String(text) if text.is_character_row() => {
                    Value::CharArray(runmat_value::CharArray::new_row(&text.runtime_text()))
                }
                MirConstant::String(text) => Value::String(text.runtime_text()),
                MirConstant::Symbol(symbol) => Value::String(symbol.0.clone()),
                MirConstant::Bool(value) => Value::Bool(*value),
                MirConstant::EmptyArray => Value::Tensor(
                    runmat_value::Tensor::new(Vec::new(), vec![0, 0])
                        .map_err(NativeExecutorError::Host)?,
                ),
            };
            Ok(state.arena.insert(value))
        }
        MirOperand::FunctionHandle(identity) => {
            if let runmat_hir::CallableIdentity::BoundFunction(function)
            | runmat_hir::CallableIdentity::AnonymousFunction(function) = identity
            {
                let function = u32::try_from(function.0)
                    .map(runmat_types::ProgramFunctionId)
                    .map_err(|_| {
                        NativeExecutorError::Host(
                            "function handle identity exceeds native schema".into(),
                        )
                    })?;
                if let Some(target) = state.program_function(function) {
                    let name = target.name.clone();
                    let captures = state.lexical_captures(function)?.unwrap_or_default();
                    let value = if captures.is_empty() {
                        Value::BoundFunctionHandle {
                            name,
                            function: function.0 as usize,
                        }
                    } else {
                        runmat_runtime::call::closures::semantic_closure_value(
                            runmat_types::FunctionId(function.0 as usize),
                            name,
                            captures.into_iter().map(|capture| capture.value).collect(),
                        )
                    };
                    return Ok(state.arena.insert(value));
                }
            }
            identity
                .display_name()
                .map(Value::FunctionHandle)
                .map(|value| state.arena.insert(value))
                .ok_or_else(|| {
                    NativeExecutorError::Host(format!(
                        "function handle {identity:?} has no runtime identity"
                    ))
                })
        }
    }
}

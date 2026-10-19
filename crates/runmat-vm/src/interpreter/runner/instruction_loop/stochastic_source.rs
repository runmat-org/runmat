//! VM publication adapter for an eligible source-level stochastic loop.
//! The Runtime coordinator owns RNG, physical selection, and per-step effects;
//! this module owns only the original VM binding-store semantics.

use super::*;
use crate::bytecode::StochasticSourceOperand;
use runmat_accelerate_api::scalar_program::ScalarLiteral;
use runmat_runtime::builtins::stats::random::stochastic_evolution::{
    positive_stochastic_source_steps, StochasticCheckpoint, StochasticSourceControls,
    StochasticSourceExecution, StochasticSourcePublication, ValueControlRef,
};

pub(super) struct StochasticSourceFrame<'a> {
    pub vars: &'a mut crate::interpreter::vars::InterpreterVars,
    pub context: &'a mut crate::bytecode::program::ExecutionContext,
    pub var_names: &'a HashMap<usize, String>,
    pub completion_assigned_slots: &'a mut Option<HashSet<usize>>,
    pub global_aliases: &'a runmat_runtime::workspace::bindings::FrameGlobalBindings,
    pub persistent_aliases: &'a mut crate::interpreter::PersistentAliases,
    pub missing_input_slots: &'a mut HashSet<usize>,
    pub current_function_name: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StochasticDispatch {
    Declined,
    Completed,
}

pub(super) struct PreparedStochasticSource<'a> {
    iteration_slot: usize,
    noise_slot: usize,
    state_slot: usize,
    prepared:
        runmat_runtime::builtins::stats::random::stochastic_evolution::StochasticPreparedSource,
    frame: StochasticSourceFrame<'a>,
}

impl PreparedStochasticSource<'_> {
    pub(super) fn runtime(&self) -> &runmat_runtime::context::RuntimeContext {
        &self.frame.context.runtime
    }
}

fn plain_source_binding<'a>(
    slot: usize,
    vars: &'a crate::interpreter::vars::InterpreterVars,
    context: &crate::bytecode::program::ExecutionContext,
    global_aliases: &runmat_runtime::workspace::bindings::FrameGlobalBindings,
    persistent_aliases: &crate::interpreter::PersistentAliases,
    missing_input_slots: &HashSet<usize>,
) -> Option<crate::interpreter::vars::InterpreterValueRef<'a>> {
    if global_aliases.has_binding_for_slot(slot)
        || persistent_aliases.contains_key(&slot)
        || context
            .lexical_binding_slots
            .iter()
            .any(|binding| binding.slot == slot)
        || missing_input_slots.contains(&slot)
        || crate::runtime::workspace::workspace_slot_assigned(slot) != Some(true)
    {
        return None;
    }
    vars.get(slot)
}

fn plain_destination_binding(
    slot: usize,
    context: &crate::bytecode::program::ExecutionContext,
    global_aliases: &runmat_runtime::workspace::bindings::FrameGlobalBindings,
    persistent_aliases: &crate::interpreter::PersistentAliases,
) -> bool {
    !global_aliases.has_binding_for_slot(slot)
        && !persistent_aliases.contains_key(&slot)
        && !context
            .lexical_binding_slots
            .iter()
            .any(|binding| binding.slot == slot)
}

enum ResolvedSourceControl<'a> {
    Slot(crate::interpreter::vars::InterpreterValueRef<'a>),
    Scalar(ScalarLiteral),
    Integer(&'a runmat_value::IntValue),
    Text(&'a str),
}

impl ResolvedSourceControl<'_> {
    fn as_control(&self) -> ValueControlRef<'_> {
        match self {
            Self::Slot(value) => ValueControlRef::Borrowed(value),
            Self::Scalar(value) => ValueControlRef::Scalar(*value),
            Self::Integer(value) => ValueControlRef::Integer(value),
            Self::Text(value) => ValueControlRef::Text(value),
        }
    }
}

fn resolve_source_control<'a>(
    operand: &'a StochasticSourceOperand,
    vars: &'a crate::interpreter::vars::InterpreterVars,
    context: &crate::bytecode::program::ExecutionContext,
    global_aliases: &runmat_runtime::workspace::bindings::FrameGlobalBindings,
    persistent_aliases: &crate::interpreter::PersistentAliases,
    missing_input_slots: &HashSet<usize>,
) -> Option<ResolvedSourceControl<'a>> {
    match operand {
        StochasticSourceOperand::Slot(slot) => plain_source_binding(
            *slot,
            vars,
            context,
            global_aliases,
            persistent_aliases,
            missing_input_slots,
        )
        .map(ResolvedSourceControl::Slot),
        StochasticSourceOperand::Number(value) => Some(ResolvedSourceControl::Scalar(
            ScalarLiteral::F64Bits(value.to_bits()),
        )),
        StochasticSourceOperand::Integer(value) => Some(ResolvedSourceControl::Integer(value)),
        StochasticSourceOperand::String(value) => Some(ResolvedSourceControl::Text(value)),
        StochasticSourceOperand::Bool(_) => None,
    }
}

/// Resolve and admit all borrowed source controls synchronously. A rejected
/// source loop returns `None` without constructing an async continuation; the
/// selected preparation owns every fact needed after the read guards retire.
pub(super) fn prepare<'a>(
    instruction: &'a Instr,
    frame: StochasticSourceFrame<'a>,
) -> Result<Option<PreparedStochasticSource<'a>>, RuntimeError> {
    let Instr::StochasticEvolution {
        iteration_slot,
        noise_slot,
        state_slot,
        steps,
        drift,
        scale,
        randn_args,
        ..
    } = instruction
    else {
        unreachable!("stochastic source dispatch requires its typed instruction")
    };
    let StochasticSourceFrame {
        vars,
        context,
        var_names,
        completion_assigned_slots,
        global_aliases,
        persistent_aliases,
        missing_input_slots,
        current_function_name,
    } = frame;
    let runtime = context.runtime.clone();
    let borrow = |operand| {
        resolve_source_control(
            operand,
            vars,
            context,
            global_aliases,
            persistent_aliases,
            missing_input_slots,
        )
    };
    let Some(steps) = borrow(steps) else {
        return Ok(None);
    };
    let positive_steps = match steps.as_control() {
        ValueControlRef::Borrowed(value) => positive_stochastic_source_steps(value),
        ValueControlRef::Scalar(ScalarLiteral::F64Bits(bits)) => {
            positive_stochastic_source_steps(&Value::Num(f64::from_bits(bits)))
        }
        _ => None,
    };
    if positive_steps.is_none() {
        return Ok(None);
    }
    // Output aliases require a separate retained publication/rollback proof.
    // They need not be assigned before the original loop executes.
    if !plain_destination_binding(*iteration_slot, context, global_aliases, persistent_aliases)
        || !plain_destination_binding(*noise_slot, context, global_aliases, persistent_aliases)
    {
        return Ok(None);
    }
    // A range with no source iterations must not consult S or any randn/body
    // control, even at the level of a speculative binding proof.
    let Some(state) = plain_source_binding(
        *state_slot,
        vars,
        context,
        global_aliases,
        persistent_aliases,
        missing_input_slots,
    ) else {
        return Ok(None);
    };
    let (Some(drift), Some(scale)) = (borrow(drift), borrow(scale)) else {
        return Ok(None);
    };
    let Some(randn_controls) = randn_args.iter().map(borrow).collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    // Runtime alone owns the parser's exact temporary footprint and the
    // retained preparation permit. No Value or shape is cloned in this VM
    // borrow-only probe. The real preparation/executor owner is joined by the
    // whole-call coordinator; this isolated source packet has no stub route.
    let prepared = {
        let arg_at = |index| {
            randn_controls
                .get(index)
                .map(ResolvedSourceControl::as_control)
        };
        runmat_runtime::builtins::stats::random::stochastic_evolution::prepare_stochastic_source(
            &runtime,
            StochasticSourceControls {
                state: &state,
                steps: steps.as_control(),
                drift: drift.as_control(),
                scale: scale.as_control(),
                randn_arg_count: randn_args.len(),
                randn_arg: &arg_at,
            },
        )?
    };
    let Some(prepared) = prepared else {
        return Ok(None);
    };
    // Preparation owns every source fact needed by execution. Release all
    // InterpreterVars read guards before the checkpoint callback mutates that
    // same shared slot control or the executor suspends.
    drop(randn_controls);
    drop(scale);
    drop(drift);
    drop(state);
    drop(steps);
    Ok(Some(PreparedStochasticSource {
        iteration_slot: *iteration_slot,
        noise_slot: *noise_slot,
        state_slot: *state_slot,
        prepared,
        frame: StochasticSourceFrame {
            vars,
            context,
            var_names,
            completion_assigned_slots,
            global_aliases,
            persistent_aliases,
            missing_input_slots,
            current_function_name,
        },
    }))
}

pub(super) async fn execute_selected(
    selected: PreparedStochasticSource<'_>,
) -> Result<StochasticDispatch, RuntimeError> {
    let PreparedStochasticSource {
        iteration_slot,
        noise_slot,
        state_slot,
        prepared,
        frame,
    } = selected;
    let StochasticSourceFrame {
        vars,
        context,
        var_names,
        completion_assigned_slots,
        global_aliases: _,
        persistent_aliases,
        missing_input_slots,
        current_function_name,
    } = frame;
    let runtime = context.runtime.clone();
    let mut publish = |checkpoint: StochasticCheckpoint| {
        let (slot, value) = match checkpoint {
            StochasticCheckpoint::IterationBinding(counter) => {
                (iteration_slot, Value::Num(counter))
            }
            StochasticCheckpoint::Noise(value) => (noise_slot, value),
            StochasticCheckpoint::State(value) => (state_slot, value),
        };
        interp_dispatch::publish_var_value(
            value,
            slot,
            interp_dispatch::BindingPublicationFrame {
                vars: &mut *vars,
                context: &mut *context,
                var_names,
                persistent_aliases: &mut *persistent_aliases,
                missing_input_slots: &mut *missing_input_slots,
                current_function_name,
            },
            |_stored_index| {},
            |stored_index| {
                after_global_store(&mut *completion_assigned_slots, stored_index);
            },
        )
    };
    let result =
        runmat_runtime::builtins::stats::random::stochastic_evolution::execute_stochastic_source(
            &runtime,
            prepared,
            StochasticSourcePublication {
                // Ordinary binding stores own shape views, projection roots
                // and namespace bookkeeping. Their finite publication quote
                // must come from those owners, not from numeric payload size.
                host_workspace: runmat_execution::ResourceByteBound::Unavailable(
                    runmat_execution::ResourceBoundUnavailable::NoScratchBound,
                ),
                checkpoint: &mut publish,
            },
        )
        .await?;
    Ok(match result {
        StochasticSourceExecution::Declined => StochasticDispatch::Declined,
        StochasticSourceExecution::Completed => StochasticDispatch::Completed,
    })
}

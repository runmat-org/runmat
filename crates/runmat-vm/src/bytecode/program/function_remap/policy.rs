use std::collections::HashMap;

use runmat_hir::FunctionId;
use runmat_types::CallableIdentity;

use super::grammar::with_instruction_function_references;
use crate::bytecode::{BytecodeSubscriptStep, FunctionBytecode, Instr};

#[derive(Clone, Copy)]
enum RemapPolicy {
    /// Eval definitions are merged into a live parent registry. References
    /// explicitly bound to the parent stay in that parent's identity domain,
    /// even when their numeric IDs overlap eval-local IDs.
    EvalLocal,
    /// Every compiler-bound reference belongs to the admitted closed unit and
    /// must be projected into its destination identity domain.
    ClosedUnit,
}

struct RemapContext<'a> {
    ids: &'a HashMap<FunctionId, FunctionId>,
    policy: RemapPolicy,
}

fn remap_function_with_policy(
    function: &mut FunctionBytecode,
    remap: &HashMap<FunctionId, FunctionId>,
    policy: RemapPolicy,
) {
    if let Some(destination) = remap.get(&function.function).copied() {
        function.rebind_identity(destination, function.portable_function);
    }
    let context = RemapContext { ids: remap, policy };
    for instruction in &mut function.instructions {
        remap_instruction_with_policy(instruction, &context);
    }
}

pub(crate) fn remap_eval_local_function(
    function: &mut FunctionBytecode,
    remap: &HashMap<FunctionId, FunctionId>,
) {
    remap_function_with_policy(function, remap, RemapPolicy::EvalLocal);
}

pub(crate) fn remap_closed_function(
    function: &mut FunctionBytecode,
    remap: &HashMap<FunctionId, FunctionId>,
) {
    remap_function_with_policy(function, remap, RemapPolicy::ClosedUnit);
}

pub(crate) fn remap_eval_local_instruction(
    instruction: &mut Instr,
    remap: &HashMap<FunctionId, FunctionId>,
) {
    let context = RemapContext {
        ids: remap,
        policy: RemapPolicy::EvalLocal,
    };
    remap_instruction_with_policy(instruction, &context);
}

fn remap_instruction_with_policy(instruction: &mut Instr, context: &RemapContext<'_>) {
    with_instruction_function_references!(
        instruction,
        context,
        remap_id_in,
        remap_external_id_in,
        remap_identity_in,
        remap_subscript_steps_in
    );
}

pub(in super::super) fn visit_instruction(instruction: &Instr, mut visit: impl FnMut(FunctionId)) {
    with_instruction_function_references!(
        instruction,
        &mut visit,
        visit_id,
        visit_id,
        visit_identity,
        visit_subscript_steps
    );
}

fn remap_subscript_steps_in(context: &RemapContext<'_>, steps: &mut [BytecodeSubscriptStep]) {
    for step in steps {
        if let BytecodeSubscriptStep::DottedInvoke {
            semantic_fallback: Some(identity),
            ..
        } = step
        {
            remap_identity_in(context, identity);
        }
    }
}

fn remap_identity_in(context: &RemapContext<'_>, identity: &mut CallableIdentity) {
    match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function) => remap_id_in(context, function),
        CallableIdentity::ExternalFunction { function, .. } => {
            remap_external_id_in(context, function)
        }
        CallableIdentity::Builtin(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_)
        | CallableIdentity::DynamicName(_)
        | CallableIdentity::ExternalName(_) => {}
    }
}

fn remap_id_in(context: &RemapContext<'_>, function: &mut FunctionId) {
    if let Some(destination) = context.ids.get(function).copied() {
        *function = destination;
    }
}

fn remap_external_id_in(context: &RemapContext<'_>, function: &mut FunctionId) {
    if matches!(context.policy, RemapPolicy::ClosedUnit) {
        remap_id_in(context, function);
    }
}

fn visit_id(visit: &mut impl FnMut(FunctionId), function: &FunctionId) {
    visit(*function);
}

fn visit_identity(visit: &mut impl FnMut(FunctionId), identity: &CallableIdentity) {
    match identity {
        CallableIdentity::BoundFunction(function)
        | CallableIdentity::AnonymousFunction(function)
        | CallableIdentity::ExternalFunction { function, .. } => visit(*function),
        CallableIdentity::Builtin(_)
        | CallableIdentity::Imported(_)
        | CallableIdentity::Method(_)
        | CallableIdentity::DynamicName(_)
        | CallableIdentity::ExternalName(_) => {}
    }
}

fn visit_subscript_steps(visit: &mut impl FnMut(FunctionId), steps: &[BytecodeSubscriptStep]) {
    for step in steps {
        if let BytecodeSubscriptStep::DottedInvoke {
            semantic_fallback: Some(identity),
            ..
        } = step
        {
            visit_identity(visit, identity);
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

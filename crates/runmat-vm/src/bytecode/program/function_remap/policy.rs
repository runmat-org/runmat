use std::collections::HashMap;

use runmat_hir::FunctionId;
use runmat_types::{CallableIdentity, ProgramFunctionId};

use super::grammar::with_instruction_function_references;
use crate::bytecode::{
    Bytecode, BytecodeSourceRef, BytecodeSubscriptStep, FunctionBytecode, Instr,
};

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
    #[cfg(feature = "native-accel")]
    let selected_stage_valid = function.validate_fusion_stage_projection().is_ok();
    let context = RemapContext { ids: remap, policy };
    for instruction in &mut function.instructions {
        remap_instruction_with_policy(instruction, &context);
    }
    #[cfg(feature = "native-accel")]
    if selected_stage_valid {
        function.reseal_fusion_stage_after_owned_remap();
    } else {
        function.fusion_stage_projection = None;
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

/// Rebind a session-owned body as one validated transform. Compiler-local IDs
/// move, while externally bound parent references retain their identity domain.
pub fn remap_session_function(
    function: &mut FunctionBytecode,
    destination: FunctionId,
    program_function: ProgramFunctionId,
    source: Option<BytecodeSourceRef>,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    #[cfg(feature = "native-accel")]
    function.validate_fusion_stage_projection()?;
    let mut next = function.clone();
    next.rebind_identity_unsealed(destination, program_function);
    next.source_id = source;
    let context = RemapContext {
        ids: remap,
        policy: RemapPolicy::EvalLocal,
    };
    for instruction in &mut next.instructions {
        remap_instruction_with_policy(instruction, &context);
    }
    #[cfg(feature = "native-accel")]
    {
        next.reseal_fusion_stage_after_owned_remap();
        next.validate_fusion_stage_projection()?;
        if function.fusion_stage_projection.is_some() && next.fusion_stage_projection.is_none() {
            return Err("selected function MIR projection cannot be resealed".into());
        }
    }
    *function = next;
    Ok(())
}

pub fn rebind_session_source(
    function: &mut FunctionBytecode,
    source: Option<BytecodeSourceRef>,
) -> Result<(), String> {
    #[cfg(feature = "native-accel")]
    function.validate_fusion_stage_projection()?;
    let mut next = function.clone();
    next.source_id = source;
    #[cfg(feature = "native-accel")]
    {
        next.reseal_fusion_stage_after_owned_remap();
        next.validate_fusion_stage_projection()?;
        if function.fusion_stage_projection.is_some() && next.fusion_stage_projection.is_none() {
            return Err("selected function MIR projection cannot be resealed".into());
        }
    }
    *function = next;
    Ok(())
}

pub fn remap_session_entrypoint(
    bytecode: &mut Bytecode,
    remap: &HashMap<FunctionId, FunctionId>,
    bindings: &[SessionCallableBinding],
) -> Result<(), String> {
    #[cfg(feature = "native-accel")]
    bytecode.validate_fusion_stage_projection()?;
    let mut next = bytecode.clone();
    let context = RemapContext {
        ids: remap,
        policy: RemapPolicy::EvalLocal,
    };
    for instruction in &mut next.instructions {
        remap_instruction_with_policy(instruction, &context);
    }
    let mut bound_pcs = HashMap::<usize, (FunctionId, &str)>::new();
    for binding in bindings {
        if let Some((function, name)) = bound_pcs.insert(
            binding.pc,
            (binding.function, binding.display_name.as_str()),
        ) {
            if function == binding.function && name == binding.display_name {
                continue;
            }
            return Err("conflicting session callable bindings at one instruction".into());
        }
        if next.function_registry.resolve_name(&binding.display_name) != Some(binding.function) {
            return Err("session callable binding disagrees with selected registry".into());
        }
        let instruction = next.instructions.get_mut(binding.pc).ok_or_else(|| {
            "session callable binding points outside instruction stream".to_string()
        })?;
        let admissible = match instruction {
            Instr::CreateFunctionHandle(name) => name == &binding.display_name,
            Instr::LoadString(text) | Instr::LoadCharRow(text) => {
                text.trim().strip_prefix('@').unwrap_or(text.trim()).trim() == binding.display_name
            }
            _ => false,
        };
        if !admissible {
            return Err("session callable binding does not match source instruction".into());
        }
        let role = next
            .function_registry
            .get(binding.function)
            .map_or(runmat_types::FunctionSourceRole::Unknown, |body| {
                body.source_role
            });
        *instruction =
            Instr::CreateBoundFunctionHandle(binding.function, binding.display_name.clone(), role);
    }
    #[cfg(feature = "native-accel")]
    {
        next.reseal_fusion_stage_after_owned_remap();
        next.validate_fusion_stage_projection()?;
        if bytecode.fusion_metadata.stage_projection.is_some()
            && next.fusion_metadata.stage_projection.is_none()
        {
            return Err("selected entrypoint MIR projection cannot be resealed".into());
        }
    }
    *bytecode = next;
    Ok(())
}

/// A Core-resolved callable-literal edit, checked against the selected VM
/// registry and original instruction before entering the sealed stream.
pub struct SessionCallableBinding {
    pub pc: usize,
    pub function: FunctionId,
    pub display_name: String,
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

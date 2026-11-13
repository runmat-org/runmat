use std::collections::HashMap;

use runmat_hir::FunctionId;
use runmat_types::CallableIdentity;

use super::grammar::with_instruction_function_references;
use super::reference_sites::FunctionReferenceKind;
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
) -> Result<(), String> {
    let lexical_parent = project_lexical_parent(function, remap)?;
    let inherited_storage = project_inherited_storage(function, remap)?;
    if let Some(destination) = remap.get(&function.function).copied() {
        function.rebind_identity(destination);
    }
    function.lexical_parent = lexical_parent;
    function.inherited_storage_slots = inherited_storage;
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
    Ok(())
}

fn project_lexical_parent(
    function: &FunctionBytecode,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<Option<FunctionId>, String> {
    match (function.lexical_parent, function.portable_lexical_parent) {
        (None, None) if function.source_role == runmat_types::FunctionSourceRole::Nested => {
            Err("nested definition has no lexical parent".into())
        }
        (None, None) => Ok(None),
        (Some(parent), Some(_)) => {
            remap.get(&parent).copied().map(Some).ok_or_else(|| {
                format!("nested definition has no admitted lexical parent {parent:?}")
            })
        }
        _ => Err("nested definition has inconsistent lexical parent identities".into()),
    }
}

fn project_inherited_storage(
    function: &FunctionBytecode,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<Vec<crate::bytecode::InheritedStorageSlot>, String> {
    function
        .inherited_storage_slots
        .iter()
        .map(|entry| {
            let mut projected = entry.clone();
            projected.owner = remap.get(&entry.owner).copied().ok_or_else(|| {
                format!(
                    "inherited canonical storage owner {:?} is outside admitted function map",
                    entry.owner
                )
            })?;
            Ok(projected)
        })
        .collect()
}

pub(crate) fn remap_eval_local_function(
    function: &mut FunctionBytecode,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    remap_function_with_policy(function, remap, RemapPolicy::EvalLocal)
}

pub(crate) fn remap_closed_function(
    function: &mut FunctionBytecode,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    remap_function_with_policy(function, remap, RemapPolicy::ClosedUnit)
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
/// and same-unit lexical parents move. Externally bound call references retain
/// their identity domain; no producer admits an external lexical parent.
pub fn remap_session_function(
    function: &mut FunctionBytecode,
    destination: FunctionId,
    source: Option<BytecodeSourceRef>,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    remap_session_function_with_policy(function, destination, source, remap, RemapPolicy::EvalLocal)
}

/// Rebind a proved closed product body after fresh session IDs have been
/// assigned. Unlike eval-local remapping, every reference to a fresh ID is
/// owned by this product and must move, including external-call encodings.
pub fn remap_closed_session_function(
    function: &mut FunctionBytecode,
    destination: FunctionId,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    let source = function.source_id;
    remap_session_function_with_policy(
        function,
        destination,
        source,
        remap,
        RemapPolicy::ClosedUnit,
    )
}

fn remap_session_function_with_policy(
    function: &mut FunctionBytecode,
    destination: FunctionId,
    source: Option<BytecodeSourceRef>,
    remap: &HashMap<FunctionId, FunctionId>,
    policy: RemapPolicy,
) -> Result<(), String> {
    #[cfg(feature = "native-accel")]
    function.validate_fusion_stage_projection()?;
    let mut next = function.clone();
    next.lexical_parent = project_lexical_parent(&next, remap)?;
    next.inherited_storage_slots = project_inherited_storage(&next, remap)?;
    next.rebind_identity_unsealed(destination);
    next.source_id = source;
    let context = RemapContext { ids: remap, policy };
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
    remap_session_entrypoint_with_policy(bytecode, remap, bindings, RemapPolicy::EvalLocal)
}

pub(crate) fn remap_closed_session_entrypoint(
    bytecode: &mut Bytecode,
    remap: &HashMap<FunctionId, FunctionId>,
) -> Result<(), String> {
    remap_session_entrypoint_with_policy(bytecode, remap, &[], RemapPolicy::ClosedUnit)
}

fn remap_session_entrypoint_with_policy(
    bytecode: &mut Bytecode,
    remap: &HashMap<FunctionId, FunctionId>,
    bindings: &[SessionCallableBinding],
    policy: RemapPolicy,
) -> Result<(), String> {
    #[cfg(feature = "native-accel")]
    bytecode.validate_fusion_stage_projection()?;
    let mut next = bytecode.clone();
    let context = RemapContext { ids: remap, policy };
    for instruction in &mut next.instructions {
        remap_instruction_with_policy(instruction, &context);
    }
    let mut bound_pcs = HashMap::new();
    for binding in bindings {
        if let Some((function, name, kind, reference)) = bound_pcs.insert(
            binding.pc,
            (
                binding.function,
                binding.display_name.as_str(),
                binding.kind,
                binding.reference,
            ),
        ) {
            if function == binding.function
                && name == binding.display_name
                && kind == binding.kind
                && reference == binding.reference
            {
                continue;
            }
            return Err("conflicting session callable bindings at one instruction".into());
        }
        let selected = match binding.kind {
            SessionCallableBindingKind::Named => {
                next.function_registry.resolve_name(&binding.display_name)
            }
            SessionCallableBindingKind::Constructor => next
                .function_registry
                .resolve_session_constructor_handle(&binding.display_name)?,
        };
        if selected != Some(binding.function) {
            return Err("session callable binding disagrees with selected registry".into());
        }
        let instruction = next.instructions.get_mut(binding.pc).ok_or_else(|| {
            "session callable binding points outside instruction stream".to_string()
        })?;
        let admissible = match binding.kind {
            SessionCallableBindingKind::Named => matches!(
                instruction,
                Instr::CreateFunctionHandle(name) if name == &binding.display_name
            ),
            SessionCallableBindingKind::Constructor => matches!(
                instruction,
                Instr::CreateFunctionHandle(name) | Instr::CreateExternalFunctionHandle(name)
                    if name == &binding.display_name
            ),
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
        // Source spelling does not establish the definition's identity domain:
        // an unqualified constructor can belong to a prior session product.
        *instruction = if binding.reference == FunctionReferenceKind::External {
            Instr::CreateExternalBoundFunctionHandle(
                binding.function,
                binding.display_name.clone(),
                role,
            )
        } else {
            Instr::CreateBoundFunctionHandle(binding.function, binding.display_name.clone(), role)
        };
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

/// A Core-resolved explicit function-handle edit, checked against the selected
/// VM registry and original instruction before entering the sealed stream.
/// Textual callback names remain runtime name selections.
pub struct SessionCallableBinding {
    pub pc: usize,
    pub function: FunctionId,
    pub display_name: String,
    kind: SessionCallableBindingKind,
    reference: FunctionReferenceKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SessionCallableBindingKind {
    Named,
    Constructor,
}

impl SessionCallableBinding {
    pub fn named(
        pc: usize,
        function: FunctionId,
        display_name: String,
        reference: FunctionReferenceKind,
    ) -> Self {
        Self {
            pc,
            function,
            display_name,
            kind: SessionCallableBindingKind::Named,
            reference,
        }
    }

    pub fn constructor(
        pc: usize,
        function: FunctionId,
        display_name: String,
        reference: FunctionReferenceKind,
    ) -> Self {
        Self {
            pc,
            function,
            display_name,
            kind: SessionCallableBindingKind::Constructor,
            reference,
        }
    }
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
            let changed = match identity.as_ref() {
                CallableIdentity::BoundFunction(function)
                | CallableIdentity::AnonymousFunction(function) => context
                    .ids
                    .get(function)
                    .is_some_and(|next| next != function),
                CallableIdentity::ExternalFunction { function, .. }
                    if matches!(context.policy, RemapPolicy::ClosedUnit) =>
                {
                    context
                        .ids
                        .get(function)
                        .is_some_and(|next| next != function)
                }
                _ => false,
            };
            if changed {
                remap_identity_in(context, std::sync::Arc::make_mut(identity));
            }
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

use std::collections::{HashMap, HashSet};

use runmat_hir::{
    BindingId, CapturedBinding, ClassId, FunctionAbi, FunctionArgumentValidation, FunctionId,
    FunctionKind, FunctionModifiers, FunctionName, HirAssembly, HirBinding, ModuleId, Span,
};

use crate::bytecode::FunctionBytecode;

/// A late source selection may rewrite call identity and output arity inside
/// this body, but it cannot change which closure value the already-running
/// frame was about to create. The body is admitted only if its lexical ABI
/// and compiler-owned capture layout still match that exact issuance site.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClosureIssuanceAbi {
    function: FunctionId,
    source_role: runmat_types::FunctionSourceRole,
    module: ModuleId,
    parent: Option<FunctionId>,
    enclosing_class: Option<ClassId>,
    name: FunctionName,
    kind: FunctionKind,
    modifiers: FunctionModifiers,
    span: Span,
    params: Vec<BindingId>,
    outputs: Vec<BindingId>,
    abi: FunctionAbi,
    argument_validations: Vec<FunctionArgumentValidation>,
    locals: Vec<BindingId>,
    captures: Vec<CapturedBinding>,
    display_name: String,
    source_id: Option<crate::BytecodeSourceRef>,
    class_method_owner: Option<runmat_types::ClassMethodOwner>,
    private_owner_scope: String,
    input_slots: Vec<usize>,
    varargin_slot: Option<usize>,
    output_slots: Vec<usize>,
    varargout_slot: Option<usize>,
    capture_slots: Vec<usize>,
    lexical_binding_slots: Vec<crate::bytecode::program::LexicalBindingSlot>,
    inherited_storage_slots: Vec<crate::bytecode::program::InheritedStorageSlot>,
    creates_lexical_environment: bool,
    lexical_parent: Option<FunctionId>,
    portable_lexical_parent: Option<runmat_types::ProgramFunctionId>,
    /// Names and declaration facts, not compiler-local BindingIds. Workspace
    /// assignment can seed new outer bindings before the same AST is lowered
    /// at this closure's issuance point.
    params_shape: Vec<HirBinding>,
    outputs_shape: Vec<HirBinding>,
    locals_shape: Vec<HirBinding>,
    captures_shape: Vec<HirBinding>,
    parent_bindings_shape: Vec<HirBinding>,
    auxiliary_bindings_shape: [Option<HirBinding>; 4],
    canonical_output_forwarding: bool,
}

#[derive(Clone, Debug)]
pub(crate) enum SelectedClosureCapture {
    Original(usize),
    AssignedWorkspace { name: String, slot: usize },
}

fn binding_shape(hir: &HirAssembly, id: BindingId) -> Option<HirBinding> {
    let mut binding = hir
        .bindings
        .iter()
        .find(|binding| binding.id == id)?
        .clone();
    // Local IDs are an assembly implementation detail; all remaining fields
    // describe the typed lexical declaration and must still agree. An
    // incoming workspace value uses the same lexical slot and capture ABI as
    // a local that the earlier invocation assigned; its different role only
    // records the MIR entry-initialization fact.
    binding.id = BindingId(0);
    if binding.role == runmat_hir::BindingRole::IncomingWorkspace {
        binding.role = runmat_hir::BindingRole::Local;
    }
    Some(binding)
}

fn binding_shapes(hir: &HirAssembly, ids: &[BindingId]) -> Option<Vec<HirBinding>> {
    ids.iter().map(|id| binding_shape(hir, *id)).collect()
}

fn optional_binding_shape(hir: &HirAssembly, id: Option<BindingId>) -> Option<Option<HirBinding>> {
    match id {
        Some(id) => binding_shape(hir, id).map(Some),
        None => Some(None),
    }
}

impl ClosureIssuanceAbi {
    pub(crate) fn original_capture_count(&self) -> usize {
        self.captures.len()
    }

    pub(super) fn capture(
        hir: &HirAssembly,
        body: &FunctionBytecode,
        function: FunctionId,
    ) -> Option<Self> {
        let selected = hir
            .functions
            .iter()
            .find(|candidate| candidate.id == function)?;
        (body.function == function).then_some(())?;
        Some(Self {
            function,
            source_role: hir.function_source_role(selected),
            module: selected.module,
            parent: selected.parent,
            enclosing_class: selected.enclosing_class,
            name: selected.name.clone(),
            kind: selected.kind.clone(),
            modifiers: selected.modifiers.clone(),
            span: selected.span,
            params: selected.params.clone(),
            outputs: selected.outputs.clone(),
            abi: selected.abi.clone(),
            argument_validations: selected.argument_validations.clone(),
            locals: selected.locals.clone(),
            captures: selected.captures.clone(),
            display_name: body.display_name.clone(),
            source_id: body.source_id,
            class_method_owner: body.class_method_owner.clone(),
            private_owner_scope: body.private_owner_scope.clone(),
            input_slots: body.input_slots.clone(),
            varargin_slot: body.varargin_slot,
            output_slots: body.output_slots.clone(),
            varargout_slot: body.varargout_slot,
            capture_slots: body.capture_slots.clone(),
            lexical_binding_slots: body.lexical_binding_slots.clone(),
            inherited_storage_slots: body.inherited_storage_slots.clone(),
            creates_lexical_environment: body.creates_lexical_environment,
            lexical_parent: body.lexical_parent,
            portable_lexical_parent: body.portable_lexical_parent,
            params_shape: binding_shapes(hir, &selected.params)?,
            outputs_shape: binding_shapes(hir, &selected.outputs)?,
            locals_shape: binding_shapes(hir, &selected.locals)?,
            captures_shape: selected
                .captures
                .iter()
                .map(|capture| binding_shape(hir, capture.binding))
                .collect::<Option<Vec<_>>>()?,
            parent_bindings_shape: hir
                .bindings
                .iter()
                .filter(|binding| {
                    Some(binding.owner) == selected.parent.map(runmat_hir::BindingOwner::Function)
                })
                .map(|binding| binding_shape(hir, binding.id))
                .collect::<Option<Vec<_>>>()?,
            auxiliary_bindings_shape: [
                optional_binding_shape(hir, selected.abi.varargin)?,
                optional_binding_shape(hir, selected.abi.varargout)?,
                optional_binding_shape(hir, selected.abi.implicit_nargin)?,
                optional_binding_shape(hir, selected.abi.implicit_nargout)?,
            ],
            canonical_output_forwarding: match selected.body.statements.as_slice() {
                [statement] => matches!(&statement.kind,
                    runmat_hir::HirStmtKind::SequenceAssign { target, .. }
                        if super::anonymous_output_forwarding_target(selected, target)
                            && selected.abi.varargout.is_some_and(|output| selected.outputs == [output])
                            && selected.abi.implicit_nargout.is_some_and(|count| selected.locals.contains(&count))),
                _ => false,
            },
        })
    }

    pub(super) fn matches_selected(&self, hir: &HirAssembly, body: &FunctionBytecode) -> bool {
        Self::capture(hir, body, self.function).is_some_and(|selected| selected == *self)
    }

    pub(in crate::call::builtins) fn capture_plan(
        &self,
        hir: &HirAssembly,
        body: &FunctionBytecode,
        assigned_slots: &HashMap<String, usize>,
    ) -> Option<Vec<SelectedClosureCapture>> {
        let selected = Self::capture(hir, body, self.function)?;
        if self.matches_selected(hir, body) {
            return Some(
                (0..self.captures.len())
                    .map(SelectedClosureCapture::Original)
                    .collect(),
            );
        }
        // A change to an existing capture or any callable signature is not
        // an assignment shadow. Only newly seeded, already-assigned outer
        // bindings may extend the capture vector of this exact anonymous AST.
        if self.kind != FunctionKind::Anonymous
            || !self.argument_validations.is_empty()
            || !selected.argument_validations.is_empty()
            || selected.function != self.function
            || selected.source_role != self.source_role
            || selected.module != self.module
            || selected.parent != self.parent
            || selected.enclosing_class != self.enclosing_class
            || selected.name != self.name
            || selected.kind != self.kind
            || selected.modifiers != self.modifiers
            || selected.span != self.span
            || selected.display_name != self.display_name
            || selected.source_id != self.source_id
            || selected.class_method_owner != self.class_method_owner
            || selected.private_owner_scope != self.private_owner_scope
            || selected.params_shape != self.params_shape
            || selected.outputs_shape != self.outputs_shape
            || selected.auxiliary_bindings_shape != self.auxiliary_bindings_shape
            || !self.parent_bindings_shape.iter().all(|binding| {
                selected.parent_bindings_shape.contains(binding)
                    || (assigned_slots.contains_key(&binding.name.0)
                        && binding.role == runmat_hir::BindingRole::Local
                        && binding.storage == runmat_hir::BindingStorage::Lexical
                        && binding.workspace_visibility
                            == runmat_hir::WorkspaceVisibility::TopLevel
                        && selected.parent_bindings_shape.iter().any(|candidate| {
                            candidate.owner == binding.owner
                                && candidate.name == binding.name
                                && candidate.role == binding.role
                                && candidate.storage == binding.storage
                                && candidate.workspace_visibility == binding.workspace_visibility
                                && candidate.declared_span == Span { start: 0, end: 0 }
                        }))
            })
            || selected.params.len() != self.params.len()
            || selected.outputs.len() != self.outputs.len()
            || selected.abi.fixed_inputs.len() != self.abi.fixed_inputs.len()
            || selected.abi.fixed_outputs.len() != self.abi.fixed_outputs.len()
            || self.abi.fixed_inputs != self.params
            || selected.abi.fixed_inputs != selected.params
            || self.abi.fixed_outputs != self.outputs
            || selected.abi.fixed_outputs != selected.outputs
            || selected.abi.varargin.is_some() != self.abi.varargin.is_some()
            || selected.abi.varargout.is_some() != self.abi.varargout.is_some()
            || selected.abi.implicit_nargin.is_some() != self.abi.implicit_nargin.is_some()
            || selected.abi.implicit_nargout.is_some() != self.abi.implicit_nargout.is_some()
            || self.abi.varargin.is_some()
            || self.abi.implicit_nargin.is_some()
            // Current anonymous bodies have an ABI-owned varargout/count
            // forwarding pair. Capture extension may retain that exact pair,
            // never admit a different output destination or user variadic ABI.
            || ((self.abi.varargout.is_some() || self.abi.implicit_nargout.is_some())
                && !(self.canonical_output_forwarding && selected.canonical_output_forwarding))
            || selected.input_slots.len() != self.input_slots.len()
            || selected.output_slots.len() != self.output_slots.len()
            || selected.varargin_slot.is_some() != self.varargin_slot.is_some()
            || selected.varargout_slot.is_some() != self.varargout_slot.is_some()
            || selected.creates_lexical_environment != self.creates_lexical_environment
            || selected.lexical_parent != self.lexical_parent
            || selected.portable_lexical_parent != self.portable_lexical_parent
            || selected.capture_slots.len() != selected.captures.len()
        {
            return None;
        }
        let mut seen_workspace = HashSet::new();
        let mut seen_original = HashSet::new();
        let plan = selected
            .captures
            .iter()
            .map(|capture| {
                let binding = hir
                    .bindings
                    .iter()
                    .find(|item| item.id == capture.binding)?;
                let name = binding.name.0.clone();
                let shape = binding_shape(hir, capture.binding)?;
                if let Some(index) = self
                    .captures_shape
                    .iter()
                    .position(|original| original == &shape)
                {
                    return seen_original
                        .insert(index)
                        .then_some(SelectedClosureCapture::Original(index));
                }
                let slot = *assigned_slots.get(&name)?;
                (binding.owner == runmat_hir::BindingOwner::Function(capture.from_function)
                    && Some(capture.from_function) == self.parent
                    && matches!(
                        binding.role,
                        runmat_hir::BindingRole::Local | runmat_hir::BindingRole::IncomingWorkspace
                    )
                    && binding.storage == runmat_hir::BindingStorage::Lexical
                    && binding.workspace_visibility == runmat_hir::WorkspaceVisibility::TopLevel
                    && binding.declared_span == Span { start: 0, end: 0 }
                    && !self.parent_bindings_shape.contains(&shape)
                    && seen_workspace.insert((name.clone(), slot)))
                .then_some(SelectedClosureCapture::AssignedWorkspace { name, slot })
            })
            .collect::<Option<Vec<_>>>()?;
        if seen_original.len() != self.captures.len() {
            return None;
        }
        // A newly assigned top-level workspace value can turn this exact
        // anonymous body's former ExternalWorkspace read into an issued
        // parent capture. That removed local is the only permitted local ABI
        // change, and only after the checked capture plan has named its
        // issuing workspace slot. All other original locals stay identical.
        let assigned_capture_names = plan
            .iter()
            .filter_map(|capture| match capture {
                SelectedClosureCapture::AssignedWorkspace { name, .. } => Some(name.as_str()),
                SelectedClosureCapture::Original(_) => None,
            })
            .collect::<HashSet<_>>();
        let original_remaining = self
            .locals_shape
            .iter()
            .filter(|binding| {
                !(binding.role == runmat_hir::BindingRole::ExternalWorkspace
                    && assigned_capture_names.contains(binding.name.0.as_str()))
            })
            .collect::<Vec<_>>();
        let selected_locals = selected.locals_shape.iter().collect::<Vec<_>>();
        (original_remaining == selected_locals).then_some(plan)
    }
}

#[cfg(test)]
#[path = "abi/tests.rs"]
mod tests;

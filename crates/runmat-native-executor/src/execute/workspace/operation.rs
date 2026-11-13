//! Borrowed operation authority for a native workspace adapter.
//!
//! The service retains only a weak canonical delegate. A RuntimeContext is
//! supplied by each effect/read operation, never captured by the service.

use std::rc::Rc;

use runmat_runtime::context::{RuntimeContext, RuntimeWorkspaceService};
use runmat_runtime::workspace::GlobalBindingTarget;
use runmat_runtime::RuntimeError;

use super::{
    native_workspace_closed, workspace_read_borrow_error, NativeWorkspaceFrame,
    NativeWorkspaceService,
};
use runmat_value::Value;

type WorkspaceOperationResult<T> = Result<T, Box<RuntimeError>>;

pub(super) fn validate_frame_context(
    frame: &NativeWorkspaceFrame,
    context: Option<&RuntimeContext>,
) -> WorkspaceOperationResult<()> {
    match (frame.runtime, context.map(RuntimeContext::local_identity)) {
        (None, None) => Ok(()),
        (Some(_), None) => Err(Box::new(context_required())),
        (Some(expected), Some(actual)) if expected == actual => Ok(()),
        _ => Err(Box::new(runmat_runtime::runtime_error::semantic_error(
            "RunMat:Workspace:MutationContextMismatch",
            "native workspace operation belongs to a different runtime context",
        ))),
    }
}

impl NativeWorkspaceService {
    fn global_target_for_name(&self, name: &str) -> Option<(usize, GlobalBindingTarget)> {
        let slot = self.local_names.slot(name)?;
        self.global_aliases
            .binding_handle_for_slot(slot)
            .map(|target| (slot, target))
    }

    fn current_global_value(
        &self,
        target: &GlobalBindingTarget,
    ) -> WorkspaceOperationResult<Option<Value>> {
        match &self.session_variables {
            Some(authority) => super::super::global_target::current_value(authority, target),
            #[cfg(test)]
            None => Ok(match target {
                GlobalBindingTarget::Named(name) => {
                    runmat_runtime::workspace::session::global_value(name.as_str())
                }
                GlobalBindingTarget::LegacySlot(slot) => {
                    runmat_runtime::workspace::session::global_slot_value(*slot)
                }
            }),
            #[cfg(not(test))]
            None => Err(Box::new(super::missing_session_variable_authority())),
        }
    }

    pub(super) fn ensure_mutation_in_operation(
        &self,
        context: Option<&RuntimeContext>,
    ) -> WorkspaceOperationResult<()> {
        let frame = self.frame()?;
        validate_frame_context(&frame, context)?;
        if !self.publish {
            if let Some(caller) = &self.caller {
                let caller = caller.upgrade().ok_or_else(native_workspace_closed)?;
                match context {
                    Some(context) => caller.ensure_mutation_authority_with_context(context)?,
                    None => caller.ensure_mutation_authority()?,
                }
            }
        }
        if let Some(canonical) = self.canonical_for_operation()? {
            match context {
                Some(context) => canonical.ensure_mutation_authority_with_context(context)?,
                None => canonical.ensure_mutation_authority()?,
            }
        }
        Ok(())
    }

    pub(super) fn canonical_for_operation(
        &self,
    ) -> WorkspaceOperationResult<Option<Rc<dyn RuntimeWorkspaceService>>> {
        let Some(delegate) = &self.canonical else {
            return Ok(None);
        };
        let delegate = delegate.upgrade().ok_or_else(native_workspace_closed)?;
        Ok(Some(delegate))
    }

    pub(super) fn ensure_effect_in_operation(
        &self,
        context: Option<&RuntimeContext>,
    ) -> WorkspaceOperationResult<()> {
        let frame = self.frame()?;
        validate_frame_context(&frame, context)?;
        if !self.publish {
            if let Some(caller) = &self.caller {
                let caller = caller.upgrade().ok_or_else(native_workspace_closed)?;
                match context {
                    Some(context) => caller.ensure_effect_authority_with_context(context)?,
                    None => caller.ensure_effect_authority()?,
                }
            }
        }
        if let Some(canonical) = self.canonical_for_operation()? {
            if let Some(context) = context {
                canonical.ensure_effect_authority_with_context(context)?;
            } else {
                canonical.ensure_effect_authority()?;
            }
        }
        Ok(())
    }

    pub(super) fn try_lookup_in_operation(
        &self,
        context: Option<&RuntimeContext>,
        name: &str,
    ) -> WorkspaceOperationResult<Option<Value>> {
        self.ensure_effect_in_operation(context)?;
        let lexical = {
            let frame = self.frame()?;
            let bindings = frame
                .lexical
                .try_borrow()
                .map_err(workspace_read_borrow_error)?;
            let lexical = bindings.as_ref().and_then(|owner| {
                owner
                    .by_name
                    .get(name)
                    .copied()
                    .map(|binding| (owner.environment.clone(), binding))
            });
            lexical
        };
        if let Some((environment, binding)) = lexical {
            if binding.kind != runmat_native_codegen::NativeLexicalBindingKind::Transit {
                return runmat_runtime::call::lexical::read_binding(&environment, binding.binding)
                    .map_err(Box::new);
            }
        }
        if let Some((_, target)) = self.global_target_for_name(name) {
            return self.current_global_value(&target);
        }
        let canonical = self
            .canonical_for_operation()?
            .ok_or_else(native_workspace_closed)?;
        match context {
            Some(context) => canonical
                .try_lookup_with_context(context, name)
                .map_err(Box::new),
            None => canonical.try_lookup(name).map_err(Box::new),
        }
    }

    pub(super) fn try_snapshot_in_operation(
        &self,
        context: Option<&RuntimeContext>,
    ) -> WorkspaceOperationResult<Vec<(String, Value)>> {
        self.ensure_effect_in_operation(context)?;
        let canonical = self
            .canonical_for_operation()?
            .ok_or_else(native_workspace_closed)?;
        let mut values = match context {
            Some(context) => canonical.try_snapshot_with_context(context)?,
            None => canonical.try_snapshot()?,
        };
        for (slot, name) in self.local_names.iter() {
            if let Some(target) = self.global_aliases.binding_handle_for_slot(slot) {
                if let Some(value) = self.current_global_value(&target)? {
                    if let Some((_, existing)) = values.iter_mut().find(|(local, _)| local == name)
                    {
                        *existing = value;
                    } else {
                        values.push((name.to_owned(), value));
                    }
                }
            }
        }
        let lexical = {
            let frame = self.frame()?;
            let bindings = frame
                .lexical
                .try_borrow()
                .map_err(workspace_read_borrow_error)?;
            bindings.clone()
        };
        if let Some(lexical) = lexical {
            for (name, binding) in lexical.by_name {
                if binding.kind == runmat_native_codegen::NativeLexicalBindingKind::Transit {
                    continue;
                }
                let observed = runmat_runtime::call::lexical::read_binding(
                    &lexical.environment,
                    binding.binding,
                )?;
                if let Some((index, _)) = values
                    .iter()
                    .enumerate()
                    .find(|(_, (local, _))| local == &name)
                {
                    if let Some(value) = observed {
                        values[index].1 = value;
                    } else {
                        values.remove(index);
                    }
                } else if let Some(value) = observed {
                    values.push((name, value));
                }
            }
        }
        Ok(values)
    }

    pub(super) fn try_global_names_in_operation(
        &self,
        context: Option<&RuntimeContext>,
    ) -> WorkspaceOperationResult<Vec<String>> {
        self.ensure_effect_in_operation(context)?;
        match self.session_variables.as_ref() {
            Some(authority) => authority.global_names().map_err(Box::new),
            #[cfg(test)]
            None => Ok(runmat_runtime::workspace::session::global_names()),
            #[cfg(not(test))]
            None => Err(Box::new(super::missing_session_variable_authority())),
        }
    }

    pub(super) fn assign_in_operation(
        &self,
        context: Option<&RuntimeContext>,
        name: &str,
        value: Value,
    ) -> WorkspaceOperationResult<()> {
        self.ensure_effect_in_operation(context)?;
        let global = self.global_target_for_name(name);
        #[cfg(not(test))]
        if global.is_some() && self.session_variables.is_none() {
            return Err(Box::new(super::missing_session_variable_authority()));
        }
        let frame = self.frame()?;
        let lexical = {
            let bindings = frame
                .lexical
                .try_borrow()
                .map_err(workspace_read_borrow_error)?;
            let lexical = bindings.as_ref().and_then(|owner| {
                owner
                    .by_name
                    .get(name)
                    .copied()
                    .map(|binding| (owner.environment.clone(), binding))
            });
            lexical
        };
        let lexical_writer = lexical.filter(|(_, binding)| {
            binding.kind != runmat_native_codegen::NativeLexicalBindingKind::Transit
        });
        if let Some((_, target)) = global {
            match (&self.session_variables, context) {
                (Some(authority), Some(context)) => {
                    let prepared = super::super::global_target::prepare_store(
                        authority, context, &target, value,
                    )?;
                    let lexical_value = lexical_writer.as_ref().map(|_| prepared.value().clone());
                    prepared.checkpoint(context)?;
                    prepared.commit();
                    frame.externally_mutated.set(true);
                    if let (Some((environment, binding)), Some(value)) =
                        (lexical_writer, lexical_value)
                    {
                        super::publish_native_lexical_binding(&environment, binding, value, true)?;
                    }
                }
                (Some(_), None) => return Err(Box::new(context_required())),
                (None, _) => {
                    #[cfg(test)]
                    {
                        let lexical_value = lexical_writer.as_ref().map(|_| value.clone());
                        super::super::global_target::store_legacy(&target, value)?;
                        frame.externally_mutated.set(true);
                        if let (Some((environment, binding)), Some(value)) =
                            (lexical_writer, lexical_value)
                        {
                            super::publish_native_lexical_binding(
                                &environment,
                                binding,
                                value,
                                true,
                            )?;
                        }
                    }
                    #[cfg(not(test))]
                    return Err(Box::new(super::missing_session_variable_authority()));
                }
            }
        } else if let Some((environment, binding)) = lexical_writer {
            // A shared lexical cell is the sole value authority for this name.
            // Reads and immutable capture retain it through the lexical lease.
            super::publish_native_lexical_binding(&environment, binding, value, false)?;
            frame.externally_mutated.set(true);
        } else {
            let canonical = self
                .canonical_for_operation()?
                .ok_or_else(native_workspace_closed)?;
            match context {
                Some(context) => canonical.assign_with_context(context, name, value)?,
                None => canonical.assign(name, value)?,
            }
            frame.externally_mutated.set(true);
        }
        Ok(())
    }

    pub(super) fn clear_in_operation(
        &self,
        context: Option<&RuntimeContext>,
    ) -> WorkspaceOperationResult<()> {
        self.ensure_effect_in_operation(context)?;
        let frame = self.frame()?;
        let lexical = frame
            .lexical
            .try_borrow()
            .map_err(workspace_read_borrow_error)?
            .clone();
        if self.publish || self.private_current {
            if let Some(canonical) = self.canonical_for_operation()? {
                if let Some(context) = context {
                    canonical.clear_with_context(context)?;
                } else {
                    canonical.clear()?;
                }
            }
        }
        frame.externally_mutated.set(true);
        self.global_aliases.unbind_all();
        if let Some(lexical) = lexical {
            super::retire_native_lexical_bindings(
                &lexical.environment,
                lexical.by_name.values().copied(),
            )?;
        }
        Ok(())
    }

    pub(super) fn declare_global_in_operation(
        &self,
        context: Option<&RuntimeContext>,
        slot: usize,
        name: &str,
    ) -> WorkspaceOperationResult<()> {
        let frame = self.frame()?;
        let active = context
            .is_none()
            .then(runmat_runtime::context::legacy::active)
            .flatten();
        let context = context
            .or(active.as_ref())
            .ok_or_else(|| Box::new(context_required()))?;
        validate_frame_context(&frame, Some(context))?;
        let authority = self
            .session_variables
            .as_ref()
            .ok_or_else(|| Box::new(super::missing_session_variable_authority()))?;
        let prepared_session = authority
            .prepare_declare_global_with_context(context, name)
            .map_err(Box::new)?;
        let prepared_alias = self
            .global_aliases
            .prepare_named(context, slot, name)
            .map_err(|cause| {
                Box::new(super::super::global_target::alias_admission_error(
                    context, cause,
                ))
            })?;
        let target = prepared_alias.target();
        prepared_session.checkpoint(context).map_err(Box::new)?;
        prepared_alias.checkpoint().map_err(|cause| {
            Box::new(super::super::global_target::alias_admission_error(
                context, cause,
            ))
        })?;
        if self.publish || self.private_current {
            if let Some(canonical) = self.canonical_for_operation()? {
                canonical.bind_base_global_target_with_context(context, name, &target)?;
            }
        }
        prepared_session.commit();
        prepared_alias.commit();
        frame.externally_mutated.set(true);
        Ok(())
    }

    pub(super) fn declare_legacy_global_in_operation(
        &self,
        context: &RuntimeContext,
        slot: usize,
        name: &str,
    ) -> WorkspaceOperationResult<()> {
        let frame = self.frame()?;
        validate_frame_context(&frame, Some(context))?;
        let authority = self
            .session_variables
            .as_ref()
            .ok_or_else(|| Box::new(super::missing_session_variable_authority()))?;
        let target = GlobalBindingTarget::LegacySlot(slot);
        let prepared_session = authority
            .prepare_declare_global_slot_with_context(context, slot)
            .map_err(Box::new)?;
        let prepared_alias = self
            .global_aliases
            .prepare_target(context, slot, &target)
            .map_err(|cause| {
                Box::new(super::super::global_target::alias_admission_error(
                    context, cause,
                ))
            })?;
        prepared_session.checkpoint(context).map_err(Box::new)?;
        prepared_alias.checkpoint().map_err(|cause| {
            Box::new(super::super::global_target::alias_admission_error(
                context, cause,
            ))
        })?;
        if self.publish || self.private_current {
            if let Some(canonical) = self.canonical_for_operation()? {
                canonical.bind_base_global_target_with_context(context, name, &target)?;
            }
        }
        prepared_session.commit();
        prepared_alias.commit();
        frame.externally_mutated.set(true);
        Ok(())
    }

    pub(super) fn bind_global_target_in_operation(
        &self,
        context: &RuntimeContext,
        name: &str,
        target: &GlobalBindingTarget,
    ) -> WorkspaceOperationResult<()> {
        let frame = self.frame()?;
        validate_frame_context(&frame, Some(context))?;
        let slot = self.local_names.slot(name).ok_or_else(|| {
            Box::new(runmat_runtime::runtime_error::semantic_error(
                "RunMat:Workspace:GlobalBindingUnavailable",
                "native workspace global has no matching frame local",
            ))
        })?;
        let prepared_alias = self
            .global_aliases
            .prepare_target(context, slot, target)
            .map_err(|cause| {
                Box::new(super::super::global_target::alias_admission_error(
                    context, cause,
                ))
            })?;
        prepared_alias.checkpoint().map_err(|cause| {
            Box::new(super::super::global_target::alias_admission_error(
                context, cause,
            ))
        })?;
        if self.publish || self.private_current {
            if let Some(canonical) = self.canonical_for_operation()? {
                canonical.bind_base_global_target_with_context(context, name, target)?;
            }
        }
        prepared_alias.commit();
        frame.externally_mutated.set(true);
        Ok(())
    }

    pub(super) fn remove_in_operation(
        &self,
        context: Option<&RuntimeContext>,
        name: &str,
    ) -> WorkspaceOperationResult<()> {
        self.ensure_effect_in_operation(context)?;
        let frame = self.frame()?;
        let lexical = frame
            .lexical
            .try_borrow()
            .map_err(workspace_read_borrow_error)?
            .as_ref()
            .and_then(|owner| {
                owner
                    .by_name
                    .get(name)
                    .copied()
                    .map(|binding| (owner.environment.clone(), binding))
            });
        if self.publish || self.private_current {
            if let Some(canonical) = self.canonical_for_operation()? {
                if let Some(context) = context {
                    canonical.remove_with_context(context, name)?;
                } else {
                    canonical.remove(name)?;
                }
            }
        }
        frame.externally_mutated.set(true);
        for (slot, local) in self.local_names.iter() {
            if local == name {
                self.global_aliases.unbind_slot(slot);
            }
        }
        if let Some((environment, binding)) = lexical {
            super::retire_native_lexical_bindings(&environment, std::iter::once(binding))?;
        }
        Ok(())
    }
}

fn context_required() -> RuntimeError {
    runmat_runtime::runtime_error::semantic_error(
        "RunMat:Workspace:PublicationContextRequired",
        "native workspace operation requires its live runtime context",
    )
}

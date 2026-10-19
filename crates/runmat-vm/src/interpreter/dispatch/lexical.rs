use crate::bytecode::program::ExecutionContext;
use crate::interpreter::vars::InterpreterVars;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

/// Refresh a typed shared binding before the ordinary slot read. Frames with
/// no selected nested environment remain on the original zero-overhead path.
pub(super) fn refresh(
    context: &ExecutionContext,
    vars: &mut InterpreterVars,
    slot: usize,
) -> Result<bool, RuntimeError> {
    let Some(environment) = context.lexical_environment.as_ref() else {
        return Ok(true);
    };
    let Some(binding) = context
        .lexical_binding_slots
        .iter()
        .find(|binding| binding.slot == slot)
    else {
        return Ok(true);
    };
    if binding.kind == crate::bytecode::program::LexicalBindingKind::Transit {
        return Ok(true);
    }
    if let Some(value) =
        runmat_runtime::call::lexical::read_binding_lease(environment, binding.binding)?
    {
        let prepared = if let Some(frame) = crate::runtime::workspace::workspace_frame_identity() {
            Some(
                crate::runtime::workspace::prepare_workspace_frame_import_lease(
                    frame,
                    &context.runtime,
                    slot,
                    value,
                )?,
            )
        } else {
            let retired = vars.replace_lexical(slot, value).ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "LexicalEnvironment",
                    "shared binding slot is outside its frame",
                )
            })?;
            drop(retired);
            None
        };
        if let Some(prepared) = prepared.as_ref() {
            prepared.checkpoint_before_publication()?;
        }
        vars.get(slot).ok_or_else(|| {
            crate::interpreter::errors::mex(
                "LexicalEnvironment",
                "shared binding slot is outside its frame",
            )
        })?;
        if let Some(prepared) = prepared {
            prepared.commit();
        }
        return Ok(true);
    }
    let prepared = crate::runtime::workspace::workspace_frame_identity()
        .map(|frame| {
            crate::runtime::workspace::prepare_workspace_frame_unassign(
                frame,
                Some(&context.runtime),
                slot,
            )
        })
        .transpose()?;
    if let Some(prepared) = &prepared {
        prepared.checkpoint_before_publication()?;
    }
    if vars.get(slot).is_some() {
        let _retired = vars.replace_owned(slot, Value::Num(0.0));
    }
    if let Some(prepared) = prepared {
        prepared.commit();
    }
    Ok(false)
}

/// Publish only a successful local store through its binding owner. A child
/// never creates a second cell for an inherited ancestor binding.
pub(super) fn publish(
    context: &ExecutionContext,
    vars: &InterpreterVars,
    slot: usize,
) -> Result<(), RuntimeError> {
    let Some(environment) = context.lexical_environment.as_ref() else {
        return Ok(());
    };
    let Some(binding) = context
        .lexical_binding_slots
        .iter()
        .find(|binding| binding.slot == slot)
    else {
        return Ok(());
    };
    if binding.kind == crate::bytecode::program::LexicalBindingKind::Transit {
        return Ok(());
    }
    let value = vars.get(slot).map(|value| value.clone()).ok_or_else(|| {
        crate::interpreter::errors::mex(
            "LexicalEnvironment",
            "shared binding slot is outside its frame",
        )
    })?;
    if binding.kind == crate::bytecode::program::LexicalBindingKind::Owned {
        runmat_runtime::call::lexical::write_owned_binding(environment, binding.binding, value)
    } else {
        runmat_runtime::call::lexical::write_binding(environment, binding.binding, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::program::{LexicalBindingKind, LexicalBindingSlot};
    use runmat_types::BindingId;

    #[test]
    fn anonymous_transit_keeps_its_snapshot_while_descendant_sees_parent_cell() {
        runmat_gc::gc_test_context(|| {
            let (environment, _roots) =
                runmat_runtime::call::lexical::allocate_environment(None).unwrap();
            let binding = BindingId(7);
            assert!(
                runmat_runtime::call::lexical::reserve_owned_binding(&environment, binding)
                    .unwrap()
            );
            runmat_runtime::call::lexical::write_owned_binding(
                &environment,
                binding,
                Value::Num(10.0),
            )
            .unwrap();
            let mut context = ExecutionContext {
                lexical_environment: Some(environment.clone()),
                lexical_binding_slots: vec![LexicalBindingSlot {
                    binding,
                    slot: 0,
                    kind: LexicalBindingKind::Transit,
                }],
                ..ExecutionContext::default()
            };
            let mut snapshot = InterpreterVars::from_owned(vec![Value::Num(3.0)]);
            assert!(refresh(&context, &mut snapshot, 0).unwrap());
            assert_eq!(snapshot.get(0).as_deref(), Some(&Value::Num(3.0)));
            let _ = snapshot.replace_owned(0, Value::Num(4.0));
            publish(&context, &snapshot, 0).unwrap();
            assert_eq!(
                runmat_runtime::call::lexical::read_binding(&environment, binding).unwrap(),
                Some(Value::Num(10.0))
            );
            context.lexical_binding_slots[0].kind = LexicalBindingKind::Inherited;
            assert!(refresh(&context, &mut snapshot, 0).unwrap());
            assert_eq!(snapshot.get(0).as_deref(), Some(&Value::Num(10.0)));
            runmat_runtime::call::lexical::remove_binding(&environment, binding).unwrap();
            assert!(!refresh(&context, &mut snapshot, 0).unwrap());
            assert_eq!(snapshot.get(0).as_deref(), Some(&Value::Num(0.0)));
        });
    }
}

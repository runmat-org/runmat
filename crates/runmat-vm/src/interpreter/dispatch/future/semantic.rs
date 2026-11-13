//! Selected pending semantic closure creation; the parent owns admission.
use super::*;

pub(super) async fn dispatch(
    mut meta: DispatchMeta<'_>,
    stack: &mut crate::interpreter::stack::InterpreterStack,
    context: &crate::bytecode::program::ExecutionContext,
) -> Result<(), RuntimeError> {
    let _call_arg_spans = meta.call_arg_spans.take();
    let Instr::CreateSemanticClosure(function, display_name, capture_count, source_role) =
        meta.instr
    else {
        unreachable!("semantic-closure route requires its creation opcode");
    };
    let inherits_shared_bindings = *source_role == runmat_types::FunctionSourceRole::Anonymous
        && meta.function_registry.get(*function).is_some_and(|target| {
            target.lexical_binding_slots.iter().any(|binding| {
                binding.kind == crate::bytecode::program::LexicalBindingKind::Transit
            })
        });
    let lexical_environment = (*source_role == runmat_types::FunctionSourceRole::Nested
        || inherits_shared_bindings)
        .then(|| context.lexical_environment.clone())
        .flatten();
    let port = context.dynamic_eval_admission.clone();
    let pending = port.as_ref().and_then(|port| {
        port.pending
            .as_ref()
            .filter(|plan| plan.function(*function).is_some())
            .cloned()
    });
    if let Some(plan) = pending {
        let port = port.expect("pending closure requires its admission port");
        let pending_function = plan
            .function(*function)
            .expect("pending closure lookup was checked above");
        if pending_function.abi.original_capture_count() != *capture_count {
            return Err(crate::interpreter::errors::mex(
                "DynamicWorkspaceCompile",
                "pending closure opcode differs from its original capture layout",
            ));
        }
        let runtime = context.runtime.clone();
        let compiled = crate::call::builtins::compile_pending_closure(
            &plan,
            pending_function,
            &runtime,
            meta.function_registry,
            *function,
            &port,
        )
        .await?;
        if crate::runtime::workspace::workspace_frame_identity() != Some(compiled.issuing_frame) {
            return Err(crate::interpreter::errors::mex(
                "DynamicWorkspaceCompile",
                "pending closure changed workspace frame after source selection",
            ));
        }
        let direct_captures = compiled.captures.len() == *capture_count
            && compiled.captures.iter().enumerate().all(|(index, capture)| {
                matches!(capture, crate::call::builtins::SelectedClosureCapture::Original(original) if *original == index)
            });
        let additional_captures = if direct_captures {
            None
        } else {
            if stack.len() < *capture_count {
                return Err(crate::interpreter::errors::mex(
                    "DynamicWorkspaceCompile",
                    "pending closure's original captures are absent from the stack",
                ));
            }
            let crate::runtime::workspace::WorkspaceBindingState {
                frame: live_frame,
                names: live_names,
                assigned,
                ..
            } = crate::runtime::workspace::workspace_binding_state().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "DynamicWorkspaceUnavailable",
                    "pending closure lost its issuing workspace frame",
                )
            })?;
            if live_frame != compiled.issuing_frame {
                return Err(crate::interpreter::errors::mex(
                    "DynamicWorkspaceCompile",
                    "pending closure changed workspace frame after source selection",
                ));
            }
            let mut values = Vec::with_capacity(compiled.captures.len());
            for capture in &compiled.captures {
                let crate::call::builtins::SelectedClosureCapture::AssignedWorkspace { name, slot } =
                    capture
                else {
                    continue;
                };
                if live_names.get(name) != Some(slot) || !assigned.contains(name) {
                    return Err(crate::interpreter::errors::mex(
                        "DynamicWorkspaceCompile",
                        "pending closure workspace capture changed after source selection",
                    ));
                }
                values.push(
                    crate::runtime::workspace::workspace_lookup(name).ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "DynamicWorkspaceCompile",
                            "pending closure workspace capture is no longer assigned",
                        )
                    })?,
                );
            }
            Some(
                runmat_runtime::execution::RootedValues::new(values)
                    .map_err(|error| error.into_runtime_error())?,
            )
        };
        let admitted = port.host().admit_pending_closure(
            crate::interpreter::dynamic_admission::PendingClosureAdmissionRequest {
                runtime: &runtime,
                source: &plan.source,
                caller_source: plan.caller_source,
                product: compiled.product,
                compile_inherited: &plan.inherited,
                executing: meta.function_registry,
                local_function: pending_function.local_id,
                execution_function: *function,
                expected_issuing_owner: &pending_function.execution_owner,
                expected_source: pending_function.execution_source,
                siblings: &pending_function.siblings,
            },
        )?;
        if admitted.binding.function != function.0 {
            return Err(crate::interpreter::errors::mex(
                "DynamicWorkspaceCompile",
                "pending closure selected a different execution identity",
            ));
        }
        if let Some(captures) = additional_captures {
            let _stack_roots = runmat_runtime::execution::RootedValueLease::for_values(
                stack.as_slice()[stack.len() - *capture_count..].iter(),
            )
            .map_err(|error| error.into_runtime_error())?;
            let mut originals = Vec::with_capacity(*capture_count);
            for _ in 0..*capture_count {
                originals.push(crate::interpreter::stack::pop_value(stack)?);
            }
            originals.reverse();
            let mut originals = originals.into_iter().map(Some).collect::<Vec<_>>();
            captures.with_unbounded_values(|captures| -> Result<(), RuntimeError> {
                let mut added = captures.into_iter();
                let mut ordered = Vec::with_capacity(compiled.captures.len());
                for source in &compiled.captures {
                    let next = match source {
                        crate::call::builtins::SelectedClosureCapture::Original(index) => {
                            originals.get_mut(*index).and_then(Option::take)
                        }
                        crate::call::builtins::SelectedClosureCapture::AssignedWorkspace {
                            ..
                        } => added.next(),
                    }
                    .ok_or_else(|| {
                        crate::interpreter::errors::mex(
                            "DynamicWorkspaceCompile",
                            "pending closure capture plan lost a selected value",
                        )
                    })?;
                    ordered.push(next);
                }
                if originals.iter().any(Option::is_some) || added.next().is_some() {
                    return Err(crate::interpreter::errors::mex(
                        "DynamicWorkspaceCompile",
                        "pending closure capture plan left an issuing value unbound",
                    ));
                }
                crate::call::closures::create_selected_semantic_closure_with_captures(
                    stack,
                    admitted.binding,
                    display_name.clone(),
                    admitted.source_file,
                    ordered,
                    *source_role,
                    lexical_environment,
                )?;
                Ok(())
            })??;
        } else {
            crate::call::closures::create_selected_semantic_closure(
                stack,
                admitted.binding,
                display_name.clone(),
                admitted.source_file,
                *capture_count,
                *source_role,
                lexical_environment,
            )?;
        }
        drop(admitted.publication);
    } else {
        handle_create_semantic_closure(
            stack,
            *function,
            display_name.clone(),
            *capture_count,
            *source_role,
            lexical_environment,
        )?;
    }
    Ok(())
}

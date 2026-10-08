use super::*;
use runmat_accelerate::fusion::FusionStoreMaterialization;
use runmat_accelerate::graph::VarBinding;
use runmat_accelerate::VarKind;
use runmat_runtime::workspace::bindings::FrameGlobalBindings;
use std::rc::Rc;

thread_local! {
    static FUSION_EXECUTION_ENTRIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn record_fusion_execution_entry() {
    FUSION_EXECUTION_ENTRIES.with(|entries| entries.set(entries.get() + 1));
}

pub(super) fn fusion_execution_entries() -> usize {
    FUSION_EXECUTION_ENTRIES.with(std::cell::Cell::get)
}

#[test]
fn second_store_failure_keeps_first_publication_and_exact_error() {
    for kind in [VarKind::Global, VarKind::Local] {
        let records_completion = kind == VarKind::Global;
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        let (runtime, activation) =
            crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
        futures::executor::block_on(activation.scope(runtime.scope(async {
            let mut vars = vec![Value::Num(0.0), Value::Num(0.0)];
            let names = HashMap::from([("mu".to_string(), 0), ("sigma".to_string(), 1)]);
            let _workspace = crate::runtime::workspace::set_workspace_state(
                names,
                HashSet::new(),
                &mut vars,
                FrameGlobalBindings::default(),
            );
            let store = |index| FusionStoreMaterialization {
                value_id: index as u32,
                binding: VarBinding {
                    kind: kind.clone(),
                    index,
                },
            };
            let mut context = crate::bytecode::program::ExecutionContext {
                runtime: runtime.clone(),
                ..Default::default()
            };
            let mut completed = Some(HashSet::new());
            let mut missing = HashSet::new();
            let aliases = FrameGlobalBindings::default();
            let mut persistent_aliases = crate::interpreter::PersistentAliases::new();
            if records_completion {
                aliases.declare_named(0, "fusion_first_store".to_string());
            } else {
                runtime
                    .session_variable_authority()
                    .store_persistent_slot("<main>", 0, Value::Num(0.0))
                    .expect("seed selected persistent slot");
                crate::interpreter::persistent::declare(
                    &mut persistent_aliases,
                    &mut vars,
                    &runtime.session_variable_authority(),
                    "<main>",
                    0,
                    crate::interpreter::persistent::PersistentAlias::LegacySlot,
                )
                .expect("declare selected persistent slot");
            }
            let var_names = HashMap::from([(0, "mu".to_string()), (1, "sigma".to_string())]);
            let error = publish_fusion_stores(
                vec![
                    (store(0), Value::Num(7.0)),
                    (store(1), Value::OutputList(vec![Value::Num(8.0)])),
                ],
                FusionPublicationFrame {
                    vars: &mut vars,
                    context: &mut context,
                    var_names: &var_names,
                    completion_assigned_slots: &mut completed,
                    global_aliases: &aliases,
                    persistent_aliases: &persistent_aliases,
                    current_function_name: "<main>",
                    missing_input_slots: &mut missing,
                },
            )
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:TransientSequenceNotStorable")
            );
            assert_eq!(vars, vec![Value::Num(7.0), Value::Num(0.0)]);
            assert_eq!(
                completed.unwrap(),
                if records_completion {
                    HashSet::from([0])
                } else {
                    HashSet::new()
                }
            );
            assert_eq!(
                crate::runtime::workspace::workspace_lookup("mu"),
                Some(Value::Num(7.0))
            );
            assert_eq!(crate::runtime::workspace::workspace_lookup("sigma"), None);
            if records_completion {
                assert_eq!(
                    runmat_runtime::workspace::session::global_value("fusion_first_store"),
                    Some(Value::Num(7.0))
                );
            } else {
                assert_eq!(
                    runtime
                        .session_variable_authority()
                        .persistent_slot_value("<main>", 0)
                        .expect("selected persistent slot"),
                    Some(Value::Num(7.0))
                );
            }
        })));
    }
}

#[test]
fn in_frame_local_store_uses_frame_offset_without_workspace_publication() {
    let mut vars = vec![Value::Num(0.0)];
    let mut context = crate::bytecode::program::ExecutionContext {
        locals: vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)],
        ..Default::default()
    };
    context
        .call_stack
        .push(crate::bytecode::program::CallFrame {
            function_name: "fusion_fixture".to_string(),
            return_address: 0,
            locals_start: 2,
            locals_count: 1,
            expected_outputs: 0,
        });
    let mut completion = Some(HashSet::new());
    let mut missing = HashSet::new();
    publish_fusion_stores(
        vec![(
            FusionStoreMaterialization {
                value_id: 0,
                binding: VarBinding {
                    kind: VarKind::Local,
                    index: 0,
                },
            },
            Value::Num(7.0),
        )],
        FusionPublicationFrame {
            vars: &mut vars,
            context: &mut context,
            var_names: &HashMap::new(),
            completion_assigned_slots: &mut completion,
            global_aliases: &FrameGlobalBindings::default(),
            persistent_aliases: &crate::interpreter::PersistentAliases::new(),
            current_function_name: "fusion_fixture",
            missing_input_slots: &mut missing,
        },
    )
    .expect("admitted frame-local publication");
    assert_eq!(
        context.locals,
        vec![Value::Num(1.0), Value::Num(2.0), Value::Num(7.0)]
    );
    assert_eq!(vars, vec![Value::Num(0.0)]);
    assert_eq!(completion, Some(HashSet::new()));
}

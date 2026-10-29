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
fn supplied_final_only_span_with_stores_reaches_ordinary_vm_before_fusion_execution() {
    use runmat_accelerate::fusion::FusionPlan;
    use runmat_runtime::context::RuntimeInvocationArity;

    assert!(
        matches!(
            accel_fusion::FusionAttemptError::from(RuntimeError::from("admission denied")),
            accel_fusion::FusionAttemptError::Terminal(_)
        ),
        "continuation admission failure cannot authorize bytecode replay"
    );

    let source = r#"
        imgs = ones(2,4,5,'single');
        mu = single(mean(imgs,[2 3],'native'));
        sigma = single(sqrt(mean((imgs-mu).^2,[2 3],'native')+single(1e-6)));
        out = single(((imgs-mu)./sigma).^single(1.8));
    "#;
    let ast = runmat_parser::parse(source).expect("parse image source");
    let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty()).expect("lower HIR");
    let mir = runmat_mir::lowering::lower_assembly(&hir.assembly).expect("lower MIR");
    let bytecode = crate::bytecode::compile(&hir.assembly, &mir, hir.assembly.entrypoints[0].id)
        .expect("compile image source");
    let ordinary_vars = Vec::new();
    let mut ordinary = crate::interpreter::state::InterpreterState::from_owned_in_context(
        bytecode.clone(),
        ordinary_vars,
        Some("<main>"),
        RuntimeInvocationArity::Script,
        runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        )),
    );
    ordinary.fusion_plan = None;
    ordinary.fusion_accel_graph = None;
    let ordinary_runtime = ordinary.context.runtime.clone();
    let (ordinary_runtime, ordinary_workspace) =
        crate::runtime::workspace_service::VmWorkspaceService::install(ordinary_runtime, None);
    let crate::interpreter::api::InterpreterOutcome::Completed(expected) =
        futures::executor::block_on(ordinary_workspace.scope(
            ordinary_runtime.scope(super::super::run_interpreter_owned(Box::new(ordinary))),
        ))
        .expect("independent ordinary selected image source");
    let expected = &expected.values;
    let publication_slots: Vec<_> = ["mu", "sigma", "out"]
        .into_iter()
        .map(|name| {
            let slot = bytecode
                .var_names
                .iter()
                .find_map(|(slot, label)| (label == name).then_some(*slot))
                .expect("selected publication slot");
            (name, slot)
        })
        .collect();
    let graph = crate::accel::graph::build_accel_graph(&bytecode.instructions, &bytecode.var_types);
    let group = graph
        .detect_fusion_groups()
        .into_iter()
        .find(|group| group.kind == runmat_accelerate::FusionKind::ImageNormalize)
        .expect("real image graph group");
    assert!(
        crate::accel::fusion::final_only_span_requires_ordinary_stores(
            &group.kind,
            &bytecode.instructions,
            &group.span,
        )
    );
    let initial_vars = Vec::new();
    let mut state = crate::interpreter::state::InterpreterState::from_owned_in_context(
        bytecode,
        initial_vars,
        Some("<main>"),
        RuntimeInvocationArity::Script,
        runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        )),
    );
    state.completion_assigned_slots = Some(HashSet::new());
    state.fusion_plan = Some(Rc::new(FusionPlan::from_graph(&graph, &[group])));
    state.fusion_accel_graph = Some(graph);
    let selected_runtime = state.context.runtime.clone();
    let (selected_runtime, selected_workspace) =
        crate::runtime::workspace_service::VmWorkspaceService::install(selected_runtime, None);
    FUSION_EXECUTION_ENTRIES.with(|entries| entries.set(0));
    let outcome = futures::executor::block_on(
        selected_workspace
            .scope(selected_runtime.scope(super::super::run_interpreter_owned(Box::new(state)))),
    )
    .expect("observable stores must take the ordinary interpreter path");
    assert_eq!(
        FUSION_EXECUTION_ENTRIES.with(std::cell::Cell::get),
        0,
        "the selected final-only plan must not reach fusion execution"
    );
    let crate::interpreter::api::InterpreterOutcome::Completed(completion) = outcome;
    for (name, slot) in publication_slots {
        assert_eq!(
            *completion.values.get(slot).unwrap(),
            *expected.get(slot).unwrap(),
            "{name} publication"
        );
        assert!(completion.assigned_slots.contains(&slot), "{name} assigned");
    }
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
            let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![
                Value::Num(0.0),
                Value::Num(0.0),
            ]);
            let names = HashMap::from([("mu".to_string(), 0), ("sigma".to_string(), 1)]);
            let aliases = FrameGlobalBindings::default();
            if records_completion {
                aliases.declare_named(0, "fusion_first_store".to_string());
                aliases.declare_named(1, "fusion_second_store".to_string());
            }
            let _workspace = crate::runtime::workspace::set_workspace_state(
                names,
                HashSet::new(),
                &mut vars,
                aliases,
            );
            let store = |index| FusionStoreMaterialization {
                value_id: index as u32,
                binding: VarBinding { kind, index },
            };
            let mut context = crate::bytecode::program::ExecutionContext {
                runtime: runtime.clone(),
                ..Default::default()
            };
            let mut completed = Some(HashSet::new());
            let mut missing = HashSet::new();
            let mut persistent_aliases = crate::interpreter::PersistentAliases::new();
            if !records_completion {
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
                    (store(1), Value::output_list(vec![Value::Num(8.0)])),
                ],
                FusionPublicationFrame {
                    vars: &mut vars,
                    context: &mut context,
                    var_names: &var_names,
                    completion_assigned_slots: &mut completed,
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
fn in_frame_local_store_uses_frame_offset_and_only_publishes_declared_globals() {
    let mut vars = crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(0.0)]);
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

    let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let (runtime, activation) =
        crate::runtime::workspace_service::VmWorkspaceService::install(runtime, None);
    futures::executor::block_on(activation.scope(runtime.scope(async {
        let mut vars =
            crate::interpreter::vars::InterpreterVars::from_owned(vec![Value::Num(0.0); 3]);
        let aliases = FrameGlobalBindings::default();
        aliases.declare_named(2, "local_global_alias".to_string());
        let _workspace = crate::runtime::workspace::set_workspace_state(
            HashMap::from([("compiler_local".to_string(), 2)]),
            HashSet::new(),
            &mut vars,
            aliases,
        );
        let mut context = crate::bytecode::program::ExecutionContext {
            runtime: runtime.clone(),
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
                Value::Num(11.0),
            )],
            FusionPublicationFrame {
                vars: &mut vars,
                context: &mut context,
                var_names: &HashMap::new(),
                completion_assigned_slots: &mut completion,
                persistent_aliases: &crate::interpreter::PersistentAliases::new(),
                current_function_name: "fusion_fixture",
                missing_input_slots: &mut missing,
            },
        )
        .expect("declared global local publication");
        assert_eq!(context.locals[2], Value::Num(11.0));
        assert_eq!(completion, Some(HashSet::new()));
        assert_eq!(
            runmat_runtime::workspace::session::global_value("local_global_alias"),
            Some(Value::Num(11.0))
        );
    })));
}

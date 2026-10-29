use super::*;
use runmat_accelerate::FusionKind;
use runmat_runtime::{context::RuntimeInvocationArity, execution::RuntimeExecutionService};
use std::rc::Rc;

#[test]
fn compiled_elementwise_persistent_read_and_store_use_selected_session() {
    runmat_gc::gc_test_context(|| {
        let source = "function y = fusedPersistent()\npersistent p\nt = p .* 2;\np = p + t + 1;\ny = p;\nend\n";
        let ast = runmat_parser::parse(source).expect("parse persistent function");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("lower persistent function");
        let mir = runmat_mir::lowering::lower_assembly(&hir.assembly)
            .expect("lower persistent function MIR");
        assert!(hir.assembly.entrypoints.is_empty());
        assert_eq!(hir.assembly.functions.len(), 1);
        let function_id = hir.assembly.functions[0].id;
        let registry = crate::bytecode::FunctionRegistry::new(
            crate::bytecode::compile_semantic_function_registry(&hir.assembly, &mir)
                .expect("compile persistent function registry"),
        );
        let layout = crate::layout::derive_layout(&hir.assembly, &mir)
            .expect("derive persistent function layout");
        let bytecode = crate::Bytecode::for_function(
            registry
                .get(function_id)
                .expect("selected persistent function"),
            registry.clone(),
            layout,
        );
        let slot = bytecode
            .var_names
            .iter()
            .find_map(|(slot, name)| (name == "p").then_some(*slot))
            .expect("compiled persistent slot");
        assert!(bytecode
            .instructions
            .iter()
            .any(|instruction| matches!(instruction, Instr::DeclarePersistentNamed(_, _))));
        let (plan, graph) = bytecode.prepare_vm_fusion_plan();
        let plan = plan.expect("compiled function VM fusion plan");
        assert!(graph.is_some(), "runtime graph is retained for execution");
        assert!(
            plan.groups.iter().any(|group| {
                group.group.kind == FusionKind::ElementwiseChain
                    && group.kernel.supported
                    && group.materialized_stores.is_empty()
                    && bytecode.instructions[group.group.span.start..=group.group.span.end]
                        .iter()
                        .all(|instruction| !matches!(instruction, Instr::StoreVar(_) | Instr::StoreLocal(_)))
                    && bytecode.instructions[group.group.span.end + 1..]
                        .iter()
                        .any(|instruction| matches!(instruction, Instr::StoreVar(index) if *index == slot))
            }),
            "operator-only fusion must leave the later persistent store to ordinary execution"
        );
        let runtime =
            runmat_runtime::context::RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        runtime
            .session_variable_authority()
            .store_persistent_named_and_slot(
                "fusedPersistent",
                "p",
                slot,
                Value::Tensor(runmat_value::Tensor::new(vec![1.0; 4], vec![2, 2]).unwrap()),
            )
            .expect("seed selected persistent tensor");

        for expected in [4.0, 13.0] {
            let vars = Vec::new();
            let (invocation_runtime, activation) =
                crate::runtime::workspace_service::VmWorkspaceService::install(
                    runtime.clone(),
                    None,
                );
            let frame = activation.frame();
            let state = crate::interpreter::state::InterpreterState::from_owned_in_context(
                bytecode.clone(),
                vars,
                Some("fusedPersistent"),
                RuntimeInvocationArity::function(0, 1),
                invocation_runtime.clone(),
            );
            assert!(state.fusion_plan.as_ref().is_some_and(|runtime_plan| {
                runtime_plan.groups.iter().any(|group| {
                    group.group.kind == FusionKind::ElementwiseChain
                        && group.kernel.supported
                        && group.materialized_stores.is_empty()
                })
            }));
            let before = super::fusion_store_tests::fusion_execution_entries();
            let outcome = futures::executor::block_on(activation.scope(
                invocation_runtime.scope(super::super::run_interpreter_owned(Box::new(state))),
            ))
            .expect("compiled persistent fusion candidate");
            let _ = crate::runtime::workspace::take_workspace_completion(frame);
            assert!(matches!(
                outcome,
                crate::interpreter::api::InterpreterOutcome::Completed(_)
            ));
            assert!(
                super::fusion_store_tests::fusion_execution_entries() > before,
                "the compiled elementwise group must reach the fusion entry"
            );
            let value = runtime
                .session_variable_authority()
                .persistent_named_value("fusedPersistent", "p")
                .expect("selected persistent read")
                .expect("persistent publication");
            let Value::Tensor(tensor) = value else {
                panic!("persistent value must remain a tensor");
            };
            assert_eq!(tensor.materialize_f64(), vec![expected; 4]);
        }
    });
}

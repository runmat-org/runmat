use super::*;

#[test]
fn compiled_persistent_elementwise_executes_ordered_provider_store_numeric_resident() {
    gc_test_context(|| {
        let _guard = accel_test_lock();
        ensure_provider_registered();
        let provider = registered_provider().expect("selected fixture provider");
        let source = "function y = fusedPersistent()\npersistent p\nt = p .* 2;\np = p + t + 1;\ny = p;\nend\n";
        let ast = runmat_parser::parse(source).expect("parse persistent function");
        let hir = runmat_hir::lower(&ast, &runmat_hir::LoweringContext::empty())
            .expect("lower persistent function");
        let mir = runmat_mir::lowering::lower_assembly(&hir.assembly)
            .expect("lower persistent function MIR");
        assert!(hir.assembly.entrypoints.is_empty());
        assert_eq!(hir.assembly.functions.len(), 1);
        let function_id = hir.assembly.functions[0].id;
        let registry = runmat_vm::FunctionRegistry::new(
            runmat_vm::compile_semantic_function_registry(&hir.assembly, &mir)
                .expect("compile persistent function registry"),
        );
        let layout = runmat_vm::derive_layout(&hir.assembly, &mir)
            .expect("derive persistent function layout");
        let bytecode = runmat_vm::Bytecode::for_function(
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
        let graph = graph_for_fusion_test(&bytecode).expect("real function fusion graph");
        let groups = bytecode.runtime_fusion_groups_for_graph(&graph);
        let plan = runmat_accelerate::prepare_fusion_plan(
            Some(&graph),
            &groups,
            bytecode.fusion_metadata.mir_fusion_candidate_group_count,
        )
        .expect("compiled function semantic fusion plan");
        assert!(
            plan.groups.iter().any(|group| {
                group.group.kind == FusionKind::ElementwiseChain
                    && group.kernel.supported
                    && bytecode.instructions[group.group.span.start..=group.group.span.end]
                        .iter()
                        .all(|instruction| !matches!(instruction, Instr::StoreVar(_) | Instr::StoreLocal(_)))
            }),
            "compiled elementwise compute must leave persistent publication to the following ordinary store"
        );
        let runtime = runmat_runtime::context::RuntimeContext::new(Rc::new(
            runmat_runtime::execution::RuntimeExecutionService::new(),
        ));
        runtime
            .session_variable_authority()
            .store_persistent_named_and_slot(
                "fusedPersistent",
                "p",
                slot,
                Value::Tensor(Tensor::new(vec![1.0; 4096], vec![1, 4096]).unwrap()),
            )
            .expect("seed selected persistent tensor");

        for expected in [4.0, 13.0] {
            let dispatches_before = provider.fused_dispatches.load(Ordering::Relaxed);
            let vars = Vec::new();
            block_on(runmat_vm::interpret_workspace(
                &bytecode,
                runmat_vm::InterpreterVars::from_owned(vars),
                Some("fusedPersistent"),
                runtime.clone(),
                None,
                None,
            ))
            .outcome
            .expect("compiled fused persistent invocation");
            assert!(
                provider.fused_dispatches.load(Ordering::Relaxed) > dispatches_before,
                "selected provider must execute the actual fused graph"
            );
            let value = runtime
                .session_variable_authority()
                .persistent_named_value("fusedPersistent", "p")
                .expect("selected persistent read")
                .expect("ordered fused store published persistent value");
            let Value::GpuTensor(handle) = value else {
                panic!("fused persistent publication must remain numeric-resident");
            };
            let (data, shape) = provider
                .pull(&handle)
                .expect("live numeric-resident persistent result");
            assert_eq!(shape, vec![1, 4096]);
            assert_eq!(data, vec![expected; 4096]);
        }
    });
}

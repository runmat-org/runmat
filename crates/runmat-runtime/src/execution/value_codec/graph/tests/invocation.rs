use super::*;
use crate::call::descriptor::{
    CallableCallKind, CallableDescriptor, CallableMetadata, CallableTarget, FunctionNameResolver,
    RetainedClosureValue,
};
use crate::execution::DeferredInvocation;
use runmat_execution::value::GraphInvocationRoots;
use runmat_value::{FunctionDefinitionOwner, LocalExport};
use std::sync::Arc;

struct NoNamedFunction;

impl FunctionNameResolver for NoNamedFunction {
    fn resolve_function(&self, _: &str) -> Option<runmat_types::FunctionId> {
        None
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn typed_bound_target_uses_selected_proof_without_name_redispatch() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let issuer = GraphSourceIssuer::capture_session(&runtime);
        let invocation = DeferredInvocation::callable(CallableDescriptor::bound(
            // The selected session ID is not the portable function ordinal.
            BoundFunctionBinding::portable(23, origin(3)),
            vec![Value::Num(7.0)],
            1,
            CallableCallKind::Feval,
        ));
        let (prepared, roots) =
            prepare_deferred_invocation_graph(&invocation, &issuer, ValueLimits::default(), None)
                .expect("typed target is captured before admission");
        assert!(prepared.direct_bound_origins().contains(&origin(3)));
        let units = prepared
            .bound_authorities()
            .iter()
            .map(|(reference, _, _)| (*reference, GraphProgramUnitId(0)))
            .collect();
        let graph = prepared
            .finish(&issuer, &source_map(), &layouts(), &proofs(), &units)
            .expect("selected body supplies canonical callable facts");
        let GraphInvocationRoots::Callable { target, arguments } = roots else {
            panic!("typed callable root")
        };
        assert_eq!(graph.roots, vec![target, arguments[0]]);
        let GraphNode::Callable(target) = &graph.nodes[target.0 as usize] else {
            panic!("callable target node")
        };
        assert_eq!(
            target.owner,
            runmat_execution::value::GraphCallableOwner::Bound(owner(3))
        );
        assert_eq!(target.qualified_name, "anonymous");
        assert_eq!(target.source_role, FunctionSourceRole::Anonymous);
        assert_eq!(target.source_file, None);
        let selected = HashMap::from([(owner(3), BoundFunctionBinding::portable(23, origin(3)))]);
        let reconstructed = decode_payload_graph_in_context(
            &graph,
            &runtime,
            &source_map(),
            &layouts(),
            &proofs(),
            &selected,
            ValueLimits::default(),
        )
        .expect("selected local ID reconstructs under the admitted portable origin");
        assert!(matches!(
            reconstructed.values().first(),
            Some(Value::BoundFunctionHandle(handle)) if handle.function == 23
        ));
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn deferred_nested_closure_keeps_one_capture_and_its_selected_source() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let issuer = GraphSourceIssuer::capture_session(&runtime);
        let (environment, _roots) = crate::call::lexical::allocate_environment(None).unwrap();
        assert!(crate::call::lexical::reserve_owned_binding(&environment, BindingId(7)).unwrap());
        let descriptor = CallableDescriptor::from_feval_value(
            Value::Closure(Closure::from_data(runmat_value::ClosureData {
                function_name: "nested".into(),
                binding: Some(BoundFunctionBinding::portable(2, origin(2))),
                source_role: FunctionSourceRole::Nested,
                source_file: None,
                captures: vec![Value::Num(11.0)].into(),
                lexical_environment: Some(environment),
            })),
            vec![Value::Num(17.0)],
            1,
            &NoNamedFunction,
        );
        assert_eq!(descriptor.args, [Value::Num(17.0)]);
        let invocation = DeferredInvocation::callable(descriptor);
        let (prepared, roots) =
            prepare_deferred_invocation_graph(&invocation, &issuer, ValueLimits::default(), None)
                .unwrap();
        let units = prepared
            .bound_authorities()
            .iter()
            .map(|(reference, _, _)| (*reference, GraphProgramUnitId(0)))
            .collect();
        let graph = prepared
            .finish(&issuer, &source_map(), &layouts(), &proofs(), &units)
            .unwrap();
        let GraphInvocationRoots::Callable { target, arguments } = roots else {
            panic!("callable graph roots")
        };
        assert_eq!(
            arguments.len(),
            1,
            "capture is not a second supplied argument"
        );
        assert_eq!(graph.roots, [target, arguments[0]]);
        let GraphNode::Callable(callable) = &graph.nodes[target.0 as usize] else {
            panic!("nested closure node")
        };
        assert_eq!(
            callable.owner,
            runmat_execution::value::GraphCallableOwner::Bound(owner(2))
        );
        assert_eq!(callable.captures.len(), 1);
        assert_eq!(callable.source_role, FunctionSourceRole::Nested);
        assert!(callable.activation.is_some());

        let rebuilt = decode_payload_graph_in_context(
            &graph,
            &runtime,
            &source_map(),
            &layouts(),
            &proofs(),
            &bindings(),
            ValueLimits::default(),
        )
        .unwrap();
        let [Value::Closure(closure), Value::Num(17.0)] = rebuilt.values() else {
            panic!("one closure target and one supplied argument")
        };
        assert_eq!(closure.captures, [Value::Num(11.0)]);
        assert_eq!(
            closure.binding,
            Some(BoundFunctionBinding::portable(2, origin(2)))
        );
        assert_eq!(closure.source_role, FunctionSourceRole::Nested);
        assert!(closure.lexical_environment.is_some());

        // V10/V11 saved graphs can carry a lexical Closure target *and* the
        // historical flattened capture prefix in their argument roots. The
        // old root roles remain decodable; only V12 changes their meaning.
        let legacy = DeferredInvocation::callable(CallableDescriptor {
            target: CallableTarget::BoundLexical {
                binding: BoundFunctionBinding::portable(2, origin(2)),
                closure: RetainedClosureValue::new(closure.clone()),
            },
            args: vec![Value::Num(11.0), Value::Num(17.0)],
            requested_outputs: 1,
            metadata: CallableMetadata::default(),
        });
        let (legacy_prepared, legacy_roots) =
            prepare_deferred_invocation_graph(&legacy, &issuer, ValueLimits::default(), None)
                .unwrap();
        let legacy_units = legacy_prepared
            .bound_authorities()
            .iter()
            .map(|(reference, _, _)| (*reference, GraphProgramUnitId(0)))
            .collect();
        let legacy_graph = legacy_prepared
            .finish(&issuer, &source_map(), &layouts(), &proofs(), &legacy_units)
            .unwrap();
        let GraphInvocationRoots::Callable { arguments, .. } = legacy_roots else {
            panic!("legacy callable roots")
        };
        assert_eq!(arguments.len(), 2);
        let legacy_values = decode_payload_graph_in_context(
            &legacy_graph,
            &runtime,
            &source_map(),
            &layouts(),
            &proofs(),
            &bindings(),
            ValueLimits::default(),
        )
        .unwrap();
        assert!(matches!(
            legacy_values.values(),
            [Value::Closure(_), Value::Num(11.0), Value::Num(17.0)]
        ));

        // A selected graph worker can only invoke an exact bound owner.
        // Preserving captures alone must not silently discard the unresolved
        // target identity and leave a request that fails after scheduling.
        let unresolved = DeferredInvocation::callable(CallableDescriptor::from_feval_value(
            Value::Closure(Closure::from_data(runmat_value::ClosureData {
                function_name: "nested".into(),
                binding: None,
                source_role: FunctionSourceRole::Nested,
                source_file: None,
                captures: vec![Value::Num(11.0)].into(),
                lexical_environment: closure.lexical_environment.clone(),
            })),
            vec![Value::Num(17.0)],
            1,
            &NoNamedFunction,
        ));
        assert!(prepare_deferred_invocation_graph(
            &unresolved,
            &issuer,
            ValueLimits::default(),
            None
        )
        .is_err());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn typed_bound_target_rejects_mismatched_local_definition_lease() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let issuer = GraphSourceIssuer::capture_session(&runtime);
        let definition = Arc::new(FunctionDefinitionOwner::default());
        let invocation = DeferredInvocation::callable(CallableDescriptor::bound(
            BoundFunctionBinding::local(
                3,
                runtime.session_lease(),
                FunctionDefinitionOwner::issue(&definition, 2),
                Vec::new(),
                LocalExport::Exact(origin(3)),
            ),
            Vec::new(),
            1,
            CallableCallKind::Feval,
        ));
        assert!(prepare_deferred_invocation_graph(
            &invocation,
            &issuer,
            ValueLimits::default(),
            None
        )
        .is_err());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn prepared_bound_target_has_no_wire_node_before_exact_unit_selection() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let issuer = GraphSourceIssuer::capture_session(&runtime);
        let invocation = DeferredInvocation::callable(CallableDescriptor::bound(
            BoundFunctionBinding::portable(3, origin(3)),
            vec![Value::Num(7.0)],
            1,
            CallableCallKind::Feval,
        ));
        let (prepared, _) =
            prepare_deferred_invocation_graph(&invocation, &issuer, ValueLimits::default(), None)
                .unwrap();
        assert_eq!(prepared.bound_authorities().len(), 1);
        assert!(prepared
            .finish(
                &issuer,
                &source_map(),
                &layouts(),
                &proofs(),
                &HashMap::new(),
            )
            .is_err());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn builtin_arguments_keep_a_nested_activation_as_graph_roots() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let issuer = GraphSourceIssuer::capture_session(&runtime);
        let (environment, _roots) = crate::call::lexical::allocate_environment(None).unwrap();
        assert!(crate::call::lexical::reserve_owned_binding(&environment, BindingId(7)).unwrap());
        let closure = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            function_name: "nested".into(),
            binding: Some(BoundFunctionBinding::portable(2, origin(2))),
            source_role: FunctionSourceRole::Nested,
            source_file: None,
            captures: Vec::new().into(),
            lexical_environment: Some(environment),
        }));
        let invocation = DeferredInvocation::Program {
            callable: runmat_execution::ProgramCallable::builtin("feval").unwrap(),
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: vec![closure],
            requested_outputs: 1,
        };
        assert!(deferred_invocation_requires_graph(&invocation));
        let (prepared, roots) =
            prepare_deferred_invocation_graph(&invocation, &issuer, ValueLimits::default(), None)
                .unwrap();
        let units = prepared
            .bound_authorities()
            .iter()
            .map(|(reference, _, _)| (*reference, GraphProgramUnitId(0)))
            .collect();
        let graph = prepared
            .finish(&issuer, &source_map(), &layouts(), &proofs(), &units)
            .unwrap();
        let GraphInvocationRoots::Builtin { name, arguments } = roots else {
            panic!("builtin graph roots")
        };
        assert_eq!(name.0, "feval");
        assert_eq!(arguments, graph.roots);
        let GraphNode::Callable(callable) = &graph.nodes[graph.roots[0].0 as usize] else {
            panic!("closure argument")
        };
        assert!(callable.activation.is_some());
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn builtin_route_inspects_nested_values_without_encoding_numeric_inputs() {
    runmat_gc::gc_test_context(|| {
        let (environment, _roots) = crate::call::lexical::allocate_environment(None).unwrap();
        let nested = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            function_name: "nested".into(),
            binding: Some(BoundFunctionBinding::portable(2, origin(2))),
            source_role: FunctionSourceRole::Nested,
            source_file: None,
            captures: Vec::new().into(),
            lexical_environment: Some(environment),
        }));
        let invocation = DeferredInvocation::Program {
            callable: runmat_execution::ProgramCallable::builtin("feval").unwrap(),
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: vec![Value::Cell(
                runmat_value::CellArray::new(vec![nested], 1, 1).unwrap(),
            )],
            requested_outputs: 1,
        };
        assert!(deferred_invocation_requires_graph(&invocation));
        let ordinary = DeferredInvocation::Program {
            callable: runmat_execution::ProgramCallable::builtin("sum").unwrap(),
            context: runmat_execution::ProgramInvocationContext::Direct,
            arguments: vec![
                Value::Num(1.0),
                Value::Closure(Closure::from_data(runmat_value::ClosureData {
                    function_name: "plain".into(),
                    binding: Some(BoundFunctionBinding::portable(3, origin(3))),
                    source_role: FunctionSourceRole::Anonymous,
                    source_file: None,
                    captures: vec![Value::Num(2.0)].into(),
                    lexical_environment: None,
                })),
            ],
            requested_outputs: 1,
        };
        assert!(!deferred_invocation_requires_graph(&ordinary));
    });
}

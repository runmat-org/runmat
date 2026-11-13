use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn capture_free_nested_closure_round_trips_without_an_activation() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let closure = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            function_name: "nested".into(),
            binding: Some(BoundFunctionBinding::portable(2, origin(2))),
            source_role: FunctionSourceRole::Nested,
            source_file: None,
            captures: Vec::new().into(),
            lexical_environment: None,
        }));
        let mut admitted_layouts = layouts();
        admitted_layouts.insert(
            owner(2),
            GraphLexicalLayout {
                activation_parent: None,
                bindings: Vec::new(),
            },
        );
        let mut admitted_proofs = proofs();
        admitted_proofs.get_mut(&owner(2)).unwrap().captures = GraphCaptureProof::NestedShared {
            vm_copied: 0,
            zero_copy: GraphNestedZeroCopy::RequiresCopies,
        };
        let graph = encode_payload_graph_in_context(
            std::slice::from_ref(&closure),
            &runtime,
            &source_map(),
            &admitted_layouts,
            &admitted_proofs,
            ValueLimits::default(),
        )
        .expect("capture-free nested closure is an admitted Value graph");
        assert!(graph
            .nodes
            .iter()
            .all(|node| !matches!(node, GraphNode::Activation(_))));
        let rebuilt = decode_payload_graph_in_context(
            &graph,
            &runtime,
            &source_map(),
            &admitted_layouts,
            &admitted_proofs,
            &bindings(),
            ValueLimits::default(),
        )
        .expect("capture-free nested closure reconstructs without an activation");
        let Value::Closure(rebuilt) = &rebuilt.values()[0] else {
            panic!("nested closure")
        };
        assert!(rebuilt.lexical_environment.is_none());
        assert_eq!(rebuilt.function_name, "nested");
        assert_eq!(rebuilt.captures.len(), 0);

        admitted_proofs.get_mut(&owner(2)).unwrap().captures = GraphCaptureProof::NestedShared {
            vm_copied: 1,
            zero_copy: GraphNestedZeroCopy::RequiresCopies,
        };
        assert!(
            decode_payload_graph_in_context(
                &graph,
                &runtime,
                &source_map(),
                &admitted_layouts,
                &admitted_proofs,
                &bindings(),
                ValueLimits::default(),
            )
            .is_err(),
            "copied capture cannot be silently omitted without an admitted owner"
        );
        admitted_proofs.get_mut(&owner(2)).unwrap().captures = GraphCaptureProof::NestedShared {
            vm_copied: 1,
            zero_copy: GraphNestedZeroCopy::CanonicalStorage,
        };
        decode_payload_graph_in_context(
            &graph,
            &runtime,
            &source_map(),
            &admitted_layouts,
            &admitted_proofs,
            &bindings(),
            ValueLimits::default(),
        )
        .expect("admitted canonical storage can back a native zero-copy nested closure");
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn admitted_nested_copies_and_anonymous_by_value_captures_round_trip() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let (environment, _roots) = crate::call::lexical::allocate_environment(None).unwrap();
        assert!(crate::call::lexical::reserve_owned_binding(&environment, BindingId(7)).unwrap());
        let nested = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            function_name: "nested".into(),
            binding: Some(BoundFunctionBinding::portable(2, origin(2))),
            source_role: FunctionSourceRole::Nested,
            source_file: None,
            captures: vec![Value::Num(11.0)].into(),
            lexical_environment: Some(environment),
        }));
        let anonymous = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            function_name: "anonymous".into(),
            binding: Some(BoundFunctionBinding::portable(3, origin(3))),
            source_role: FunctionSourceRole::Anonymous,
            source_file: None,
            captures: vec![Value::Num(17.0)].into(),
            lexical_environment: None,
        }));
        let graph = encode_payload_graph_in_context(
            &[nested, anonymous],
            &runtime,
            &source_map(),
            &layouts(),
            &proofs(),
            ValueLimits::default(),
        )
        .unwrap();
        let decoded = decode_payload_graph_in_context(
            &graph,
            &runtime,
            &source_map(),
            &layouts(),
            &proofs(),
            &bindings(),
            ValueLimits::default(),
        )
        .unwrap();
        let Value::Closure(nested) = &decoded.values()[0] else {
            panic!("nested")
        };
        let Value::Closure(anonymous) = &decoded.values()[1] else {
            panic!("anonymous")
        };
        assert_eq!(nested.captures, [Value::Num(11.0)]);
        assert!(nested.lexical_environment.is_some());
        assert_eq!(anonymous.captures, [Value::Num(17.0)]);
        assert!(anonymous.lexical_environment.is_none());

        let mut missing_capture = graph.clone();
        let anonymous_root = missing_capture.roots[1];
        let GraphNode::Callable(callable) = &mut missing_capture.nodes[anonymous_root.0 as usize]
        else {
            panic!("anonymous graph node")
        };
        callable.captures.clear();
        assert!(validate_bound_callable_proofs(&missing_capture, &proofs()).is_err());
    });
}

use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn closure_activation_cycle_and_argument_alias_round_trip_under_one_rooted_batch() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let (environment, original_roots) =
            crate::call::lexical::allocate_environment(None).unwrap();
        assert!(crate::call::lexical::reserve_owned_binding(&environment, BindingId(7)).unwrap());
        let closure = Value::Closure(Closure::from_data(runmat_value::ClosureData {
            function_name: "nested".into(),
            binding: Some(BoundFunctionBinding::portable(2, origin(2))),
            source_role: FunctionSourceRole::Nested,
            source_file: None,
            captures: Vec::new().into(),
            lexical_environment: Some(environment.clone()),
        }));
        crate::call::lexical::write_owned_binding(&environment, BindingId(7), closure.clone())
            .unwrap();
        let argument = Value::Cell(CellArray::new(vec![closure.clone()], 1, 1).unwrap());
        let tight_cycle = encode_payload_graph_in_context(
            std::slice::from_ref(&closure),
            &runtime,
            &source_map(),
            &layouts(),
            &proofs(),
            ValueLimits {
                max_depth: 1,
                ..ValueLimits::default()
            },
        )
        .expect("an activation SCC counts as one canonical graph level");
        assert!(tight_cycle
            .validate_for_program(
                ValueLimits {
                    max_depth: 1,
                    ..ValueLimits::default()
                },
                Some(&source_map()),
                &layouts(),
                1,
            )
            .is_ok());
        let graph = encode_payload_graph_in_context(
            &[closure, argument],
            &runtime,
            &source_map(),
            &layouts(),
            &proofs(),
            ValueLimits::default(),
        )
        .unwrap();
        let mut malformed = graph.clone();
        let activation = malformed
            .nodes
            .iter()
            .position(|node| matches!(node, runmat_execution::value::GraphNode::Activation(_)))
            .expect("cycle includes an activation");
        malformed.roots[0] = runmat_execution::value::GraphNodeRef(activation as u32);
        assert!(
            decode_payload_graph_in_context(
                &malformed,
                &runtime,
                &source_map(),
                &layouts(),
                &proofs(),
                &bindings(),
                ValueLimits::default(),
            )
            .is_err(),
            "an activation is not a Value root"
        );
        let mut impossible_value_cycle = graph.clone();
        let first_root = impossible_value_cycle.roots[0];
        let GraphNode::Callable(callable) =
            &mut impossible_value_cycle.nodes[first_root.0 as usize]
        else {
            panic!("nested graph root")
        };
        callable.captures.push(first_root);
        assert!(
            impossible_value_cycle
                .validate_for_program(ValueLimits::default(), Some(&source_map()), &layouts(), 1)
                .is_err(),
            "an activation-containing SCC cannot hide a by-value cycle"
        );
        assert!(
            decode_payload_graph_in_context(
                &impossible_value_cycle,
                &runtime,
                &source_map(),
                &layouts(),
                &proofs(),
                &bindings(),
                ValueLimits::default(),
            )
            .is_err(),
            "a by-value Value cycle must fail before activation allocation"
        );
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
        let Value::Closure(first) = &decoded.values()[0] else {
            panic!("target closure")
        };
        assert_eq!(
            first
                .binding
                .as_ref()
                .unwrap()
                .authority
                .export_origin(&runtime.session_lease()),
            Some(origin(2)),
        );
        let rebuilt = first
            .lexical_environment
            .as_ref()
            .expect("activation rebuilt");
        assert_ne!(rebuilt.handle(), environment.handle());
        let Value::Cell(cell) = &decoded.values()[1] else {
            panic!("argument cell")
        };
        let Value::Closure(aliased) = &cell.data()[0] else {
            panic!("argument closure")
        };
        assert_eq!(
            aliased.lexical_environment.as_ref().unwrap().handle(),
            rebuilt.handle()
        );
        let Value::Closure(back_edge) = crate::call::lexical::read_binding(rebuilt, BindingId(7))
            .unwrap()
            .expect("self cycle")
        else {
            panic!("binding callback")
        };
        assert_eq!(
            back_edge.lexical_environment.as_ref().unwrap().handle(),
            rebuilt.handle()
        );
        drop(original_roots);
        runmat_gc::gc_collect_major().unwrap();
        assert!(crate::call::lexical::read_binding(rebuilt, BindingId(7))
            .unwrap()
            .is_some());
    });
}

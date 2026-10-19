use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn admitted_callable_metadata_and_capture_arity_are_not_wire_authority() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let (environment, _roots) = crate::call::lexical::allocate_environment(None).unwrap();
        assert!(crate::call::lexical::reserve_owned_binding(&environment, BindingId(7)).unwrap());
        let value = Value::Closure(Closure::new(
            "nested".into(),
            Some(BoundFunctionBinding::portable(2, origin(2))),
            FunctionSourceRole::Nested,
            None,
            Vec::new(),
            Some(environment),
        ));
        let Value::Closure(wrong_closure) = value.clone() else {
            unreachable!()
        };
        let mut wrong_closure = wrong_closure.into_data();
        wrong_closure.function_name = "unadmitted".into();
        assert!(
            encode_payload_graph_in_context(
                &[Value::Closure(Closure::from_data(wrong_closure))],
                &runtime,
                &source_map(),
                &layouts(),
                &proofs(),
                ValueLimits::default(),
            )
            .is_err(),
            "snapshot must reject unadmitted callable metadata"
        );
        let graph = encode_payload_graph_in_context(
            &[value],
            &runtime,
            &source_map(),
            &layouts(),
            &proofs(),
            ValueLimits::default(),
        )
        .unwrap();
        for alteration in 0..4 {
            let mut forged = graph.clone();
            let root = forged.roots[0];
            let extra = runmat_execution::value::GraphNodeRef(forged.nodes.len() as u32);
            if alteration == 3 {
                forged.nodes.push(GraphNode::Leaf {
                    value: runmat_execution::value::ValuePayload::Inline(Box::new(
                        runmat_execution::value::InlineValue::Null,
                    )),
                });
            }
            let GraphNode::Callable(callable) = &mut forged.nodes[root.0 as usize] else {
                panic!("bound closure node")
            };
            match alteration {
                0 => callable.qualified_name = "other".into(),
                1 => callable.source_role = FunctionSourceRole::Anonymous,
                2 => callable.source_file = Some("other.m".into()),
                3 => callable.captures.extend([extra, extra]),
                _ => unreachable!(),
            }
            callable.callable_digest = CallableValue::identity_digest(
                callable.owner.portable_owner(),
                &callable.qualified_name,
                callable.form,
                callable.source_role,
                callable.source_file.as_deref(),
            );
            assert!(validate_bound_callable_proofs(&forged, &proofs()).is_err());
            assert!(
                decode_payload_graph_in_context(
                    &forged,
                    &runtime,
                    &source_map(),
                    &layouts(),
                    &proofs(),
                    &bindings(),
                    ValueLimits::default(),
                )
                .is_err(),
                "forged callable attribute {alteration} must not certify itself"
            );
        }
    });
}

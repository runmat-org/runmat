use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn mean_without_sign_evidence_and_sqrt_keep_their_existing_inference() {
    runmat_gc::gc_test_context(|| {
        let source = plan(
            vec![FusionOp::Builtin {
                origin: Default::default(),
                name: "mean".into(),
                inputs: vec![0],
                output: Some(2),
            }],
            2,
        );
        let program = resolve(&source, &[input(), epsilon()]).expect("mean without sign evidence");
        assert!(!program.sign.contains_key(&2));
        assert_eq!(program.floating_class(2).unwrap(), NumericClass::Single);

        let source = plan(vec![cast("sqrt", 0, 2)], 2);
        let program = resolve(&source, &[input(), epsilon()]).expect("unproved sqrt source");
        assert!(!program.sign.contains_key(&2));
        assert_eq!(
            program.fact(2).unwrap().kind,
            runmat_types::ValueKindFact::Unknown
        );
        let bindings = semantic_bindings::capture(&source).unwrap();
        assert!(bindings[0].unwrap().declaration.spec.elementwise.is_none());
    });
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn signed_add_keeps_the_static_resolution_path_and_identity() {
    runmat_gc::gc_test_context(|| {
        let mut source = plan(
            vec![
                FusionOp::Primitive {
                    op: PrimitiveOp::ElemPow,
                    inputs: vec![0, 2],
                    output: Some(3),
                },
                cast("single", 3, 6),
                FusionOp::Primitive {
                    op: PrimitiveOp::Add,
                    inputs: vec![6, 1],
                    output: Some(4),
                },
                cast("sqrt", 4, 5),
            ],
            5,
        );
        source.const_values.insert(2, Value::Num(2.0));
        let input = input();
        let epsilon = epsilon();
        let bindings = semantic_bindings::capture(&source).unwrap();
        let static_program = ResolvedProgram::resolve_static_captured(
            &source,
            &[&input, &epsilon],
            ProviderPrecision::F64,
            &bindings,
        )
        .expect("static admission")
        .expect("signed Add must not force asynchronous resolution");
        let async_program = futures::executor::block_on(ResolvedProgram::resolve_captured(
            &source,
            &[&input, &epsilon],
            ProviderPrecision::F64,
            None,
            &bindings,
        ))
        .expect("async admission");
        assert_eq!(static_program.sign.get(&4), async_program.sign.get(&4));
        assert_eq!(
            static_program.floating_class(5).unwrap(),
            NumericClass::Single
        );
        assert_eq!(
            static_program.cache_identity().unwrap(),
            async_program.cache_identity().unwrap()
        );
    });
}

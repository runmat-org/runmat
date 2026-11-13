use super::*;

#[test]
fn explicit_color_keeps_original_capture_tail_storage_and_gc_registration() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let root = runmat_gc::gc_allocate_rooted(Value::Num(47.)).unwrap();
        let handle = root.handle();
        let color = runmat_value::Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]).unwrap();
        let color_pointer = color.as_f64_slice().unwrap().as_ptr();
        let callable = crate::execution::RetainedValueReadLease::prepare_owned(
            Value::Closure(runmat_value::Closure::new(
                "surface_capture_fixture".into(),
                None,
                runmat_types::FunctionSourceRole::Anonymous,
                None,
                vec![
                    Value::Num(1.),
                    Value::Num(2.),
                    Value::Num(3.),
                    Value::Tensor(color),
                    Value::from("UserData"),
                    Value::HandleObject(runmat_value::HandleRef {
                        class_name: "SurfaceCapturedEdge".into(),
                        target: handle,
                        valid: true,
                    }),
                ],
                None,
            )),
            &runtime,
        )
        .unwrap();
        let args = crate::call::BuiltinArguments::new(
            crate::call::RootedFunctionArguments::captured(
                callable,
                crate::execution::RootedValues::new(Vec::new()).unwrap(),
            )
            .unwrap(),
        );
        let tail_pointer = args.get(4).unwrap() as *const Value;
        root.unroot().unwrap();
        let parsed = parse(args).unwrap();
        assert_eq!(parsed.x, Value::Num(1.));
        assert_eq!(parsed.y, Value::Num(2.));
        assert_eq!(parsed.z, Value::Num(3.));
        let Some(Value::Tensor(color)) = &parsed.color else {
            panic!("numeric color")
        };
        assert_eq!(color.as_f64_slice().unwrap().as_ptr(), color_pointer);
        assert_eq!(parsed.rest.len(), 2);
        assert_eq!(parsed.rest.get(0).unwrap() as *const Value, tail_pointer);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_ok());
        drop(parsed);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_err());
    });
}

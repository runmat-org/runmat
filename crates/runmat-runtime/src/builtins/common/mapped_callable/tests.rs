use super::*;

#[test]
fn retained_unpolled_callback_keeps_original_capture_roots_until_drop() {
    runmat_gc::gc_test_context(|| {
        let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
            crate::execution::RuntimeExecutionService::new(),
        ));
        let initial = runmat_gc::gc_allocate_rooted(Value::Num(29.)).unwrap();
        let handle = initial.handle();
        let tensor = runmat_value::Tensor::new(vec![3., 5.], vec![1, 2]).unwrap();
        let original = tensor.as_f64_slice().unwrap().as_ptr();
        let closure = runmat_value::Closure::new(
            "unpolled_capture_fixture".into(),
            None,
            runmat_types::FunctionSourceRole::Anonymous,
            None,
            vec![
                Value::Tensor(tensor),
                Value::HandleObject(runmat_value::HandleRef {
                    class_name: "MappedCaptureFixture".into(),
                    target: handle,
                    valid: true,
                }),
            ],
            None,
        );
        let callable =
            MappedCallable::Closure(crate::call::descriptor::RetainedClosureValue::new(closure));
        let arguments = crate::call::BuiltinArguments::from_borrowed(&[], Some(&runtime)).unwrap();
        let future = callable.invoke_retained(Some(&runtime), arguments, 1);
        initial.unroot().unwrap();
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
        let MappedCallable::Closure(retained) = &callable else {
            unreachable!()
        };
        let Value::Tensor(capture) = &retained.as_closure().captures[0] else {
            unreachable!()
        };
        assert_eq!(capture.as_f64_slice().unwrap().as_ptr(), original);
        drop(future);
        runmat_gc::gc_collect_major().unwrap();
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
    });
}

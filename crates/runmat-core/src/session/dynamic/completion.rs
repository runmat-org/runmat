use runmat_runtime::RuntimeError;

pub(super) fn finish_dynamic_invocation<T>(
    result: Result<runmat_runtime::execution::RootedValueSequence, RuntimeError>,
    retirement: T,
    preserve_error: impl FnOnce(&mut RuntimeError),
) -> Result<runmat_runtime::execution::RootedValueSequence, RuntimeError> {
    match result {
        Ok(rooted) => {
            drop(retirement);
            Ok(rooted)
        }
        Err(mut error) => {
            preserve_error(&mut error);
            drop(retirement);
            Err(error)
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use runmat_value::Value;
    use std::cell::Cell;
    use std::rc::Rc;

    struct CollectOnDrop(Option<runmat_gc::ExplicitRoot>);

    impl Drop for CollectOnDrop {
        fn drop(&mut self) {
            self.0
                .take()
                .expect("source root")
                .unroot()
                .expect("transfer to completion root");
            runmat_gc::gc_collect_major().expect("collect during retirement");
        }
    }

    #[test]
    fn successful_outputs_stay_rooted_while_dynamic_owners_retire() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("dynamic output".into()))
                .expect("allocate returned target");
            let handle = initial.handle();
            let sequence =
                runmat_value::ValueSequence::single(Value::HandleObject(runmat_value::HandleRef {
                    class_name: "DynamicOutput".into(),
                    target: handle,
                    valid: true,
                }))
                .expect("valid output");
            let completed = finish_dynamic_invocation(
                Ok(
                    runmat_runtime::execution::RootedValueSequence::new(sequence)
                        .expect("root output before retirement"),
                ),
                CollectOnDrop(Some(initial)),
                |_| panic!("successful output has no diagnostic"),
            )
            .expect("retirement preserves output");
            assert_eq!(
                runmat_gc::gc_clone_value(&handle).expect("output remains live at handoff"),
                Value::String("dynamic output".into())
            );
            drop(completed);
            runmat_gc::gc_collect_major().expect("collect after caller drops output");
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }

    #[test]
    fn failure_diagnostics_are_captured_before_dynamic_owners_retire() {
        struct AssertPreservedOnDrop(Rc<Cell<bool>>);
        impl Drop for AssertPreservedOnDrop {
            fn drop(&mut self) {
                assert!(self.0.get(), "retirement ran before diagnostic capture");
            }
        }

        let preserved = Rc::new(Cell::new(false));
        let error = finish_dynamic_invocation(
            Err(runmat_runtime::build_runtime_error("dynamic failure").build()),
            AssertPreservedOnDrop(Rc::clone(&preserved)),
            |_| preserved.set(true),
        )
        .expect_err("failure remains a failure");
        assert_eq!(error.message, "dynamic failure");
        assert!(preserved.get());
    }
}

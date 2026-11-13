use super::*;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn recursive_activation_owns_its_binding_while_child_writes_parent_cell() {
    runmat_gc::gc_test_context(|| {
        let binding = BindingId(17);
        let (parent, _parent_roots) = allocate_environment(None).expect("parent activation");
        assert!(reserve_owned_binding(&parent, binding).unwrap());
        write_owned_binding(&parent, binding, Value::Num(2.0)).expect("parent write");

        let (child, _child_roots) =
            allocate_environment(Some(parent.clone())).expect("child activation");
        assert_eq!(
            read_binding(&child, binding).unwrap(),
            Some(Value::Num(2.0))
        );
        write_binding(&child, binding, Value::Num(3.0)).expect("child writes parent");
        assert_eq!(
            read_binding(&parent, binding).unwrap(),
            Some(Value::Num(3.0))
        );

        let (recursive, _recursive_roots) =
            allocate_environment(Some(child.clone())).expect("recursive activation");
        assert!(reserve_owned_binding(&recursive, binding).unwrap());
        write_owned_binding(&recursive, binding, Value::Num(11.0))
            .expect("recursive activation has a distinct cell");
        assert_eq!(
            read_binding(&recursive, binding).unwrap(),
            Some(Value::Num(11.0))
        );
        assert_eq!(
            read_binding(&child, binding).unwrap(),
            Some(Value::Num(3.0))
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn clear_retires_old_value_then_reuses_only_its_owned_slot() {
    runmat_gc::gc_test_context(|| {
        let binding = BindingId(23);
        let (parent, _roots) = allocate_environment(None).expect("activation");
        assert!(reserve_owned_binding(&parent, binding).unwrap());
        let original =
            runmat_gc::gc_allocate_rooted(Value::String("old".into())).expect("managed old value");
        let handle = original.handle();
        write_owned_binding(
            &parent,
            binding,
            Value::HandleObject(runmat_value::HandleRef {
                class_name: "OldCellValue".into(),
                target: handle,
                valid: true,
            }),
        )
        .expect("publish original value");
        original.unroot().expect("transfer root to cell");
        runmat_gc::gc_collect_major().expect("collect while published");
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_ok());

        remove_owned_binding(&parent, binding).expect("clear binding");
        assert_eq!(read_binding(&parent, binding).unwrap(), None);
        runmat_gc::gc_collect_major().expect("collect retired value");
        assert!(runmat_gc::gc_with_value(&handle, |_| ()).is_err());
        write_owned_binding(&parent, binding, Value::Num(7.0)).expect("reuse vacant slot");
        assert_eq!(
            read_binding(&parent, binding).unwrap(),
            Some(Value::Num(7.0))
        );
    });
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn retired_environment_handle_preserves_typed_gc_cause() {
    use std::error::Error;

    runmat_gc::gc_test_context(|| {
        let binding = BindingId(29);
        let (environment, roots) = allocate_environment(None).expect("activation");
        assert!(reserve_owned_binding(&environment, binding).unwrap());
        write_owned_binding(&environment, binding, Value::Num(4.0)).expect("published cell");
        drop(roots);
        runmat_gc::gc_collect_major().expect("retire unrooted activation");

        for error in [
            root_environment(&environment).expect_err("retired root"),
            read_binding(&environment, binding).expect_err("retired cell"),
        ] {
            assert_eq!(error.identifier(), Some("RunMat:LexicalEnvironment"));
            assert!(
                error
                    .source()
                    .and_then(|source| source.downcast_ref::<runmat_gc::GcError>())
                    .is_some(),
                "operational GC cause must remain typed"
            );
        }
    });
}

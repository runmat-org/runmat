use super::VariableError;
use std::error::Error;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn grouping_error_retains_original_runtime_cause() {
    let original = crate::build_runtime_error("original grouping cause")
        .with_identifier("RunMat:GroupingCause")
        .build();
    let error = VariableError::with_source("grouping input", original)
        .attach(crate::build_runtime_error("outer grouping error").build());
    let wrapper = error
        .source()
        .unwrap()
        .downcast_ref::<VariableError>()
        .unwrap();
    let original = wrapper
        .source()
        .unwrap()
        .downcast_ref::<crate::RuntimeError>()
        .unwrap();
    assert_eq!(original.identifier.as_deref(), Some("RunMat:GroupingCause"));
    assert_eq!(original.message, "original grouping cause");
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test]
fn grouping_error_keeps_browser_local_cause_alive() {
    use std::rc::Rc;

    #[derive(Debug)]
    struct LocalCause(Rc<()>);
    impl std::fmt::Display for LocalCause {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("worker-local grouping cause")
        }
    }
    impl Error for LocalCause {}

    let owner = Rc::new(());
    let error = VariableError::with_source("grouping input", LocalCause(Rc::clone(&owner)))
        .attach(crate::build_runtime_error("outer grouping error").build());
    let wrapper = error
        .source()
        .unwrap()
        .downcast_ref::<VariableError>()
        .unwrap();
    let cause = wrapper
        .source()
        .unwrap()
        .downcast_ref::<LocalCause>()
        .unwrap();
    assert!(Rc::ptr_eq(&owner, &cause.0));
    assert_eq!(Rc::strong_count(&owner), 2);
    drop(error);
    assert_eq!(Rc::strong_count(&owner), 1);
}

use super::*;

#[test]
fn exact_binding_rejects_an_ordinary_owner_before_default_creation() {
    let class =
        runmat_types::ClassIdentity::new("MissingBoundConstructor").expect("valid class identity");
    let owner = runmat_types::ClassMethodOwner {
        declaring_class: class.clone(),
        method: runmat_types::MethodName::from("MissingBoundConstructor"),
        is_static: false,
        role: runmat_types::ClassMethodRole::Ordinary,
    };
    let error = futures::executor::block_on(super::call_class_constructor(
        class,
        &[],
        Some(1),
        None,
        ConstructorInvocationTarget::ExactBound {
            owner,
            binding: runmat_value::BoundFunctionBinding::unavailable(1),
        },
    ))
    .expect_err("ordinary method cannot enter constructor orchestration");
    assert_eq!(error.identifier(), Some("RunMat:ConstructorOwnerInvalid"));
}

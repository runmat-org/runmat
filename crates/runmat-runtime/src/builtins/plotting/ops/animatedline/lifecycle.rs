use super::*;
use crate::builtins::plotting::state;

#[test]
fn animatedline_unpolled_drop_does_not_publish_graphics_state() {
    let _lock = lock_plot_registry();
    state::reset_plot_state();
    let future = animatedline_builtin(crate::builtins::common::test_support::builtin_arguments(
        vec![],
    ));
    drop(future);
    assert!(state::figure_handles().is_empty());
    assert!(state::take_recent_figures().is_empty());
}

#[test]
fn animatedline_output_error_precedes_state_and_discard_still_constructs() {
    let _lock = lock_plot_registry();
    ensure_plot_test_env();
    state::reset_plot_state();
    {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:animatedline:TooManyOutputs")
        );
        assert!(state::figure_handles().is_empty());
    }
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let handle = block_on(animatedline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    assert_eq!(
        get_builtin(vec![handle, Value::from("Type")]).unwrap(),
        Value::from("animatedline")
    );
    assert_eq!(state::figure_handles().len(), 1);
}

#[test]
fn animatedline_structured_errors_preserve_details_and_causes() {
    let _guard = setup();
    let error = block_on(animatedline_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::from("MaximumNumPoints"),
            Value::Num(0.0),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:animatedline:InvalidArgument")
    );
    assert!(error.message().contains("MaximumNumPoints"));
    let source = std::io::Error::other("provider cause");
    let error = crate::build_runtime_error("precise provider failure")
        .with_source(source)
        .build();
    let mapped = contextual_error(error, "RunMat:animatedline:InvalidArgument");
    assert_eq!(mapped.message(), "precise provider failure");
    assert_eq!(
        mapped.source.as_ref().unwrap().to_string(),
        "provider cause"
    );
}

#[test]
fn animatedline_preserves_nd_flattening_and_empty_vector_rules() {
    let _guard = setup();
    for shape in [vec![1, 2, 3], vec![2, 1, 3]] {
        let input = Value::Tensor(Tensor::new(vec![1., 2., 3., 4., 5., 6.], shape).unwrap());
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![input.clone(), input]),
        ))
        .unwrap();
        assert_eq!(
            tensor_data(get_builtin(vec![handle, Value::from("XData")]).unwrap()),
            vec![1., 2., 3., 4., 5., 6.]
        );
    }
    for shape in [vec![1, 0], vec![0, 1]] {
        let input = Value::Tensor(Tensor::new(vec![], shape).unwrap());
        let handle = block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![input.clone(), input]),
        ))
        .unwrap();
        assert!(tensor_data(get_builtin(vec![handle, Value::from("XData")]).unwrap()).is_empty());
    }
    let input = Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
    assert_eq!(
        block_on(animatedline_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![input.clone(), input])
        ))
        .unwrap_err()
        .identifier(),
        Some("RunMat:animatedline:InvalidArgument")
    );
}

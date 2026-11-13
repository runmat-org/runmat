use super::*;
fn matrix(values: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(values, shape).unwrap())
}
fn arguments(parents: Value) -> crate::call::BuiltinArguments {
    crate::builtins::common::test_support::builtin_arguments(vec![
        parents,
        super::super::super::inputs::raw::empty(),
        Value::Num(1.),
        Value::Struct(runmat_value::UnboundedStructValueBuilder::new().finish()),
        matrix(vec![4., 0.], vec![2, 1]),
        matrix(vec![0., 2.], vec![2, 1]),
    ])
}
#[test]
fn heuristic_helper_uses_requested_parent_scores_and_rejects_invalid_pairs() {
    futures::executor::block_on(async {
        let parents = matrix(vec![1., 2.], vec![1, 2]);
        let result = execute(Kind::Heuristic, None, arguments(parents))
            .await
            .unwrap();
        let Value::Tensor(result) = result else {
            panic!("double crossover child")
        };
        assert_eq!(result.shape(), [1, 1]);
        assert!((result.as_f64_slice().unwrap()[0] - 2.4).abs() < 1e-14);
        let error = execute(Kind::Heuristic, None, arguments(Value::Num(1.)))
            .await
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::GA_ERROR_INVALID_ARGUMENT.identifier
        );
        let error = execute(
            Kind::Heuristic,
            None,
            arguments(matrix(vec![1., 3.], vec![1, 2])),
        )
        .await
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            runmat_builtins::GA_ERROR_INVALID_ARGUMENT.identifier
        );
    });
}

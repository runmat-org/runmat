use super::*;
use futures::executor::block_on;
use runmat_value::{ComplexTensor, IntValue, LogicalArray, Tensor};

#[test]
fn unary_numeric_values_preserve_canonical_class_shape_and_exact_integers() {
    let values = [
        Value::Bool(true),
        Value::Num(1.0),
        Value::Complex(2.0, -3.0),
        Value::Int(IntValue::I64(9_007_199_254_740_993)),
        Value::Tensor(Tensor::new(vec![1.0, -2.0], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::from_f32(vec![1.0, -2.0], vec![1, 2]).unwrap()),
        Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
        Value::ComplexTensor(ComplexTensor::new(vec![(1.0, 2.0)], vec![1, 1]).unwrap()),
    ];
    for value in values {
        let plus = block_on(evaluate(UnaryArithmeticOperation::Plus, value.clone())).unwrap();
        assert_eq!(plus, value);
        let expected = block_on(crate::call_builtin_async(
            "times",
            &[value.clone(), Value::Num(-1.0)],
        ))
        .unwrap();
        let minus = block_on(evaluate(UnaryArithmeticOperation::Minus, value)).unwrap();
        assert_eq!(minus, expected);
    }
    assert_eq!(
        block_on(evaluate(
            UnaryArithmeticOperation::Minus,
            Value::Int(IntValue::I64(9_007_199_254_740_993))
        ))
        .unwrap(),
        Value::Int(IntValue::I64(-9_007_199_254_740_993)),
    );
}

#[test]
fn unary_rejects_typed_complex_integers_consistently() {
    let value = Value::ComplexTensor(
        ComplexTensor::new_integer(
            runmat_value::IntegerComplexStorage::new(
                runmat_value::IntegerStorage::I64(vec![9_007_199_254_740_993]),
                runmat_value::IntegerStorage::I64(vec![1]),
            )
            .unwrap(),
            vec![1, 1],
        )
        .unwrap(),
    );
    for operation in [
        UnaryArithmeticOperation::Plus,
        UnaryArithmeticOperation::Minus,
    ] {
        let error = block_on(evaluate(operation, value.clone())).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:ComplexIntegerArithmetic"));
    }
}

#[test]
fn unary_object_overloads_preserve_errors_and_missing_method_fallback() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    let runtime = crate::context::RuntimeContext::new(std::rc::Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _activation = runtime.enter();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let _resolver =
        crate::user_functions::install_semantic_function_resolver(Some(Arc::new(|name| {
            (name == "unary_fixture_minus").then_some(91)
        })))
        .expect("call-port scope");
    let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
        move |function, _, requested| {
            assert_eq!(function, 91);
            assert_eq!(requested, 1);
            observed.fetch_add(1, Ordering::SeqCst);
            Box::pin(async {
                Err(crate::runtime_error::semantic_error(
                    "UnaryFixtureError",
                    "body failed",
                ))
            })
        },
    )))
    .expect("call-port scope");
    crate::class_registry::register_class(crate::class_registry::RuntimeClass {
        events: Default::default(),
        name: "UnaryFixture".into(),
        parent: None,
        properties: Default::default(),
        methods: [(
            "uminus".into(),
            crate::class_registry::RuntimeMethod {
                name: "uminus".into(),
                is_static: false,
                is_abstract: false,
                is_sealed: false,
                access: runmat_types::MemberAccess::Public,
                function_name: "unary_fixture_minus".into(),
                implicit_class_argument: None,
                role: runmat_types::ClassMethodRole::Ordinary,
            },
        )]
        .into_iter()
        .collect(),
    });
    let object = Value::Object(runmat_value::ObjectInstance::new("UnaryFixture"));
    assert_eq!(
        block_on(evaluate(UnaryArithmeticOperation::Plus, object.clone())).unwrap(),
        object
    );
    let error = block_on(evaluate(UnaryArithmeticOperation::Minus, object)).unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:UnaryFixtureError"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

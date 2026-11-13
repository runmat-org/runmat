use runmat_runtime::{
    call_builtin_async_with_outputs, compatibility::push_runmat_extensions_enabled,
};
use runmat_value::Value;

#[test]
fn core_operator_extensions_reject_before_semantic_dispatch_in_matlab_mode() {
    let _matlab_mode = push_runmat_extensions_enabled(false);
    let like_cases = [
        ("plus", "RunMat:compatibility:PlusLikePrototypeExtension"),
        ("minus", "RunMat:compatibility:MinusLikePrototypeExtension"),
        ("times", "RunMat:compatibility:TimesLikePrototypeExtension"),
        (
            "rdivide",
            "RunMat:compatibility:RdivideLikePrototypeExtension",
        ),
        ("power", "RunMat:compatibility:PowerLikePrototypeExtension"),
    ];
    for (name, identifier) in like_cases {
        let args = [
            Value::Num(2.0),
            Value::Num(1.0),
            Value::String("like".to_string().into()),
            Value::Num(0.0),
        ];
        let error = futures::executor::block_on(call_builtin_async_with_outputs(name, &args, 1))
            .expect_err("MATLAB mode must reject RunMat-only like prototype");
        assert_eq!(error.identifier(), Some(identifier), "{name}");
    }

    for (name, identifier) in [
        ("or", "RunMat:compatibility:OrComplexInputExtension"),
        ("xor", "RunMat:compatibility:XorComplexInputExtension"),
    ] {
        let args = [Value::Complex(0.0, 1.0), Value::Bool(false)];
        let error = futures::executor::block_on(call_builtin_async_with_outputs(name, &args, 1))
            .expect_err("MATLAB mode must reject RunMat-only complex logical input");
        assert_eq!(error.identifier(), Some(identifier), "{name}");
    }

    let args = [
        Value::CharArray(runmat_value::CharArray::new_row("A")),
        Value::Bool(false),
    ];
    let error = futures::executor::block_on(call_builtin_async_with_outputs("or", &args, 1))
        .expect_err("MATLAB mode must reject RunMat-only character logical input");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:OrCharacterInputExtension")
    );
    let xor = futures::executor::block_on(call_builtin_async_with_outputs("xor", &args, 1))
        .expect("character xor belongs to the supported strict subset");
    assert!(matches!(xor.value(), Value::Bool(true)));
}

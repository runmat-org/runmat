use super::*;

#[test]
fn trig_execution_conversion_and_range_failures_are_terminal_and_typed() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for (name, hook) in [
        ("sin", "unary_sin"),
        ("cos", "unary_cos"),
        ("tan", "unary_tan"),
    ] {
        check(name, hook, false, false);
        check_case(
            name,
            "complex_from_real",
            &[hook, "complex_from_real"],
            vec![2, 1],
            GpuTensorStorage::Real,
            |left, prototype| {
                vec![
                    Value::GpuTensor(left.clone()),
                    Value::String("like".into()),
                    Value::GpuTensor(prototype.clone()),
                ]
            },
        );
    }
    for name in ["asin", "acos", "acosh"] {
        check(name, "reduce_min", false, false);
    }
    for name in ["asin", "acos"] {
        check_case(
            name,
            "reduce_max",
            &["reduce_min", "reduce_max"],
            vec![2, 1],
            GpuTensorStorage::Real,
            |left, _| vec![Value::GpuTensor(left.clone())],
        );
    }
}

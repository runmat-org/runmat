use super::*;

#[test]
fn unary_and_complex_dispatch_preserve_exact_terminal_causes() {
    for (name, hook, probe) in [
        ("erf", "unary_erf", false),
        ("erfcinv", "unary_erfcinv", false),
        ("factorial", "unary_factorial", false),
        ("gamma", "unary_gamma", false),
        ("gammaln", "unary_gammaln", true),
        ("sqrt", "unary_sqrt", true),
        ("realsqrt", "unary_sqrt", true),
        ("pow2", "unary_pow2", false),
        ("nextpow2", "unary_nextpow2", false),
        ("log", "unary_log", true),
        ("log2", "unary_log2", true),
        ("log10", "unary_log10", true),
        ("log1p", "unary_log1p", true),
        ("real", "unary_real", false),
        ("imag", "unary_imag", false),
        ("conj", "unary_conj", false),
        ("complex", "complex_from_real", false),
    ] {
        check(name, hook, probe, false);
    }
    check("complex", "complex_from_real_imag", false, true);
}

#[test]
fn matrix_dispatch_preserves_each_provider_boundary_and_owned_transpose() {
    for (name, hook, shape) in [
        ("dot", "dot", vec![3, 1]),
        ("cross", "cross", vec![3, 1]),
        ("mtimes", "matmul", vec![2, 2]),
    ] {
        check_case(
            name,
            hook,
            &[hook],
            shape,
            GpuTensorStorage::Real,
            |left, right| {
                vec![
                    Value::GpuTensor(left.clone()),
                    Value::GpuTensor(right.clone()),
                ]
            },
        );
    }
    for scalar_first in [false, true] {
        check_case(
            "mtimes",
            "scalar_mul",
            &["scalar_mul"],
            vec![2, 2],
            GpuTensorStorage::Real,
            |left, _| {
                let resident = Value::GpuTensor(left.clone());
                if scalar_first {
                    vec![Value::Num(2.0), resident]
                } else {
                    vec![resident, Value::Num(2.0)]
                }
            },
        );
    }
    for (hook, shape, storage, trace) in [
        (
            "transpose",
            vec![2, 3],
            GpuTensorStorage::Real,
            vec!["transpose"],
        ),
        (
            "permute",
            vec![2, 3, 1],
            GpuTensorStorage::Real,
            vec!["permute"],
        ),
        (
            "unary_conj",
            vec![2, 3],
            GpuTensorStorage::ComplexInterleaved,
            vec!["transpose", "unary_conj"],
        ),
    ] {
        check_case("ctranspose", hook, &trace, shape, storage, |left, _| {
            vec![Value::GpuTensor(left.clone())]
        });
    }
}

use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{HostIntegerDataOwned, HostNumericDataOwned};

const OPERATIONS: [Operation; 4] = [
    Operation::Add,
    Operation::Sub,
    Operation::Mul,
    Operation::Div,
];

fn binary(
    provider: &InProcessProvider,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
    op: Operation,
) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
    block_on(match op {
        Operation::Add => provider.elem_add(lhs, rhs),
        Operation::Sub => provider.elem_sub(lhs, rhs),
        Operation::Mul => provider.elem_mul(lhs, rhs),
        Operation::Div => provider.elem_div(lhs, rhs),
    })
}

fn upload(
    provider: &InProcessProvider,
    values: &[f64],
    shape: &[usize],
    single: bool,
    complex: bool,
) -> GpuTensorHandle {
    provider
        .allocate_numeric_data(
            if single {
                InProcessNumericData::F32(values.iter().map(|&v| v as f32).collect())
            } else {
                InProcessNumericData::F64(values.to_vec())
            },
            shape.to_vec(),
            if complex {
                GpuTensorStorage::ComplexInterleaved
            } else {
                GpuTensorStorage::Real
            },
        )
        .unwrap()
}

fn assert_output(
    provider: &InProcessProvider,
    output: &GpuTensorHandle,
    shape: &[usize],
    expected: &[f64],
    single: bool,
    complex: bool,
) {
    assert_eq!(output.shape(), shape);
    assert_eq!(
        output.physical().storage(),
        if complex {
            GpuTensorStorage::ComplexInterleaved
        } else {
            GpuTensorStorage::Real
        }
    );
    match block_on(provider.download_numeric(output)).unwrap().data {
        HostNumericDataOwned::F32(actual) => {
            assert!(single, "Double narrowed");
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                let expected = *expected as f32;
                assert!(
                    actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan(),
                    "{actual:?} != {expected:?}"
                );
            }
        }
        HostNumericDataOwned::F64(actual) => {
            assert!(!single, "Single widened");
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan(),
                    "{actual:?} != {expected:?}"
                );
            }
        }
        _ => panic!("floating output changed class"),
    }
}

fn real(op: Operation, left: f64, right: f64, both_single: bool) -> f64 {
    if both_single {
        let (left, right) = (left as f32, right as f32);
        f64::from(match op {
            Operation::Add => left + right,
            Operation::Sub => left - right,
            Operation::Mul => left * right,
            Operation::Div => left / right,
        })
    } else {
        match op {
            Operation::Add => left + right,
            Operation::Sub => left - right,
            Operation::Mul => left * right,
            Operation::Div => left / right,
        }
    }
}

#[test]
fn four_binary_hooks_broadcast_native_and_mixed_floats() {
    let provider = InProcessProvider::new();
    for ls in [true, false] {
        for rs in [true, false] {
            let left = [1.0 / 3.0, -2.5];
            let right = [0.25, 3.1, -4.0];
            let lhs = upload(&provider, &left, &[2, 1, 1], ls, false);
            let rhs = upload(&provider, &right, &[1, 3, 1], rs, false);
            for op in OPERATIONS {
                let output = binary(&provider, &lhs, &rhs, op).unwrap();
                let expected: Vec<f64> = right
                    .iter()
                    .flat_map(|&r| {
                        left.iter().map(move |&l| {
                            real(
                                op,
                                if ls { f64::from(l as f32) } else { l },
                                if rs { f64::from(r as f32) } else { r },
                                ls && rs,
                            )
                        })
                    })
                    .collect();
                assert_output(&provider, &output, &[2, 3, 1], &expected, ls || rs, false);
                assert!(!output.shares_allocation_with(&lhs));
                assert!(!output.shares_allocation_with(&rhs));
            }
        }
    }
}

#[test]
fn six_scalar_hooks_preserve_double_scalar_rounding_and_ieee_division() {
    let provider = InProcessProvider::new();
    for single in [true, false] {
        let values = [
            -0.0,
            0.0,
            f64::from(f32::from_bits(1)),
            1.0,
            -2.0,
            f64::INFINITY,
            f64::NAN,
        ];
        let input = upload(&provider, &values, &[1, 7], single, false);
        // Not exactly representable in Single: catches premature scalar rounding.
        for scalar in [1.0 + f64::from(f32::EPSILON) * 0.75, 0.0, -0.0] {
            for (op, reverse, output) in [
                (Operation::Add, false, provider.scalar_add(&input, scalar)),
                (Operation::Sub, false, provider.scalar_sub(&input, scalar)),
                (Operation::Mul, false, provider.scalar_mul(&input, scalar)),
                (Operation::Div, false, provider.scalar_div(&input, scalar)),
                (Operation::Sub, true, provider.scalar_rsub(&input, scalar)),
                (Operation::Div, true, provider.scalar_rdiv(&input, scalar)),
            ] {
                let expected: Vec<f64> = values
                    .iter()
                    .map(|&x| {
                        let x = if single { f64::from(x as f32) } else { x };
                        if reverse {
                            real(op, scalar, x, false)
                        } else {
                            real(op, x, scalar, false)
                        }
                    })
                    .collect();
                assert_output(
                    &provider,
                    &output.unwrap(),
                    input.shape(),
                    &expected,
                    single,
                    false,
                );
            }
        }
    }
}

#[test]
fn complex_and_real_mixed_storage_match_domain_formulas() {
    let provider = InProcessProvider::new();
    for single in [true, false] {
        for (lc, rc) in [(true, true), (true, false), (false, true)] {
            let left = if lc {
                vec![1.25, -0.5, -2.0, 0.75]
            } else {
                vec![1.25, -2.0]
            };
            let right = if rc { vec![0.5, 1.5] } else { vec![0.5] };
            let lhs = upload(&provider, &left, &[2, 1], single, lc);
            let rhs = upload(&provider, &right, &[1, 1], single, rc);
            for op in OPERATIONS {
                let mut expected = Vec::new();
                for i in 0..2 {
                    let ar = left[i * if lc { 2 } else { 1 }];
                    let ai = if lc { left[i * 2 + 1] } else { 0.0 };
                    let br = right[0];
                    let bi = if rc { right[1] } else { 0.0 };
                    // Native complex arithmetic is the domain oracle; real RHS
                    // division is component-wise (not complex-zero synthesis).
                    let pair = if single {
                        let a = num_complex::Complex32::new(ar as f32, ai as f32);
                        let b = num_complex::Complex32::new(br as f32, bi as f32);
                        let v = match op {
                            Operation::Add => a + b,
                            Operation::Sub => a - b,
                            Operation::Mul => a * b,
                            Operation::Div => {
                                if rc {
                                    a / b
                                } else {
                                    a / br as f32
                                }
                            }
                        };
                        (f64::from(v.re), f64::from(v.im))
                    } else {
                        let a = num_complex::Complex64::new(ar, ai);
                        let b = num_complex::Complex64::new(br, bi);
                        let v = match op {
                            Operation::Add => a + b,
                            Operation::Sub => a - b,
                            Operation::Mul => a * b,
                            Operation::Div => {
                                if rc {
                                    a / b
                                } else {
                                    a / br
                                }
                            }
                        };
                        (v.re, v.im)
                    };
                    expected.extend([pair.0, pair.1]);
                }
                assert_output(
                    &provider,
                    &binary(&provider, &lhs, &rhs, op).unwrap(),
                    &[2, 1],
                    &expected,
                    single,
                    true,
                );
            }
        }
        let input = upload(
            &provider,
            &[2.0, -0.0, f64::INFINITY, 1.0],
            &[2, 1],
            single,
            true,
        );
        assert_output(
            &provider,
            &provider.scalar_add(&input, 1.0).unwrap(),
            &[2, 1],
            &[3.0, -0.0, f64::INFINITY, 1.0],
            single,
            true,
        );
        assert_output(
            &provider,
            &provider.scalar_mul(&input, 2.0).unwrap(),
            &[2, 1],
            &[4.0, -0.0, f64::INFINITY, 2.0],
            single,
            true,
        );
        assert_output(
            &provider,
            &provider.scalar_div(&input, 2.0).unwrap(),
            &[2, 1],
            &[1.0, -0.0, f64::INFINITY, 0.5],
            single,
            true,
        );
        for scale in if single {
            [1.0e30, 1.0e-30, f64::from(f32::from_bits(1))]
        } else {
            [1.0e200, 1.0e-200, f64::from_bits(1)]
        } {
            let numerator = upload(&provider, &[scale, 0.0], &[1, 1], single, true);
            let divisor = upload(&provider, &[scale, scale], &[1, 1], single, true);
            let quotient = binary(&provider, &numerator, &divisor, Operation::Div).unwrap();
            assert_output(&provider, &quotient, &[1, 1], &[0.5, -0.5], single, true);
        }
    }
}

#[test]
fn transpose_empty_and_cancelled_operations_keep_exact_ownership() {
    let provider = InProcessProvider::new();
    for single in [true, false] {
        let input = upload(
            &provider,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            single,
            false,
        );
        let transposed = provider.transpose(&input).unwrap();
        let row = upload(&provider, &[10.0, 20.0], &[1, 2], single, false);
        let future = provider.elem_add(&transposed, &row);
        drop(future); // No output is published by an unpolled provider future.
        let output = binary(&provider, &transposed, &row, Operation::Add).unwrap();
        assert_output(
            &provider,
            &output,
            &[3, 2],
            &[11.0, 13.0, 15.0, 22.0, 24.0, 26.0],
            single,
            false,
        );
        let wire = output.wire_descriptor();
        drop(output);
        assert!(provider.acquire_live(&wire).is_err());
        assert!(provider.acquire_live(&input.wire_descriptor()).is_ok());
        assert!(provider.acquire_live(&transposed.wire_descriptor()).is_ok());
        let empty = upload(&provider, &[], &[usize::MAX, usize::MAX, 0], single, false);
        let scalar = upload(&provider, &[1.0], &[1, 1], single, false);
        for op in OPERATIONS {
            assert_output(
                &provider,
                &binary(&provider, &empty, &scalar, op).unwrap(),
                empty.shape(),
                &[],
                single,
                false,
            );
        }
    }
}

#[test]
fn mixed_precision_complex_and_reverse_scalar_keep_domain_rounding() {
    let provider = InProcessProvider::new();
    for left_single in [true, false] {
        let left = upload(&provider, &[0.3, -0.7], &[1, 1], left_single, true);
        let right = upload(&provider, &[1.1, 0.2], &[1, 1], !left_single, true);
        let round = |v: f64, single| if single { f64::from(v as f32) } else { v };
        let a = num_complex::Complex64::new(round(0.3, left_single), round(-0.7, left_single));
        let b = num_complex::Complex64::new(round(1.1, !left_single), round(0.2, !left_single));
        for op in OPERATIONS {
            let value = match op {
                Operation::Add => a + b,
                Operation::Sub => a - b,
                Operation::Mul => a * b,
                Operation::Div => a / b,
            };
            assert_output(
                &provider,
                &binary(&provider, &left, &right, op).unwrap(),
                &[1, 1],
                &[value.re, value.im],
                true,
                true,
            );
        }
        let difference = num_complex::Complex64::new(2.0, 0.0) - a;
        assert_output(
            &provider,
            &provider.scalar_rsub(&left, 2.0).unwrap(),
            &[1, 1],
            &[difference.re, difference.im],
            left_single,
            true,
        );
        let quotient = num_complex::Complex64::new(2.0, 0.0) / a;
        assert_output(
            &provider,
            &provider.scalar_rdiv(&left, 2.0).unwrap(),
            &[1, 1],
            &[quotient.re, quotient.im],
            left_single,
            true,
        );
    }
}

#[test]
fn invalid_owner_and_shape_are_not_unsupported_and_integer_fallback_stays_exact() {
    let provider = InProcessProvider::new();
    let other = InProcessProvider::new();
    let foreign = upload(&other, &[1.0], &[1, 1], true, false);
    let numeric = upload(&provider, &[1.0], &[1, 1], true, false);
    let logical = provider
        .allocate_produced(
            InProcessNumericData::F32(vec![1.0]),
            vec![1, 1],
            GpuTensorStorage::Real,
            true,
        )
        .unwrap();
    let classes = [
        HostIntegerDataOwned::I8(vec![1]),
        HostIntegerDataOwned::I16(vec![1]),
        HostIntegerDataOwned::I32(vec![1]),
        HostIntegerDataOwned::I64(vec![i64::MAX]),
        HostIntegerDataOwned::U8(vec![1]),
        HostIntegerDataOwned::U16(vec![1]),
        HostIntegerDataOwned::U32(vec![1]),
        HostIntegerDataOwned::U64(vec![u64::MAX]),
    ];
    let mut unsupported = vec![logical];
    for data in classes {
        unsupported.push(
            provider
                .allocate_numeric_data(
                    InProcessNumericData::Integer(data),
                    vec![1, 1],
                    GpuTensorStorage::Real,
                )
                .unwrap(),
        );
    }
    for input in unsupported {
        for op in OPERATIONS {
            assert!(binary(&provider, &input, &numeric, op)
                .unwrap_err()
                .is_unsupported());
            assert!(!binary(&provider, &input, &foreign, op)
                .unwrap_err()
                .is_unsupported());
        }
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &provider.scalar_add(&input, 1.0).unwrap_err()
        ));
    }
    let a = upload(&provider, &[1.0, 2.0], &[2, 1], true, false);
    let b = upload(&provider, &[1.0, 2.0, 3.0], &[3, 1], true, false);
    assert!(!binary(&provider, &a, &b, Operation::Add)
        .unwrap_err()
        .is_unsupported());
}

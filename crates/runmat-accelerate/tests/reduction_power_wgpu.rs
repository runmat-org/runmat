#![cfg(feature = "wgpu")]
//! Actual-adapter parity: the legacy reduction geometry consumes the selected
//! ordinary POWER producer, rather than independently reimplementing POWER.

#[path = "support/fusion_plan.rs"]
mod fixture;
#[path = "support/selected_reduction.rs"]
mod selected_reduction;

use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate::fusion::{FusionKind, FusionOp};
use runmat_accelerate::fusion_exec::{execute_elementwise, FusionExecutionRequest};
use runmat_accelerate::graph::{PrimitiveOp, ShapeInfo};
use runmat_accelerate_api::{
    AccelProvider, GpuHandleProvenance, GpuTensorStorage, HostNumericDataOwned,
    HostNumericDataView, HostNumericTensorView, ReductionFlavor,
};
use runmat_value::{Tensor, Value};

fn public_power_then_sum(base: &[f32], exponent: f32) -> f32 {
    let power = block_on(runmat_runtime::call_builtin_async(
        "power",
        &[
            Value::Tensor(Tensor::from_f32(base.to_vec(), vec![base.len(), 1]).unwrap()),
            Value::Tensor(Tensor::from_f32(vec![exponent], vec![1, 1]).unwrap()),
        ],
    ))
    .expect("public Runtime POWER");
    let power = block_on(runmat_runtime::dispatcher::gather_if_needed_async(&power)).unwrap();
    let Value::Tensor(power) = power else {
        panic!("public Single POWER tensor expected")
    };
    assert_eq!(power.shape(), [base.len(), 1]);
    assert!(
        power.as_f32_slice().is_some(),
        "public POWER must publish Single"
    );
    let sum = block_on(runmat_runtime::call_builtin_async(
        "sum",
        &[Value::Tensor(power), Value::Num(1.0)],
    ))
    .expect("public Runtime sum after POWER");
    let sum = block_on(runmat_runtime::dispatcher::gather_if_needed_async(&sum)).unwrap();
    let Value::Tensor(sum) = sum else {
        panic!("public Single sum tensor expected")
    };
    assert_eq!(sum.shape(), [1, 1]);
    let values = sum.as_f32_slice().expect("public sum must publish Single");
    values[0]
}

#[test]
fn selected_native_single_power_then_sum_matches_ordinary_power_output() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required");
    for exponent in [3.0_f32, -3.0, 0.0] {
        let base = [1.000_000_1_f32, -0.0, -2.0, 0.999_999_94];
        let base_handle = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&base),
                shape: &[4, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        assert_eq!(
            base_handle.physical().element_type(),
            runmat_accelerate_api::NumericElementType::F32
        );
        let exponent_value = Value::Tensor(Tensor::from_f32(vec![exponent], vec![1, 1]).unwrap());
        let mut ordinary = fixture::plan(
            vec![FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![0, 1],
                output: Some(2),
            }],
            2,
        );
        ordinary.inputs = vec![0, 1];
        ordinary.stack_pattern = vec![0, 1];
        ordinary.group.shape = ShapeInfo::Tensor(vec![Some(4), Some(1)]);
        let ordinary_result = block_on(execute_elementwise(FusionExecutionRequest {
            plan: &ordinary,
            inputs: vec![
                Value::GpuTensor(base_handle.clone()),
                exponent_value.clone(),
            ],
            placement: None,
            runtime: None,
        }))
        .expect("ordinary admitted POWER");
        let Value::GpuTensor(ordinary_handle) = ordinary_result.final_value else {
            panic!("ordinary POWER must stay resident")
        };
        let ordinary_data = block_on(provider.download_numeric(&ordinary_handle)).unwrap();
        let HostNumericDataOwned::F32(ordinary_values) = ordinary_data.data else {
            panic!("native Single ordinary POWER expected")
        };

        let mut reduction = ordinary.clone();
        reduction.group.kind = FusionKind::Reduction;
        reduction.kernel.kind = FusionKind::Reduction;
        select_sum_terminal(&mut reduction);
        reduction.reduction_data = Some(2);
        reduction.reduction_flavor = Some(ReductionFlavor::Sum);
        let reduced = block_on(selected_reduction::execute(
            FusionExecutionRequest {
                plan: &reduction,
                inputs: vec![Value::GpuTensor(base_handle.clone()), exponent_value],
                placement: None,
                runtime: None,
            },
            4,
            1,
            &[1, 1],
            64,
        ))
        .expect("selected reduction execution");
        match reduced {
            None => {}
            Some(value) => {
                let Value::GpuTensor(reduced_handle) = value else {
                    panic!("selected reduction must publish resident output");
                };

                let reduced_data = block_on(provider.download_numeric(&reduced_handle)).unwrap();
                let HostNumericDataOwned::F32(actual) = reduced_data.data else {
                    panic!("native Single reduction expected")
                };
                let expected = public_power_then_sum(&base, exponent);
                let ordinary_sum: f32 = ordinary_values.iter().copied().sum();
                assert!(
                    (ordinary_sum - expected).abs() <= 2e-6 * expected.abs().max(1.0)
                        || (ordinary_sum.is_infinite() && ordinary_sum == expected),
                    "ordinary fusion POWER diverged from public POWER→sum"
                );
                if expected.is_infinite() {
                    assert_eq!(actual[0], expected, "exponent {exponent}");
                } else {
                    assert!(
                        (actual[0] - expected).abs() <= 2e-6 * expected.abs().max(1.0),
                        "exponent {exponent}: reduction={} ordinary POWER then sum={expected}",
                        actual[0]
                    );
                }
            }
        }
    }
}

#[test]
fn singleton_reduction_matches_public_power_then_sum_exact_single_bits() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required");
    for (base, exponent) in [
        (1.000_000_1_f32, 3.0_f32),
        (0.999_999_94, -3.0),
        (-0.0, 3.0),
        (-2.0, 3.0),
        (0.0, 0.0),
    ] {
        let expected = public_power_then_sum(&[base], exponent);
        let base_handle = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[base]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        assert_eq!(
            base_handle.physical().element_type(),
            runmat_accelerate_api::NumericElementType::F32
        );
        let mut reduction = fixture::plan(
            vec![FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![0, 1],
                output: Some(2),
            }],
            2,
        );
        reduction.inputs = vec![0, 1];
        reduction.stack_pattern = vec![0, 1];
        reduction.group.kind = FusionKind::Reduction;
        reduction.group.shape = ShapeInfo::Tensor(vec![Some(1), Some(1)]);
        reduction.kernel.kind = FusionKind::Reduction;
        select_sum_terminal(&mut reduction);
        reduction.reduction_data = Some(2);
        reduction.reduction_flavor = Some(ReductionFlavor::Sum);
        let result = block_on(selected_reduction::execute(
            FusionExecutionRequest {
                plan: &reduction,
                inputs: vec![
                    Value::GpuTensor(base_handle),
                    Value::Tensor(Tensor::from_f32(vec![exponent], vec![1, 1]).unwrap()),
                ],
                placement: None,
                runtime: None,
            },
            1,
            1,
            &[1, 1],
            64,
        ))
        .expect("singleton selected reduction");
        match result {
            None => {}
            Some(value) => {
                let Value::GpuTensor(handle) = value else {
                    panic!("selected reduction must publish resident output");
                };

                assert_eq!(handle.shape(), [1, 1]);
                assert_eq!(
                    handle.physical().element_type(),
                    runmat_accelerate_api::NumericElementType::F32
                );
                let downloaded = block_on(provider.download_numeric(&handle)).unwrap();
                assert_eq!(downloaded.shape, [1, 1]);
                let HostNumericDataOwned::F32(actual) = downloaded.data else {
                    panic!("Single output expected")
                };
                assert_eq!(
                    actual[0].to_bits(),
                    expected.to_bits(),
                    "base={base:?}, exponent={exponent:?}"
                );
            }
        }
    }
}

#[test]
fn broadcast_reduction_matches_public_power_then_sum_across_columns() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required");
    let base = [1.000_000_1_f32, -0.0, -2.0, 0.999_999_94];
    let exponents = [2.0_f32, 3.0, 4.0];
    let public = block_on(runmat_runtime::call_builtin_async(
        "power",
        &[
            Value::Tensor(Tensor::from_f32(base.to_vec(), vec![4, 1]).unwrap()),
            Value::Tensor(Tensor::from_f32(exponents.to_vec(), vec![1, 3]).unwrap()),
        ],
    ))
    .expect("public broadcast POWER");
    let public = block_on(runmat_runtime::dispatcher::gather_if_needed_async(&public)).unwrap();
    let Value::Tensor(public) = public else {
        panic!("public broadcast POWER tensor")
    };
    assert_eq!(public.shape(), [4, 3]);
    assert!(public.as_f32_slice().is_some());
    let expected = block_on(runmat_runtime::call_builtin_async(
        "sum",
        &[Value::Tensor(public), Value::Num(1.0)],
    ))
    .expect("public sum of broadcast POWER");
    let expected = block_on(runmat_runtime::dispatcher::gather_if_needed_async(
        &expected,
    ))
    .unwrap();
    let Value::Tensor(expected) = expected else {
        panic!("public sum tensor")
    };
    assert_eq!(expected.shape(), [1, 3]);
    let expected = expected.as_f32_slice().expect("public Single sum");

    let base_handle = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&base),
            shape: &[4, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    assert_eq!(
        base_handle.physical().element_type(),
        runmat_accelerate_api::NumericElementType::F32
    );
    let mut reduction = fixture::plan(
        vec![FusionOp::Primitive {
            op: PrimitiveOp::ElemPow,
            inputs: vec![0, 1],
            output: Some(2),
        }],
        2,
    );
    reduction.inputs = vec![0, 1];
    reduction.stack_pattern = vec![0, 1];
    reduction.group.kind = FusionKind::Reduction;
    reduction.group.shape = ShapeInfo::Tensor(vec![Some(4), Some(3)]);
    reduction.kernel.kind = FusionKind::Reduction;
    select_sum_terminal(&mut reduction);
    reduction.reduction_data = Some(2);
    reduction.reduction_flavor = Some(ReductionFlavor::Sum);
    let result = block_on(selected_reduction::execute(
        FusionExecutionRequest {
            plan: &reduction,
            inputs: vec![
                Value::GpuTensor(base_handle),
                Value::Tensor(Tensor::from_f32(exponents.to_vec(), vec![1, 3]).unwrap()),
            ],
            placement: None,
            runtime: None,
        },
        4,
        3,
        &[1, 3],
        64,
    ))
    .expect("selected broadcast reduction");
    match result {
        None => {}
        Some(value) => {
            let Value::GpuTensor(handle) = value else {
                panic!("selected reduction must publish resident output");
            };

            assert_eq!(handle.shape(), [1, 3]);
            let actual = block_on(provider.download_numeric(&handle)).unwrap();
            assert_eq!(actual.shape, [1, 3]);
            let HostNumericDataOwned::F32(actual) = actual.data else {
                panic!("Single output")
            };
            for (column, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    (actual - expected).abs() <= 2e-6 * expected.abs().max(1.0),
                    "column {column}: reduction={actual}, public POWER→sum={expected}"
                );
            }
        }
    }
}

// Retain the actual canonical terminal owner, not a flavor-only proxy plan.
fn select_sum_terminal(plan: &mut runmat_accelerate::fusion::FusionGroupPlan) {
    plan.operations.push(FusionOp::Builtin {
        name: "sum".into(),
        origin: runmat_accelerate::graph::FusionCallOrigin::named("sum"),
        inputs: vec![2, 4],
        output: Some(3),
    });
    plan.const_values.insert(4, Value::Num(1.0));
    plan.output = Some(3);
    plan.reduction_dim = Some(4);
    plan.group.span.end = plan.operations.len();
}

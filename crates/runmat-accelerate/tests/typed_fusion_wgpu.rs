#![cfg(feature = "wgpu")]
//! Qualification tests require a real adapter. No missing-adapter early return
//! is counted as a pass; F64-specific coverage is an explicit separate gate.
#[path = "support/typed_fusion_plan.rs"]
mod fixture;
use fixture::{cast, plan};
use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
use runmat_accelerate::fusion::{FusionOp, FusionStoreMaterialization};
use runmat_accelerate::fusion_exec::{execute_elementwise, FusionExecutionRequest};
use runmat_accelerate::graph::{PrimitiveOp, ShapeInfo, VarBinding, VarKind};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataOwned, HostTensorView, NumericElementType,
    ProviderPrecision,
};
use runmat_value::{Tensor, Value};

fn values() -> Vec<f64> {
    vec![
        16_777_217.0,
        -16_777_217.0,
        f64::from(f32::MIN_POSITIVE),
        -0.0,
        0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ]
}

fn same_f32(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn actual_adapter_single_final_has_typed_f32_storage_and_preserves_nd_shape() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required");
    let values = values();
    let shape = vec![2, 2, 2];
    let mut plan = plan(vec![cast("single", 0, 2)], 2);
    plan.group.shape = ShapeInfo::Tensor(shape.iter().copied().map(Some).collect());
    let result = block_on(execute_elementwise(FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::Tensor(
            Tensor::new(values.clone(), shape.clone()).unwrap(),
        )],
        placement: None,
        runtime: None,
    }))
    .unwrap();
    let Value::GpuTensor(output) = result.final_value else {
        panic!("resident result required")
    };
    assert_eq!(output.shape, shape);
    assert_eq!(
        output.descriptor.element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(output.descriptor.storage, Some(GpuTensorStorage::Real));
    let downloaded = block_on(provider.download_numeric(&output)).unwrap();
    let HostNumericDataOwned::F32(actual) = downloaded.data else {
        panic!("physical F32 payload required")
    };
    same_f32(
        &actual,
        &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
    );
    provider.free(&output).unwrap();
}

#[test]
#[ignore = "requires an actual F64 WGPU adapter; run explicitly during F64 qualification"]
fn actual_f64_adapter_preserves_single_interior_rounding_and_mixed_final_buffers() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual F64 WGPU adapter required");
    assert_eq!(
        provider.precision(),
        ProviderPrecision::F64,
        "this gate must not silently test an F32 adapter"
    );
    let values = values();
    let shape = vec![2, 2, 2];
    let source = provider
        .upload(&HostTensorView {
            data: &values,
            shape: &shape,
        })
        .unwrap();
    let direct_single = block_on(provider.unary_single(&source)).unwrap();
    let HostNumericDataOwned::F32(direct) = block_on(provider.download_numeric(&direct_single))
        .unwrap()
        .data
    else {
        panic!("typed direct Single")
    };
    let mut plan = plan(
        vec![
            cast("single", 0, 2),
            FusionOp::Primitive {
                op: PrimitiveOp::Add,
                inputs: vec![2, 2],
                output: Some(3),
            },
            cast("double", 3, 4),
        ],
        4,
    );
    plan.group.shape = ShapeInfo::Tensor(shape.iter().copied().map(Some).collect());
    plan.materialized_stores.push(FusionStoreMaterialization {
        value_id: 2,
        binding: VarBinding {
            kind: VarKind::Global,
            index: 0,
        },
    });
    let result = block_on(execute_elementwise(FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(source.clone())],
        placement: None,
        runtime: None,
    }))
    .unwrap();
    let Value::GpuTensor(final_output) = result.final_value else {
        panic!("resident Double")
    };
    let Value::GpuTensor(single_output) = &result.materialized_stores[0].1 else {
        panic!("resident Single")
    };
    assert_eq!(
        single_output.descriptor.element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(
        final_output.descriptor.element_type,
        Some(NumericElementType::F64)
    );
    assert_eq!(single_output.shape, shape);
    assert_eq!(final_output.shape, shape);
    let HostNumericDataOwned::F32(single) = block_on(provider.download_numeric(single_output))
        .unwrap()
        .data
    else {
        panic!("physical F32")
    };
    same_f32(&single, &direct);
    let HostNumericDataOwned::F64(final_values) =
        block_on(provider.download_numeric(&final_output))
            .unwrap()
            .data
    else {
        panic!("physical F64")
    };
    for (actual, input) in final_values.iter().zip(&values) {
        let rounded = *input as f32;
        let expected = f64::from(rounded + rounded);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    for handle in [&source, &direct_single, single_output, &final_output] {
        provider.free(handle).unwrap();
    }
}

fn transposed_fusion(
    provider: &'static runmat_accelerate::backend::wgpu::provider::WgpuProvider,
    source: runmat_accelerate_api::GpuTensorHandle,
) {
    let transposed = provider.transpose(&source).unwrap();
    assert_eq!(transposed.shape, vec![3, 2]);
    assert_eq!(
        transposed.descriptor.element_type,
        source.descriptor.element_type
    );
    assert!(runmat_accelerate_api::handle_is_transposed(&transposed));
    provider.free(&source).unwrap(); // Alias owns its buffer independently.
    let before = runmat_accelerate_api::snapshot_handle_metadata(&transposed);
    let mut plan = plan(vec![cast("single", 0, 2), cast("single", 0, 3)], 3);
    plan.group.shape = ShapeInfo::Tensor(vec![Some(3), Some(2)]);
    plan.materialized_stores.push(FusionStoreMaterialization {
        value_id: 2,
        binding: VarBinding {
            kind: VarKind::Global,
            index: 0,
        },
    });
    let result = block_on(execute_elementwise(FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(transposed.clone())],
        placement: None,
        runtime: None,
    }))
    .unwrap();
    assert_eq!(
        runmat_accelerate_api::snapshot_handle_metadata(&transposed),
        before
    );
    let Value::GpuTensor(output) = result.final_value else {
        panic!("resident materialized result")
    };
    assert_eq!(output.shape, vec![3, 2]);
    assert_eq!(
        output.descriptor.element_type,
        Some(NumericElementType::F32)
    );
    assert!(!runmat_accelerate_api::handle_is_transposed(&output));
    let HostNumericDataOwned::F32(actual) =
        block_on(provider.download_numeric(&output)).unwrap().data
    else {
        panic!("physical Single output")
    };
    same_f32(&actual, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    provider.free(&output).unwrap();
    let Value::GpuTensor(stored) = &result.materialized_stores[0].1 else {
        panic!("resident second fused output")
    };
    assert_eq!(stored.shape, vec![3, 2]);
    let HostNumericDataOwned::F32(actual) =
        block_on(provider.download_numeric(stored)).unwrap().data
    else {
        panic!("physical Single stored output")
    };
    same_f32(&actual, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    provider.free(stored).unwrap();
    let materialized = provider.reshape(&transposed, &[3, 2]).unwrap();
    assert_eq!(
        materialized.descriptor.element_type,
        transposed.descriptor.element_type
    );
    assert!(!runmat_accelerate_api::handle_is_transposed(&materialized));
    provider.free(&transposed).unwrap();
    let actual = block_on(provider.download_numeric(&materialized)).unwrap();
    match actual.data {
        HostNumericDataOwned::F32(values) => same_f32(&values, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
        HostNumericDataOwned::F64(values) => assert_eq!(values, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
        _ => panic!("preserved floating storage"),
    }
    provider.free(&materialized).unwrap();
}

#[test]
fn actual_adapter_materializes_asymmetric_native_transpose_on_device() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required");
    let source = provider
        .upload(&HostTensorView {
            data: &[1., 2., 3., 4., 5., 6.],
            shape: &[2, 3],
        })
        .unwrap();
    transposed_fusion(provider, source);
}

#[test]
#[ignore = "requires an actual F64 WGPU adapter; run explicitly during F64 qualification"]
fn actual_f64_adapter_materializes_single_transpose_without_reinterpreting_bytes() {
    use runmat_accelerate_api::{HostNumericDataView, HostNumericTensorView};
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual F64 WGPU adapter required");
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    let source = runmat_accelerate_api::upload_numeric_protecting(
        provider,
        &HostNumericTensorView {
            data: HostNumericDataView::F32(&[1., 2., 3., 4., 5., 6.]),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        },
        &[],
    )
    .unwrap();
    transposed_fusion(provider, source);
}

#[test]
fn actual_adapter_power_classifies_exact_transposed_pairs_and_broadcasts() {
    use runmat_accelerate_api::{snapshot_handle_metadata, PowerResultDomain};
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required");
    let source = provider
        .upload(&HostTensorView {
            data: &[4., -9., 16., 25., 36., 49.],
            shape: &[2, 3],
        })
        .unwrap();
    let base = provider.transpose(&source).unwrap();
    let before = snapshot_handle_metadata(&base);
    // Logical column-major values are [4,16,36,-9,25,49], not the original bytes.
    for (shape, exponents, domain, expected) in [
        (
            vec![3, 2],
            vec![0.5, 0.5, 0.5, 2., 0.5, 0.5],
            PowerResultDomain::Real,
            vec![2., 4., 6., 81., 5., 7.],
        ),
        (
            vec![3, 2],
            vec![0.5, 2., 0.5, 0.5, 0.5, 0.5],
            PowerResultDomain::Complex,
            vec![],
        ),
        (
            vec![1, 2],
            vec![0.5, 2.],
            PowerResultDomain::Real,
            vec![2., 4., 6., 81., 625., 2401.],
        ),
    ] {
        let exponent = provider
            .upload(&HostTensorView {
                data: &exponents,
                shape: &shape,
            })
            .unwrap();
        assert_eq!(
            block_on(provider.classify_real_power(&base, &exponent)).unwrap(),
            domain
        );
        assert_eq!(snapshot_handle_metadata(&base), before);
        if domain == PowerResultDomain::Real {
            let mut plan = plan(
                vec![FusionOp::Primitive {
                    op: PrimitiveOp::ElemPow,
                    inputs: vec![0, 1],
                    output: Some(2),
                }],
                2,
            );
            plan.inputs = vec![0, 1];
            plan.stack_pattern = vec![0, 1];
            plan.group.shape = ShapeInfo::Tensor(vec![Some(3), Some(2)]);
            let result = block_on(execute_elementwise(FusionExecutionRequest {
                plan: &plan,
                inputs: vec![
                    Value::GpuTensor(base.clone()),
                    Value::GpuTensor(exponent.clone()),
                ],
                placement: None,
                runtime: None,
            }))
            .unwrap();
            let Value::GpuTensor(output) = result.final_value else {
                panic!("resident exact power")
            };
            let actual = block_on(provider.download(&output)).unwrap();
            assert_eq!(actual.shape, vec![3, 2]);
            for (actual, expected) in actual.data.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-4 * expected.abs().max(1.0));
            }
            provider.free(&output).unwrap();
        }
        let mut foreign = exponent.clone();
        foreign.device_id ^= 0x8000_0000;
        assert!(block_on(provider.classify_real_power(&base, &foreign)).is_err());
        let mut wrong_class = exponent.clone();
        wrong_class.descriptor.element_type = Some(NumericElementType::I32);
        assert!(block_on(provider.classify_real_power(&base, &wrong_class)).is_err());
        runmat_accelerate_api::record_handle_transpose(&base, 3, 2);
        assert!(block_on(provider.classify_real_power(&base, &exponent)).is_err());
        runmat_accelerate_api::restore_handle_metadata(&base, &before);
        provider.free(&exponent).unwrap();
    }
    provider.free(&source).unwrap();
    assert_eq!(snapshot_handle_metadata(&base), before);
    provider.free(&base).unwrap();
}

#![cfg(feature = "wgpu")]
//! Qualification tests require a real adapter. No missing-adapter early return
//! is counted as a pass; F64-specific coverage is an explicit separate gate.
#[path = "support/typed_fusion_plan.rs"]
mod fixture;
#[path = "support/retirement.rs"]
mod retirement;
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
    assert_eq!(output.shape(), shape);
    assert_eq!(
        output.descriptor().element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(output.descriptor().storage, Some(GpuTensorStorage::Real));
    let downloaded = block_on(provider.download_numeric(&output)).unwrap();
    let HostNumericDataOwned::F32(actual) = downloaded.data else {
        panic!("physical F32 payload required")
    };
    same_f32(
        &actual,
        &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
    );
    retirement::release_all([output]);
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
    let Value::GpuTensor(single_output) = result
        .materialized_stores
        .into_iter()
        .next()
        .expect("Single materialized store")
        .1
    else {
        panic!("resident Single")
    };
    assert_eq!(
        single_output.descriptor().element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(
        final_output.descriptor().element_type,
        Some(NumericElementType::F64)
    );
    assert_eq!(single_output.shape(), shape);
    assert_eq!(final_output.shape(), shape);
    let HostNumericDataOwned::F32(single) = block_on(provider.download_numeric(&single_output))
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
    assert_eq!(final_values.len(), values.len());
    for (actual, input) in final_values.iter().zip(&values) {
        let rounded = *input as f32;
        let expected = f64::from(rounded + rounded);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    retirement::release_all([source, direct_single, single_output, final_output]);
}

fn transposed_fusion(
    provider: &'static runmat_accelerate::backend::wgpu::provider::WgpuProvider,
    source: runmat_accelerate_api::GpuTensorHandle,
) {
    let transposed = provider.transpose(&source).unwrap();
    assert_eq!(transposed.shape(), vec![3, 2]);
    assert_eq!(
        transposed.descriptor().element_type,
        source.descriptor().element_type
    );
    assert!(runmat_accelerate_api::handle_is_transposed(&transposed));
    assert!(source.shares_allocation_with(&transposed));
    assert_ne!(source.view_identity(), transposed.view_identity());
    let source_weak = source.allocation_lease().downgrade();
    assert_eq!(
        source.release().unwrap(),
        runmat_accelerate_api::allocation::AllocationReleaseOutcome::ReferenceReleased
    );
    assert!(source_weak.acquire().is_some()); // The admitted transpose view retains the allocation.
    let before = transposed.wire_descriptor();
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
    assert_eq!(transposed.wire_descriptor(), before);
    let Value::GpuTensor(output) = result.final_value else {
        panic!("resident materialized result")
    };
    assert_eq!(output.shape(), vec![3, 2]);
    assert_eq!(
        output.descriptor().element_type,
        Some(NumericElementType::F32)
    );
    assert!(!runmat_accelerate_api::handle_is_transposed(&output));
    let HostNumericDataOwned::F32(actual) =
        block_on(provider.download_numeric(&output)).unwrap().data
    else {
        panic!("physical Single output")
    };
    same_f32(&actual, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    let output_weak = output.allocation_lease().downgrade();
    output.release().unwrap();
    let Value::GpuTensor(stored) = result
        .materialized_stores
        .into_iter()
        .next()
        .expect("second materialized store")
        .1
    else {
        panic!("resident second fused output")
    };
    assert_eq!(stored.shape(), vec![3, 2]);
    let HostNumericDataOwned::F32(actual) =
        block_on(provider.download_numeric(&stored)).unwrap().data
    else {
        panic!("physical Single stored output")
    };
    same_f32(&actual, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    retirement::release_all([stored]);
    assert!(output_weak.acquire().is_none());
    let materialized = provider.reshape(&transposed, &[3, 2]).unwrap();
    assert_eq!(
        materialized.descriptor().element_type,
        transposed.descriptor().element_type
    );
    assert!(!runmat_accelerate_api::handle_is_transposed(&materialized));
    assert!(!materialized.shares_allocation_with(&transposed));
    retirement::release_all([transposed]);
    assert!(source_weak.acquire().is_none());
    let actual = block_on(provider.download_numeric(&materialized)).unwrap();
    match actual.data {
        HostNumericDataOwned::F32(values) => same_f32(&values, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
        HostNumericDataOwned::F64(values) => assert_eq!(values, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
        _ => panic!("preserved floating storage"),
    }
    retirement::release_all([materialized]);
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
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual WGPU adapter required");
    transposed_power(provider, false);
}

#[test]
#[ignore = "requires an actual F64 WGPU adapter; run explicitly during F64 qualification"]
fn actual_f64_adapter_classifies_typed_f32_views_without_widening_inputs() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("actual F64 WGPU adapter required");
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    transposed_power(provider, true);
}

fn transposed_power(
    provider: &'static runmat_accelerate::backend::wgpu::provider::WgpuProvider,
    physical_single: bool,
) {
    use runmat_accelerate_api::PowerResultDomain;
    let upload = |data: &[f64], shape: &[usize]| {
        if physical_single {
            let data: Vec<f32> = data.iter().map(|value| *value as f32).collect();
            runmat_accelerate_api::upload_numeric_protecting(
                provider,
                &runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F32(&data),
                    shape,
                    storage: GpuTensorStorage::Real,
                },
                &[],
            )
            .unwrap()
        } else {
            provider.upload(&HostTensorView { data, shape }).unwrap()
        }
    };
    let source = upload(&[4., -9., 16., 25., 36., 49.], &[2, 3]);
    let base = provider.transpose(&source).unwrap();
    let before = base.wire_descriptor();
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
        let exponent = upload(&exponents, &shape);
        assert_eq!(
            block_on(provider.classify_real_power(&base, &exponent)).unwrap(),
            domain
        );
        assert_eq!(base.wire_descriptor(), before);
        if physical_single {
            let error = block_on(provider.real_power_in_domain(&base, &exponent, domain))
                .expect_err("classifier support does not widen standalone power execution");
            assert!(runmat_accelerate_api::is_unsupported_provider_operation(
                &error
            ));
            let native_exponent = provider
                .upload(&HostTensorView {
                    data: &exponents,
                    shape: &shape,
                })
                .unwrap();
            let error = block_on(provider.classify_real_power(&base, &native_exponent))
                .expect_err("mixed operand precision is not an admitted pair");
            assert!(runmat_accelerate_api::is_unsupported_provider_operation(
                &error
            ));
            retirement::release_all([native_exponent]);
        }
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
            if physical_single {
                assert_eq!(
                    output.descriptor().element_type,
                    Some(NumericElementType::F32)
                );
            }
            let actual = block_on(provider.download(&output)).unwrap();
            assert_eq!(actual.shape, vec![3, 2]);
            assert_eq!(actual.data.len(), expected.len());
            for (actual, expected) in actual.data.iter().zip(expected) {
                assert!((actual - expected).abs() <= 1e-4 * expected.abs().max(1.0));
            }
            retirement::release_all([output]);
        }
        let foreign = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                shape.clone(),
                exponent.physical().element_type(),
                GpuTensorStorage::Real,
            )
            .unwrap();
        assert_ne!(foreign.handle.device_id(), exponent.device_id());
        assert!(block_on(provider.classify_real_power(&base, &foreign.handle)).is_err());
        retirement::release_all([foreign.handle]);
        assert_eq!(foreign.probe.retirements(), 1);
        let integers = vec![2_i32; exponents.len()];
        let wrong_class = runmat_accelerate_api::upload_numeric_protecting(
            provider,
            &runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::I32(&integers),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            },
            &[&base, &exponent],
        )
        .expect("genuine same-provider integer payload");
        assert_eq!(wrong_class.device_id(), base.device_id());
        assert_eq!(
            wrong_class.physical().element_type(),
            NumericElementType::I32
        );
        assert!(block_on(provider.classify_real_power(&base, &wrong_class)).is_err());
        retirement::release_all([wrong_class]);
        let malformed = base.view().clone().with_layout(
            runmat_accelerate_api::allocation::GpuTensorLayout::Transpose(
                runmat_accelerate_api::TransposeInfo {
                    base_rows: 3,
                    base_cols: 2,
                },
            ),
        );
        assert!(
            base.with_view(malformed).is_err(),
            "inconsistent transpose must fail admission"
        );
        assert_eq!(base.wire_descriptor(), before);
        assert_eq!(
            block_on(provider.classify_real_power(&base, &exponent)).unwrap(),
            domain
        );
        retirement::release_all([exponent]);
    }
    let source_weak = source.allocation_lease().downgrade();
    assert!(source.shares_allocation_with(&base));
    source.release().unwrap();
    assert!(source_weak.acquire().is_some());
    assert_eq!(base.wire_descriptor(), before);
    retirement::release_all([base]);
    assert!(source_weak.acquire().is_none());
    for (bases, exponents, domain) in [
        (
            vec![-1e-30, -0.0, f64::NAN],
            vec![0.5, 3.0, 0.0],
            PowerResultDomain::Complex,
        ),
        (
            vec![-1.0, f64::NAN, 1.0],
            vec![16_777_217.0, 0.0, f64::NAN],
            PowerResultDomain::Real,
        ),
        (vec![], vec![], PowerResultDomain::Real),
    ] {
        let shape = [1, bases.len()];
        let base = upload(&bases, &shape);
        let exponent = upload(&exponents, &shape);
        assert_eq!(
            block_on(provider.classify_real_power(&base, &exponent)).unwrap(),
            domain
        );
        retirement::release_all([base]);
        retirement::release_all([exponent]);
    }
}

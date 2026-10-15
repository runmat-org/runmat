//! Layout qualification uses the real lazy view and the real provider recipe.
use super::*;
use runmat_accelerate_api::power_service::{
    execute_resident_scalar_power, plan_resident_scalar_power, plan_scalar_power_for_dense_output,
    prepare_resident_scalar_power, prepare_resident_scalar_power_against_plan,
};
use runmat_accelerate_api::{
    ExactSingleBasePower, PowerResultDomain, PowerScalarControl, ProviderFeasibility,
    ProviderOperationIdentity, RealPowerOperand,
};

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual WGPU adapter; qualifies exact POWER lazy transpose and zero-work allocation"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn certified_power_preserves_live_layout_and_empty_backing_authority() {
    let provider = owner().await;
    let values = [
        f32::from_bits(1),
        -0.0,
        -2.0,
        f32::from_bits(0x1fffffff),
        f32::INFINITY,
        f32::NAN,
    ];
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let transpose = provider.transpose(&input).unwrap();
    assert!(transpose.shares_allocation_with(&input));
    assert!(runmat_accelerate_api::handle_is_transposed(&transpose));
    let empty = input
        .with_view(input.view().clone().with_shape(vec![0, 3]))
        .unwrap();
    assert_eq!(
        empty.physical(),
        input.physical(),
        "empty view retains actual source backing"
    );
    assert!(provider.allocations.validate_empty_view(&empty).is_ok());
    assert!(
        provider.get_entry_raw(&empty).is_err(),
        "zero carrier must not expose retained physical length to generic kernels"
    );
    assert!(
        provider.download_numeric(&empty).await.is_err(),
        "ordinary transfer must not read retained backing through an empty view"
    );
    assert!(
        input
            .with_view(input.view().clone().with_shape(vec![1, 3]))
            .is_err(),
        "nonempty prefix carrier guard remains unchanged"
    );
    assert!(
        input
            .with_view(
                input
                    .view()
                    .clone()
                    .with_shape(vec![0, 3])
                    .with_logical(true)
            )
            .is_err(),
        "empty shape cannot forge logical producer authority"
    );
    assert!(provider.get_entry_raw(&input).is_ok());
    for source in [&input, &empty] {
        let representation = source_representation(source).unwrap();
        let projected = provider
            .plan_dense_real_output_buffer_bytes(&representation)
            .unwrap()
            .unwrap();
        let control = PowerScalarControl::DoubleBits(2.0_f64.to_bits());
        let operation =
            ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name);
        let geometry = plan_resident_scalar_power(
            provider,
            representation.clone(),
            control,
            RealPowerOperand::Base,
            &[1, 1],
            operation.clone(),
        )
        .unwrap()
        .unwrap();
        assert!(
            geometry
                .feasibility_query(PowerResultDomain::Real)
                .unwrap()
                .workload
                .real_power_scalar
                .unwrap()
                .input
                .is_none(),
            "geometry-only projection stays available without fabricating backing"
        );
        let planned = plan_scalar_power_for_dense_output(
            provider,
            representation,
            control,
            RealPowerOperand::Base,
            &[1, 1],
            operation.clone(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            provider
                .plan_power_output_buffer_bytes(&planned, PowerResultDomain::Real)
                .unwrap(),
            Some(projected)
        );
        let bound = prepare_resident_scalar_power_against_plan(
            provider,
            source,
            &planned,
            control,
            RealPowerOperand::Base,
            &[1, 1],
            operation,
        );
        assert_eq!(
            bound.is_ok(),
            source.physical().requested_bytes() == projected,
            "future allocator extent must match actual live backing, not only shape"
        );
    }
    for source in [&transpose, &empty] {
        for (exponent, operation) in [
            (0.0_f64, ExactSingleBasePower::Zero),
            (-0.0, ExactSingleBasePower::Zero),
            (1.0, ExactSingleBasePower::Identity),
            (2.0, ExactSingleBasePower::Square),
        ] {
            let control = PowerScalarControl::DoubleBits(exponent.to_bits());
            let prepared = prepare_resident_scalar_power(
                provider,
                source,
                control,
                RealPowerOperand::Base,
                &[1, 1],
                ProviderOperationIdentity::new(runmat_builtins::POWER_CATALOG_ENTRY.identity.name),
            )
            .unwrap()
            .expect("certified actual view must have an owned recipe");
            assert_eq!(prepared.control(), control);
            let query = prepared
                .symbolic_plan()
                .feasibility_query(PowerResultDomain::Real)
                .unwrap();
            let description = query
                .workload
                .real_power_scalar
                .as_ref()
                .unwrap()
                .input
                .as_ref()
                .unwrap();
            let runmat_accelerate_api::PowerInputDescription::Live { physical, view } = description
            else {
                panic!("live preparation must bind authentic input facts");
            };
            assert_eq!(physical, source.physical());
            assert_eq!(view, source.view());
            let mut hypothetical = query.clone();
            hypothetical
                .workload
                .real_power_scalar
                .as_mut()
                .unwrap()
                .input = None;
            match provider.query_feasibility(&hypothetical) {
                ProviderFeasibility::Supported { estimate } => assert!(
                    estimate.transient_bytes.is_none(),
                    "shape-only planning cannot fabricate retained backing extent"
                ),
                ProviderFeasibility::Rejected { .. } => {}
            }
            if source.shape()[0] == 0 {
                let ProviderFeasibility::Supported { estimate } =
                    provider.query_feasibility(&query)
                else {
                    panic!("known retained backing must be costable");
                };
                assert_eq!(estimate.output_bytes, Some(4));
                assert_eq!(estimate.dispatches, Some(0));
                assert!(
                    estimate.transient_bytes.unwrap() >= input.physical().requested_bytes() + 4
                );
                let request = ProviderRealElementwiseRequest {
                    operation: Operation::ExactSingleBasePower { control },
                    source: source_representation(source).unwrap(),
                    source_buffer_bytes: source.physical().requested_bytes(),
                };
                let recipe = ProviderRealElementwiseRecipe::plan(provider, request)
                    .unwrap()
                    .unwrap();
                let physical = recipe.physical();
                assert_eq!(physical.output_buffer_bytes, 4);
                assert_eq!(physical.dispatches, 0);
                assert_eq!(physical.queue_submissions, 0);
                assert_eq!(physical.uniform_buffer_bytes, 0);
                assert!(physical.pipeline_key.is_none());
            }
            let output = execute_resident_scalar_power(prepared).await.unwrap();
            assert_eq!(output.handle.shape(), source.shape());
            assert!(!runmat_accelerate_api::handle_is_transposed(&output.handle));
            assert!(!output.handle.shares_allocation_with(source));
            let HostNumericDataOwned::F32(actual) = provider
                .download_numeric(&output.handle)
                .await
                .unwrap()
                .data
            else {
                panic!("Single publication");
            };
            let order: &[usize] = if source.shape()[0] == 0 {
                &[]
            } else {
                &[0, 2, 4, 1, 3, 5]
            };
            assert_eq!(actual.len(), order.len());
            for (&actual, &index) in actual.iter().zip(order) {
                let expected = operation.evaluate(values[index]);
                if expected.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "layout index {index}, exponent {exponent}"
                    );
                }
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual WGPU adapter; qualifies receipt-based ordinary POWER fusion"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn certified_power_ordinary_fusion_executes_words_on_uniform_data_abi() {
    qualify_ordinary_fusion(owner().await).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual F64 WGPU owner; qualifies F32 input/output ABI without widening or publication adapters"]
async fn f64_owner_preserves_binary32_words_in_single_and_multi_fusion() {
    let provider = owner().await;
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64,
        "actual F64 owner is required, never skipped or substituted"
    );
    let data = [1.0_f64 + f64::EPSILON, 2.0];
    let double = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F64(&data),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let inputs = [double.clone()];
    let shader = crate::backend::wgpu::shaders::exact_power::shader_for_layout(
        ExactSingleBasePower::Identity,
        true,
    );
    let abi = runmat_accelerate_api::ProviderFusedElementwiseAbi::BroadcastReal {
        precision: runmat_accelerate_api::ProviderPrecision::F32,
    };
    let before = provider.allocations.record_count().unwrap();
    for classes in [
        &[runmat_accelerate_api::ProviderElementType::F32][..],
        &[runmat_accelerate_api::ProviderElementType::F32; 2][..],
    ] {
        let request = runmat_accelerate_api::ProviderFusedElementwiseRequest {
            shader: &shader,
            requirements: runmat_accelerate_api::ProviderShaderRequirements::Basic,
            inputs: &inputs,
            output_shape: &[1, 2],
            len: 2,
            output_classes: classes,
            abi,
        };
        if classes.len() == 1 {
            assert!(provider.fused_elementwise_typed(request).await.is_err());
        } else {
            assert!(provider
                .fused_elementwise_multi_typed(request)
                .await
                .is_err());
        }
        assert_eq!(
            provider.allocations.record_count().unwrap(),
            before,
            "wrong physical Double data must reject before output allocation"
        );
        double.allocation_lease().validate_live().unwrap();
    }
    qualify_ordinary_fusion(provider).await;
}

async fn qualify_ordinary_fusion(provider: &'static WgpuProvider) {
    use crate::fusion::{FusionGroup, FusionGroupPlan, FusionKernelSpec, FusionKind, FusionOp};
    use crate::fusion_exec::{
        execute_elementwise_placed, FusionExecutionOutcome, FusionExecutionRequest,
    };
    use crate::graph::{InstrSpan, PrimitiveOp, ShapeInfo};
    use runmat_accelerate_api::ThreadProviderGuard;
    use runmat_value::Value;
    use std::collections::HashMap;

    let _selected = ThreadProviderGuard::set(Some(provider));
    let values = [f32::from_bits(1), -0.0, -2.0, f32::from_bits(0x1fffffff)];
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let transposed = provider.transpose(&input).unwrap();
    let mut source = FusionGroupPlan {
        index: 0,
        group: FusionGroup {
            id: 0,
            kind: FusionKind::ElementwiseChain,
            nodes: vec![],
            shape: ShapeInfo::Tensor(vec![Some(2), Some(2)]),
            span: InstrSpan { start: 0, end: 3 },
            pattern: None,
            stack_layout: None,
        },
        operations: vec![
            FusionOp::Primitive {
                op: PrimitiveOp::ElemPow,
                inputs: vec![0, 1],
                output: Some(2),
            },
            FusionOp::Primitive {
                op: PrimitiveOp::UPlus,
                inputs: vec![2],
                output: Some(3),
            },
        ],
        inputs: vec![0, 1],
        stack_pattern: vec![0, 1],
        constants: HashMap::new(),
        const_values: HashMap::new(),
        materialized_stores: vec![],
        output: Some(3),
        kernel: FusionKernelSpec {
            kind: FusionKind::ElementwiseChain,
            supported: true,
        },
        reduction_data: None,
        reduction_dim: None,
        reduction_flavor: None,
        reduction_axes: None,
        pattern: None,
    };
    for multi in [false, true] {
        source.materialized_stores = if multi {
            vec![crate::fusion::FusionStoreMaterialization {
                value_id: 0,
                binding: crate::graph::VarBinding {
                    kind: crate::graph::VarKind::Global,
                    index: 0,
                },
            }]
        } else {
            vec![]
        };
        for handle in [&input, &transposed] {
            for (exponent, operation) in [
                (0.0_f64, ExactSingleBasePower::Zero),
                (-0.0, ExactSingleBasePower::Zero),
                (1.0, ExactSingleBasePower::Identity),
                (2.0, ExactSingleBasePower::Square),
            ] {
                let result = execute_elementwise_placed(FusionExecutionRequest {
                    plan: &source,
                    inputs: vec![Value::GpuTensor(handle.clone()), Value::Num(exponent)],
                    placement: None,
                    runtime: None,
                })
                .await
                .unwrap();
                let FusionExecutionOutcome::Executed(result) = result else {
                    panic!("ordinary scalar-expression fusion must execute with binary32 data ABI");
                };
                let Value::GpuTensor(output) = result.final_value else {
                    panic!("resident fused result");
                };
                assert_eq!(
                    output.physical().element_type(),
                    runmat_accelerate_api::NumericElementType::F32
                );
                assert_eq!(output.physical().requested_bytes(), 16);
                assert!(!output.shares_allocation_with(handle));
                assert_eq!(result.materialized_stores.len(), usize::from(multi));
                for (_, stored) in result.materialized_stores {
                    let Value::GpuTensor(stored) = stored else {
                        panic!("resident stored result");
                    };
                    assert_eq!(
                        stored.physical().element_type(),
                        runmat_accelerate_api::NumericElementType::F32
                    );
                    assert!(!stored.shares_allocation_with(handle));
                }
                let HostNumericDataOwned::F32(actual) =
                    provider.download_numeric(&output).await.unwrap().data
                else {
                    panic!("Single result");
                };
                let order = if runmat_accelerate_api::handle_is_transposed(handle) {
                    [0, 2, 1, 3]
                } else {
                    [0, 1, 2, 3]
                };
                for (&actual, index) in actual.iter().zip(order) {
                    assert_eq!(
                        actual.to_bits(),
                        operation.evaluate(values[index]).to_bits()
                    );
                }
            }
        }
    }
}

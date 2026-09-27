#![cfg(feature = "wgpu")]
#[path = "support/typed_fusion_plan.rs"]
mod fixture;
use runmat_accelerate::backend::wgpu::provider::{
    register_wgpu_provider, WgpuProvider, WgpuProviderOptions,
};
use runmat_accelerate::fusion_exec::{execute_elementwise, FusionExecutionRequest};
use runmat_accelerate::graph::ShapeInfo;
use runmat_accelerate_api::{
    allocation::AllocationReleaseOutcome, AccelProvider, GpuTensorStorage, HostNumericDataOwned,
    HostTensorView, NumericElementType, ProviderPrecision,
};
use runmat_value::Value;

#[tokio::test]
#[ignore = "requires an actual compatible WGPU adapter"]
async fn round_hardware_preserves_integral_values_and_rounds_ties_away() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("rounding qualification requires a compatible WGPU adapter");
    check_direct_and_fused_round(provider).await;
}

#[tokio::test]
#[ignore = "requires an actual F64 WGPU adapter; run explicitly during F64 qualification"]
async fn round_hardware_f64_direct_and_fused_share_represented_value_semantics() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default())
        .expect("rounding qualification requires an actual F64 WGPU adapter");
    assert_eq!(
        provider.precision(),
        ProviderPrecision::F64,
        "an F32 adapter does not satisfy this F64 gate"
    );
    check_direct_and_fused_round(provider).await;
}

async fn check_direct_and_fused_round(provider: &'static WgpuProvider) {
    let large = if provider.precision() == ProviderPrecision::F32 {
        8_388_609.0
    } else {
        4_503_599_627_370_497.0
    };
    let source = [
        0.5,
        -0.5,
        2.5,
        -2.5,
        large,
        -large,
        0.0,
        -0.0,
        -0.25,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let expected = [
        1.0,
        -1.0,
        3.0,
        -3.0,
        large,
        -large,
        0.0,
        -0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let input = provider
        .upload(&HostTensorView {
            data: &source,
            shape: &[2, 2, 3],
        })
        .unwrap();
    let metadata = input.wire_descriptor();
    let input_view = input.view_identity();
    let output = provider.unary_round(&input).await.unwrap();
    let mut plan = fixture::plan(vec![fixture::cast("round", 0, 2)], 2);
    plan.group.shape = ShapeInfo::Tensor(vec![Some(2), Some(2), Some(3)]);
    let result = execute_elementwise(FusionExecutionRequest {
        plan: &plan,
        inputs: vec![Value::GpuTensor(input.clone())],
        placement: None,
        runtime: None,
    })
    .await
    .expect("actual fused Round execution");
    assert!(result.materialized_stores.is_empty());
    let Value::GpuTensor(fused) = result.final_value else {
        panic!("fused Round must publish a resident result");
    };
    assert_eq!(input.wire_descriptor(), metadata);
    assert_eq!(input.view_identity(), input_view);
    assert_ne!(output.allocation_identity(), fused.allocation_identity());
    assert!(!output.shares_allocation_with(&fused));
    for handle in [&output, &fused] {
        assert_eq!(handle.shape(), input.shape());
        assert_eq!(handle.device_id(), provider.device_id());
        assert_ne!(handle.allocation_identity(), input.allocation_identity());
        assert!(!handle.shares_allocation_with(&input));
        assert_eq!(handle.physical().storage(), GpuTensorStorage::Real);
        let downloaded = provider.download_numeric(handle).await.unwrap();
        assert_eq!(downloaded.shape, input.shape());
        let actual = match (provider.precision(), downloaded.data) {
            (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
                assert_eq!(handle.physical().element_type(), NumericElementType::F32);
                values.into_iter().map(f64::from).collect::<Vec<_>>()
            }
            (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => {
                assert_eq!(handle.physical().element_type(), NumericElementType::F64);
                values
            }
            _ => panic!("round output must retain the actual provider-native floating storage"),
        };
        assert_eq!(
            actual.len(),
            expected.len(),
            "no truncated download may satisfy the oracle"
        );
        for (actual, expected) in actual.into_iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }
    for handle in [output, fused] {
        let weak = handle.allocation_lease().downgrade();
        let alias = handle.clone();
        assert_eq!(
            handle.release().unwrap(),
            AllocationReleaseOutcome::ReferenceReleased
        );
        assert!(weak.acquire().is_some());
        provider.download(&alias).await.unwrap();
        assert_eq!(alias.release().unwrap(), AllocationReleaseOutcome::Retired);
        assert!(weak.acquire().is_none());
    }
    let downloaded = provider.download(&input).await.unwrap();
    assert_eq!(downloaded.data.len(), source.len());
    for (actual, expected) in downloaded.data.iter().zip(source) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    let weak = input.allocation_lease().downgrade();
    assert_eq!(input.release().unwrap(), AllocationReleaseOutcome::Retired);
    assert!(weak.acquire().is_none());
}

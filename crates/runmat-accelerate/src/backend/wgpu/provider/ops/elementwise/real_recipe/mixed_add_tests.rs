//! Real device witnesses; incapable/F32 browser is a pre-effect decline only.
use super::*;
use runmat_accelerate_api::{ProviderMixedRealAddRecipe, ProviderPrecision};

fn request(input: &GpuTensorHandle, scalar: f64) -> ProviderRealElementwiseRequest {
    ProviderRealElementwiseRequest {
        operation: Operation::ScalarAdd {
            control: ScalarLiteral::F64Bits(scalar.to_bits()),
            mode: runmat_types::RealFloatingAddMode::MixedDoubleThenSingle,
        },
        source: source_representation(input).unwrap(),
        source_buffer_bytes: input.physical().requested_bytes(),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual F32 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f32"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn no_f64_mixed_service_declines_before_any_allocation_or_submission() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, -0.0]),
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let before = provider.allocations.record_count().unwrap();
    assert!(
        ProviderMixedRealAddRecipe::plan(provider, request(&input, 1.0))
            .unwrap()
            .is_none()
    );
    assert_eq!(provider.allocations.record_count().unwrap(), before);
    input.allocation_lease().validate_live().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual shader-f64 adapter; RUNMAT_WGPU_FORCE_PRECISION=f64"]
async fn actual_mixed_three_recipe_add_preserves_tie_sides_bits_class_and_warm_resources() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    let below = f32::from_bits(0x3f7f_ffff);
    let values = [below, 1.0, -0.0, f32::INFINITY, f32::NEG_INFINITY];
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[values.len(), 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let half = f64::from(f32::EPSILON) / 4.0;
    for scalar in [
        half - f64::EPSILON,
        half + f64::EPSILON,
        -0.0,
        f64::INFINITY,
        f64::from_bits(0x7ff8_0000_0000_1234),
    ] {
        let recipe = ProviderMixedRealAddRecipe::plan(provider, request(&input, scalar))
            .unwrap()
            .unwrap();
        let cold = recipe.execute(&input).await.unwrap();
        let warm = recipe
            .execute_protected(&input, &[&cold.output])
            .await
            .unwrap();
        assert_eq!(cold.observed, warm.observed);
        assert_eq!(cold.observed.resources, *recipe.resources());
        assert_eq!(cold.observed.steps.len(), 3);
        assert_eq!(
            cold.output.physical().element_type(),
            runmat_accelerate_api::NumericElementType::F32
        );
        assert_eq!(cold.output.shape(), input.shape());
        let HostNumericDataOwned::F32(output) =
            provider.download_numeric(&cold.output).await.unwrap().data
        else {
            panic!("mixed result must be Single");
        };
        assert_eq!(output.len(), values.len());
        for (input, actual) in values.iter().zip(output) {
            let expected = (f64::from(*input) + scalar) as f32;
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        assert!(!cold.output.shares_allocation_with(&input));
        input.allocation_lease().validate_live().unwrap();
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual shader-f64 adapter; RUNMAT_WGPU_FORCE_PRECISION=f64"]
async fn actual_double_source_single_control_add_uses_two_owner_recipes_and_single_output() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    let half = f64::from(f32::EPSILON) / 2.0;
    let values = [
        1.0 + half - f64::EPSILON,
        1.0 + half + f64::EPSILON,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F64(&values),
            shape: &[values.len(), 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    for control in [
        0.0_f32,
        -0.0,
        f32::from_bits(1),
        1.0,
        f32::INFINITY,
        f32::from_bits(0x7fc0_1234),
    ] {
        let original = ProviderRealElementwiseRequest {
            operation: Operation::ScalarAdd {
                control: ScalarLiteral::F32Bits(control.to_bits()),
                mode: runmat_types::RealFloatingAddMode::MixedDoubleThenSingle,
            },
            source: source_representation(&input).unwrap(),
            source_buffer_bytes: input.physical().requested_bytes(),
        };
        let recipe = ProviderMixedRealAddRecipe::plan(provider, original.clone())
            .unwrap()
            .unwrap();
        assert_eq!(*recipe.request(), original);
        let cold = recipe.execute(&input).await.unwrap();
        let warm = recipe
            .execute_protected(&input, &[&cold.output])
            .await
            .unwrap();
        assert_eq!(cold.observed, warm.observed);
        assert_eq!(cold.observed.resources, *recipe.resources());
        assert!(matches!(
            cold.observed.steps,
            runmat_accelerate_api::ProviderMixedRealAddStepObservations::AddNarrow(_)
        ));
        assert_eq!(cold.output.shape(), input.shape());
        assert_eq!(
            cold.output.physical().element_type(),
            runmat_accelerate_api::NumericElementType::F32
        );
        assert!(!cold.output.shares_allocation_with(&input));
        let HostNumericDataOwned::F32(output) =
            provider.download_numeric(&cold.output).await.unwrap().data
        else {
            panic!("mixed Double source must publish Single");
        };
        assert_eq!(output.len(), values.len());
        for (&source, actual) in values.iter().zip(output) {
            let expected = (source + f64::from(control)) as f32;
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        input.allocation_lease().validate_live().unwrap();
        cold.output.allocation_lease().validate_live().unwrap();
    }
}

//! Actual allocation and ordinary execution qualification, never a mock timing
//! result. Native adapter tests are explicit opt-in; browser uses real WebGPU.
#[path = "real_recipe/mixed_add_tests.rs"]
mod mixed_add;
#[path = "real_recipe/cast_tests.rs"]
mod casts;
#[path = "real_recipe/exact_layout_tests.rs"]
mod exact_layouts;
#[path = "real_recipe/exponential_tests.rs"]
mod exponentials;
use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
};

async fn owner() -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let provider =
        WgpuProvider::new(WgpuProviderOptions::default()).expect("actual WGPU adapter required");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(WgpuProviderOptions::default())
        .await
        .expect("actual WebGPU adapter required");
    let provider = Box::leak(Box::new(provider));
    // SAFETY: process-lifetime facade and uniquely minted actual WGPU device.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual WGPU adapter; qualifies integer-word POWER, including FTZ boundaries"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn exact_double_scalar_power_family_matches_independent_binary64_oracle() {
    use runmat_accelerate_api::power_service::{
        execute_resident_scalar_power, prepare_resident_scalar_power, PowerPhysicalRoute,
    };
    use runmat_accelerate_api::{
        ExactSingleBasePower as Exact, PowerScalarControl, RealPowerArithmeticMode,
        RealPowerOperand,
    };
    let provider = owner().await;
    let mut bits = Vec::new();
    // Every exponent field, both signs, and mantissa carry/normalization
    // boundaries; includes NaNs, infinities and subnormals without float ops.
    for exponent in 0..=255_u32 {
        for fraction in [0, 1, 0x3fffff, 0x400000, 0x400001, 0x7ffffe, 0x7fffff] {
            for sign in [0, 0x80000000] {
                bits.push(sign | (exponent << 23) | fraction);
            }
        }
    }
    for center in [
        0x19ffffff_u32,
        0x1a000000,
        0x1fffffff,
        0x20000000,
        0x5f7fffff,
        0x5f800000,
    ] {
        for delta in 0..=64 {
            bits.push(center - 32 + delta);
        }
    }
    let mut state = 0x71c38e29_u32;
    for _ in 0..4096 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        bits.push(state);
    }
    let values = bits
        .iter()
        .map(|&bits| f32::from_bits(bits))
        .collect::<Vec<_>>();
    for shape in [
        vec![values.len(), 1],
        vec![1, values.len()],
        vec![values.len() / 2, 1, 2],
    ] {
        let input = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F32(&values),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        for (exponent, operation) in [
            (0.0_f64, Exact::Zero),
            (-0.0, Exact::Zero),
            (1.0, Exact::Identity),
            (2.0, Exact::Square),
        ] {
            let control = PowerScalarControl::DoubleBits(exponent.to_bits());
            for _ in 0..2 {
                let prepared = prepare_resident_scalar_power(
                    provider,
                    &input,
                    control,
                    RealPowerOperand::Base,
                    &[1, 1],
                    runmat_accelerate_api::ProviderOperationIdentity::new(
                        runmat_builtins::POWER_CATALOG_ENTRY.identity.name,
                    ),
                )
                .unwrap()
                .expect("actual provider must propose certified route");
                assert_eq!(prepared.control(), control);
                assert_eq!(
                    prepared.mode(),
                    RealPowerArithmeticMode::MixedDoubleThenSingle {
                        single_operand: RealPowerOperand::Base
                    }
                );
                assert_eq!(
                    prepared.physical_recipe().route,
                    PowerPhysicalRoute::ExactSingleBase
                );
                assert!(
                    !prepared
                        .symbolic_plan()
                        .physical_work(runmat_accelerate_api::PowerResultDomain::Real)
                        .unwrap()
                        .paired_domain_probe
                );
                let output = execute_resident_scalar_power(prepared).await.unwrap();
                assert_eq!(output.handle.shape(), shape);
                assert!(!output.handle.shares_allocation_with(&input));
                let HostNumericDataOwned::F32(actual) = provider
                    .download_numeric(&output.handle)
                    .await
                    .unwrap()
                    .data
                else {
                    panic!("Single publication");
                };
                for ((&base, &actual), &base_bits) in values.iter().zip(&actual).zip(&bits) {
                    let expected = operation.evaluate(base);
                    if expected.is_nan() {
                        assert!(actual.is_nan(), "{base_bits:08x}^{exponent}");
                    } else {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{base_bits:08x}^{exponent}"
                        );
                        // Independent platform powf oracle, not a second copy
                        // of the shader's integer multiplication/packing.
                        assert_eq!(
                            expected.to_bits(),
                            (f64::from(base).powf(exponent) as f32).to_bits(),
                            "binary64 powf equivalence {base_bits:08x}^{exponent}"
                        );
                    }
                }
            }
        }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn scalar_recipe_requires_one_enabled_uniform_binding_before_effects() {
    let mut limits = wgpu::Limits::default();
    assert!(scalar_uniform_available(&limits));
    limits.max_uniform_buffers_per_shader_stage = 0;
    assert!(!scalar_uniform_available(&limits));
}

async fn qualify(provider: &'static WgpuProvider, len: usize) {
    exponentials::qualify(provider, false).await;
    let values = vec![4.0_f32; len];
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[len, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let source = source_representation(&input).unwrap();
    let add = ProviderRealElementwiseRecipe::plan(
        provider,
        ProviderRealElementwiseRequest {
            operation: Operation::ScalarAdd {
                control: ScalarLiteral::F32Bits(5.0_f32.to_bits()),
                mode: runmat_types::RealFloatingAddMode::Single,
            },
            source: source.clone(),
            source_buffer_bytes: input.physical().requested_bytes(),
        },
    )
    .unwrap()
    .unwrap();
    let cold = add.execute(&input).await.unwrap();
    let warm = add.execute(&input).await.unwrap();
    assert_eq!(cold.observed, warm.observed);
    let chunks = u32::try_from((len as u64).div_ceil(add.physical().chunk_capacity)).unwrap();
    assert_eq!(cold.observed.dispatches, chunks);
    assert_eq!(cold.observed.queue_submissions, chunks);
    assert_eq!(cold.observed.uniform_allocations, chunks);
    assert_eq!(cold.observed.uniform_buffer_bytes, u64::from(chunks) * 256);
    assert_eq!(
        cold.observed.output_buffer_bytes,
        Some(cold.output.physical().requested_bytes())
    );
    let ordinary_add = provider.scalar_add(&input, 5.0).unwrap();
    assert_eq!(
        ordinary_add.physical().element_type(),
        cold.output.physical().element_type()
    );
    let sqrt = ProviderRealElementwiseRecipe::plan(
        provider,
        ProviderRealElementwiseRequest {
            operation: Operation::Sqrt,
            source,
            source_buffer_bytes: cold.output.physical().requested_bytes(),
        },
    )
    .unwrap()
    .unwrap();
    let first = sqrt.execute(&cold.output).await.unwrap();
    let second = sqrt.execute(&cold.output).await.unwrap();
    assert_eq!(first.observed, second.observed);
    assert_eq!(
        first.observed.queue_submissions,
        first.observed.dispatches + 1,
        "warm pipeline still emits the actual unary warmup"
    );
    assert_eq!(first.observed.uniform_allocations, 1);
    assert_eq!(
        first.observed.uniform_buffer_bytes,
        std::mem::size_of::<crate::backend::wgpu::params::FusionParams>() as u64
    );
    let ordinary_sqrt = provider.unary_sqrt(&cold.output).await.unwrap();
    for handle in [&first.output, &second.output, &ordinary_sqrt] {
        assert!(!handle.shares_allocation_with(&input));
        assert!(!handle.shares_allocation_with(&cold.output));
        assert_eq!(
            provider.download_numeric(handle).await.unwrap().data,
            HostNumericDataOwned::F32(vec![3.0; len])
        );
    }
    input.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual WGPU adapter; run explicitly"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_native_add_sqrt_planned_allocations_uniforms_and_warm_submissions_match() {
    qualify(owner().await, 65).await;
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual WGPU adapter; run explicitly"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn native_class_mul_max_share_ordinary_scalar_physical_execution() {
    let provider = owner().await;
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[4.0; 65]),
            shape: &[65, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let source = source_representation(&input).unwrap();
    let extent = input.physical().requested_bytes();
    for (operation, expected) in [
        (
            Operation::ScalarMul {
                control: ScalarLiteral::F32Bits(2.0_f32.to_bits()),
            },
            8.0_f32,
        ),
        (
            Operation::ScalarMax {
                control: ScalarLiteral::F32Bits(3.0_f32.to_bits()),
            },
            4.0_f32,
        ),
    ] {
        let request = ProviderRealElementwiseRequest {
            operation,
            source: source.clone(),
            source_buffer_bytes: extent,
        };
        let recipe = ProviderRealElementwiseRecipe::plan(provider, request.clone())
            .unwrap()
            .unwrap();
        let result = recipe.execute(&input).await.unwrap();
        assert_eq!(
            result.observed.output_buffer_bytes,
            Some(recipe.physical().output_buffer_bytes)
        );
        assert_eq!(result.observed.dispatches, recipe.physical().dispatches);
        assert_eq!(
            result.observed.queue_submissions,
            recipe.physical().queue_submissions
        );
        assert!(!result.output.shares_allocation_with(&input));
        assert_eq!(
            provider
                .download_numeric(&result.output)
                .await
                .unwrap()
                .data,
            HostNumericDataOwned::F32(vec![expected; 65])
        );
        let wrong_class = ProviderRealElementwiseRequest {
            operation: match operation {
                Operation::ScalarMul { .. } => Operation::ScalarMul {
                    control: ScalarLiteral::F64Bits(2.0_f64.to_bits()),
                },
                Operation::ScalarMax { .. } => Operation::ScalarMax {
                    control: ScalarLiteral::F64Bits(3.0_f64.to_bits()),
                },
                Operation::ScalarAdd { .. }
                | Operation::Sqrt
                | Operation::Spacing
                | Operation::Expm1
                | Operation::UnaryMath(_)
                | Operation::FloatingCast { .. }
                | Operation::ExactSingleBasePower { .. } => {
                    unreachable!("test scalar case")
                }
            },
            ..request
        };
        assert!(ProviderRealElementwiseRecipe::plan(provider, wrong_class)
            .unwrap()
            .is_none());
    }
    input.allocation_lease().validate_live().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual shader-f64 adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
async fn actual_f64_owner_keeps_native_single_recipe_class() {
    let provider = owner().await;
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    qualify(provider, 65).await;
    exponentials::qualify(provider, true).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual shader-f64 adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
async fn actual_double_scalar_multichunk_retains_offset_tail_and_observed_resources() {
    let provider = owner().await;
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let launch = provider
        .real_scalar_pipeline(NumericPrecision::F64)
        .unwrap()
        .scalar_launch()
        .unwrap();
    let len = usize::try_from(launch.chunk_capacity().unwrap() + 1).unwrap();
    let mut values = vec![4.0_f64; len];
    values[len - 1] = 16.0;
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F64(&values),
            shape: &[len, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let recipe = ProviderRealElementwiseRecipe::plan(
        provider,
        ProviderRealElementwiseRequest {
            operation: Operation::ScalarAdd {
                control: ScalarLiteral::F64Bits(5.0_f64.to_bits()),
                mode: runmat_types::RealFloatingAddMode::Double,
            },
            source: source_representation(&input).unwrap(),
            source_buffer_bytes: input.physical().requested_bytes(),
        },
    )
    .unwrap()
    .unwrap();
    let cold = recipe.execute(&input).await.unwrap();
    let warm = recipe.execute(&input).await.unwrap();
    assert_eq!(cold.observed, warm.observed);
    assert_eq!(cold.observed.dispatches, 2);
    assert_eq!(cold.observed.queue_submissions, 2);
    assert_eq!(cold.observed.uniform_allocations, 2);
    assert_eq!(cold.observed.uniform_buffer_bytes, 512);
    values.iter_mut().for_each(|value| *value += 5.0);
    for handle in [
        &cold.output,
        &warm.output,
        &provider.scalar_add(&input, 5.0).unwrap(),
    ] {
        assert_eq!(
            provider.download_numeric(handle).await.unwrap().data,
            HostNumericDataOwned::F64(values.clone())
        );
    }
    input.allocation_lease().validate_live().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual adapter and RUNMAT_WG=64; run alone with --test-threads=1"]
async fn actual_nondefault_scalar_bundle_survives_selector_change_before_and_after_warm_execution()
{
    let provider = owner().await;
    let launch = provider
        .real_scalar_pipeline(NumericPrecision::F32)
        .unwrap()
        .scalar_launch()
        .unwrap();
    assert_eq!(launch.workgroup(), 64, "run with RUNMAT_WG=64");
    struct Restore(u32);
    impl Drop for Restore {
        fn drop(&mut self) {
            crate::backend::wgpu::config::set_effective_workgroup_size(self.0);
        }
    }
    let _restore = Restore(crate::backend::wgpu::config::effective_workgroup_size());
    let key = launch.key();
    crate::backend::wgpu::config::set_effective_workgroup_size(128);
    qualify(provider, 65).await;
    assert_eq!(launch.key(), key);
    assert_eq!(launch.workgroup(), 64);
    qualify(
        provider,
        usize::try_from(launch.chunk_capacity().unwrap() + 1).unwrap(),
    )
    .await;
    assert_eq!(launch.key(), key);
}

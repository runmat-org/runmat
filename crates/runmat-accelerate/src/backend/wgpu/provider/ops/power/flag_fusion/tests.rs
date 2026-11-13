use super::*;
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn full_flag_symbolic_recipe_preserves_both_precisions_and_production_shader() {
    let limits = wgpu::Limits::default();
    let features = wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_INT64;
    for workgroup in [64, 128, 256] {
        for (input, flags, caps, bytes) in [
            (
                NumericPrecision::F32,
                NumericPrecision::F32,
                RequiredCaps::None,
                160,
            ),
            (
                NumericPrecision::F32,
                NumericPrecision::F64,
                RequiredCaps::Float64Int64,
                320,
            ),
            (
                NumericPrecision::F64,
                NumericPrecision::F64,
                RequiredCaps::Float64Int64,
                320,
            ),
        ] {
            let recipe = PowerFlagFusionRecipe::plan(
                input,
                flags,
                &[2, 4, 5],
                FrozenLinearLaunch::new(workgroup, 2).unwrap(),
                &limits,
                features,
            )
            .unwrap();
            assert_eq!(recipe.len, 40);
            assert_eq!(recipe.fusion.expected().output_buffer_bytes, Some(bytes));
            assert_eq!(
                recipe.fusion.expected().uniform_buffer_bytes,
                crate::backend::wgpu_geometry::fusion_broadcast_uniform_bytes(
                    2,
                    crate::backend::wgpu::params::BCAST_MAX_RANK
                )
                .unwrap()
            );
            assert_eq!(recipe.fusion.expected().queue_submissions, 2);
            validate_generated(
                "frozen-power-flag-fusion",
                &recipe.fusion.pipeline().shader,
                workgroup,
                8,
                caps,
            );
        }
    }
    assert!(PowerFlagFusionRecipe::plan(
        NumericPrecision::F64,
        NumericPrecision::F32,
        &[2, 4, 5],
        FrozenLinearLaunch::new(64, 2).unwrap(),
        &limits,
        features
    )
    .is_err());
    assert!(PowerFlagFusionRecipe::plan(
        NumericPrecision::F32,
        NumericPrecision::F32,
        &[2, 0, 5],
        FrozenLinearLaunch::new(64, 2).unwrap(),
        &limits,
        features
    )
    .is_err());
    assert!(PowerFlagFusionRecipe::plan(
        NumericPrecision::F32,
        NumericPrecision::F32,
        &[usize::MAX, 2],
        FrozenLinearLaunch::new(64, 2).unwrap(),
        &limits,
        features
    )
    .is_err());
    for missing in [
        wgpu::Features::empty(),
        wgpu::Features::SHADER_F64,
        wgpu::Features::SHADER_INT64,
    ] {
        assert!(PowerFlagFusionRecipe::plan(
            NumericPrecision::F32,
            NumericPrecision::F64,
            &[2, 4, 5],
            FrozenLinearLaunch::new(64, 2).unwrap(),
            &limits,
            missing
        )
        .is_err());
    }
}

async fn owner() -> &'static WgpuProvider {
    let options = crate::backend::wgpu::provider::WgpuProviderOptions::default();
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(options).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(options)
        .await
        .expect("actual WebGPU adapter");
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this independently created actual owner lives for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

async fn actual_flags(provider: &'static WgpuProvider, input_precision: NumericPrecision) {
    let single = [2.0_f32, -2.0, -0.0, 0.0, f32::INFINITY, f32::NAN];
    let double: Vec<f64> = single.iter().copied().map(f64::from).collect();
    let exponent_single = [0.5_f32];
    let exponent_double = [0.5_f64];
    let base = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: if input_precision == NumericPrecision::F32 {
                HostNumericDataView::F32(&single)
            } else {
                HostNumericDataView::F64(&double)
            },
            shape: &[2, 3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let exponent = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: if input_precision == NumericPrecision::F32 {
                HostNumericDataView::F32(&exponent_single)
            } else {
                HostNumericDataView::F64(&exponent_double)
            },
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let mut previous = None;
    for _ in 0..2 {
        let (flags, receipt) = provider
            .execute_power_flag_fusion_exec(&base, &exponent, &[2, 3, 1], 6, input_precision)
            .unwrap();
        receipt.require_output(&flags).unwrap();
        assert!(!flags.shares_allocation_with(&base));
        assert!(!flags.shares_allocation_with(&exponent));
        assert_eq!(
            receipt.resources.output_buffer_bytes,
            Some(flags.physical().requested_bytes())
        );
        assert_eq!(receipt.resources.dispatches, 1);
        assert_eq!(receipt.resources.queue_submissions, 2);
        assert_eq!(receipt.resources.uniform_allocations, 1);
        if let Some(previous) = &previous {
            assert_eq!(previous, &receipt.resources);
        }
        previous = Some(receipt.resources.clone());
        let host = provider.download(&flags).await.unwrap();
        assert_eq!(host.data.len(), 6);
        assert_eq!(host.data, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(receipt.require_output(&base).is_err());
        let alias = provider.reshape_exec(&flags, &[6, 1]).unwrap();
        assert!(alias.shares_allocation_with(&flags));
        assert!(
            receipt.require_output(&alias).is_err(),
            "receipt binds the exact output view"
        );
        receipt.require_output(&flags).unwrap();
        base.allocation_lease().validate_live().unwrap();
        exponent.allocation_lease().validate_live().unwrap();
    }
    let before = provider.telemetry_snapshot();
    let allocations = provider.allocations.record_count().unwrap();
    assert!(provider
        .execute_power_flag_fusion_exec(&base, &exponent, &[6, 1], 6, input_precision)
        .is_err());
    assert_eq!(before, provider.telemetry_snapshot());
    assert_eq!(allocations, provider.allocations.record_count().unwrap());
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual WGPU adapter and probe IEEE capabilities"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_single_cold_warm_flag_resources_and_borrowed_liveness() {
    actual_flags(owner().await, NumericPrecision::F32).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual enabled SHADER_F64 and SHADER_INT64; F32/Naga do not qualify F64"]
async fn actual_double_cold_warm_flag_resources_and_borrowed_liveness() {
    assert_eq!(
        std::env::var("RUNMAT_WGPU_FORCE_PRECISION").as_deref(),
        Ok("f64"),
        "run this opt-in witness with RUNMAT_WGPU_FORCE_PRECISION=f64 before owner construction"
    );
    let provider = owner().await;
    assert_eq!(provider.precision, NumericPrecision::F64);
    assert!(
        provider
            .enabled_features
            .contains(wgpu::Features::SHADER_F64 | wgpu::Features::SHADER_INT64),
        "F64 probe qualification requires both features actually enabled on the constructed device"
    );
    provider.require_ieee_words(NumericPrecision::F64).unwrap();
    actual_flags(provider, NumericPrecision::F64).await;
}

// Recompile the exact frozen shader and its two-input pipeline only after the
// ordinary executor's output has been read. The existing gated API captures
// browser validation errors; compilation messages are awaited separately.
async fn diagnostic_shader_and_pipeline_errors(
    provider: &WgpuProvider,
    recipe: &FrozenFusionRecipe,
) -> String {
    let shader = recipe.pipeline().shader.as_str();
    let module_receipt = provider
        .access
        .with_scoped(|emission| {
            crate::backend::wgpu::pipelines::create_shader_module(
                emission,
                "power-flag-diagnostic-module",
                shader,
            )
        })
        .expect("diagnostic shader scope acquisition");
    let module = match module_receipt.resolve().await {
        Ok(Ok(module)) => module,
        Ok(Err(error)) => return format!("module_preparation={error:#}"),
        Err(errors) => {
            return errors.with_errors(|inner| format!("module_scope={inner:?}"));
        }
    };
    let messages = module.get_compilation_info().await.messages;
    let pipeline_receipt = provider
        .access
        .with_scoped(|emission| {
            let binding_layout = crate::backend::wgpu::bindings::build_fusion_bgl(emission, 2);
            let layout = crate::backend::wgpu::pipelines::create_pipeline_layout(
                emission,
                "power-flag-diagnostic-pipeline",
                &binding_layout,
            );
            emission.device().create_compute_pipeline(
                &crate::backend::wgpu::compat::wgpu_compute_pipeline_descriptor! {
                    label: Some("power-flag-diagnostic-pipeline"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: "main",
                },
            )
        })
        .expect("diagnostic pipeline scope acquisition");
    let pipeline = match pipeline_receipt.resolve().await {
        Ok(_) => "pipeline_scope=ok".to_string(),
        Err(errors) => errors.with_errors(|inner| format!("pipeline_scope={inner:?}")),
    };
    format!("module_messages={messages:?}; {pipeline}")
}

// Diagnostic only: the qualifying no-prefetch fixture and its oracle above
// remain unchanged. This separate test intentionally synchronizes readback
// before flags to distinguish bad upload/flush from shader classification.
async fn diagnose_single_flag_inputs_and_predicates(provider: &'static WgpuProvider) {
    let base_values = [2.0_f32, -2.0, -0.0, 0.0, f32::INFINITY, f32::NAN];
    let exponent_values = [0.5_f32];
    let base = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&base_values),
            shape: &[2, 3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let exponent = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&exponent_values),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let HostNumericDataOwned::F32(read_base) = provider.download_numeric(&base).await.unwrap().data
    else {
        panic!("diagnostic base must retain native F32 storage")
    };
    let HostNumericDataOwned::F32(read_exponent) =
        provider.download_numeric(&exponent).await.unwrap().data
    else {
        panic!("diagnostic exponent must retain native F32 storage")
    };
    let base_bits = read_base
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let exponent_bits = read_exponent
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    eprintln!("power flag diagnostic input bits: base={base_bits:?} exponent={exponent_bits:?}");
    assert_eq!(
        base_bits,
        base_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "upload/readback changed a base bit before classification"
    );
    assert_eq!(
        exponent_bits,
        exponent_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "upload/readback changed an exponent bit before classification"
    );

    let precision = provider.precision;
    provider.require_ieee_words(precision).unwrap();
    let source = classification_shader(NumericPrecision::F32, precision);
    let entry = crate::kernel_math::power::entrypoint(match precision {
        NumericPrecision::F32 => runmat_accelerate_api::ProviderPrecision::F32,
        NumericPrecision::F64 => runmat_accelerate_api::ProviderPrecision::F64,
    });
    let marker = format!("Out.data[idx] = select(T(0.0), T(1.0), {entry}_needs_complex(a, b));");
    assert_eq!(source.matches(&marker).count(), 1);
    let probes = [
        (
            "constant_one",
            "Out.data[idx] = T(1.0);".to_string(),
            [1.0; 6],
        ),
        (
            "lane_ramp",
            "Out.data[idx] = T(idx + 1u);".to_string(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        (
            "a_lt_zero",
            "Out.data[idx] = select(T(0.0), T(1.0), a < T(0.0));".to_string(),
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "finite_a",
            format!("Out.data[idx] = select(T(0.0), T(1.0), {entry}_finite(a));"),
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ),
        (
            "fractional_b",
            "Out.data[idx] = select(T(0.0), T(1.0), trunc(b) != b);".to_string(),
            [1.0; 6],
        ),
        (
            "full_needs_complex",
            format!("Out.data[idx] = select(T(0.0), T(1.0), {entry}_needs_complex(a, b));"),
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ];
    let mut observed = Vec::new();
    for (name, body, expected) in probes {
        // Test-only derivative of the production shader: only its final
        // predicate changes. Keep the broadcast uniform ABI and ordinary
        // prepared executor identical to the production flag route.
        let shader = source.replacen(&marker, &body, 1);
        let work = FusedElementwiseWork {
            output_shape: &[2, 3, 1],
            len: 6,
            invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
            representation: flag_representation(NumericPrecision::F32, precision),
            uniform_abi: Some(FusionUniformAbi::Broadcast),
        };
        let recipe = FrozenFusionRecipe::plan(
            &shader,
            2,
            &work,
            provider.linear_launch,
            &provider.adapter_limits,
        )
        .unwrap();
        let (output, _) = provider
            .fused_elementwise_prepared_with_telemetry_exec(
                &shader,
                &[base.clone(), exponent.clone()],
                FusedElementwiseWork {
                    output_shape: &[2, 3, 1],
                    len: 6,
                    invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
                    representation: flag_representation(NumericPrecision::F32, precision),
                    uniform_abi: Some(FusionUniformAbi::Broadcast),
                },
                &recipe,
            )
            .unwrap();
        let values = provider.download(&output).await.unwrap().data;
        let shader_pipeline = diagnostic_shader_and_pipeline_errors(provider, &recipe).await;
        eprintln!("power flag diagnostic {name}: actual={values:?} expected={expected:?}");
        observed.push((name, values, expected, shader_pipeline));
    }
    assert!(
        observed
            .iter()
            .all(|(_, values, expected, _)| values.as_slice() == expected.as_slice()),
        "power flag predicates differ; base_bits={base_bits:?} exponent_bits={exponent_bits:?} all_observed={observed:?}"
    );
    base.allocation_lease().validate_live().unwrap();
    exponent.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual WGPU adapter; diagnostic does not qualify the original fixture"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn diagnostic_actual_single_flag_inputs_and_shader_predicates() {
    diagnose_single_flag_inputs_and_predicates(owner().await).await;
}

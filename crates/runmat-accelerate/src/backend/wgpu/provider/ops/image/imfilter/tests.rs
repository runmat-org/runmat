use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use futures::executor::block_on;
use runmat_accelerate_api::{AccelProvider, GpuHandleProvenance, HostNumericDataView};

fn provider() -> WgpuProvider {
    assert!(
        std::env::var("RUNMAT_WGPU_DISABLE_IMFILTER").is_err(),
        "hardware qualification must not disable the filter kernel"
    );
    WgpuProvider::new(WgpuProviderOptions::default()).expect("actual WGPU adapter required")
}

fn single_input(
    provider: &WgpuProvider,
    explicit: bool,
    values: &[f32],
    shape: &[usize],
) -> GpuTensorHandle {
    provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(values),
            shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
        .with_provenance(if explicit {
            GpuHandleProvenance::Explicit
        } else {
            GpuHandleProvenance::Automatic
        })
        .unwrap()
}

#[test]
#[ignore = "requires an actual WGPU adapter; execute explicitly in native GPU qualification"]
fn actual_imfilter_lazy_image_materializes_without_host_image_transfer() {
    let provider = provider();
    let original_values = [1.0f32, 4.0, -2.0, 8.0, 3.0, 7.0];
    let original = single_input(&provider, true, &original_values, &[2, 3]);
    let image = provider.transpose_exec(&original).unwrap();
    assert!(runmat_accelerate_api::handle_is_transposed(&image));
    assert!(image.shares_allocation_with(&original));
    let original_view = original.view().clone();
    let image_view = image.view().clone();
    let image_identity = image.view_identity();
    let kernel = single_input(&provider, false, &[1.0], &[1, 1]);
    let before = provider.telemetry_snapshot().download_bytes;
    let output =
        block_on(provider.imfilter_exec(&image, &kernel, &ImfilterOptions::default())).unwrap();
    assert_eq!(
        provider.telemetry_snapshot().download_bytes - before,
        4,
        "only the kernel is downloaded"
    );
    assert!(provider.pipelines.imfilter_single.is_initialized());
    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output.physical().element_type(), NumericElementType::F32);
    assert!(!output.shares_allocation_with(&original));
    assert!(!output.shares_allocation_with(&kernel));
    assert_eq!(image.view_identity(), image_identity);
    assert_eq!(image.view(), &image_view);
    assert_eq!(original.view(), &original_view);
    assert_eq!(
        block_on(provider.download_numeric_exec(&output))
            .unwrap()
            .data,
        HostNumericDataOwned::F32(vec![1.0, -2.0, 3.0, 4.0, 8.0, 7.0])
    );
    drop(image);
    assert_eq!(
        block_on(provider.download_numeric_exec(&original))
            .unwrap()
            .data,
        HostNumericDataOwned::F32(original_values.to_vec())
    );
}

#[test]
#[ignore = "requires an actual WGPU adapter; execute explicitly in native GPU qualification"]
fn actual_imfilter_single_kernel_uses_native_bytes_and_preserves_input_aliases() {
    let provider = provider();
    assert!(!provider.pipelines.imfilter_single.is_initialized());
    let values = [1.0f32, 4.0, -2.0, 8.0, 3.0, 7.0];
    let input = single_input(&provider, true, &values, &[2, 3]);
    let caller = input.clone();
    let weights = [0.25f32, -2.0];
    let kernel = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&weights),
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let options = ImfilterOptions {
        padding: ImfilterPadding::Replicate,
        ..Default::default()
    };
    let output = block_on(provider.imfilter_exec(&input, &kernel, &options)).unwrap();
    assert_eq!(
        provider.telemetry_snapshot().download_bytes,
        8,
        "only H may be read back; A must execute on device"
    );
    assert!(
        provider.pipelines.imfilter_single.is_initialized(),
        "production call must compile/execute the native shader route"
    );
    assert!(!provider.pipelines.imfilter_single_double.is_initialized());
    assert_eq!(output.physical().element_type(), NumericElementType::F32);
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(provider.get_entry_raw(&output).unwrap().allocated_bytes, 24);
    let HostNumericDataOwned::F32(actual) = block_on(provider.download_numeric_exec(&output))
        .unwrap()
        .data
    else {
        panic!("native Single result")
    };
    let reference = apply_imfilter_native(
        &Tensor::from_f32(values.to_vec(), vec![2, 3]).unwrap(),
        &Tensor::from_f32(weights.to_vec(), vec![2, 1]).unwrap(),
        &options,
        true,
        "imfilter",
    )
    .unwrap();
    assert_eq!(actual, reference.as_f32_slice().unwrap());
    assert!(!output.shares_allocation_with(&input));
    assert!(!output.shares_allocation_with(&kernel));
    drop(input);
    assert_eq!(
        block_on(provider.download_numeric_exec(&caller))
            .unwrap()
            .data,
        HostNumericDataOwned::F32(values.to_vec())
    );
}

#[test]
#[ignore = "requires enabled shader-f64; missing capability fails rather than silently skipping"]
fn actual_imfilter_mixed_single_double_recipe_preserves_accumulation_and_storage() {
    let provider = provider();
    assert!(
        provider
            .with_emission(|emission, _| emission
                .device()
                .features()
                .contains(wgpu::Features::SHADER_F64))
            .unwrap(),
        "F64 qualification requires shader-f64 enabled on the actual device"
    );
    assert!(!provider.pipelines.imfilter_single_double.is_initialized());
    let values = [16_777_216.0f32, 1.0, -16_777_216.0];
    let input = single_input(&provider, false, &values, &[3, 1]);
    let weights = [1.0 + 2.0_f64.powi(-24), 1.0, 1.0];
    let kernel = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F64(&weights),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let output =
        block_on(provider.imfilter_exec(&input, &kernel, &ImfilterOptions::default())).unwrap();
    assert_eq!(
        provider.telemetry_snapshot().download_bytes,
        24,
        "mixed arithmetic must not download A"
    );
    assert!(provider.pipelines.imfilter_single_double.is_initialized());
    assert!(!provider.pipelines.imfilter_double.is_initialized());
    assert_eq!(provider.get_entry_raw(&output).unwrap().allocated_bytes, 12);
    let HostNumericDataOwned::F32(actual) = block_on(provider.download_numeric_exec(&output))
        .unwrap()
        .data
    else {
        panic!("native Single output")
    };
    assert_eq!(
        actual[1], 2.0,
        "HostDouble accumulation with one final Single rounding"
    );
    let explicit = input
        .with_provenance(GpuHandleProvenance::Explicit)
        .unwrap();
    let single_output =
        block_on(provider.imfilter_exec(&explicit, &kernel, &ImfilterOptions::default())).unwrap();
    let HostNumericDataOwned::F32(actual) =
        block_on(provider.download_numeric_exec(&single_output))
            .unwrap()
            .data
    else {
        panic!("native Single output")
    };
    assert_eq!(
        actual[1], 0.0,
        "explicit Single keeps binary32 products and sums"
    );
    assert!(provider.pipelines.imfilter_single.is_initialized());
    let double_values = values.map(f64::from);
    let double_input = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F64(&double_values),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let double_output =
        block_on(provider.imfilter_exec(&double_input, &kernel, &ImfilterOptions::default()))
            .unwrap();
    assert!(provider.pipelines.imfilter_double.is_initialized());
    let HostNumericDataOwned::F64(actual) =
        block_on(provider.download_numeric_exec(&double_output))
            .unwrap()
            .data
    else {
        panic!("native Double output")
    };
    assert_eq!(actual[1], 2.0);
    assert_eq!(
        provider
            .get_entry_raw(&double_output)
            .unwrap()
            .allocated_bytes,
        24
    );
    assert_eq!(
        block_on(provider.download_numeric_exec(&input))
            .unwrap()
            .data,
        HostNumericDataOwned::F32(values.to_vec())
    );
}

#[test]
#[ignore = "requires an actual WGPU adapter; qualifies the existing typed host-mediated fallback"]
fn actual_imfilter_host_mediated_fallback_preserves_single_and_arithmetic_policy() {
    let provider = provider();
    let values = [16_777_216.0f32, 1.0, -16_777_216.0];
    let kernel = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 1.0, 1.0]),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    for explicit in [false, true] {
        let input = single_input(&provider, explicit, &values, &[3, 1]);
        let output =
            block_on(provider.imfilter_exec_fallback(&input, &kernel, &ImfilterOptions::default()))
                .unwrap();
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
        let HostNumericDataOwned::F32(actual) = block_on(provider.download_numeric_exec(&output))
            .unwrap()
            .data
        else {
            panic!("native Single fallback output")
        };
        assert_eq!(actual[1], if explicit { 0.0 } else { 1.0 });
        input.allocation_lease().validate_live().unwrap();
        kernel.allocation_lease().validate_live().unwrap();
    }
}

#[test]
#[ignore = "requires a device created without shader-f64; run with RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn actual_imfilter_unavailable_double_accumulation_rejects_before_readback() {
    let provider = provider();
    assert!(
        !provider
            .with_emission(|emission, _| emission
                .device()
                .features()
                .contains(wgpu::Features::SHADER_F64))
            .unwrap(),
        "this capability-negative gate requires a device without shader-f64"
    );
    let input = single_input(&provider, false, &[1.0, 2.0, 3.0], &[3, 1]);
    let kernel = provider
        .upload_numeric_exec(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 1.0, 1.0]),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let before = provider.telemetry_snapshot();
    let error =
        block_on(provider.imfilter_exec(&input, &kernel, &ImfilterOptions::default())).unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &error
    ));
    let after = provider.telemetry_snapshot();
    assert_eq!(after.download_bytes, before.download_bytes);
    assert_eq!(after.upload_bytes, before.upload_bytes);
    assert!(!provider.pipelines.imfilter_single_double.is_initialized());
    input.allocation_lease().validate_live().unwrap();
    kernel.allocation_lease().validate_live().unwrap();
}

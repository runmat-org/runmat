use super::super::*;
use runmat_accelerate_api::{AccelProvider, HostNumericDataView, HostNumericTensorView};

async fn actual_pipeline_source_witness() {
    let options = crate::backend::wgpu::provider::WgpuProviderOptions::default();
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(options).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(options)
        .await
        .expect("actual browser WebGPU adapter");
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this independently created actual owner lives for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    let prepared = provider.prepare_power_any_pipeline_exec().unwrap();
    assert_eq!(
        prepared.shader_workgroup(),
        provider.linear_launch.workgroup()
    );
    prepared
        .require(
            provider.device_id(),
            provider.precision,
            &provider.pipelines.reduce_dim_sum_mean,
        )
        .unwrap();
    let (source, workgroup) = provider
        .pipelines
        .reduce_dim_sum_mean
        .frozen_reduction_source()
        .unwrap();
    assert_eq!(workgroup, provider.linear_launch.workgroup());
    assert!(!source.contains("@WG@"));
    assert!(source.contains(&format!("@compute @workgroup_size({workgroup})")));
    provider
        .access
        .with_shared(|emission| -> anyhow::Result<()> {
            let cold = provider.pipelines.reduce_dim_sum_mean.get(emission)?;
            let warm = provider.pipelines.reduce_dim_sum_mean.get(emission)?;
            assert!(std::ptr::eq(cold, warm));
            Ok(())
        })
        .expect("WGPU access")
        .expect("cold/warm dimensional pipeline");
    prepared
        .require(
            provider.device_id(),
            provider.precision,
            &provider.pipelines.reduce_dim_sum_mean,
        )
        .unwrap();
    let wg = usize::try_from(workgroup).unwrap();
    for columns in [
        wg.checked_add(1).unwrap(),
        wg.checked_mul(2).unwrap().checked_add(1).unwrap(),
    ] {
        let mut single: Vec<f32> = (0..columns)
            .map(|index| if index % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        *single.last_mut().unwrap() = 1.0;
        let double: Vec<f64> = single.iter().map(|value| f64::from(*value)).collect();
        let data = match provider.precision {
            NumericPrecision::F32 => HostNumericDataView::F32(&single),
            NumericPrecision::F64 => HostNumericDataView::F64(&double),
        };
        let input = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data,
                shape: &[1, columns],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let reduced = provider.reduce_any_dim_exec(&input, 0, true).unwrap();
        let host = provider.download_exec(&reduced).await.unwrap();
        assert_eq!(host.shape, [1, columns]);
        assert_eq!(host.data, double);
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual WGPU adapter and enabled dimensional shader"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_dim_pipeline_cold_warm_frozen_source() {
    actual_pipeline_source_witness().await;
}

use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataView, HostNumericTensorView, ProviderPrecision,
};

async fn provider() -> WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("numeric output extent qualification requires an actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(WgpuProviderOptions::default())
        .await
        .expect("numeric output extent qualification requires an actual WebGPU adapter");
    provider
}

fn representation(element_type: ProviderElementType, shape: Vec<u64>) -> ProviderRepresentation {
    ProviderRepresentation {
        element_type,
        storage: ProviderStorage::DenseReal,
        layout: ProviderLayout::ColumnMajorContiguous,
        shape,
        residency: ProviderResidency::Device,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual WGPU adapter; run explicitly with --ignored"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn dense_real_projection_matches_actual_allocator_and_rejects_other_forms() {
    let mut provider = provider().await;
    let source = representation(ProviderElementType::F32, vec![2, 4, 5]);
    let projected = provider
        .plan_dense_real_output_buffer_bytes(&source)
        .unwrap();
    assert_eq!(projected, Some(160));
    let values = [1.0_f32; 40];
    let actual = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[2, 4, 5],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert_eq!(actual.physical().requested_bytes(), projected.unwrap());
    assert!(actual.allocation_lease().validate_live().is_ok());

    let mut unsupported = source.clone();
    unsupported.storage = ProviderStorage::Sparse;
    assert_eq!(
        provider
            .plan_dense_real_output_buffer_bytes(&unsupported)
            .unwrap(),
        None
    );
    unsupported = source.clone();
    unsupported.layout = ProviderLayout::Strided;
    assert_eq!(
        provider
            .plan_dense_real_output_buffer_bytes(&unsupported)
            .unwrap(),
        None
    );
    unsupported = source.clone();
    unsupported.element_type = ProviderElementType::I32;
    assert_eq!(
        provider
            .plan_dense_real_output_buffer_bytes(&unsupported)
            .unwrap(),
        None
    );
    unsupported = source.clone();
    unsupported.shape = vec![0, 4, 5];
    assert_eq!(
        provider
            .plan_dense_real_output_buffer_bytes(&unsupported)
            .unwrap(),
        Some(4)
    );
    unsupported.shape = vec![u64::MAX, 2];
    assert_eq!(
        provider
            .plan_dense_real_output_buffer_bytes(&unsupported)
            .unwrap(),
        None
    );

    let f64 = representation(ProviderElementType::F64, vec![2, 4, 5]);
    if provider.precision() == ProviderPrecision::F64 {
        let double_values = [1.0_f64; 40];
        let double_actual = provider
            .upload_numeric_raw(&HostNumericTensorView {
                data: HostNumericDataView::F64(&double_values),
                shape: &[2, 4, 5],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        assert_eq!(
            provider.plan_dense_real_output_buffer_bytes(&f64).unwrap(),
            Some(double_actual.physical().requested_bytes())
        );
    } else {
        assert_eq!(
            provider.plan_dense_real_output_buffer_bytes(&f64).unwrap(),
            None
        );
    }

    provider.adapter_limits.max_buffer_size = 159;
    assert_eq!(
        provider
            .plan_dense_real_output_buffer_bytes(&source)
            .unwrap(),
        None
    );
    provider.adapter_limits.max_buffer_size = 160;
    provider.adapter_limits.max_storage_buffer_binding_size = 159;
    assert_eq!(
        provider
            .plan_dense_real_output_buffer_bytes(&source)
            .unwrap(),
        None
    );
}

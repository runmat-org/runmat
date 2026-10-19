#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider::{WgpuProvider, WgpuProviderOptions};
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, NumericElementType, ProviderPrecision,
};

#[tokio::test]
#[ignore = "requires a compatible WGPU adapter; run on both F32 and F64 qualification adapters"]
async fn hann_shader_preserves_cospi_cardinals_on_the_selected_adapter() {
    let provider = WgpuProvider::new(WgpuProviderOptions::default())
        .expect("window shader qualification requires an actual WGPU adapter");
    for (length, periodic, expected) in [
        (5, false, vec![0.0, 0.5, 1.0, 0.5, 0.0]),
        (4, true, vec![0.0, 0.5, 1.0, 0.5]),
    ] {
        let handle = provider
            .hann_window(length, periodic)
            .expect("resident Hann window");
        assert_eq!(handle.shape(), [length, 1]);
        assert_eq!(handle.device_id(), provider.device_id());
        let downloaded = provider
            .download_numeric(&handle)
            .await
            .expect("download window");
        let actual = match (provider.precision(), downloaded.data) {
            (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
                assert_eq!(handle.physical().element_type(), NumericElementType::F32);
                values.into_iter().map(f64::from).collect::<Vec<_>>()
            }
            (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => {
                assert_eq!(handle.physical().element_type(), NumericElementType::F64);
                values
            }
            _ => panic!("window must retain the adapter's physical floating precision"),
        };
        assert_eq!(actual, expected);
    }
}

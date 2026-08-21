#![cfg(feature = "wgpu")]

use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderQrOptions, ProviderQrPivot};

fn register_provider() -> &'static dyn AccelProvider {
    #[cfg(target_os = "windows")]
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
    #[cfg(not(target_os = "windows"))]
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f64");

    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default());
    runmat_accelerate_api::provider().expect("wgpu provider registered")
}

#[test]
fn qr_precision_boundary_uses_expected_execution_path() {
    let provider = register_provider();

    let rows = 48usize;
    let cols = 16usize;
    let mut data = vec![0.0f64; rows * cols];
    for c in 0..cols {
        for r in 0..rows {
            data[r + c * rows] = ((r + c + 1) as f64).sin() + (c as f64) * 0.05;
        }
    }

    let input = provider
        .upload(&HostTensorView {
            data: &data,
            shape: &[rows, cols],
        })
        .expect("upload matrix");

    provider.reset_telemetry();
    let options = ProviderQrOptions {
        economy: true,
        pivot: ProviderQrPivot::Matrix,
    };
    let qr = block_on(provider.qr(&input, options)).expect("qr");
    let telemetry = provider.telemetry_snapshot();

    assert!(
        telemetry.download_bytes > 0,
        "direct QR should use its correctness-oriented host fallback at {:?} precision",
        provider.precision()
    );

    let _ = provider.free(&input);
    let _ = provider.free(&qr.q);
    let _ = provider.free(&qr.r);
    let _ = provider.free(&qr.perm_matrix);
    let _ = provider.free(&qr.perm_vector);
}

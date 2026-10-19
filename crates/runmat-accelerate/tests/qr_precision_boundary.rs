#![cfg(feature = "wgpu")]

#[path = "support/retirement.rs"]
mod retirement;
use futures::executor::block_on;
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
use runmat_accelerate_api::{
    AccelProvider, HostTensorView, ProviderQrColumnOrder, ProviderQrOptions, ProviderQrOutputs,
    ProviderQrResult,
};

fn register_provider() -> &'static dyn AccelProvider {
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");

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
        ..ProviderQrOptions::default()
    };
    let qr = block_on(provider.qr(&input, options)).expect("qr");
    let telemetry = provider.telemetry_snapshot();

    assert!(
        telemetry.download_bytes > 0,
        "direct QR should use its correctness-oriented host fallback at {:?} precision",
        provider.precision()
    );

    let ProviderQrResult::FullEnvelope {
        q,
        r,
        perm_matrix,
        perm_vector,
    } = qr
    else {
        panic!("full provider request must return its full envelope");
    };
    let unpivoted = ProviderQrOptions {
        economy: true,
        column_order: ProviderQrColumnOrder::Unpivoted,
        outputs: ProviderQrOutputs::R,
    };
    let ProviderQrResult::R { r: r_only } =
        block_on(provider.qr(&input, unpivoted)).expect("R-only provider QR")
    else {
        panic!("R-only request must not publish unused Q or permutation handles");
    };
    let ProviderQrResult::Qr {
        q: q_pair,
        r: r_pair,
    } = block_on(provider.qr(
        &input,
        ProviderQrOptions {
            outputs: ProviderQrOutputs::Qr,
            ..unpivoted
        },
    ))
    .expect("two-output provider QR")
    else {
        panic!("two-output request must not publish unused permutation handles");
    };
    retirement::release_all([
        input,
        q,
        r,
        perm_matrix,
        perm_vector,
        r_only,
        q_pair,
        r_pair,
    ]);
}

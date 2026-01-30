#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
use runmat_accelerate_api::{AccelProvider, HostTensorView};

fn register_provider() -> &'static dyn AccelProvider {
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default());
    runmat_accelerate_api::provider().expect("wgpu provider registered")
}

#[tokio::test]
async fn matmul_output_reuse_from_pool() {
    let provider = register_provider();

    let lhs_data: Vec<f64> = (0..16).map(|v| (v + 1) as f64 * 0.01).collect();
    let rhs_data: Vec<f64> = (0..16).map(|v| (v + 1) as f64 * 0.02).collect();

    let lhs_view = HostTensorView {
        data: lhs_data.as_slice(),
        shape: &[4, 4],
    };
    let rhs_view = HostTensorView {
        data: rhs_data.as_slice(),
        shape: &[4, 4],
    };

    let lhs_handle = provider.upload(&lhs_view).expect("upload lhs");
    let rhs_handle = provider.upload(&rhs_view).expect("upload rhs");

    let mut cpu_out = [0.0f64; 16];
    for row in 0..4 {
        for col in 0..4 {
            let mut acc = 0.0f64;
            for k in 0..4 {
                acc += lhs_data[row + k * 4] * rhs_data[k + col * 4];
            }
            cpu_out[row + col * 4] = acc;
        }
    }

    for _ in 0..8 {
        let gpu_out = provider
            .matmul(&lhs_handle, &rhs_handle)
            .await
            .expect("matmul");
        let host = provider.download(&gpu_out).await.expect("download");
        assert_eq!(host.shape, vec![4, 4]);

        let mut max_err = 0.0f64;
        let mut max_val = 0.0f64;
        for (a, b) in host.data.iter().zip(cpu_out.iter()) {
            max_err = max_err.max((a - b).abs());
            max_val = max_val.max(a.abs());
        }
        assert!(
            max_val > 1.0e-6,
            "expected non-zero matmul result, got max_abs={}",
            max_val
        );
        assert!(
            max_err < 1.0e-6,
            "matmul result deviates from CPU reference: max_err={}",
            max_err
        );
        provider.free(&gpu_out).expect("free matmul output");
    }

    provider.free(&lhs_handle).expect("free lhs");
    provider.free(&rhs_handle).expect("free rhs");
}

#![cfg(feature = "wgpu")]

use bytemuck::cast_slice;
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderQrOptions, ProviderQrPivot};
use std::{fs, path::Path};

fn register_provider() -> &'static dyn AccelProvider {
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default());
    runmat_accelerate_api::provider().expect("wgpu provider registered")
}

#[ignore]
#[tokio::test]
async fn matmul_pca_zero_regression() {
    let provider = register_provider();
    provider.reset_telemetry();

    let dump_base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/matmul_zero");
    let lhs_path = dump_base.join("lhs_26_1048576.bin");
    let rhs_path = dump_base.join("rhs_26_8192.bin");

    let lhs_bytes = fs::read(lhs_path).expect("load lhs dump");
    let rhs_bytes = fs::read(rhs_path).expect("load rhs dump");

    let lhs_data: Vec<f64> = cast_slice(&lhs_bytes).to_vec();
    let rhs_data: Vec<f64> = cast_slice(&rhs_bytes).to_vec();

    assert_eq!(lhs_data.len(), 1024 * 1024);
    assert_eq!(rhs_data.len(), 1024 * 8);

    let lhs_view = HostTensorView {
        data: lhs_data.as_slice(),
        shape: &[1024usize, 1024usize],
    };
    let rhs_view = HostTensorView {
        data: rhs_data.as_slice(),
        shape: &[1024usize, 8usize],
    };

    let lhs_handle = provider.upload(&lhs_view).expect("upload lhs");
    let mut rhs_handle = provider.upload(&rhs_view).expect("upload rhs");

    std::env::set_var("RUNMAT_DEBUG_QR_ZEROHOST", "1");

    let options = ProviderQrOptions {
        economy: true,
        pivot: ProviderQrPivot::Matrix,
    };

    for iter in 0..15 {
        let rhs_host = provider
            .download(&rhs_handle)
            .await
            .expect("download rhs for inspection");
        let max_rhs = rhs_host
            .data
            .iter()
            .fold(0.0f64, |acc, value| acc.max(value.abs()));
        println!("iter {} input Q max_abs={:.6e}", iter, max_rhs);
        println!("iter {} Q sample {:?}", iter, &rhs_host.data[0..8]);

        let mut cpu_product_iter = vec![0.0f64; 1024 * 8];
        let mut cpu_product_iter_f32 = vec![0.0f64; 1024 * 8];
        for col in 0..8 {
            for row in 0..1024 {
                let mut acc = 0.0f64;
                let mut acc_f32 = 0.0f32;
                for k in 0..1024 {
                    let lhs_idx = row + k * 1024;
                    let rhs_idx = k + col * 1024;
                    acc += lhs_data[lhs_idx] * rhs_host.data[rhs_idx];
                    let lhs_val = lhs_data[lhs_idx] as f32;
                    let rhs_val = rhs_host.data[rhs_idx] as f32;
                    acc_f32 += lhs_val * rhs_val;
                }
                cpu_product_iter[row + col * 1024] = acc;
                cpu_product_iter_f32[row + col * 1024] = acc_f32 as f64;
            }
        }
        let max_cpu_iter = cpu_product_iter
            .iter()
            .fold(0.0f64, |acc, value| acc.max(value.abs()));
        println!("iter {} CPU product max_abs={:.6e}", iter, max_cpu_iter);
        let max_cpu_iter_f32 = cpu_product_iter_f32
            .iter()
            .fold(0.0f64, |acc, value| acc.max(value.abs()));
        println!(
            "iter {} CPU product f32 max_abs={:.6e}",
            iter, max_cpu_iter_f32
        );

        let product_handle = provider
            .matmul(&lhs_handle, &rhs_handle)
            .await
            .expect("gpu product check");
        let gpu_product_host = provider
            .download(&product_handle)
            .await
            .expect("download gpu product check");
        let max_gpu_product = gpu_product_host
            .data
            .iter()
            .fold(0.0f64, |acc, value| acc.max(value.abs()));
        println!(
            "iter {} standalone GPU product max_abs={:.6e}",
            iter, max_gpu_product
        );
        println!(
            "iter {} GPU product sample {:?}",
            iter,
            &gpu_product_host.data[0..8]
        );

        let result = provider
            .qr_power_iter(&product_handle, Some(&lhs_handle), &rhs_handle, &options)
            .await
            .expect("invoke qr_power_iter")
            .expect("device qr path");
        provider.free(&product_handle).ok();

        let gpu_host = provider
            .download(&result.q)
            .await
            .expect("download q result");
        let max_gpu = gpu_host
            .data
            .iter()
            .fold(0.0f64, |acc, value| acc.max(value.abs()));

        println!("iter {} GPU Q max_abs={:.6e}", iter, max_gpu);
        assert!(
            max_gpu > 1.0e-6,
            "gpu qr_power_iter produced zero Q (iter={}, max_abs={:.6e})",
            iter,
            max_gpu
        );

        rhs_handle = result.q;
    }

    let mut cpu_product = vec![0.0f64; 1024 * 8];
    for col in 0..8 {
        for row in 0..1024 {
            let mut acc = 0.0f64;
            for k in 0..1024 {
                let lhs_idx = row + k * 1024;
                let rhs_idx = k + col * 1024;
                acc += lhs_data[lhs_idx] * rhs_data[rhs_idx];
            }
            cpu_product[row + col * 1024] = acc;
        }
    }

    let max_cpu = cpu_product
        .iter()
        .fold(0.0f64, |acc, value| acc.max(value.abs()));

    println!("CPU product max_abs={:.6e}", max_cpu);
}

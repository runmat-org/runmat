#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
use runmat_accelerate_api::{HostTensorView, ProviderPrecision};

fn cpu_transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; data.len()];
    for col in 0..cols {
        for row in 0..rows {
            let src_idx = row + col * rows;
            let dst_idx = col + row * cols;
            out[dst_idx] = data[src_idx];
        }
    }
    out
}

fn cpu_matmul(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    assert_eq!(ac, br);
    let mut out = vec![0.0f64; ar * bc];
    for col in 0..bc {
        for row in 0..ar {
            let mut sum = 0.0;
            for k in 0..ac {
                sum += a[row + k * ar] * b[k + col * br];
            }
            out[row + col * ar] = sum;
        }
    }
    out
}

#[test]
fn transpose_roundtrip_matches_cpu() {
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");

    let rows = 37usize;
    let cols = 29usize;
    let mut data = Vec::with_capacity(rows * cols);
    for col in 0..cols {
        for row in 0..rows {
            data.push(((row * 13 + col * 7) % 101) as f64 * 0.03125 + 0.5);
        }
    }

    let handle = p
        .upload(&HostTensorView {
            data: &data,
            shape: &[rows, cols],
        })
        .expect("upload");
    let transposed = p.transpose(&handle).expect("transpose");
    assert!(runmat_accelerate_api::handle_transpose_info(&transposed).is_some());
    let host_t = p.download(&transposed).expect("download transpose");
    assert_eq!(host_t.shape, vec![cols, rows]);

    let expected_t = cpu_transpose(&data, rows, cols);
    for (idx, (&got, &want)) in host_t.data.iter().zip(expected_t.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= 1e-9,
            "transpose mismatch at {idx}: got={got} want={want} diff={diff}"
        );
    }

    let transposed_back = p.transpose(&transposed).expect("double transpose");
    assert!(runmat_accelerate_api::handle_transpose_info(&transposed_back).is_none());
    let host_tt = p
        .download(&transposed_back)
        .expect("download double transpose");
    assert_eq!(host_tt.shape, vec![rows, cols]);
    for (idx, (&got, &want)) in host_tt.data.iter().zip(data.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= 1e-9,
            "double transpose mismatch at {idx}: got={got} want={want} diff={diff}"
        );
    }
}

#[test]
fn matmul_with_transposed_operand_matches_cpu() {
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");
    if p.precision() != ProviderPrecision::F64 {
        eprintln!(
            "skipping matmul_with_transposed_operand_matches_cpu: precision {:?}",
            p.precision()
        );
        return;
    }

    let m = 48usize;
    let k = 32usize;
    let n = 27usize;

    let mut a = Vec::with_capacity(m * k);
    for col in 0..k {
        for row in 0..m {
            a.push(((row * 5 + col * 3) % 17) as f64 * 0.125 - 0.75);
        }
    }

    let mut b = Vec::with_capacity(m * n);
    for col in 0..n {
        for row in 0..m {
            b.push(((row + col * 11) % 23) as f64 * 0.0625 + 0.25);
        }
    }

    let handle_a = p
        .upload(&HostTensorView {
            data: &a,
            shape: &[m, k],
        })
        .expect("upload A");
    let handle_b = p
        .upload(&HostTensorView {
            data: &b,
            shape: &[m, n],
        })
        .expect("upload B");

    let a_t = p.transpose(&handle_a).expect("transpose A");
    let result = p.matmul(&a_t, &handle_b).expect("matmul A^T * B");
    let host = p.download(&result).expect("download result");
    assert_eq!(host.shape, vec![k, n]);

    let a_t_cpu = cpu_transpose(&a, m, k);
    let expected = cpu_matmul(&a_t_cpu, k, m, &b, m, n);
    for (idx, (&got, &want)) in host.data.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= 1e-9,
            "matmul transpose mismatch at {idx}: got={got} want={want} diff={diff}"
        );
    }
}

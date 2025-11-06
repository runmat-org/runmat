#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::HostTensorView;

#[cfg(feature = "wgpu")]
fn cpu_matmul(a: &[f64], ar: usize, ac: usize, b: &[f64], br: usize, bc: usize) -> Vec<f64> {
    assert_eq!(ac, br);
    let mut out = vec![0.0; ar * bc];
    for j in 0..bc {
        for i in 0..ar {
            let mut sum = 0.0;
            for k in 0..ac {
                sum += a[i + k * ar] * b[k + j * br];
            }
            out[i + j * ar] = sum;
        }
    }
    out
}

#[cfg(feature = "wgpu")]
#[test]
fn matmul_small_k_threshold() {
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");

    let m = 64usize;
    let n = 32usize;
    let k = 4usize;
    let mut a = Vec::with_capacity(m * k);
    for col in 0..k {
        for row in 0..m {
            a.push(((row + 1) as f64) + (col as f64) * 0.25);
        }
    }
    let mut b = Vec::with_capacity(k * n);
    for col in 0..n {
        for row in 0..k {
            b.push(((row + 2 * col) % 7) as f64);
        }
    }

    let ha = p
        .upload(&HostTensorView {
            data: &a,
            shape: &[m, k],
        })
        .expect("upload A");
    let hb = p
        .upload(&HostTensorView {
            data: &b,
            shape: &[k, n],
        })
        .expect("upload B");
    let hc = p.matmul(&ha, &hb).expect("matmul");
    let host = p.download(&hc).expect("download");
    assert_eq!(host.shape, vec![m, n]);

    let expected = cpu_matmul(&a, m, k, &b, k, n);
    for idx in 0..expected.len() {
        let diff = (host.data[idx] - expected[idx]).abs();
        assert!(
            diff < 1e-9,
            "small-k mismatch at {}: got={} want={} diff={}",
            idx,
            host.data[idx],
            expected[idx],
            diff
        );
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn matmul_small_k_exact_threshold() {
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");

    let m = 48usize;
    let n = 24usize;
    let k = 8usize; // equal to threshold
    let mut a = Vec::with_capacity(m * k);
    for col in 0..k {
        for row in 0..m {
            a.push(((row * 3 + col) % 11) as f64 + 0.1 * col as f64);
        }
    }
    let mut b = Vec::with_capacity(k * n);
    for col in 0..n {
        for row in 0..k {
            b.push(((row + col) % 5) as f64 - 0.2 * row as f64);
        }
    }

    let ha = p
        .upload(&HostTensorView {
            data: &a,
            shape: &[m, k],
        })
        .expect("upload A");
    let hb = p
        .upload(&HostTensorView {
            data: &b,
            shape: &[k, n],
        })
        .expect("upload B");
    let hc = p.matmul(&ha, &hb).expect("matmul");
    let host = p.download(&hc).expect("download");
    assert_eq!(host.shape, vec![m, n]);

    let expected = cpu_matmul(&a, m, k, &b, k, n);
    for idx in 0..expected.len() {
        let diff = (host.data[idx] - expected[idx]).abs();
        assert!(
            diff < 1e-8,
            "threshold mismatch at {}: got={} want={} diff={}",
            idx,
            host.data[idx],
            expected[idx],
            diff
        );
    }
}

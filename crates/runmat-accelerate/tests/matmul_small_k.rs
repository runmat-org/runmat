#[cfg(feature = "wgpu")]
use once_cell::sync::Lazy;
#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::{HostTensorView, ProviderPrecision};
#[cfg(feature = "wgpu")]
use std::sync::Mutex;

#[cfg(feature = "wgpu")]
static TEST_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

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
fn cpu_matmul_f32(a: &[f32], ar: usize, ac: usize, b: &[f32], br: usize, bc: usize) -> Vec<f32> {
    assert_eq!(ac, br);
    let mut out = vec![0.0f32; ar * bc];
    for j in 0..bc {
        for i in 0..ar {
            let mut sum = 0.0f32;
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
    let _guard = TEST_MUTEX.lock().unwrap();
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
    let _guard = TEST_MUTEX.lock().unwrap();
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
            diff < 5e-5,
            "threshold mismatch at {}: got={} want={} diff={}",
            idx,
            host.data[idx],
            expected[idx],
            diff
        );
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn matmul_vec4_f32_matches_cpu() {
    let _guard = TEST_MUTEX.lock().unwrap();
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");
    if p.precision() != ProviderPrecision::F32 {
        eprintln!(
            "skipping matmul_vec4_f32_matches_cpu: provider precision is {:?}",
            p.precision()
        );
        return;
    }

    let m = 32usize;
    let n = 28usize;
    let k = 36usize;

    let mut a = Vec::with_capacity(m * k);
    for col in 0..k {
        for row in 0..m {
            a.push(((row + col * 3) % 19) as f32 * 0.03125 + 0.5);
        }
    }

    let mut b = Vec::with_capacity(k * n);
    for col in 0..n {
        for row in 0..k {
            b.push(((row * 2 + col) % 23) as f32 * 0.015625 - 0.25);
        }
    }

    let a_host: Vec<f64> = a.iter().map(|&v| v as f64).collect();
    let b_host: Vec<f64> = b.iter().map(|&v| v as f64).collect();

    let ha = p
        .upload(&HostTensorView {
            data: &a_host,
            shape: &[m, k],
        })
        .expect("upload A");
    let hb = p
        .upload(&HostTensorView {
            data: &b_host,
            shape: &[k, n],
        })
        .expect("upload B");
    let hc = p.matmul(&ha, &hb).expect("matmul");
    let host = p.download(&hc).expect("download");
    assert_eq!(host.shape, vec![m, n]);

    let expected = cpu_matmul_f32(&a, m, k, &b, k, n);
    for idx in 0..expected.len() {
        let got = host.data[idx] as f32;
        let want = expected[idx];
        let diff = (got - want).abs();
        let tol = 5e-4 * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "vec4 mismatch at {}: got={} want={} diff={}",
            idx,
            got,
            want,
            diff
        );
    }
}

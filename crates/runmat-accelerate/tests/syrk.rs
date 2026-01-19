#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use once_cell::sync::Lazy;
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::{HostTensorView, ProviderPrecision};
#[cfg(feature = "wgpu")]
use std::sync::Mutex;

#[cfg(feature = "wgpu")]
static TEST_MUTEX: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[cfg(feature = "wgpu")]
fn cpu_syrk(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; cols * cols];
    for col in 0..cols {
        for row in 0..=col {
            let mut acc = 0.0;
            for k in 0..rows {
                let lhs = a[k + row * rows];
                let rhs = a[k + col * rows];
                acc += lhs * rhs;
            }
            out[row + col * cols] = acc;
            if row != col {
                out[col + row * cols] = acc;
            }
        }
    }
    out
}

#[cfg(feature = "wgpu")]
fn cpu_syrk_f32(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols * cols];
    for col in 0..cols {
        for row in 0..=col {
            let mut acc = 0.0f32;
            for k in 0..rows {
                let lhs = a[k + row * rows];
                let rhs = a[k + col * rows];
                acc += lhs * rhs;
            }
            out[row + col * cols] = acc;
            if row != col {
                out[col + row * cols] = acc;
            }
        }
    }
    out
}

#[cfg(feature = "wgpu")]
#[test]
fn syrk_matches_cpu() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");

    let rows = 16usize;
    let cols = 5usize;
    let mut data = Vec::with_capacity(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            data.push((r + 1 + c * 3) as f64);
        }
    }

    let handle = p
        .upload(&HostTensorView {
            data: &data,
            shape: &[rows, cols],
        })
        .expect("upload");
    let gpu = p.syrk(&handle).expect("syrk");
    let host = p.download(&gpu).expect("download");
    assert_eq!(host.shape, vec![cols, cols]);

    match p.precision() {
        ProviderPrecision::F64 => {
            let expected = cpu_syrk(&data, rows, cols);
            for (idx, (got, want)) in host.data.iter().zip(expected.iter()).enumerate() {
                let diff = (got - want).abs();
                assert!(
                    diff < 1e-9,
                    "mismatch at {}: got={} want={} diff={}",
                    idx,
                    got,
                    want,
                    diff
                );
            }
        }
        ProviderPrecision::F32 => {
            let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
            let expected = cpu_syrk_f32(&data_f32, rows, cols);
            for (idx, (got64, want)) in host.data.iter().zip(expected.iter()).enumerate() {
                let got = *got64 as f32;
                let diff = (got - want).abs();
                let tol = 1e-3 * want.abs().max(1.0);
                assert!(
                    diff <= tol,
                    "mismatch at {}: got={} want={} diff={}",
                    idx,
                    got,
                    want,
                    diff
                );
            }
        }
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn syrk_large_rows_chunks() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");

    let rows = 131072usize; // exceeds chunk threshold
    let cols = 3usize;
    let mut data = Vec::with_capacity(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            data.push(((r % 97) as f64) + (c as f64) * 0.5);
        }
    }

    let handle = p
        .upload(&HostTensorView {
            data: &data,
            shape: &[rows, cols],
        })
        .expect("upload");
    let gpu = p.syrk(&handle).expect("syrk");
    let host = p.download(&gpu).expect("download");
    assert_eq!(host.shape, vec![cols, cols]);

    let expected = cpu_syrk(&data, rows, cols);
    for (idx, (got, want)) in host.data.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        let tol = 1e-3 * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "chunk mismatch at {}: got={} want={} diff={}",
            idx,
            got,
            want,
            diff
        );
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn syrk_vec4_f32_matches_cpu() {
    let _guard = TEST_MUTEX.lock().unwrap();
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");
    if p.precision() != ProviderPrecision::F32 {
        eprintln!(
            "skipping syrk_vec4_f32_matches_cpu: provider precision is {:?}",
            p.precision()
        );
        return;
    }

    let rows = 64usize;
    let cols = 12usize;
    let mut data = Vec::with_capacity(rows * cols);
    for c in 0..cols {
        for r in 0..rows {
            data.push(((r * 5 + c) % 29) as f32 * 0.02 + 0.1 * (c as f32));
        }
    }

    let data_host: Vec<f64> = data.iter().map(|&v| v as f64).collect();

    let handle = p
        .upload(&HostTensorView {
            data: &data_host,
            shape: &[rows, cols],
        })
        .expect("upload");
    let gpu = p.syrk(&handle).expect("syrk");
    let host = p.download(&gpu).expect("download");
    assert_eq!(host.shape, vec![cols, cols]);

    let expected = cpu_syrk_f32(&data, rows, cols);
    for (idx, (got64, want)) in host.data.iter().zip(expected.iter()).enumerate() {
        let got = *got64 as f32;
        let diff = (got - want).abs();
        let tol = 1e-3 * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "vec4 syrk mismatch at {}: got={} want={} diff={}",
            idx,
            got,
            want,
            diff
        );
    }
}

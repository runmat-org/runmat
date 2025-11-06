#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::HostTensorView;

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
#[test]
fn syrk_matches_cpu() {
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

    let expected = cpu_syrk(&data, rows, cols);
    for idx in 0..expected.len() {
        let got = host.data[idx];
        let want = expected[idx];
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

#[cfg(feature = "wgpu")]
#[test]
fn syrk_large_rows_chunks() {
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
    for idx in 0..expected.len() {
        let got = host.data[idx];
        let want = expected[idx];
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

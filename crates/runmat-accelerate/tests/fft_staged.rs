#![cfg(feature = "wgpu")]

use num_complex::Complex;
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
use runmat_accelerate_api::{HostTensorView, ProviderPrecision};
use rustfft::FftPlanner;

struct EnvGuard {
    key: &'static str,
    prev: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self { key, prev }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(v) = &self.prev {
            std::env::set_var(self.key, v);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

fn tolerance(precision: ProviderPrecision) -> f64 {
    match precision {
        ProviderPrecision::F32 => 1e-3,
        ProviderPrecision::F64 => 1e-9,
    }
}

fn cpu_fft_forward(data: &[f64]) -> Vec<(f64, f64)> {
    let mut planner = FftPlanner::<f64>::new();
    let plan = planner.plan_fft_forward(data.len());
    let mut buf = data.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>();
    if data.len() > 1 {
        plan.process(&mut buf);
    }
    buf.into_iter().map(|z| (z.re, z.im)).collect()
}

#[tokio::test]
async fn staged_fft_forward_matches_cpu_for_pow2_family() {
    let _staged = EnvGuard::set("RUNMAT_FFT_USE_STAGED", "1");
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");
    let tol = tolerance(p.precision());

    for &n in &[4usize, 8usize, 16usize] {
        let input = (1..=n).map(|v| v as f64).collect::<Vec<_>>();
        let handle = p
            .upload(&HostTensorView {
                data: &input,
                shape: &[n],
            })
            .expect("upload");
        let out = p.fft_dim(&handle, None, 0).await.expect("fft_dim");
        let host = p.download(&out).await.expect("download fft");
        let expected = cpu_fft_forward(&input);

        assert_eq!(host.shape, vec![n, 2], "shape mismatch for n={n}");
        for (idx, &(er, ei)) in expected.iter().enumerate() {
            let got_re = host.data[idx * 2];
            let got_im = host.data[idx * 2 + 1];
            assert!(
                (got_re - er).abs() <= tol && (got_im - ei).abs() <= tol,
                "pow2 n={n} idx={idx}: got=({got_re},{got_im}) expected=({er},{ei}) tol={tol}"
            );
        }

        p.free(&handle).ok();
        p.free(&out).ok();
    }
}

#[tokio::test]
async fn staged_fft_ifft_roundtrip_recovers_input_for_pow2_family() {
    let _staged = EnvGuard::set("RUNMAT_FFT_USE_STAGED", "1");
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");
    let tol = tolerance(p.precision());

    for &n in &[4usize, 8usize, 16usize] {
        let input = (1..=n).map(|v| v as f64).collect::<Vec<_>>();
        let handle = p
            .upload(&HostTensorView {
                data: &input,
                shape: &[n],
            })
            .expect("upload");
        let freq = p.fft_dim(&handle, None, 0).await.expect("fft_dim");
        let back = p.ifft_dim(&freq, None, 0).await.expect("ifft_dim");
        let host = p.download(&back).await.expect("download ifft");

        assert_eq!(host.shape, vec![n, 2], "shape mismatch for n={n}");
        for (idx, &want) in input.iter().enumerate() {
            let got_re = host.data[idx * 2];
            let got_im = host.data[idx * 2 + 1];
            assert!(
                (got_re - want).abs() <= tol && got_im.abs() <= tol,
                "pow2 n={n} idx={idx}: got=({got_re},{got_im}) expected=({want},0) tol={tol}"
            );
        }

        p.free(&handle).ok();
        p.free(&freq).ok();
        p.free(&back).ok();
    }
}

#[tokio::test]
async fn staged_fft_forward_matches_cpu_for_non_pow2_families() {
    let _staged = EnvGuard::set("RUNMAT_FFT_USE_STAGED", "1");
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");
    let tol = tolerance(p.precision());

    for &n in &[9usize, 25usize, 15usize, 7usize] {
        let input = (1..=n).map(|v| v as f64).collect::<Vec<_>>();
        let handle = p
            .upload(&HostTensorView {
                data: &input,
                shape: &[n],
            })
            .expect("upload");
        let out = p.fft_dim(&handle, None, 0).await.expect("fft_dim");
        let host = p.download(&out).await.expect("download fft");
        let expected = cpu_fft_forward(&input);

        assert_eq!(host.shape, vec![n, 2], "shape mismatch for n={n}");
        for (idx, &(er, ei)) in expected.iter().enumerate() {
            let got_re = host.data[idx * 2];
            let got_im = host.data[idx * 2 + 1];
            assert!(
                (got_re - er).abs() <= tol && (got_im - ei).abs() <= tol,
                "n={n} idx={idx}: got=({got_re},{got_im}) expected=({er},{ei}) tol={tol}"
            );
        }

        p.free(&handle).ok();
        p.free(&out).ok();
    }
}

#[tokio::test]
async fn staged_fft_ifft_roundtrip_recovers_input_for_non_pow2_families() {
    let _staged = EnvGuard::set("RUNMAT_FFT_USE_STAGED", "1");
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");
    let tol = tolerance(p.precision());

    for &n in &[9usize, 25usize, 15usize, 7usize] {
        let input = (1..=n).map(|v| v as f64).collect::<Vec<_>>();
        let handle = p
            .upload(&HostTensorView {
                data: &input,
                shape: &[n],
            })
            .expect("upload");
        let freq = p.fft_dim(&handle, None, 0).await.expect("fft_dim");
        let back = p.ifft_dim(&freq, None, 0).await.expect("ifft_dim");
        let host = p.download(&back).await.expect("download ifft");

        assert_eq!(host.shape, vec![n, 2], "shape mismatch for n={n}");
        for (idx, &want) in input.iter().enumerate() {
            let got_re = host.data[idx * 2];
            let got_im = host.data[idx * 2 + 1];
            assert!(
                (got_re - want).abs() <= tol && got_im.abs() <= tol,
                "n={n} idx={idx}: got=({got_re},{got_im}) expected=({want},0) tol={tol}"
            );
        }

        p.free(&handle).ok();
        p.free(&freq).ok();
        p.free(&back).ok();
    }
}

#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::{HostTensorView, ImageNormalizeDescriptor};

#[cfg(feature = "wgpu")]
#[allow(clippy::too_many_arguments)]
fn cpu_image_normalize(
    data: &[f64],
    batch: usize,
    height: usize,
    width: usize,
    epsilon: f64,
    gain: Option<f64>,
    bias: Option<f64>,
    gamma: Option<f64>,
) -> Vec<f64> {
    let plane = height * width;
    let stride_h = batch;
    let stride_w = batch * height;
    let mut out = data.to_vec();

    for b in 0..batch {
        let mut sum = 0.0;
        for w in 0..width {
            let base_w = w * stride_w;
            for h in 0..height {
                let idx = b + h * stride_h + base_w;
                sum += data[idx];
            }
        }
        let mean = sum / plane as f64;

        let mut sq_sum = 0.0;
        for w in 0..width {
            let base_w = w * stride_w;
            for h in 0..height {
                let idx = b + h * stride_h + base_w;
                let diff = data[idx] - mean;
                sq_sum += diff * diff;
            }
        }
        let variance = sq_sum / plane as f64;
        let sigma = (variance + epsilon).sqrt();
        let inv_sigma = if sigma > 0.0 { 1.0 / sigma } else { 0.0 };

        for w in 0..width {
            let base_w = w * stride_w;
            for h in 0..height {
                let idx = b + h * stride_h + base_w;
                let mut value = (data[idx] - mean) * inv_sigma;
                if let Some(g) = gain {
                    value *= g;
                }
                if let Some(bias) = bias {
                    value += bias;
                }
                value = value.max(0.0);
                if let Some(gamma) = gamma {
                    value = value.powf(gamma);
                }
                out[idx] = value;
            }
        }
    }

    out
}

#[cfg(feature = "wgpu")]
#[tokio::test]
async fn image_normalize_matches_cpu() {
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");

    let batch = 3usize;
    let height = 4usize;
    let width = 5usize;
    let epsilon = 1e-6;
    let gain = Some(1.05);
    let bias = Some(-0.02);
    let gamma = Some(1.8);

    let mut data = Vec::with_capacity(batch * height * width);
    for w in 0..width {
        for h in 0..height {
            for b in 0..batch {
                let value = (b as f64) + 0.1 * (h as f64) + 0.01 * (w as f64);
                data.push(value);
            }
        }
    }

    let handle = p
        .upload(&HostTensorView {
            data: &data,
            shape: &[batch, height, width],
        })
        .expect("upload input");

    let desc = ImageNormalizeDescriptor {
        batch,
        height,
        width,
        epsilon,
        gain,
        bias,
        gamma,
    };

    let gpu = p
        .image_normalize(&handle, &desc)
        .await
        .expect("image_normalize");
    let host = p.download(&gpu).await.expect("download");
    assert_eq!(host.shape, vec![batch, height, width]);

    let expected = cpu_image_normalize(&data, batch, height, width, epsilon, gain, bias, gamma);

    for (idx, (&got, &want)) in host.data.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= 5e-4,
            "mismatch at {}: got={} want={} diff={}",
            idx,
            got,
            want,
            diff
        );
    }
}

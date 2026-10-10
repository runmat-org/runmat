#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
use runmat_accelerate_api::{
    AccelProvider, GpuTensorStorage, HostNumericDataOwned, HostNumericDataView,
    HostNumericTensorView, HostTensorView, ImageNormalizeDescriptor, NumericElementType,
};

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
        clamp_zero: true,
    };

    let gpu = p
        .image_normalize(&handle, &desc)
        .await
        .expect("image_normalize");
    assert_eq!(
        gpu.physical().element_type(),
        handle.physical().element_type()
    );
    assert!(!gpu.shares_allocation_with(&handle));
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

#[cfg(feature = "wgpu")]
#[tokio::test]
async fn wgpu_single_image_normalize_keeps_near_constant_variance() {
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");
    assert_eq!(
        p.precision(),
        runmat_accelerate_api::ProviderPrecision::F32,
        "this witness must exercise the native single shader"
    );
    let base = 16_384.0f32;
    let deltas = [-0.25f32, 0.25, -0.5, 0.5];
    let mut data = Vec::with_capacity(8);
    for delta in deltas {
        data.extend([base + delta, base + delta]);
    }
    let shape = [2, 2, 2];
    let input = p
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&data),
            shape: &shape,
            storage: GpuTensorStorage::Real,
        })
        .expect("upload near-constant single input");
    let output = p
        .image_normalize(
            &input,
            &ImageNormalizeDescriptor {
                batch: 2,
                height: 2,
                width: 2,
                epsilon: 1e-3,
                gain: None,
                bias: None,
                gamma: None,
                clamp_zero: false,
            },
        )
        .await
        .expect("native single image normalization");
    assert_eq!(output.shape(), shape);
    assert_eq!(output.device_id(), p.device_id());
    assert_eq!(output.physical().element_type(), NumericElementType::F32);
    assert!(!output.shares_allocation_with(&input));
    let host = p
        .download_numeric(&output)
        .await
        .expect("download single output");
    let HostNumericDataOwned::F32(actual) = host.data else {
        panic!("native single shader changed output class");
    };
    let variance = deltas.iter().map(|delta| delta * delta).sum::<f32>() / 4.0;
    let sigma = (variance + 1e-3f32).sqrt();
    for (index, value) in actual.iter().enumerate() {
        let expected = deltas[index / 2] / sigma;
        assert!(
            (*value - expected).abs() <= 1e-3,
            "near-constant single variance mismatch at {index}: {value} vs {expected}"
        );
    }
}

#[cfg(feature = "wgpu")]
#[tokio::test]
async fn wgpu_cpu_fallback_image_normalize_rejects_invalid_epsilon() {
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let p = runmat_accelerate_api::provider().expect("provider");

    let handle = p
        .upload(&HostTensorView {
            data: &[],
            shape: &[1, 0, 2],
        })
        .expect("upload empty input");

    for (epsilon, expected) in [
        (f64::NAN, "image_normalize: epsilon must be finite"),
        (-1.0, "image_normalize: epsilon must be non-negative"),
    ] {
        let desc = ImageNormalizeDescriptor {
            batch: 1,
            height: 0,
            width: 2,
            epsilon,
            gain: None,
            bias: None,
            gamma: None,
            clamp_zero: true,
        };

        let err = p
            .image_normalize(&handle, &desc)
            .await
            .expect_err("invalid epsilon should be rejected by fallback path");
        assert!(
            err.to_string().contains(expected),
            "unexpected error for epsilon {epsilon:?}: {err}"
        );
    }
}

#[tokio::test]
async fn simple_provider_image_normalize_preserves_native_single_storage() {
    let provider = runmat_accelerate::simple_provider::InProcessProvider::new();
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let shape = [1, 2, 2];
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&data),
            shape: &shape,
            storage: GpuTensorStorage::Real,
        })
        .expect("upload native single input");
    let output = provider
        .image_normalize(
            &input,
            &ImageNormalizeDescriptor {
                batch: 1,
                height: 2,
                width: 2,
                epsilon: 0.0,
                gain: None,
                bias: None,
                gamma: None,
                clamp_zero: false,
            },
        )
        .await
        .expect("normalize native single input");
    assert_eq!(output.shape(), shape);
    assert_eq!(output.physical().element_type(), NumericElementType::F32);
    assert!(!output.shares_allocation_with(&input));
    let host = provider
        .download_numeric(&output)
        .await
        .expect("download native single output");
    let HostNumericDataOwned::F32(actual) = host.data else {
        panic!("image normalize changed native single output class");
    };
    let mean = 2.5f32;
    let inverse_sigma = 1.0f32 / 1.25f32.sqrt();
    for (actual, input) in actual.iter().zip(data) {
        assert!((actual - (input - mean) * inverse_sigma).abs() < 1e-6);
    }
}

#[tokio::test]
async fn simple_provider_image_normalize_rejects_invalid_epsilon() {
    let provider = runmat_accelerate::simple_provider::InProcessProvider::new();
    let data = [1.0, 2.0, 3.0, 4.0];
    let handle = provider
        .upload(&HostTensorView {
            data: &data,
            shape: &[1, 2, 2],
        })
        .expect("upload input");

    for (epsilon, expected) in [
        (f64::NAN, "image_normalize: epsilon must be finite"),
        (-1.0, "image_normalize: epsilon must be non-negative"),
    ] {
        let desc = ImageNormalizeDescriptor {
            batch: 1,
            height: 2,
            width: 2,
            epsilon,
            gain: None,
            bias: None,
            gamma: None,
            clamp_zero: true,
        };

        let err = provider
            .image_normalize(&handle, &desc)
            .await
            .expect_err("invalid epsilon should be rejected");
        assert!(
            err.to_string().contains(expected),
            "unexpected error for epsilon {epsilon:?}: {err}"
        );
    }
}

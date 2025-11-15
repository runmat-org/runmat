#![cfg(feature = "wgpu")]

use runmat_accelerate::backend::wgpu::provider::register_wgpu_provider;
use runmat_accelerate::backend::wgpu::provider_impl::WgpuProviderOptions;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorView};

fn upload_vector(provider: &dyn AccelProvider, data: &[f64]) -> GpuTensorHandle {
    provider
        .upload(&HostTensorView {
            data,
            shape: &[data.len(), 1],
        })
        .expect("upload vector")
}

#[test]
fn stochastic_evolution_handles_zero_scale() {
    let provider = register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");
    let drift = 0.05;
    let scale = 0.0;
    let steps = 4u32;
    let state = upload_vector(provider.as_ref(), &[1.0, 2.0, 3.0]);
    let evolved = provider
        .stochastic_evolution(&state, drift, scale, steps)
        .expect("stochastic evolution");
    let gathered = provider.download(&evolved).expect("download");
    let factor = (drift * steps as f64).exp();
    for (idx, value) in gathered.data.iter().enumerate() {
        let expected = (idx as f64 + 1.0) * factor;
        assert!(
            (value - expected).abs() < 1e-9,
            "value {} expected {}",
            value,
            expected
        );
    }
}

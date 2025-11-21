#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::HostTensorView;

#[cfg(feature = "wgpu")]
fn run_copy_shader(len: usize) {
    let provider = runmat_accelerate_api::provider().expect("provider");
    let data: Vec<f64> = (0..len).map(|i| (i % 7) as f64).collect();
    let shape = &[len];
    let view = HostTensorView { data: &data, shape };
    let handle = provider.upload(&view).expect("upload");

    // Elementwise copy shader with @WG@ sentinel
    let shader = r#"
struct Tensor { data: array<f32> };
struct Params { len: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<storage,read> input0: Tensor;
@group(0) @binding(1) var<storage,read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }
  output.data[i] = input0.data[i];
}
"#;

    let out = provider
        .fused_elementwise(shader, std::slice::from_ref(&handle), shape, len)
        .expect("fused_elementwise");
    let host = provider.download(&out).expect("download");
    assert_eq!(host.data.len(), len);
    for (i, (got, want)) in host.data.iter().zip(data.iter()).enumerate().take(len) {
        assert_eq!(got, want, "mismatch at {}", i);
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn elementwise_smoke_env_wg_256_512() {
    // Force f32 precision so the test shader (f32) matches device side
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu");

    // Flip to 256 and test odd sizes around boundaries
    std::env::set_var("RUNMAT_WG", "256");
    run_copy_shader(511);
    run_copy_shader(512);
    run_copy_shader(513);
    run_copy_shader(200_123);

    // Flip to 512 and re-run
    std::env::set_var("RUNMAT_WG", "512");
    run_copy_shader(511);
    run_copy_shader(512);
    run_copy_shader(513);
    run_copy_shader(200_123);
}

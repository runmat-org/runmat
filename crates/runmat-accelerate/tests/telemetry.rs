#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::{AccelProvider, HostTensorView};

#[cfg(feature = "wgpu")]
fn register_provider() -> &'static dyn AccelProvider {
    std::env::set_var("RUNMAT_WGPU_FORCE_PRECISION", "f32");
    // Registration is idempotent; ignore duplicate registrations from other tests.
    let _ = provider::register_wgpu_provider(WgpuProviderOptions::default());
    runmat_accelerate_api::provider().expect("wgpu provider registered")
}

#[cfg(feature = "wgpu")]
#[test]
fn telemetry_records_basic_dispatches() {
    let provider = register_provider();
    provider.reset_telemetry();

    let elem_shape = [8usize];
    let elem_len = elem_shape[0];
    let elem_data: Vec<f64> = (0..elem_len).map(|i| i as f64).collect();
    let elem = provider
        .upload(&HostTensorView {
            data: &elem_data,
            shape: &elem_shape,
        })
        .expect("upload elementwise tensor");

    // Simple copy shader adapted from workgroup smoke test to ensure GPU execution.
    std::env::set_var("RUNMAT_WG", "64");
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
  output.data[i] = input0.data[i] + 1.0;
}
"#;
    let _ = provider
        .fused_elementwise(shader, &[elem.clone()], &elem_shape, elem_len)
        .expect("fused_elementwise");

    let mat_shape = [2usize, 2usize];
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];
    let a = provider
        .upload(&HostTensorView {
            data: &a_data,
            shape: &mat_shape,
        })
        .expect("upload a");
    let b = provider
        .upload(&HostTensorView {
            data: &b_data,
            shape: &mat_shape,
        })
        .expect("upload b");

    let _prod = provider.matmul(&a, &b).expect("matmul");

    let telemetry = provider.telemetry_snapshot();
    assert!(telemetry.upload_bytes > 0, "expected upload bytes > 0");
    assert!(
        telemetry.fused_elementwise.count > 0,
        "expected fused elementwise dispatch count > 0"
    );
    assert!(
        telemetry.matmul.count > 0,
        "expected matmul dispatch count > 0"
    );
    assert!(
        !telemetry.kernel_launches.is_empty(),
        "expected at least one kernel launch to be recorded"
    );

    provider.reset_telemetry();
    let reset = provider.telemetry_snapshot();
    assert_eq!(reset.upload_bytes, 0);
    assert_eq!(reset.fused_elementwise.count, 0);
    assert_eq!(reset.matmul.count, 0);
    assert!(reset.kernel_launches.is_empty());
}

#[cfg(feature = "wgpu")]
#[test]
fn telemetry_records_bind_group_cache_hits_for_chunked_matmul() {
    let provider = register_provider();
    provider.reset_telemetry();

    let m = 32usize;
    let n = 32usize;
    let k = 131_072usize; // ensure chunked path (K_CHUNK_SWITCH = 65536)

    let a = provider.zeros(&[m, k]).expect("zeros a");
    let b = provider.zeros(&[k, n]).expect("zeros b");
    let c = provider.matmul(&a, &b).expect("matmul");

    let telemetry = provider.telemetry_snapshot();
    assert!(
        telemetry.bind_group_cache_hits > 0,
        "expected bind group cache hits > 0 for chunked matmul, got {}",
        telemetry.bind_group_cache_hits
    );

    let _ = provider.free(&a);
    let _ = provider.free(&b);
    let _ = provider.free(&c);
}

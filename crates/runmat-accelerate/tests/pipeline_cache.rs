#[cfg(feature = "wgpu")]
use runmat_accelerate::backend::wgpu::provider::{self, WgpuProviderOptions};
#[cfg(feature = "wgpu")]
use runmat_accelerate_api::HostTensorView;

#[cfg(feature = "wgpu")]
#[test]
fn pipeline_cache_meta_and_hits_increase_on_second_run() {
    // Initialize provider
    let _ =
        provider::register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");
    let provider = runmat_accelerate_api::provider().expect("provider");

    // Small matrix and simple elementwise fuse (add) via fused_elementwise path
    let rows = 64usize;
    let cols = 4usize;
    let data = vec![1.0f64; rows * cols];
    let view = HostTensorView {
        data: &data,
        shape: &[rows, cols],
    };
    let handle = provider.upload(&view).expect("upload");

    // Track initial cache stats
    let (before_hits, before_misses) = provider.fused_cache_counters();

    // Minimal elementwise shader that copies input to output
    let shader = r#"
struct Tensor { data: array<f32> };
struct Params { len: u32, _pad0: u32, _pad1: u32, _pad2: u32 }
@group(0) @binding(0) var<storage,read> input0: Tensor;
@group(0) @binding(1) var<storage,read_write> output: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }
  output.data[i] = input0.data[i];
}
"#;

    // First run: expect miss and metadata creation
    let _out = provider
        .fused_elementwise(shader, &[handle.clone()], &[rows * cols], rows * cols)
        .expect("fused_elementwise");
    let (hits1, misses1) = provider.fused_cache_counters();
    assert!(misses1 >= before_misses, "misses should not decrease");

    // Second run: should hit cache
    let _out2 = provider
        .fused_elementwise(shader, &[handle.clone()], &[rows * cols], rows * cols)
        .expect("fused_elementwise 2");
    let (hits2, _misses2) = provider.fused_cache_counters();
    assert!(
        hits2 > before_hits || hits2 > hits1,
        "hits should increase on second run"
    );

    // Warmup-from-disk smoke: ensure meta/wgsl were persisted; simulate a new process by
    // recomputing the hash and checking that files exist. We cannot reinitialize the provider
    // in-process here, so this simply validates that persistence occurred.
    let p = provider::ensure_wgpu_provider().unwrap().unwrap();
    let layout_tag = format!("runmat-fusion-layout-{}", 1);
    let key = p.compute_pipeline_hash_bytes(shader.as_bytes(), &layout_tag, Some(256));
    let cache_dir = std::env::var("RUNMAT_PIPELINE_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .ok()
        .or_else(|| {
            dirs::cache_dir().map(|b| {
                b.join("runmat")
                    .join("pipelines")
                    .join(format!("device-{}", p.device_id()))
            })
        })
        .unwrap_or_else(|| {
            std::path::PathBuf::from("target")
                .join("tmp")
                .join(format!("wgpu-pipeline-cache-{}", p.device_id()))
        });
    let wgsl_path = cache_dir.join(format!("{:016x}.wgsl", key));
    let json_path = cache_dir.join(format!("{:016x}.json", key));
    assert!(
        wgsl_path.exists(),
        "expected wgsl persisted at {:?}",
        wgsl_path
    );
    assert!(
        json_path.exists(),
        "expected json meta persisted at {:?}",
        json_path
    );
}

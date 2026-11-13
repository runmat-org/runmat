//! Native filesystem persistence and warm-cache integration test.
#![cfg(not(target_arch = "wasm32"))]

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
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.len) { return; }
  output.data[i] = input0.data[i];
}
"#;
    let (shader, output_class) = match provider.precision() {
        runmat_accelerate_api::ProviderPrecision::F32 => (
            shader.to_owned(),
            runmat_accelerate_api::ProviderElementType::F32,
        ),
        runmat_accelerate_api::ProviderPrecision::F64 => (
            shader.replace("f32", "f64"),
            runmat_accelerate_api::ProviderElementType::F64,
        ),
    };

    // First run: expect miss and metadata creation
    let out = futures::executor::block_on(provider.fused_elementwise(
        &shader,
        runmat_accelerate_api::ProviderShaderRequirements::Basic,
        std::slice::from_ref(&handle),
        &[rows * cols],
        rows * cols,
        output_class,
    ))
    .expect("fused_elementwise");
    assert_eq!(
        futures::executor::block_on(provider.download(&out))
            .expect("first fused output")
            .data,
        data
    );
    let (hits1, misses1) = provider.fused_cache_counters();
    assert!(misses1 >= before_misses, "misses should not decrease");

    // Second run: should hit cache
    let out2 = futures::executor::block_on(provider.fused_elementwise(
        &shader,
        runmat_accelerate_api::ProviderShaderRequirements::Basic,
        std::slice::from_ref(&handle),
        &[rows * cols],
        rows * cols,
        output_class,
    ))
    .expect("fused_elementwise 2");
    assert_eq!(
        futures::executor::block_on(provider.download(&out2))
            .expect("warm fused output")
            .data,
        data
    );
    let (hits2, _misses2) = provider.fused_cache_counters();
    assert!(
        hits2 > before_hits || hits2 > hits1,
        "hits should increase on second run"
    );

    // Warmup-from-disk smoke: ensure meta/wgsl were persisted; simulate a new process by
    // recomputing the hash and checking that files exist. We cannot reinitialize the provider
    // in-process here, so this simply validates that persistence occurred.
    let p = provider::ensure_wgpu_provider().unwrap().unwrap();
    let workgroup = runmat_accelerate::backend::wgpu::config::effective_workgroup_size();
    let selected_shader = shader.replace("@WG@", &workgroup.to_string());
    let layout_tag = "runmat-fusion-layout-1";
    let key =
        p.compute_pipeline_hash_bytes(selected_shader.as_bytes(), layout_tag, Some(workgroup));
    let cache_dir = std::env::var("RUNMAT_PIPELINE_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .ok()
        .or_else(|| {
            dirs::cache_dir().map(|b| {
                b.join("runmat")
                    .join("pipelines")
                    .join(format!("device-{}", p.pipeline_cache_device_id()))
            })
        })
        .unwrap_or_else(|| {
            std::path::PathBuf::from("target").join("tmp").join(format!(
                "wgpu-pipeline-cache-{}",
                p.pipeline_cache_device_id()
            ))
        });
    let wgsl_path = cache_dir.join(format!("{:016x}.wgsl", key));
    let json_path = cache_dir.join(format!("{:016x}.json", key));
    assert_eq!(
        std::fs::read(&wgsl_path).expect("selected WGSL persisted at canonical key"),
        selected_shader.as_bytes(),
        "persisted shader differs from executed source at {wgsl_path:?}"
    );
    let meta: runmat_accelerate::backend::wgpu::cache::persist::PipelineMeta =
        serde_json::from_slice(
            &std::fs::read(&json_path).expect("pipeline metadata persisted at canonical key"),
        )
        .expect("pipeline metadata schema");
    assert_eq!(meta.label, "runmat-fusion-pipeline");
    assert_eq!(meta.layout_tag.as_deref(), Some(layout_tag));
    assert_eq!(meta.workgroup_size, Some(workgroup));
    assert_eq!(
        meta.version,
        Some(runmat_accelerate::backend::wgpu::cache::persist::PIPELINE_CACHE_VERSION)
    );
    assert_eq!(
        meta.precision.as_deref(),
        Some(match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F32 => "f32",
            runmat_accelerate_api::ProviderPrecision::F64 => "f64",
        })
    );
}

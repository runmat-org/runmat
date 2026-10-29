//! Explicit hardware qualification of successful native disk replay.
use super::{WgpuProvider, WgpuProviderOptions};
use runmat_accelerate_api::{
    AccelProvider, HostTensorView, ProviderElementType, ProviderShaderRequirements,
};
use std::path::PathBuf;

struct OwnedCacheDirectory(PathBuf);

impl OwnedCacheDirectory {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "runmat-valid-cache-replay-{}-{nonce}",
            std::process::id()
        ));
        // Only clean up a directory that this fixture created exclusively.
        std::fs::create_dir(&path).expect("create isolated replay cache");
        Self(path)
    }
}

impl Drop for OwnedCacheDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires actual compatible WGPU adapter; run explicitly with --ignored"]
fn valid_disk_replay_preserves_canonical_provenance_and_device_usability() {
    let cache = OwnedCacheDirectory::new();
    let mut provider =
        WgpuProvider::new(WgpuProviderOptions::default()).expect("required actual WGPU adapter");
    provider.pipeline_cache_dir = Some(cache.0.clone());
    assert!(
        provider.ieee_words_available(provider.precision),
        "selected precision must admit disk replay on this adapter"
    );
    assert!(
        std::env::var_os("RUNMAT_DISABLE_PIPELINE_WARMUP").is_none(),
        "qualification requires disk warmup enabled"
    );
    assert!(!provider.access.has_raw_export());

    // A real, valid storage-writing compute shader with the existing fill layout.
    // Extra unused bindings in the explicit layout are permitted by WGPU.
    let source = "@group(0) @binding(0) var<storage, read_write> output: array<u32>;\n\
                  @compute @workgroup_size(1) fn main() { output[0] = 42u; }";
    let tag = "runmat-fill-layout";
    let key = provider.compute_pipeline_hash_bytes(source.as_bytes(), tag, Some(1));
    crate::backend::wgpu::cache::persist::persist_pipeline_meta(
        &cache.0,
        key,
        "valid-disk-replay-test",
        Some(tag),
        Some(1),
        provider.precision,
        Some(source.as_bytes()),
    );
    assert!(cache.0.join(format!("{key:016x}.json")).is_file());
    assert!(cache.0.join(format!("{key:016x}.wgsl")).is_file());
    let (word, class) = match provider.precision {
        crate::backend::wgpu::types::NumericPrecision::F32 => ("f32", ProviderElementType::F32),
        crate::backend::wgpu::types::NumericPrecision::F64 => ("f64", ProviderElementType::F64),
    };
    let fused_source = format!(
        "@group(0) @binding(0) var<storage, read> input: array<{word}>;
         @group(0) @binding(1) var<storage, read_write> output: array<{word}>;
         struct Params {{ len: u32, offset: u32, rank: u32, pad: u32 }};
         @group(0) @binding(2) var<uniform> params: Params;
         @compute @workgroup_size(@WG@)
         fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
             if (id.x >= params.len) {{ return; }}
             let j = params.offset + id.x;
             output[j] = input[j] + {word}(1.0);
         }}"
    );
    let fused_recipe = super::super::pipeline_recipe::PipelineRecipe::fusion(
        &fused_source,
        1,
        super::super::pipeline_recipe::FusionPipelineLayout::Single,
        provider.linear_launch.workgroup(),
    );
    let fused_key = fused_recipe.key();
    crate::backend::wgpu::cache::persist::persist_pipeline_meta(
        &cache.0,
        fused_key,
        "disk-label-is-not-a-runtime-kind",
        Some(&fused_recipe.layout),
        Some(fused_recipe.workgroup),
        provider.precision,
        Some(fused_recipe.shader.as_bytes()),
    );
    let before = provider.fused_pipeline_cache.lock().unwrap().len();
    assert!(!provider
        .fused_pipeline_cache
        .lock()
        .unwrap()
        .contains_key(&key));

    provider.warmup_from_disk();

    let pipelines = provider.fused_pipeline_cache.lock().unwrap();
    assert!(
        pipelines.contains_key(&key),
        "successful replay must publish"
    );
    assert_eq!(pipelines.len(), before + 2);
    drop(pipelines);
    let warmed = provider
        .warmed_pipeline_cache
        .lock()
        .unwrap()
        .get(&fused_key)
        .cloned()
        .expect("exact fused recipe has a driver-validated warm receipt");
    assert_eq!(warmed.recipe, fused_recipe);
    assert!(!provider
        .validated_fusion_cache
        .lock()
        .unwrap()
        .contains_key(&fused_key));
    assert!(
        !provider.access.has_raw_export(),
        "internal disk replay must retain canonical device provenance"
    );

    // Exercise production upload, compute, submission, and readback after replay.
    // The legacy fill shader above is not dispatched; the exact fused shader is.
    pollster::block_on(async {
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0],
                shape: &[1, 3],
            })
            .expect("post-replay upload");
        let output = provider
            .elem_add(&input, &input)
            .await
            .expect("post-replay actual compute");
        let host = provider
            .download(&output)
            .await
            .expect("post-replay readback");
        assert_eq!(host.data, [2.0, 4.0, 6.0]);
        let misses = provider.telemetry_snapshot().fusion_cache_misses;
        let fused = provider
            .fused_elementwise(
                &fused_source,
                ProviderShaderRequirements::Basic,
                std::slice::from_ref(&input),
                &[3],
                3,
                class,
            )
            .await
            .expect("actual fused dispatch after disk replay");
        assert_eq!(
            provider.download(&fused).await.unwrap().data,
            [2.0, 3.0, 4.0]
        );
        let selected = provider
            .validated_fusion_cache
            .lock()
            .unwrap()
            .get(&fused_key)
            .cloned()
            .expect("typed runtime admission must promote the warmed receipt");
        assert!(selected.pipeline.ptr_eq(&warmed.pipeline));
        assert_eq!(provider.telemetry_snapshot().fusion_cache_misses, misses);
    });
}

//! Explicit hardware qualification of successful native disk replay.
use super::{WgpuProvider, WgpuProviderOptions};
use runmat_accelerate_api::{AccelProvider, HostTensorView};
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
    assert_eq!(pipelines.len(), before + 1);
    drop(pipelines);
    assert!(
        !provider.access.has_raw_export(),
        "internal disk replay must retain canonical device provenance"
    );

    // Exercise production upload, compute, submission, and readback after replay.
    // This does not qualify execution of the replayed shader itself.
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
    });
}

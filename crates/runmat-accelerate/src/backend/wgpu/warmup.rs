use super::cache::persist::PipelineMeta;
use super::cache::persist::PIPELINE_CACHE_VERSION;
use super::types::NumericPrecision;
use std::path::Path;

mod shader;

pub fn warmup_from_disk<FHash, FCompile>(
    cache_dir: Option<&Path>,
    target_precision: NumericPrecision,
    device_features: wgpu::Features,
    compute_hash: FHash,
    compile: FCompile,
) where
    FHash: Fn(&[u8], &str, Option<u32>) -> u64,
    FCompile: Fn(u64, &str, &str, &[u8], Option<u32>) -> anyhow::Result<bool>,
{
    let Some(dir) = cache_dir else {
        return;
    };
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let mut compiled = 0usize;
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };
        let meta_bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let meta: PipelineMeta = match serde_json::from_slice(&meta_bytes) {
            Ok(m) => m,
            Err(error) => {
                log::warn!("warmup: removing malformed pipeline metadata {stem}: {error}");
                let _ = std::fs::remove_file(&path);
                let _ = std::fs::remove_file(dir.join(format!("{stem}.wgsl")));
                continue;
            }
        };
        // Skip stale or incompatible cache entries silently
        if meta.version.unwrap_or(0) != PIPELINE_CACHE_VERSION {
            continue;
        }
        match meta.precision.as_deref() {
            Some(stored) if stored == target_precision.as_str() => {}
            Some(_) => {
                continue;
            }
            None => {
                // Missing precision metadata (likely stale entry); skip
                continue;
            }
        }
        let layout_tag = match meta.layout_tag.as_deref() {
            Some(t) => t,
            None => continue,
        };
        let wgsl_path = dir.join(format!("{stem}.wgsl"));
        let wgsl_bytes = match std::fs::read(&wgsl_path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        let wgsl_str = match std::str::from_utf8(&wgsl_bytes) {
            Ok(s) => s,
            Err(error) => {
                log::warn!("warmup: removing non-UTF8 cached shader {stem}: {error}");
                let _ = std::fs::remove_file(&path);
                let _ = std::fs::remove_file(&wgsl_path);
                continue;
            }
        };
        let key = compute_hash(&wgsl_bytes, layout_tag, meta.workgroup_size);
        if stem != format!("{key:016x}") {
            log::warn!(
                "warmup: removing pipeline cache entry {} because its shader contents do not match its cache key",
                stem
            );
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(&wgsl_path);
            continue;
        }
        match shader::validate(wgsl_str, device_features) {
            Ok(()) => {}
            Err(shader::CachedShaderAdmissionError::InvalidSource(error)) => {
                log::warn!("warmup: removing invalid cached pipeline {stem}: {error}");
                let _ = std::fs::remove_file(&path);
                let _ = std::fs::remove_file(&wgsl_path);
                continue;
            }
            Err(shader::CachedShaderAdmissionError::IncompatibleAdapter(error)) => {
                log::debug!("warmup: retaining cache entry {stem} for another adapter: {error}");
                continue;
            }
        }
        // Parsing/IO stay outside device access. The owner validates driver
        // construction before publishing or reporting a successfully warmed entry.
        // Validation failures are ordinary per-entry rejections. An unexpected
        // driver panic must propagate: its exclusive device gate may be
        // poisoned, so continuing would falsely claim a recovered provider.
        match compile(key, layout_tag, wgsl_str, &wgsl_bytes, meta.workgroup_size) {
            Ok(true) => {
                compiled += 1;
            }
            Ok(false) => continue,
            Err(error) => {
                // Naga already proved source validity independently of this
                // adapter. Scoped driver failures can be capability, limit,
                // memory, or device failures; none proves disk corruption.
                log::warn!("warmup: could not precompile retained pipeline {stem}: {error}");
                if let Some(scoped) = error.chain().find_map(|cause| {
                    cause.downcast_ref::<runmat_accelerate_api::WgpuScopeErrors>()
                }) {
                    scoped.with_errors(|errors| {
                        for original in errors {
                            log::warn!(
                                "warmup: retained pipeline {stem} scoped driver cause: {original}"
                            );
                        }
                    });
                }
                continue;
            }
        }
    }
    if compiled > 0 {
        log::info!(
            "warmup: precompiled {} pipelines from on-disk cache",
            compiled
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::cache::{key, persist};
    use std::cell::Cell;
    use std::path::PathBuf;

    struct OwnedCacheDirectory(PathBuf);

    impl OwnedCacheDirectory {
        fn new() -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "runmat-warmup-admission-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir(&path).expect("create isolated warmup cache");
            Self(path)
        }
    }

    impl Drop for OwnedCacheDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn cache_abi_version_is_checked_before_compilation_and_stale_records_are_retained() {
        let cache = OwnedCacheDirectory::new();
        let source = "@compute @workgroup_size(1) fn main() {}";
        let tag = "runmat-fill-layout";
        let key = key::compute_pipeline_hash_bytes(source.as_bytes(), tag, Some(1));
        persist::persist_pipeline_meta(
            &cache.0,
            key,
            "warmup-version-test",
            Some(tag),
            Some(1),
            NumericPrecision::F32,
            Some(source.as_bytes()),
        );
        let meta_path = cache.0.join(format!("{key:016x}.json"));
        let mut meta: PipelineMeta =
            serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
        assert_eq!(meta.version, Some(PIPELINE_CACHE_VERSION));
        let compiled = Cell::new(0);
        let warm = || {
            warmup_from_disk(
                Some(&cache.0),
                NumericPrecision::F32,
                wgpu::Features::empty(),
                key::compute_pipeline_hash_bytes,
                |_, _, _, _, _| {
                    compiled.set(compiled.get() + 1);
                    Ok(true)
                },
            );
        };
        warm();
        assert_eq!(compiled.get(), 1);
        meta.version = Some(PIPELINE_CACHE_VERSION - 1);
        let stale_metadata = serde_json::to_vec(&meta).unwrap();
        std::fs::write(&meta_path, &stale_metadata).unwrap();
        warm();
        assert_eq!(compiled.get(), 1, "stale ABI must not reach compilation");
        assert_eq!(std::fs::read(meta_path).unwrap(), stale_metadata);
        assert_eq!(
            std::fs::read(cache.0.join(format!("{key:016x}.wgsl"))).unwrap(),
            source.as_bytes(),
        );
    }

    #[test]
    fn invalid_source_is_evicted_but_valid_adapter_incompatible_source_is_retained() {
        let cache = OwnedCacheDirectory::new();
        let tag = "runmat-fill-layout";
        let invalid = "@compute @workgroup_size(@WG@) fn main() {}";
        let incompatible = "@compute @workgroup_size(1) fn main() { let x = f64(1.0); }";
        let write = |source: &str| {
            let key = key::compute_pipeline_hash_bytes(source.as_bytes(), tag, Some(1));
            persist::persist_pipeline_meta(
                &cache.0,
                key,
                "warmup-admission-test",
                Some(tag),
                Some(1),
                NumericPrecision::F32,
                Some(source.as_bytes()),
            );
            key
        };
        let invalid_key = write(invalid);
        let incompatible_key = write(incompatible);
        let compiled = Cell::new(0);
        warmup_from_disk(
            Some(&cache.0),
            NumericPrecision::F32,
            wgpu::Features::empty(),
            key::compute_pipeline_hash_bytes,
            |_, _, _, _, _| {
                compiled.set(compiled.get() + 1);
                Ok(true)
            },
        );
        assert_eq!(compiled.get(), 0);
        for extension in ["json", "wgsl"] {
            assert!(!cache
                .0
                .join(format!("{invalid_key:016x}.{extension}"))
                .exists());
            assert!(cache
                .0
                .join(format!("{incompatible_key:016x}.{extension}"))
                .exists());
        }
    }
}

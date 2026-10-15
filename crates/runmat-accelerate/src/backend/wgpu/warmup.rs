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
            Err(_) => continue,
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
            Err(_) => continue,
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
        if let Err(error) = shader::validate(wgsl_str, device_features) {
            log::warn!("warmup: rejecting cached pipeline {stem}: {error}");
            continue;
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
                log::warn!(
                    "warmup: failed to precompile pipeline {stem}; removing incompatible cache entry: {error}"
                );
                let _ = std::fs::remove_file(&path);
                let _ = std::fs::remove_file(&wgsl_path);
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

use super::cache::persist::PipelineMeta;
use super::cache::persist::PIPELINE_CACHE_VERSION;
use super::types::NumericPrecision;
use std::panic::{self, AssertUnwindSafe};
use std::path::Path;

pub fn warmup_from_disk<FHash, FCompile>(
    cache_dir: Option<&Path>,
    target_precision: NumericPrecision,
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
        // Parsing/IO stay outside device access. The owner encloses one complete
        // compile+submit transaction and releases access before progress waits.
        let compiled_pipeline = panic::catch_unwind(AssertUnwindSafe(|| {
            compile(key, layout_tag, wgsl_str, &wgsl_bytes, meta.workgroup_size)
        }));
        match compiled_pipeline {
            Ok(Ok(true)) => {
                compiled += 1;
            }
            Ok(Ok(false)) => continue,
            Ok(Err(_)) | Err(_) => {
                log::warn!(
                    "warmup: failed to precompile pipeline {}; removing incompatible cache entry",
                    stem
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

pub(crate) fn noop_after_create(
    emission: &runmat_accelerate_api::WgpuEmission<'_>,
    pending: &mut super::dispatch::observation::PendingSubmissions,
    pipeline: &wgpu::ComputePipeline,
) -> anyhow::Result<()> {
    let mut enc = emission
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("warmup-noop-precompiled-enc"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("warmup-noop-precompiled-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
    }
    super::dispatch::common::submit(emission, pending, enc)?;
    Ok(())
}

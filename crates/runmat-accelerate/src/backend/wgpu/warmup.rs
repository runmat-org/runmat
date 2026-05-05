use std::panic::{self, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use futures::executor::block_on;

use super::bindings::build_bgl_for_layout_tag;
use super::cache::persist::PipelineMeta;
use super::cache::persist::PIPELINE_CACHE_VERSION;
use super::types::NumericPrecision;

#[cfg(not(target_arch = "wasm32"))]
fn pop_validation_scope(device: &wgpu::Device) -> Option<wgpu::Error> {
    device.poll(wgpu::Maintain::Wait);
    block_on(device.pop_error_scope())
}

fn remove_cache_entry(meta_path: &Path, wgsl_path: &Path) {
    let _ = std::fs::remove_file(meta_path);
    let _ = std::fs::remove_file(wgsl_path);
}

pub fn warmup_from_disk<FHash, FCreate, FNoop, FRemove>(
    device: &wgpu::Device,
    cache_dir: Option<&Path>,
    target_precision: NumericPrecision,
    compute_hash: FHash,
    get_or_create: FCreate,
    after_create_noop: FNoop,
    remove_cached_pipeline: FRemove,
) where
    FHash: Fn(&[u8], &str, Option<u32>) -> u64,
    FCreate: Fn(
        u64,
        &wgpu::PipelineLayout,
        &wgpu::ShaderModule,
        &str,
        Option<&[u8]>,
        Option<&str>,
        Option<u32>,
    ) -> Arc<wgpu::ComputePipeline>,
    FNoop: Fn(&wgpu::ComputePipeline),
    FRemove: Fn(u64),
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
        let bgl = match build_bgl_for_layout_tag(device, layout_tag) {
            Some(b) => b,
            None => continue,
        };
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("warmup-pipeline-layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        // Apply the @WG@ substitution used by regular pipeline creation
        let module = crate::backend::wgpu::pipelines::create_shader_module(
            device,
            "warmup-shader-module",
            wgsl_str,
        );
        let key = compute_hash(&wgsl_bytes, layout_tag, meta.workgroup_size);
        let compiled_pipeline = panic::catch_unwind(AssertUnwindSafe(|| -> bool {
            #[cfg(not(target_arch = "wasm32"))]
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = get_or_create(
                key,
                &pl,
                &module,
                "warmup-precompiled-pipeline",
                Some(&wgsl_bytes),
                Some(layout_tag),
                meta.workgroup_size,
            );

            #[cfg(not(target_arch = "wasm32"))]
            if let Some(err) = pop_validation_scope(device) {
                log::warn!(
                    "warmup: invalid cached compute pipeline {}: {}; removing incompatible cache entry",
                    stem,
                    err
                );
                remove_cached_pipeline(key);
                remove_cache_entry(&path, &wgsl_path);
                return false;
            }

            #[cfg(not(target_arch = "wasm32"))]
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            after_create_noop(&pipeline);

            #[cfg(not(target_arch = "wasm32"))]
            if let Some(err) = pop_validation_scope(device) {
                log::warn!(
                    "warmup: cached pipeline {} failed noop validation: {}; removing incompatible cache entry",
                    stem,
                    err
                );
                remove_cached_pipeline(key);
                remove_cache_entry(&path, &wgsl_path);
                return false;
            }
            true
        }));
        match compiled_pipeline {
            Ok(true) => {
                compiled += 1;
            }
            Ok(false) => continue,
            Err(_) => {
                log::warn!(
                    "warmup: failed to precompile pipeline {}; removing incompatible cache entry",
                    stem
                );
                remove_cached_pipeline(key);
                remove_cache_entry(&path, &wgsl_path);
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

pub fn noop_after_create(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("warmup-noop-precompiled-enc"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("warmup-noop-precompiled-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
    }
    queue.submit(Some(enc.finish()));
    device.poll(wgpu::Maintain::Wait);
}

use std::path::Path;
use std::sync::Arc;

use super::bindings::build_bgl_for_layout_tag;
use super::cache::persist::PipelineMeta;

pub fn warmup_from_disk<FHash, FCreate, FNoop>(
    device: &wgpu::Device,
    cache_dir: Option<&Path>,
    compute_hash: FHash,
    get_or_create: FCreate,
    after_create_noop: FNoop,
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
{
    let Some(dir) = cache_dir else { return; };
    let Ok(rd) = std::fs::read_dir(dir) else { return; };
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
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("warmup-shader-module"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl_str)),
        });
        let key = compute_hash(&wgsl_bytes, layout_tag, meta.workgroup_size);
        let pipeline = get_or_create(
            key,
            &pl,
            &module,
            "warmup-precompiled-pipeline",
            Some(&wgsl_bytes),
            Some(layout_tag),
            meta.workgroup_size,
        );
        after_create_noop(&pipeline);
        compiled += 1;
    }
    if compiled > 0 {
        log::info!("warmup: precompiled {} pipelines from on-disk cache", compiled);
    }
}

pub fn noop_after_create(device: &wgpu::Device, queue: &wgpu::Queue, pipeline: &wgpu::ComputePipeline) {
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



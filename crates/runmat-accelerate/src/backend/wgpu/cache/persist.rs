use serde::{Deserialize, Serialize};

use crate::backend::wgpu::types::NumericPrecision;

/// Bump this when bind group layouts or shader-binding schemas change.
pub const PIPELINE_CACHE_VERSION: u32 = 4;

#[derive(Serialize, Deserialize)]
pub struct PipelineMeta {
    pub label: String,
    pub layout_tag: Option<String>,
    pub workgroup_size: Option<u32>,
    /// Optional to allow reading older cache files; when absent, treat as incompatible.
    pub version: Option<u32>,
    pub precision: Option<String>,
}

pub fn persist_pipeline_meta(
    cache_dir: &std::path::Path,
    hash_key: u64,
    label: &str,
    layout_tag: Option<&str>,
    workgroup_size: Option<u32>,
    precision: NumericPrecision,
    wgsl_src: Option<&[u8]>,
) {
    let _ = std::fs::create_dir_all(cache_dir);
    if let Some(src) = wgsl_src {
        let wgsl_path = cache_dir.join(format!("{hash_key:016x}.wgsl"));
        let _ = std::fs::write(&wgsl_path, src);
    }
    let meta = PipelineMeta {
        label: label.to_string(),
        layout_tag: layout_tag.map(|s| s.to_string()),
        workgroup_size,
        version: Some(PIPELINE_CACHE_VERSION),
        precision: Some(precision.as_str().to_string()),
    };
    let meta_path = cache_dir.join(format!("{hash_key:016x}.json"));
    if let Ok(json) = serde_json::to_vec_pretty(&meta) {
        let _ = std::fs::write(&meta_path, json);
    }
}

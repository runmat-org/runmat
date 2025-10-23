use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct PipelineMeta {
    pub label: String,
    pub layout_tag: Option<String>,
    pub workgroup_size: Option<u32>,
}

pub fn persist_pipeline_meta(
    cache_dir: &std::path::Path,
    hash_key: u64,
    label: &str,
    layout_tag: Option<&str>,
    workgroup_size: Option<u32>,
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
    };
    let meta_path = cache_dir.join(format!("{hash_key:016x}.json"));
    if let Ok(json) = serde_json::to_vec_pretty(&meta) {
        let _ = std::fs::write(&meta_path, json);
    }
}

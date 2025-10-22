use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn compute_pipeline_hash_bytes(
    shader_bytes: &[u8],
    layout_tag: &str,
    workgroup_size: Option<u32>,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    shader_bytes.hash(&mut hasher);
    layout_tag.hash(&mut hasher);
    if let Some(wg) = workgroup_size {
        wg.hash(&mut hasher);
    }
    hasher.finish()
}



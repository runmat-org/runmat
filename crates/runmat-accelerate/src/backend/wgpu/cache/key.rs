use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn compute_pipeline_hash_bytes(
    shader_bytes: &[u8],
    layout_tag: &str,
    workgroup_size: Option<u32>,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    crate::wgsl::f64_math::COMPILER_REVISION.hash(&mut hasher);
    shader_bytes.hash(&mut hasher);
    layout_tag.hash(&mut hasher);
    if let Some(wg) = workgroup_size {
        wg.hash(&mut hasher);
    }
    hasher.finish()
}

pub fn has_unresolved_launch_tokens(source: &[u8]) -> bool {
    source.windows(b"@WG@".len()).any(|part| part == b"@WG@")
        || source.windows(b"@MT@".len()).any(|part| part == b"@MT@")
}

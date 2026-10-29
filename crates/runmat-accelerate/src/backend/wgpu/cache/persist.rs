use runmat_execution::resource::{ProviderHostMetadataAdmission, RetainedHostAllocationClaim};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::backend::wgpu::types::NumericPrecision;

/// Bump this when bind group layouts, shader-binding schemas, or the accepted
/// WGSL language version changes.
// Version 8 retires the four-binding softmax ABI. Its former serialized layout
// tag is now reconstructed with three bindings, so version 7 cannot be warmed
// using the current layout recipes even when its source hash is intact.
// Version 9 introduces typed binary64 legalization and controlled SPIR-V;
// original-source identities now include the compiler revision domain.
pub const PIPELINE_CACHE_VERSION: u32 = 9;

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
    // Offline warmup is explicitly unmanaged. Selected finite calls prepare
    // the same owner with their actual metadata admission before publication.
    if let Ok(prepared) = PreparedPipelinePersistence::prepare(cache_dir, hash_key, None) {
        prepared.write(
            cache_dir,
            label,
            layout_tag,
            workgroup_size,
            precision,
            wgsl_src,
        );
    }
}

/// One exact first-party path backing; metadata and WGSL remain borrowed and
/// JSON streams directly to the file. OS/filesystem internals are not quoted.
pub(crate) struct PreparedPipelinePersistence {
    path: Vec<u8>,
    // Field order keeps this claim through physical path deallocation.
    _claim: Option<Arc<dyn RetainedHostAllocationClaim>>,
}

impl PreparedPipelinePersistence {
    pub(crate) fn prepare(
        cache_dir: &std::path::Path,
        hash_key: u64,
        metadata: Option<&dyn ProviderHostMetadataAdmission>,
    ) -> anyhow::Result<Self> {
        let prefix = cache_dir.as_os_str().as_encoded_bytes();
        let needs_separator = !prefix.is_empty()
            && !prefix
                .last()
                .is_some_and(|byte| std::path::is_separator(char::from(*byte)));
        #[cfg(windows)]
        // Match Path::join for a drive-relative directory such as `C:`.
        let needs_separator = needs_separator
            && !matches!(
                cache_dir.components().next_back(),
                Some(std::path::Component::Prefix(prefix))
                    if matches!(prefix.kind(), std::path::Prefix::Disk(_))
            );
        let separator = if needs_separator {
            std::path::MAIN_SEPARATOR_STR.as_bytes()
        } else {
            b""
        };
        let len = prefix
            .len()
            .checked_add(separator.len())
            .and_then(|n| n.checked_add(21))
            .ok_or_else(|| anyhow::anyhow!("pipeline persistence path extent overflow"))?;
        let bytes = u64::try_from(len)?;
        let claim = metadata.map(|port| port.claim_record(bytes)).transpose()?;
        let mut path = runmat_value::exact_capacity_vec::<u8>(len)?;
        path.extend_from_slice(prefix);
        path.extend_from_slice(separator);
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for shift in (0..16).rev() {
            path.push(HEX[((hash_key >> (shift * 4)) & 15) as usize]);
        }
        path.extend_from_slice(b".wgsl");
        Ok(Self {
            path,
            _claim: claim,
        })
    }

    fn path(&self) -> &std::path::Path {
        // SAFETY: the unchanged prefix came from this process's OsStr encoded
        // bytes, followed only by complete ASCII separators/filename units.
        std::path::Path::new(unsafe { std::ffi::OsStr::from_encoded_bytes_unchecked(&self.path) })
    }

    pub(crate) fn write(
        mut self,
        cache_dir: &std::path::Path,
        label: &str,
        layout_tag: Option<&str>,
        workgroup_size: Option<u32>,
        precision: NumericPrecision,
        wgsl_src: Option<&[u8]>,
    ) {
        let _ = std::fs::create_dir_all(cache_dir);
        if let Some(source) = wgsl_src {
            let _ = std::fs::write(self.path(), source);
        }
        let len = self.path.len();
        self.path[len - 4..].copy_from_slice(b"json");
        #[derive(Serialize)]
        struct BorrowedMeta<'a> {
            label: &'a str,
            layout_tag: Option<&'a str>,
            workgroup_size: Option<u32>,
            version: Option<u32>,
            precision: Option<&'a str>,
        }
        let meta = BorrowedMeta {
            label,
            layout_tag,
            workgroup_size,
            version: Some(PIPELINE_CACHE_VERSION),
            precision: Some(precision.as_str()),
        };
        if let Ok(file) = std::fs::File::create(self.path()) {
            let _ = serde_json::to_writer_pretty(file, &meta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_path_scratch_preserves_directory_join_and_encoded_prefix() {
        let check = |directory: &std::path::Path| {
            let prepared = PreparedPipelinePersistence::prepare(directory, 42, None).unwrap();
            assert_eq!(prepared.path(), directory.join("000000000000002a.wgsl"));
            assert_eq!(prepared.path.len(), prepared.path.capacity());
        };
        for directory in ["", "cache", "cache/", "/"] {
            check(std::path::Path::new(directory));
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            check(std::path::Path::new(std::ffi::OsStr::from_bytes(
                b"cache-\xff",
            )));
        }
        #[cfg(windows)]
        for directory in ["C:", "C:\\", "\\\\server\\share"] {
            check(std::path::Path::new(directory));
        }
    }
}

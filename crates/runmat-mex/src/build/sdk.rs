use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest as _, Sha256};

use super::MexBuildError;

const FILES: [(&str, &[u8]); 28] = [
    ("include/matrix.h", include_bytes!("../../include/matrix.h")),
    ("include/mex.h", include_bytes!("../../include/mex.h")),
    ("include/fintrf.h", include_bytes!("../../include/fintrf.h")),
    (
        "include/gpu/mxGPUArray.h",
        include_bytes!("../../include/gpu/mxGPUArray.h"),
    ),
    ("include/mex.hpp", include_bytes!("../../include/mex.hpp")),
    (
        "include/mexAdapter.hpp",
        include_bytes!("../../include/mexAdapter.hpp"),
    ),
    (
        "include/MatlabDataArray.hpp",
        include_bytes!("../../include/MatlabDataArray.hpp"),
    ),
    (
        "include/MatlabEngine/Exception.hpp",
        include_bytes!("../../include/MatlabEngine/Exception.hpp"),
    ),
    (
        "include/MatlabEngine/FutureResult.hpp",
        include_bytes!("../../include/MatlabEngine/FutureResult.hpp"),
    ),
    (
        "include/MatlabEngine/StreamBuffer.hpp",
        include_bytes!("../../include/MatlabEngine/StreamBuffer.hpp"),
    ),
    (
        "include/MatlabEngine/TypeConversion.hpp",
        include_bytes!("../../include/MatlabEngine/TypeConversion.hpp"),
    ),
    (
        "include/MatlabDataArray/Array.hpp",
        include_bytes!("../../include/MatlabDataArray/Array.hpp"),
    ),
    (
        "include/MatlabDataArray/TypedArray.hpp",
        include_bytes!("../../include/MatlabDataArray/TypedArray.hpp"),
    ),
    (
        "include/MatlabDataArray/String.hpp",
        include_bytes!("../../include/MatlabDataArray/String.hpp"),
    ),
    (
        "include/MatlabDataArray/StringArray.hpp",
        include_bytes!("../../include/MatlabDataArray/StringArray.hpp"),
    ),
    (
        "include/MatlabDataArray/SparseArray.hpp",
        include_bytes!("../../include/MatlabDataArray/SparseArray.hpp"),
    ),
    (
        "include/MatlabDataArray/EnumArray.hpp",
        include_bytes!("../../include/MatlabDataArray/EnumArray.hpp"),
    ),
    (
        "include/MatlabDataArray/CellArray.hpp",
        include_bytes!("../../include/MatlabDataArray/CellArray.hpp"),
    ),
    (
        "include/MatlabDataArray/StructArray.hpp",
        include_bytes!("../../include/MatlabDataArray/StructArray.hpp"),
    ),
    (
        "include/MatlabDataArray/ArrayFactory.hpp",
        include_bytes!("../../include/MatlabDataArray/ArrayFactory.hpp"),
    ),
    (
        "include/runmat_mex_host.h",
        include_bytes!("../../include/runmat_mex_host.h"),
    ),
    (
        "src-c/runmat_mex_support.c",
        include_bytes!("../../src-c/runmat_mex_support.c"),
    ),
    (
        "src-c/data_engine.inc",
        include_bytes!("../../src-c/data_engine.inc"),
    ),
    (
        "src-c/sparse_index_compat.inc",
        include_bytes!("../../src-c/sparse_index_compat.inc"),
    ),
    (
        "src-c/matrix_api.inc",
        include_bytes!("../../src-c/matrix_api.inc"),
    ),
    (
        "src-c/mex_api.inc",
        include_bytes!("../../src-c/mex_api.inc"),
    ),
    (
        "src-c/fortran_api.inc",
        include_bytes!("../../src-c/fortran_api.inc"),
    ),
    (
        "src-c/gpu_api.inc",
        include_bytes!("../../src-c/gpu_api.inc"),
    ),
];

static TEMPORARY_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) struct MexSdk {
    pub include_directory: PathBuf,
    pub support_source: PathBuf,
}

pub(super) fn prepare() -> Result<MexSdk, MexBuildError> {
    let root = std::env::temp_dir()
        .join("runmat-mex-sdk")
        .join(format!("{:016x}", content_fingerprint()));
    for (relative, contents) in FILES {
        materialize(&root.join(relative), contents)?;
    }
    Ok(MexSdk {
        include_directory: root.join("include"),
        support_source: root.join("src-c/runmat_mex_support.c"),
    })
}

fn materialize(path: &Path, contents: &[u8]) -> Result<(), MexBuildError> {
    if fs::read(path).is_ok_and(|existing| existing == contents) {
        return Ok(());
    }
    let parent = path.parent().expect("embedded MEX SDK files have parents");
    fs::create_dir_all(parent).map_err(|source| MexBuildError::PrepareSdk {
        path: parent.to_path_buf(),
        source,
    })?;
    let sequence = TEMPORARY_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = path.with_extension(format!("tmp-{}-{sequence}", std::process::id()));
    let write_result = (|| {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(contents)?;
        file.sync_all()?;
        fs::rename(&temporary, path)
    })();
    if let Err(source) = write_result {
        let _ = fs::remove_file(&temporary);
        if fs::read(path).is_ok_and(|existing| existing == contents) {
            return Ok(());
        }
        return Err(MexBuildError::PrepareSdk {
            path: path.to_path_buf(),
            source,
        });
    }
    Ok(())
}

fn content_fingerprint() -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    FILES
        .iter()
        .flat_map(|(path, contents)| path.as_bytes().iter().chain(contents.iter()))
        .fold(FNV_OFFSET, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
        })
}

pub(super) fn content_digest() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"runmat-mex-sdk-v1\0");
    for (path, contents) in FILES {
        hasher.update((path.len() as u64).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update((contents.len() as u64).to_le_bytes());
        hasher.update(contents);
    }
    let digest = hasher.finalize();
    let mut encoded = String::with_capacity(71);
    encoded.push_str("sha256:");
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_sdk_materializes_complete_compiler_inputs() {
        let sdk = prepare().unwrap();
        assert!(sdk.include_directory.join("matrix.h").is_file());
        assert!(sdk.include_directory.join("mex.h").is_file());
        assert!(sdk.include_directory.join("mex.hpp").is_file());
        assert!(sdk.include_directory.join("mexAdapter.hpp").is_file());
        assert!(sdk.include_directory.join("gpu/mxGPUArray.h").is_file());
        assert!(sdk
            .include_directory
            .join("MatlabDataArray/ArrayFactory.hpp")
            .is_file());
        assert!(sdk.include_directory.join("runmat_mex_host.h").is_file());
        assert!(sdk.support_source.is_file());
        assert_eq!(
            sdk.support_source
                .file_name()
                .and_then(|name| name.to_str()),
            Some("runmat_mex_support.c")
        );
        assert_eq!(
            FILES
                .iter()
                .filter(|(path, _)| path.starts_with("src-c/") && path.ends_with(".c"))
                .count(),
            1,
            "the embedded SDK must expose one automatically compiled support unit"
        );
    }
}

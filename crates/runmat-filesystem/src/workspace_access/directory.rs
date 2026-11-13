//! Admit directory response storage before retaining its complete entry vector.
use crate::DirEntry;
use std::{fs, io, path::Path};

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "directory listing exceeds its resident byte budget",
    )
}
/// Requested response storage, excluding allocator rounding and OS iterator internals.
/// `result_root` labels returned paths; it grants no authority to read that root.
pub(super) fn read(native: &Path, result_root: &Path, maximum: u64) -> io::Result<Vec<DirEntry>> {
    let mut count = 0usize;
    let mut required = 0u64;
    for entry in fs::read_dir(native)? {
        let entry = entry?;
        let name = entry.file_name();
        let path = result_root.join(&name);
        let response = DirEntry::new(path, name, entry.file_type()?.into());
        required = required
            .checked_add(response.resident_bytes().ok_or_else(invalid)?)
            .ok_or_else(invalid)?;
        if required > maximum {
            return Err(super::resource::exceeded(required, maximum));
        }
        count = count.checked_add(1).ok_or_else(invalid)?;
    }
    // The scope retains the existing recovery fence and serializes both passes.
    let mut entries = Vec::new();
    entries.try_reserve_exact(count).map_err(io::Error::other)?;
    let mut actual = 0u64;
    for entry in fs::read_dir(native)? {
        let entry = entry?;
        let name = entry.file_name();
        let shape = result_root.join(&name);
        let response = DirEntry::new(shape, name, entry.file_type()?.into());
        actual = actual
            .checked_add(response.resident_bytes().ok_or_else(invalid)?)
            .ok_or_else(invalid)?;
        if entries.len() == count || actual > required {
            return Err(invalid());
        }
        entries.push(response);
    }
    if entries.len() != count || actual != required {
        return Err(io::Error::other(
            "directory inventory changed after admission",
        ));
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listing_admission_counts_headers_and_names_without_gathering_file_content() {
        let root = tempfile::tempdir().unwrap();
        for name in ["first", "second name", "third"] {
            fs::File::create(root.path().join(name))
                .unwrap()
                .set_len(1 << 30)
                .unwrap();
        }
        assert!(read(root.path(), Path::new("virtual-root"), 1).is_err());
        let entries = read(root.path(), Path::new("virtual-root"), 4096).unwrap();
        assert_eq!(entries.len(), 3);
        for entry in entries {
            assert_eq!(
                entry.path(),
                Path::new("virtual-root").join(entry.file_name())
            );
            assert_eq!(entry.file_type(), crate::FsFileType::File);
        }
    }

    #[test]
    fn empty_directory_needs_no_response_allocation_but_long_result_paths_are_charged() {
        let root = tempfile::tempdir().unwrap();
        assert!(read(root.path(), Path::new("ignored"), 0)
            .unwrap()
            .is_empty());
        fs::write(root.path().join("a"), b"original").unwrap();
        assert!(read(root.path(), Path::new("root"), 256).is_ok());
        assert!(read(root.path(), Path::new(&"x".repeat(512)), 256).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn original_non_utf8_names_are_preserved_and_charged_as_owned_bytes() {
        use std::os::unix::ffi::OsStringExt;
        let root = tempfile::tempdir().unwrap();
        let name = std::ffi::OsString::from_vec(vec![b'a', 0xff]);
        fs::write(root.path().join(&name), b"original").unwrap();
        let entries = read(root.path(), Path::new("virtual"), 256).unwrap();
        assert_eq!(entries[0].file_name(), &name);
        assert_eq!(entries[0].path(), Path::new("virtual").join(name));
    }
}

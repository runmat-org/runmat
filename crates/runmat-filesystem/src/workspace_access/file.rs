//! Reserve only admitted original content; no read-to-end buffer growth.
use std::{
    fs,
    io::{self, Read},
    path::Path,
};

pub(crate) fn read(path: &Path, maximum: u64) -> io::Result<Vec<u8>> {
    read_opened(fs::File::open(path)?, maximum)
}

pub(crate) fn read_opened(mut input: fs::File, maximum: u64) -> io::Result<Vec<u8>> {
    let metadata = input.metadata()?;
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "workspace file is not regular",
        ));
    }
    if metadata.len() > maximum {
        return Err(super::resource::exceeded(metadata.len(), maximum));
    }
    let length = usize::try_from(metadata.len()).map_err(io::Error::other)?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(io::Error::other)?;
    bytes.resize(length, 0);
    input.read_exact(&mut bytes)?;
    let mut trailing = [0u8; 1];
    if input.read(&mut trailing)? != 0 {
        return Err(io::Error::other("workspace file grew after read admission"));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_content_and_exact_ceiling_are_preserved_and_oversize_rejects() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("file");
        fs::write(&path, b"original content").unwrap();
        let error = read(&path, 15).unwrap_err();
        let limit = super::super::resource_limit(&error).unwrap();
        assert_eq!((limit.required_bytes, limit.maximum_bytes), (16, 15));
        assert_eq!(read(&path, 16).unwrap(), b"original content");
        fs::File::create(&path).unwrap().set_len(1 << 30).unwrap();
        assert!(read(&path, 16).is_err());
    }

    #[test]
    fn empty_regular_file_fits_zero_bytes_but_directory_is_not_content() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("empty");
        fs::write(&path, []).unwrap();
        assert!(read(&path, 0).unwrap().is_empty());
        assert!(super::super::resource_limit(&read(root.path(), 4096).unwrap_err()).is_none());
    }
}

#[cfg(not(target_arch = "wasm32"))]
use crate::{DirEntry, FsFileType, FsMetadata, FsProvider, OpenFlags};
#[cfg(not(target_arch = "wasm32"))]
use std::ffi::OsString;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::io;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Component, Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
/// Filesystem provider that sandboxes all operations under a fixed root directory.
///
/// Incoming paths (absolute or relative) are normalized and resolved relative to the sandbox root.
/// Attempts to traverse outside the root using `..` simply clamp to the root, preventing escape.
pub struct SandboxFsProvider {
    root: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl SandboxFsProvider {
    /// Create a new sandbox rooted at `root`. The directory is created if it does not exist.
    pub fn new(root: PathBuf) -> io::Result<Self> {
        if !root.exists() {
            fs::create_dir_all(&root)?;
        }
        let canonical = fs::canonicalize(root)?;
        Ok(Self { root: canonical })
    }

    /// Return the sandbox root on the host filesystem.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn resolve(&self, path: &Path) -> PathBuf {
        let mut segments: Vec<OsString> = Vec::new();
        for component in path.components() {
            match component {
                Component::Prefix(_) | Component::RootDir => {
                    segments.clear();
                }
                Component::CurDir => {}
                Component::ParentDir => {
                    segments.pop();
                }
                Component::Normal(seg) => segments.push(seg.to_os_string()),
            }
        }
        let mut target = self.root.clone();
        for seg in segments {
            target.push(seg);
        }
        target
    }

    fn virtualize(&self, real: &Path) -> PathBuf {
        let relative = real.strip_prefix(&self.root).unwrap_or(Path::new(""));
        let mut virt = PathBuf::from(std::path::MAIN_SEPARATOR.to_string());
        if !relative.as_os_str().is_empty() {
            virt.push(relative);
        }
        virt
    }

    fn make_dir_entry(&self, real_path: PathBuf, file_name: OsString) -> DirEntry {
        let file_type = fs::metadata(&real_path)
            .ok()
            .map(|m| FsFileType::from(m.file_type()))
            .unwrap_or(FsFileType::Unknown);
        DirEntry {
            path: self.virtualize(&real_path),
            file_name,
            file_type,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl FsProvider for SandboxFsProvider {
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn crate::FileHandle>> {
        let target = self.resolve(path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut opts = fs::OpenOptions::new();
        opts.read(flags.read);
        opts.write(flags.write);
        opts.append(flags.append);
        opts.truncate(flags.truncate);
        opts.create(flags.create);
        opts.create_new(flags.create_new);
        let file = opts.open(&target)?;
        Ok(Box::new(file))
    }

    fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let target = self.resolve(path);
        fs::read(target)
    }

    fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let target = self.resolve(path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(target, data)
    }

    fn remove_file(&self, path: &Path) -> io::Result<()> {
        let target = self.resolve(path);
        if target.exists() {
            fs::remove_file(target)?;
        }
        Ok(())
    }

    fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let target = self.resolve(path);
        fs::metadata(target).map(FsMetadata::from)
    }

    fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let target = self.resolve(path);
        fs::symlink_metadata(target).map(FsMetadata::from)
    }

    fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let target = self.resolve(path);
        let entries = fs::read_dir(&target)?;
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry?;
            out.push(self.make_dir_entry(entry.path(), entry.file_name()));
        }
        Ok(out)
    }

    fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        let target = self.resolve(path);
        let real = fs::canonicalize(target)?;
        Ok(self.virtualize(&real))
    }

    fn create_dir(&self, path: &Path) -> io::Result<()> {
        let target = self.resolve(path);
        fs::create_dir(&target)
    }

    fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let target = self.resolve(path);
        fs::create_dir_all(&target)
    }

    fn remove_dir(&self, path: &Path) -> io::Result<()> {
        let target = self.resolve(path);
        fs::remove_dir(&target)
    }

    fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        let target = self.resolve(path);
        if target.exists() {
            fs::remove_dir_all(&target)?;
        }
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let src = self.resolve(from);
        let dst = self.resolve(to);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(src, dst)
    }

    fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        let target = self.resolve(path);
        let mut perms = fs::metadata(&target)?.permissions();
        perms.set_readonly(readonly);
        fs::set_permissions(target, perms)
    }
}

#[cfg(all(not(target_arch = "wasm32"), test))]
mod tests {
    use super::SandboxFsProvider;
    use crate::FsProvider;
    use std::path::Path;
    use tempfile::tempdir;

    #[test]
    fn sandbox_prevents_root_escape_and_virtualizes_paths() {
        let temp = tempdir().expect("tempdir");
        let provider = SandboxFsProvider::new(temp.path().to_path_buf()).expect("sandbox");
        provider
            .create_dir_all(Path::new("nested/sub"))
            .expect("create dir");
        provider
            .write(Path::new("nested/sub/file.txt"), b"hello")
            .expect("write");

        // Attempt to escape root should clamp to sandbox.
        provider
            .write(Path::new("../evil.txt"), b"nope")
            .expect("write outside clamped");
        let entries = provider.read_dir(Path::new(".")).expect("read root");
        assert!(entries.iter().any(|entry| entry.file_name() == "evil.txt"));

        let listing = provider.read_dir(Path::new("nested")).expect("list nested");
        let names: Vec<_> = listing
            .iter()
            .map(|entry| entry.path().display().to_string())
            .collect();
        assert!(names.iter().any(|p| p.ends_with("nested/sub")));

        let sandbox_read = provider
            .read(Path::new("/nested/sub/file.txt"))
            .expect("vfs read");
        assert_eq!(sandbox_read, b"hello");
    }

    #[test]
    fn canonicalize_returns_virtual_paths() {
        let temp = tempdir().expect("tempdir");
        let provider = SandboxFsProvider::new(temp.path().to_path_buf()).expect("sandbox");
        provider
            .create_dir_all(Path::new("data"))
            .expect("create dir");
        provider
            .write(Path::new("data/file.bin"), b"bytes")
            .expect("write");
        let canonical = provider
            .canonicalize(Path::new("./data/./file.bin"))
            .expect("canonicalize");
        assert!(canonical.to_string_lossy().ends_with("data/file.bin"));
        assert!(canonical.is_absolute());
    }
}

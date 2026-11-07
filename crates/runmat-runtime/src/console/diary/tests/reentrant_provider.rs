use super::*;
use futures::executor::block_on;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
use std::collections::BTreeMap;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc, Mutex};

#[derive(Clone, Copy)]
enum Callback {
    ConsoleLine,
    ReplaceOwnerAndFail,
}

struct ReentrantProvider {
    callback: Callback,
    fired: Arc<AtomicBool>,
    files: Arc<Mutex<BTreeMap<PathBuf, Vec<u8>>>>,
}

impl ReentrantProvider {
    fn new(callback: Callback) -> Self {
        Self {
            callback,
            fired: Arc::new(AtomicBool::new(false)),
            files: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }
}

struct ReentrantHandle {
    callback: Callback,
    fired: Arc<AtomicBool>,
    files: Arc<Mutex<BTreeMap<PathBuf, Vec<u8>>>>,
    path: PathBuf,
    cursor: Cursor<Vec<u8>>,
}

impl Read for ReentrantHandle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl Seek for ReentrantHandle {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl Write for ReentrantHandle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if !self.fired.swap(true, Ordering::SeqCst) {
            match self.callback {
                Callback::ConsoleLine => {
                    crate::console::record_console_line(ConsoleStream::Stdout, "from provider")
                        .expect("reentrant console output");
                }
                Callback::ReplaceOwnerAndFail => {
                    set_diary_filename_checked("/replacement.log").expect("nested replacement");
                    return Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "original handle failed after owner replacement",
                    ));
                }
            }
        }
        self.cursor.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.files
            .lock()
            .expect("fixture files")
            .insert(self.path.clone(), self.cursor.get_ref().clone());
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl FileHandle for ReentrantHandle {}

fn unsupported() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "fixture operation is unused")
}

#[async_trait::async_trait(?Send)]
impl FsProvider for ReentrantProvider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        runmat_filesystem::FsPathNamespace::VirtualRooted
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/"))
    }

    fn open(&self, path: &Path, _flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Ok(Box::new(ReentrantHandle {
            callback: self.callback,
            fired: self.fired.clone(),
            files: self.files.clone(),
            path: path.to_path_buf(),
            cursor: Cursor::new(Vec::new()),
        }))
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.files
            .lock()
            .expect("fixture files")
            .get(path)
            .cloned()
            .ok_or_else(unsupported)
    }

    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unsupported())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unsupported())
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unsupported())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unsupported())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unsupported())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(unsupported())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(unsupported())
    }
}

#[test]
fn file_write_reentry_returns_would_block_without_borrow_panic() {
    let _lock = fs::provider_override_lock();
    set_diary_enabled(false);
    let _ = take_diary_error();
    let provider = Arc::new(ReentrantProvider::new(Callback::ConsoleLine));
    let _owner = fs::replace_provider(provider);
    set_diary_filename_checked("/original.log").expect("enable");
    record_diary_command("outer");
    assert!(!diary_enabled());
    assert_eq!(
        take_diary_error().expect("reentry error").kind(),
        io::ErrorKind::WouldBlock
    );
}

#[test]
fn old_handle_failure_cannot_disable_reentrant_replacement_owner() {
    let _lock = fs::provider_override_lock();
    set_diary_enabled(false);
    let _ = take_diary_error();
    let provider = Arc::new(ReentrantProvider::new(Callback::ReplaceOwnerAndFail));
    let _owner = fs::replace_provider(provider.clone());
    set_diary_filename_checked("/original.log").expect("enable");
    record_diary_command("failed outer");
    assert!(diary_enabled());
    assert_eq!(diary_filename(), PathBuf::from("/replacement.log"));
    assert!(take_diary_error().is_none());
    record_diary_command("after");
    let bytes = block_on(provider.read(Path::new("/replacement.log"))).expect("replacement log");
    assert_eq!(bytes.as_slice(), b"after\n");
    set_diary_enabled(false);
}

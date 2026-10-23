use super::*;
use async_trait::async_trait;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
use runmat_value::{CellArray, CharArray, StringArray};
use std::{
    io,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

struct CountingGeometryFs {
    reads: AtomicUsize,
    owner: std::thread::ThreadId,
}

impl Default for CountingGeometryFs {
    fn default() -> Self {
        Self {
            reads: AtomicUsize::new(0),
            owner: std::thread::current().id(),
        }
    }
}

fn unexpected() -> io::Error {
    io::Error::other("unexpected filesystem operation")
}

#[async_trait(?Send)]
impl FsProvider for CountingGeometryFs {
    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/"))
    }
    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        Err(unexpected())
    }
    async fn read(&self, _: &Path) -> io::Result<Vec<u8>> {
        // The provider is process-wide; unrelated tests can read through it
        // while this test's synchronous calls are running on the owner thread.
        if std::thread::current().id() == self.owner {
            self.reads.fetch_add(1, Ordering::SeqCst);
        }
        Ok(TRIANGLE_STL.as_bytes().to_vec())
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unexpected())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unexpected())
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unexpected())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unexpected())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unexpected())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(unexpected())
    }
}

#[test]
fn invalid_path_classes_and_shapes_have_zero_filesystem_reads() {
    let _provider_lock = runmat_filesystem::provider_override_lock();
    let provider = Arc::new(CountingGeometryFs::default());
    runmat_filesystem::with_provider_override(provider.clone(), || {
        let invalid = [
            Value::Num(7.0),
            Value::from(7_i32),
            Value::Bool(true),
            Value::Cell(CellArray::new(vec![Value::String("part.stl".into())], 1, 1).unwrap()),
            Value::CharArray(
                CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap(),
            ),
            Value::CharArray(
                CharArray::from_unicode_scalars_with_shape(vec!['a', 'b'], vec![1, 2, 1]).unwrap(),
            ),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
        ];
        for value in invalid {
            let load = block_on(geometry_load_builtin(
                value.clone(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect_err("load path admission");
            assert_eq!(load.identifier(), GEOMETRY_LOAD_ERROR_INPUT.identifier);
            let inspect =
                block_on(geometry_inspect_builtin(value)).expect_err("inspect path admission");
            assert_eq!(
                inspect.identifier(),
                GEOMETRY_INSPECT_ERROR_INPUT.identifier
            );
            assert_eq!(provider.reads.load(Ordering::SeqCst), 0);
        }
        for value in [
            Value::String("part.stl".into()),
            Value::CharArray(CharArray::new_row("part.stl")),
            Value::StringArray(StringArray::new(vec!["part.stl".into()], vec![1, 1]).unwrap()),
        ] {
            let asset = block_on(geometry_load_builtin(
                value.clone(),
                crate::builtins::common::test_support::builtin_arguments(Vec::new()),
            ))
            .expect("supported text load");
            assert!(matches!(asset, Value::Object(_)));
            let info = block_on(geometry_inspect_builtin(value)).expect("supported text inspect");
            assert!(matches!(info, Value::Object(_)));
        }
        assert_eq!(provider.reads.load(Ordering::SeqCst), 6);
    });
}

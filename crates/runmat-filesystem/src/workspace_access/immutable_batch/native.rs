//! Native group commit: staged files, one barrier, a durable batch record, then links.
use super::{ImmutableBatch, ImmutableStage};
use crate::workspace_access::NativeAccess;
use std::{
    collections::BTreeSet,
    fs, io,
    io::Write as _,
    path::{Path, PathBuf},
};

const BATCH_RECORDS: &str = "immutable-batches";

fn records_directory(root: &Path) -> PathBuf {
    root.join(crate::workspace_native::JOURNAL)
        .join(BATCH_RECORDS)
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BatchRecord {
    schema_version: u16,
    /// Root-relative (staged, published) pairs.
    links: Vec<(PathBuf, PathBuf)>,
}

const BATCH_RECORD_SCHEMA_VERSION: u16 = 1;
const MAXIMUM_BATCH_RECORD_BYTES: u64 = 64 << 20;

struct Entry {
    staged: PathBuf,
    published: PathBuf,
}

pub(in crate::workspace_access) struct NativeImmutableBatch<'a> {
    access: &'a NativeAccess,
    entries: Vec<Entry>,
    /// Existing published files whose durability the commit confirms.
    existing: Vec<PathBuf>,
    committed: bool,
}

impl<'a> NativeImmutableBatch<'a> {
    pub(in crate::workspace_access) fn new(access: &'a NativeAccess) -> Self {
        Self {
            access,
            entries: Vec::new(),
            existing: Vec::new(),
            committed: false,
        }
    }

    fn relative(&self, path: &Path) -> io::Result<PathBuf> {
        path.strip_prefix(&self.access.root)
            .map(Path::to_path_buf)
            .map_err(|_| io::Error::other("immutable batch path escapes its root"))
    }
}

impl ImmutableBatch for NativeImmutableBatch<'_> {
    fn stage(
        &mut self,
        path: &Path,
        bytes: &[u8],
        maximum_existing_bytes: u64,
    ) -> io::Result<ImmutableStage> {
        let maximum_existing_bytes = maximum_existing_bytes.min(self.access.maximum_resident_bytes);
        let access = self.access;
        access.access(true, || {
            let published = access.path(path, true)?;
            let parent = published.parent().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "immutable publication path has no parent directory",
                )
            })?;
            fs::create_dir_all(parent)?;
            if let Some(existing) = existing_regular_file(&published, maximum_existing_bytes)? {
                self.existing.push(published);
                return Ok(ImmutableStage::Existing(existing));
            }
            if let Some(entry) = self
                .entries
                .iter()
                .find(|entry| entry.published == published)
            {
                let staged =
                    crate::workspace_access::file::read(&entry.staged, maximum_existing_bytes)?;
                return Ok(ImmutableStage::Existing(staged));
            }
            let staged = stage_file(parent, &published, bytes)?;
            self.entries.push(Entry { staged, published });
            Ok(ImmutableStage::Staged)
        })
    }

    fn commit(mut self: Box<Self>) -> io::Result<()> {
        if self.entries.is_empty() && self.existing.is_empty() {
            self.committed = true;
            return Ok(());
        }
        let access = self.access;
        let result = access.access(true, || {
            // 1. Every staged and confirmed file reaches the device, then one barrier.
            let mut last = None;
            for path in self
                .entries
                .iter()
                .map(|entry| &entry.staged)
                .chain(&self.existing)
            {
                #[cfg(windows)]
                let file = fs::OpenOptions::new().write(true).open(path)?;
                #[cfg(not(windows))]
                let file = fs::File::open(path)?;
                send_to_device(&file)?;
                last = Some(file);
            }
            if let Some(file) = &last {
                device_barrier(file)?;
            }
            fault::point(fault::Point::BeforeRecord)?;
            // 2. The durable record is the commit point.
            let record = if self.entries.is_empty() {
                None
            } else {
                Some(write_record(&access.root, &self.entries, |path| {
                    self.relative(path)
                })?)
            };
            fault::point(fault::Point::AfterRecord)?;
            // 3. Publish names, then persist the directories that expose them.
            for (index, entry) in self.entries.iter().enumerate() {
                fault::point(fault::Point::AfterLinks(index))?;
                link(&entry.staged, &entry.published)?;
            }
            let directories = self
                .entries
                .iter()
                .map(|entry| &entry.published)
                .chain(&self.existing)
                .filter_map(|path| path.parent())
                .collect::<BTreeSet<_>>();
            sync_exposing_directories(&access.root, directories)?;
            // 4. Retire the record and the staging names; recovery tolerates either remaining.
            for entry in &self.entries {
                remove_if_present(&entry.staged)?;
            }
            if let Some(record) = record {
                remove_if_present(&record)?;
            }
            Ok(())
        });
        if result.is_ok() || !fault::crashed() {
            self.committed = result.is_ok();
        } else {
            // A simulated crash leaves the namespace exactly as it was at the fault point.
            self.committed = true;
        }
        result
    }
}

impl Drop for NativeImmutableBatch<'_> {
    fn drop(&mut self) {
        if !self.committed {
            for entry in &self.entries {
                let _ = fs::remove_file(&entry.staged);
            }
        }
    }
}

/// Rolls forward every durable batch record under `root`. Runs inside the scope fence before
/// any operation, so no participant observes a partially linked batch.
pub(in crate::workspace_access) fn recover(
    root: &Path,
    maximum_resident_bytes: u64,
) -> io::Result<()> {
    let directory = records_directory(root);
    match fs::symlink_metadata(&directory) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
        Ok(_) => {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "immutable batch journal is not a directory",
            ))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    }
    let mut records = fs::read_dir(&directory)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<io::Result<Vec<_>>>()?;
    records.sort();
    for path in records {
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.ends_with(".json") {
            // An unfinished record write never became a commit point.
            remove_if_present(&path)?;
            continue;
        }
        let bytes = crate::workspace_access::file::read(
            &path,
            MAXIMUM_BATCH_RECORD_BYTES.min(maximum_resident_bytes),
        )?;
        let record: BatchRecord = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
        if record.schema_version != BATCH_RECORD_SCHEMA_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported immutable batch record",
            ));
        }
        let mut directories = BTreeSet::new();
        let mut links = Vec::with_capacity(record.links.len());
        for (staged, published) in &record.links {
            let staged = confined(root, staged)?;
            let published = confined(root, published)?;
            if fs::symlink_metadata(&published).is_err() {
                link(&staged, &published)?;
            }
            links.push((staged, published));
        }
        for (_, published) in &links {
            if let Some(parent) = published.parent() {
                directories.insert(parent);
            }
        }
        sync_exposing_directories(root, directories)?;
        for (staged, _) in &links {
            remove_if_present(staged)?;
        }
        remove_if_present(&path)?;
    }
    Ok(())
}

fn confined(root: &Path, relative: &Path) -> io::Result<PathBuf> {
    if relative
        .components()
        .any(|part| !matches!(part, std::path::Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "immutable batch record path escapes its root",
        ));
    }
    let path = root.join(relative);
    crate::workspace_native::ensure_no_symlink(root, &path)?;
    Ok(path)
}

fn existing_regular_file(path: &Path, maximum: u64) -> io::Result<Option<Vec<u8>>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "atomic publication target must not be a symbolic link",
        )),
        Ok(metadata) if !metadata.is_file() => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "atomic publication target is not a regular file",
        )),
        Ok(_) => crate::workspace_access::file::read(path, maximum).map(Some),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn stage_file(parent: &Path, published: &Path, bytes: &[u8]) -> io::Result<PathBuf> {
    let name = published
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("artifact");
    for _ in 0..16 {
        let candidate = parent.join(format!(
            ".{name}.publish-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                if let Err(error) = file.write_all(bytes) {
                    let _ = fs::remove_file(&candidate);
                    return Err(error);
                }
                return Ok(candidate);
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not allocate a unique immutable staging path",
    ))
}

/// Publishes one name; an existing name must already hold the same bytes.
fn link(staged: &Path, published: &Path) -> io::Result<()> {
    match fs::hard_link(staged, published) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            if fs::read(staged)? == fs::read(published)? {
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    "immutable publication identity already contains different content",
                ))
            }
        }
        Err(error) => Err(error),
    }
}

fn write_record(
    root: &Path,
    entries: &[Entry],
    relative: impl Fn(&Path) -> io::Result<PathBuf>,
) -> io::Result<PathBuf> {
    let directory = records_directory(root);
    fs::create_dir_all(&directory)?;
    let record = BatchRecord {
        schema_version: BATCH_RECORD_SCHEMA_VERSION,
        links: entries
            .iter()
            .map(|entry| Ok((relative(&entry.staged)?, relative(&entry.published)?)))
            .collect::<io::Result<_>>()?,
    };
    let bytes = serde_json::to_vec(&record).map_err(io::Error::other)?;
    let id = uuid::Uuid::new_v4();
    // Written under a temporary name and renamed, so a torn record is never a commit point.
    let partial = directory.join(format!("{id}.partial"));
    let path = directory.join(format!("{id}.json"));
    {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    fs::rename(&partial, &path)?;
    sync_exposing_directories(root, [directory.as_path()])?;
    Ok(path)
}

/// Persists each directory and its ancestors up to and including the root's parent, the same
/// boundary as `synchronize_immutable`. A relative root (whose parent is the empty path) is
/// persisted up to the current directory. Each directory is synchronized once.
fn sync_exposing_directories<'a>(
    root: &Path,
    directories: impl IntoIterator<Item = &'a Path>,
) -> io::Result<()> {
    let boundary = root
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    let mut synced = BTreeSet::new();
    for directory in directories {
        let mut current = Some(directory);
        while let Some(path) = current {
            if boundary.is_some_and(|boundary| !path.starts_with(boundary)) {
                break;
            }
            let effective = if path.as_os_str().is_empty() {
                Path::new(".")
            } else {
                path
            };
            if !synced.insert(effective.to_path_buf()) {
                break;
            }
            crate::native::directory_sync::sync(effective)?;
            if Some(path) == boundary || path.as_os_str().is_empty() {
                break;
            }
            current = path.parent();
        }
    }
    Ok(())
}

fn remove_if_present(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Sends the file's data and metadata to the device. On Apple platforms `fsync(2)` does not
/// flush the device cache; the batch's single `device_barrier` does that for every file.
fn send_to_device(file: &fs::File) -> io::Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        use std::os::fd::AsRawFd as _;
        // SAFETY: the descriptor is owned by `file` and stays open for the call.
        if unsafe { libc::fsync(file.as_raw_fd()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        file.sync_all()
    }
}

/// One full device flush covering everything previously sent (`F_FULLFSYNC` on Apple).
fn device_barrier(file: &fs::File) -> io::Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        file.sync_all()
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = file;
        Ok(())
    }
}

/// Crash injection for tests: an armed point fails the commit and leaves the namespace as a
/// killed process would (no staging cleanup, no record removal).
pub(crate) mod fault {
    use std::io;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Point {
        BeforeRecord,
        AfterRecord,
        AfterLinks(usize),
    }

    #[cfg(test)]
    thread_local! {
        static ARMED: std::cell::Cell<Option<Point>> = const { std::cell::Cell::new(None) };
        static CRASHED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    #[cfg(test)]
    pub(crate) fn arm(point: Option<Point>) {
        ARMED.with(|armed| armed.set(point));
        CRASHED.with(|crashed| crashed.set(false));
    }

    pub(super) fn point(point: Point) -> io::Result<()> {
        #[cfg(test)]
        if ARMED.with(|armed| armed.get()) == Some(point) {
            CRASHED.with(|crashed| crashed.set(true));
            return Err(io::Error::other(format!("injected crash at {point:?}")));
        }
        let _ = point;
        Ok(())
    }

    pub(super) fn crashed() -> bool {
        #[cfg(test)]
        return CRASHED.with(|crashed| crashed.get());
        #[cfg(not(test))]
        false
    }
}

#[cfg(test)]
mod tests {
    use super::fault::{arm, Point};
    use super::ImmutableStage;
    use crate::{FsProvider, NativeFsProvider};
    use std::path::Path;

    const OBJECTS: usize = 5;

    fn object(index: usize) -> (String, Vec<u8>) {
        (
            format!("objects/sha256_{index:04}.object"),
            format!("payload {index} ")
                .repeat(1000 + index)
                .into_bytes(),
        )
    }

    /// Runs one batch with an optional injected crash, then reopens the workspace (which
    /// recovers) and returns how many batch objects are visible, checking each one's content.
    fn crash_and_reopen(point: Option<Point>) -> usize {
        let root = tempfile::tempdir().unwrap();
        {
            let scope = NativeFsProvider
                .workspace_access(root.path(), true, 1 << 26)
                .unwrap();
            let mut batch = scope.immutable_batch().unwrap();
            for index in 0..OBJECTS {
                let (path, bytes) = object(index);
                assert_eq!(
                    batch.stage(Path::new(&path), &bytes, 1 << 26).unwrap(),
                    ImmutableStage::Staged
                );
            }
            arm(point);
            let outcome = batch.commit();
            arm(None);
            assert_eq!(outcome.is_ok(), point.is_none());
        }
        let scope = NativeFsProvider
            .workspace_access(root.path(), true, 1 << 26)
            .unwrap();
        let visible = (0..OBJECTS)
            .filter(|index| {
                let (path, bytes) = object(*index);
                match scope.read(Path::new(&path), 1 << 26) {
                    Ok(content) => {
                        assert_eq!(content, bytes, "a visible object must be complete");
                        true
                    }
                    Err(error) => {
                        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
                        false
                    }
                }
            })
            .count();
        // Recovery retires the journal record once the batch is fully visible.
        let records = super::records_directory(root.path());
        assert!(
            !records.exists() || std::fs::read_dir(&records).unwrap().next().is_none(),
            "no batch record may outlive recovery"
        );
        visible
    }

    #[test]
    fn a_crash_mid_batch_leaves_all_or_none_of_the_batch_visible() {
        assert_eq!(crash_and_reopen(None), OBJECTS);
        assert_eq!(crash_and_reopen(Some(Point::BeforeRecord)), 0);
        assert_eq!(crash_and_reopen(Some(Point::AfterRecord)), OBJECTS);
        for linked in 0..OBJECTS {
            assert_eq!(
                crash_and_reopen(Some(Point::AfterLinks(linked))),
                OBJECTS,
                "crash after {linked} links"
            );
        }
    }

    #[test]
    fn an_uncommitted_batch_publishes_nothing_and_leaves_no_staging() {
        let root = tempfile::tempdir().unwrap();
        let scope = NativeFsProvider
            .workspace_access(root.path(), true, 1 << 26)
            .unwrap();
        {
            let mut batch = scope.immutable_batch().unwrap();
            let (path, bytes) = object(0);
            batch.stage(Path::new(&path), &bytes, 1 << 26).unwrap();
        }
        let listing = std::fs::read_dir(root.path().join("objects"))
            .unwrap()
            .count();
        assert_eq!(listing, 0);
    }

    #[test]
    fn existing_and_repeated_names_report_their_content() {
        let root = tempfile::tempdir().unwrap();
        let scope = NativeFsProvider
            .workspace_access(root.path(), true, 1 << 26)
            .unwrap();
        scope
            .atomic_create_new(Path::new("objects/a"), b"first", 1 << 20)
            .unwrap();
        let mut batch = scope.immutable_batch().unwrap();
        assert_eq!(
            batch
                .stage(Path::new("objects/a"), b"other", 1 << 20)
                .unwrap(),
            ImmutableStage::Existing(b"first".to_vec())
        );
        assert_eq!(
            batch
                .stage(Path::new("objects/b"), b"second", 1 << 20)
                .unwrap(),
            ImmutableStage::Staged
        );
        assert_eq!(
            batch
                .stage(Path::new("objects/b"), b"second", 1 << 20)
                .unwrap(),
            ImmutableStage::Existing(b"second".to_vec())
        );
        batch.commit().unwrap();
        assert_eq!(
            scope.read(Path::new("objects/b"), 1 << 20).unwrap(),
            b"second"
        );
        let hidden = std::fs::read_dir(root.path().join("objects"))
            .unwrap()
            .filter(|entry| {
                entry
                    .as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with('.')
            })
            .count();
        assert_eq!(hidden, 0, "committed staging names are retired");
    }

    #[test]
    fn a_single_component_relative_root_persists_up_to_the_current_directory() {
        // `Path::new("src").parent()` is the empty path; the walk ends at ".".
        super::sync_exposing_directories(
            Path::new("src"),
            [Path::new("src/workspace_access"), Path::new("src")],
        )
        .unwrap();
    }
}

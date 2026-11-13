//! Exact metadata inventory. Complete discovery occurs once per cache owner or
//! after another writer changes its durable epoch, never for each local stage.
use super::*;
use std::collections::VecDeque;
#[derive(Clone)]
pub(super) struct Receipt {
    pub bytes: u64,
    pub digests: BTreeSet<String>,
}
struct Object {
    bytes: u64,
    references: usize,
}
pub(super) struct Inventory {
    pub epoch: [u8; 16],
    receipts: BTreeMap<PathBuf, Receipt>,
    oldest: VecDeque<PathBuf>,
    objects: BTreeMap<String, Object>,
    orphaned: Vec<PathBuf>,
    abandoned: Vec<PathBuf>,
    bytes: u64,
    index_bytes: u64,
    references: usize,
}
impl Inventory {
    pub fn read(
        index: &Path,
        objects: &Path,
        check: &dyn Fn() -> Result<(), MeshingStageExecutionError>,
    ) -> Result<Self, MeshingStageExecutionError> {
        let mut result = Self {
            epoch: [0; 16],
            receipts: BTreeMap::new(),
            oldest: VecDeque::new(),
            objects: BTreeMap::new(),
            orphaned: Vec::new(),
            abandoned: Vec::new(),
            bytes: 0,
            index_bytes: 0,
            references: 0,
        };
        let mut paths = Vec::new();
        for path in files(index, &mut result.abandoned, check)? {
            check()?;
            let modified = std::fs::metadata(&path)
                .map_err(error)?
                .modified()
                .map_err(error)?;
            paths.push((modified, path));
        }
        runmat_cpu_execution::sort_in_place_with_checkpoint(
            &mut paths,
            |left, right| left.cmp(right),
            check,
        )?;
        for (_, path) in paths {
            check()?;
            let bytes = NativeFsProvider
                .read_bounded_sync(&path, RECEIPT_BYTES)
                .map_err(error)?;
            let receipt: AcceptedMeshingStageReceipt =
                serde_json::from_slice(&bytes).map_err(error)?;
            result.add(
                path,
                Receipt {
                    bytes: bytes.len() as u64,
                    digests: receipt.object_digests().map(filename).collect(),
                },
                objects,
                check,
            )?;
        }
        for path in files(objects, &mut result.abandoned, check)? {
            if !result
                .objects
                .contains_key(path.file_name().unwrap().to_str().unwrap())
            {
                result.orphaned.push(path);
            }
        }
        Ok(result)
    }
    pub fn add(
        &mut self,
        path: PathBuf,
        receipt: Receipt,
        root: &Path,
        check: &dyn Fn() -> Result<(), MeshingStageExecutionError>,
    ) -> Result<(), MeshingStageExecutionError> {
        // A duplicate immutable receipt already owns exactly these references.
        if let Some(existing) = self.receipts.get(&path) {
            if existing.bytes != receipt.bytes || existing.digests != receipt.digests {
                return Err(error(
                    "immutable cache receipt changed its object inventory",
                ));
            }
            return Ok(());
        }
        self.index_bytes = add(self.index_bytes, receipt.bytes)?;
        self.bytes = add(self.bytes, receipt.bytes)?;
        self.references = self
            .references
            .checked_add(receipt.digests.len())
            .ok_or_else(|| error("cache reference count overflow"))?;
        if self.references > MAXIMUM_REFERENCES || self.index_bytes > MAXIMUM_INDEX_BYTES {
            return Err(error("cache metadata exceeds its lifecycle bounds"));
        }
        for (position, digest) in receipt.digests.iter().enumerate() {
            if position.is_multiple_of(128) {
                check()?;
            }
            if let Some(object) = self.objects.get_mut(digest) {
                object.references += 1;
            } else {
                let metadata = std::fs::symlink_metadata(root.join(digest)).map_err(error)?;
                if !metadata.is_file() {
                    return Err(error("cache object is not a regular file"));
                }
                self.bytes = add(self.bytes, metadata.len())?;
                self.objects.insert(
                    digest.clone(),
                    Object {
                        bytes: metadata.len(),
                        references: 1,
                    },
                );
            }
        }
        self.oldest.push_back(path.clone());
        self.receipts.insert(path, receipt);
        Ok(())
    }
    pub fn fits(&self, requested: u64, references: usize) -> bool {
        self.bytes
            .checked_add(requested)
            .is_some_and(|v| v <= MAXIMUM_BYTES)
            && self
                .index_bytes
                .checked_add(if requested == 0 { 0 } else { RECEIPT_BYTES })
                .is_some_and(|v| v <= MAXIMUM_INDEX_BYTES)
            && self
                .references
                .checked_add(references)
                .is_some_and(|v| v <= MAXIMUM_REFERENCES)
    }
    pub fn remove_orphans(
        &mut self,
        check: &dyn Fn() -> Result<(), MeshingStageExecutionError>,
    ) -> Result<(), MeshingStageExecutionError> {
        for path in self.orphaned.drain(..) {
            check()?;
            std::fs::remove_file(path).map_err(error)?;
        }
        // Each name is unique to one dead writer, so unlinking it cannot race a
        // live publication; another reclaimer may already have removed it.
        for path in self.abandoned.drain(..) {
            check()?;
            match std::fs::remove_file(&path) {
                Ok(()) => tracing::info!(
                    event_code = "meshing.cache.abandoned_publication_reclaimed",
                    path = %path.display(),
                    "removed a publication staging file whose writer process has exited"
                ),
                Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => {}
                Err(failure) => return Err(error(failure)),
            }
        }
        Ok(())
    }
    pub fn evict_oldest(
        &mut self,
        root: &Path,
        check: &dyn Fn() -> Result<(), MeshingStageExecutionError>,
    ) -> Result<bool, MeshingStageExecutionError> {
        check()?;
        let Some(path) = self.oldest.pop_front() else {
            return Ok(false);
        };
        let receipt = self.receipts.remove(&path).expect("ordered cache receipt");
        std::fs::remove_file(&path).map_err(error)?;
        self.bytes -= receipt.bytes;
        self.index_bytes -= receipt.bytes;
        self.references -= receipt.digests.len();
        for (position, digest) in receipt.digests.into_iter().enumerate() {
            if position.is_multiple_of(128) {
                check()?;
            }
            let object = self
                .objects
                .get_mut(&digest)
                .expect("referenced cache object");
            object.references -= 1;
            if object.references == 0 {
                let bytes = object.bytes;
                std::fs::remove_file(root.join(&digest)).map_err(error)?;
                self.objects.remove(&digest);
                self.bytes -= bytes;
            }
        }
        Ok(true)
    }
    pub fn remove_unreferenced(
        &self,
        root: &Path,
        digests: impl Iterator<Item = Digest>,
        check: &dyn Fn() -> Result<(), MeshingStageExecutionError>,
    ) -> Result<(), MeshingStageExecutionError> {
        for (position, digest) in digests.enumerate() {
            if position.is_multiple_of(128) {
                check()?;
            }
            let name = filename(digest);
            if !self.objects.contains_key(&name) {
                match std::fs::remove_file(root.join(name)) {
                    Ok(()) => {}
                    Err(failure) if failure.kind() == std::io::ErrorKind::NotFound => {}
                    Err(failure) => return Err(error(failure)),
                }
            }
        }
        Ok(())
    }
}
fn add(total: u64, bytes: u64) -> Result<u64, MeshingStageExecutionError> {
    total
        .checked_add(bytes)
        .ok_or_else(|| error("cache byte count overflow"))
}
/// Canonical receipts or objects in `root`. Publication staging files left by
/// a writer that has exited go to `abandoned`; staging files of a writer that
/// may still run are skipped. Every other name is refused.
fn files(
    root: &Path,
    abandoned: &mut Vec<PathBuf>,
    check: &dyn Fn() -> Result<(), MeshingStageExecutionError>,
) -> Result<Vec<PathBuf>, MeshingStageExecutionError> {
    let mut result = Vec::new();
    for (position, entry) in std::fs::read_dir(root).map_err(error)?.enumerate() {
        if position.is_multiple_of(128) {
            check()?;
        }
        if position >= MAXIMUM_REFERENCES.saturating_mul(2) {
            return Err(error("cache filesystem inventory exceeds lifecycle bound"));
        }
        let entry = entry.map_err(error)?;
        let name = entry.file_name();
        let name = name
            .to_str()
            .ok_or_else(|| error("cache filename is not an identity"))?;
        let regular = entry.file_type().map_err(error)?.is_file();
        if regular {
            if let Some(writer) = publication_writer(name) {
                if writer_has_exited(writer) {
                    abandoned.push(entry.path());
                } else {
                    tracing::info!(
                        event_code = "meshing.cache.publication_in_progress",
                        path = %entry.path().display(),
                        "skipped a publication staging file whose writer may still run"
                    );
                }
                continue;
            }
        }
        if !is_identity(name) || !regular {
            return Err(error(
                "cache lifecycle encountered a noncanonical object or receipt",
            ));
        }
        if result.len() >= MAXIMUM_REFERENCES {
            return Err(error("cache filesystem inventory exceeds lifecycle bound"));
        }
        result.push(entry.path());
    }
    Ok(result)
}
fn is_identity(name: &str) -> bool {
    name.len() == 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}
/// The writer pid of a staging file named exactly as
/// `runmat_filesystem` atomic creation names it:
/// `.<64 lowercase hex>.publish-<pid>-<lowercase hyphenated UUIDv4>`.
fn publication_writer(name: &str) -> Option<u32> {
    let rest = name.strip_prefix('.')?;
    let (target, rest) = rest.split_at_checked(64)?;
    let rest = rest.strip_prefix(".publish-")?;
    let (pid, nonce) = rest.split_once('-')?;
    if !is_identity(target)
        || pid.is_empty()
        || pid.starts_with('0')
        || !pid.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let pid = pid.parse::<u32>().ok()?;
    let uuid = uuid::Uuid::try_parse(nonce).ok()?;
    (uuid.get_version_num() == 4 && uuid.hyphenated().to_string() == nonce).then_some(pid)
}
/// True only when no process with `pid` exists. Pid reuse can make an exited
/// writer look alive, which only delays reclamation; it never makes a running
/// writer look exited. A reused pid's publisher stages under a fresh UUID name.
#[cfg(unix)]
fn writer_has_exited(pid: u32) -> bool {
    let Ok(raw) = libc::pid_t::try_from(pid) else {
        return false;
    };
    if raw <= 0 || pid == std::process::id() {
        return false;
    }
    // SAFETY: signal 0 performs only the existence and permission check.
    let result = unsafe { libc::kill(raw, 0) };
    result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}
#[cfg(not(unix))]
fn writer_has_exited(_pid: u32) -> bool {
    false
}

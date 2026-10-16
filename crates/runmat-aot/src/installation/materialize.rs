use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use runmat_execution::Digest;
use runmat_package::NormalizedRelativePath;

use super::{invalid, NativeInstallationBundle, NativeInstallationManifest};
use crate::tree::physical;
use crate::{AotError, AotResult};

/// Existing explicit installation store and a single atomic publication slot.
/// Generation directories are persistent, not temporary references. Replacing
/// a slot never erases an older installation held by another receipt.
/// In addition to payload entries (including implicit directories), the
/// protocol reserves six wrapper/publication entries, three file occurrences,
/// manifest bytes and an 8192-byte pointer-publication allowance. These are
/// bounded physical-path/byte permissions, not complete installer RAM, RSS,
/// kernel/filesystem-journal scratch or actual loader-execution certificates.
pub struct NativeInstallationRequest<'a> {
    pub store: &'a Path,
    pub slot: NormalizedRelativePath,
    pub overwrite: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct InstallationPointer {
    schema_version: u16,
    generation: NormalizedRelativePath,
    installation: Digest,
}
const POINTER_SCHEMA_VERSION: u16 = 1;
const POINTER_MAX_BYTES: u64 = 4096;
// Checked per-call physical allowance, not an installer-memory/RSS or kernel
// scratch certificate: generation + payload dirs and manifest (3 entries),
// publication TempDir + source/destination pointer names (3 more). Reserve two
// pointer occurrences even where no-clobber hard-links share physical blocks.
const INSTALLATION_OVERHEAD_ENTRIES: u64 = 6;
const INSTALLATION_OVERHEAD_FILES: u64 = 3; // manifest + two pointer names
const POINTER_PUBLICATION_BYTES: u64 = 2 * POINTER_MAX_BYTES;

struct InstalledRecord {
    generation: PathBuf,
    manifest: NativeInstallationManifest,
    bytes: Vec<u8>,
    identity: Digest,
    revoked: Cell<bool>,
    source_evaluation_issuer: Option<runmat_execution::host::VerifiedNativeServiceIssuer>,
}

/// A revocable physical receipt, not a runtime grant or loaded provider proof.
/// Its drop retires this in-process receipt, not the durable user's files.
#[derive(Clone)]
pub struct InstalledNativeArtifact(Rc<InstalledRecord>);
impl InstalledNativeArtifact {
    pub fn manifest(&self) -> &NativeInstallationManifest {
        &self.0.manifest
    }
    pub fn identity(&self) -> Digest {
        self.0.identity
    }
    pub fn payload_root(&self) -> PathBuf {
        self.0.generation.join("payload")
    }
    pub fn executable_path(&self) -> PathBuf {
        self.payload_root()
            .join(self.0.manifest.layout.executable.as_str())
    }
    pub fn revoke(&self) {
        self.0.revoked.set(true);
    }

    /// Revalidation is required again at a later loader/provider handoff. It
    /// establishes current physical bytes, not unobserved future OS loading.
    pub fn revalidate(&self) -> AotResult<()> {
        if self.0.revoked.get() {
            return Err(invalid("installation receipt was revoked"));
        }
        verify_generation(&self.0.generation, &self.0.manifest, &self.0.bytes)
    }

    /// Separate actual-host admission. Cross-target physical materialization
    /// alone cannot make an image executable on the build machine.
    pub fn require_current_host(&self) -> AotResult<()> {
        self.revalidate()?;
        self.0
            .manifest
            .target
            .require_host_executable()
            .map_err(invalid)
    }

    /// Host composition may pair this verified installed-service identity with
    /// the actual live evaluator only after revalidating current installation.
    /// The host, not this receipt, vouches for the evaluator's implementation.
    pub fn source_evaluation_issuer(
        &self,
    ) -> AotResult<runmat_execution::host::VerifiedNativeServiceIssuer> {
        self.require_current_host()?;
        self.0
            .source_evaluation_issuer
            .clone()
            .ok_or_else(|| invalid("installation has no selected source-evaluation service"))
    }
}

pub fn install_native_bundle(
    bundle: NativeInstallationBundle<'_>,
    request: &NativeInstallationRequest<'_>,
) -> AotResult<InstalledNativeArtifact> {
    let generation = materialize(
        &bundle.tree,
        &bundle.manifest.payload,
        &bundle.bytes,
        bundle.identity,
        &bundle.manifest.storage,
        request,
    )?;
    receipt(bundle, generation)
}

fn materialize(
    tree: &crate::tree::ResolvedNativeTree<'_>,
    payload: &runmat_package_cache::TreeManifest,
    manifest_bytes: &[u8],
    identity: Digest,
    storage: &crate::tree::NativeStorageGrant,
    request: &NativeInstallationRequest<'_>,
) -> AotResult<PathBuf> {
    admit_materialization(storage, payload, manifest_bytes.len() as u64)?;
    let store = store(request)?;
    let generation = tempfile::Builder::new()
        .prefix("runmat-installed-")
        .tempdir_in(&store)
        .map_err(|error| AotError::io("create installation generation", &store, error))?;
    let name = generation
        .path()
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| invalid("installation generation has no portable locator"))?;
    let pointer = InstallationPointer {
        schema_version: POINTER_SCHEMA_VERSION,
        generation: NormalizedRelativePath::new(name).map_err(invalid)?,
        installation: identity,
    };
    let pointer_bytes =
        crate::tree::canonical_json(&pointer, storage.metadata_bytes.min(POINTER_MAX_BYTES))?;
    storage.file(pointer_bytes.len() as u64)?;
    tree.stage(&generation.path().join("payload"), storage)?;
    physical::write_new(&generation.path().join("manifest.json"), manifest_bytes)?;
    verify_generation_bytes(generation.path(), payload, manifest_bytes)?;
    sync_directories(generation.path())?;
    // Transfer cleanup ownership and persist the store's generation entry
    // before attempting the pointer, including its no-clobber failure path.
    let generation = retain_generation_with(generation, &store, sync_directory)?;
    if let Err(source) = publish_pointer(&store, &request.slot, &pointer_bytes, request.overwrite) {
        return Err(AotError::InstallationPublication {
            generation,
            source: Box::new(source),
        });
    }
    Ok(generation)
}

fn admit_materialization(
    storage: &crate::tree::NativeStorageGrant,
    payload: &runmat_package_cache::TreeManifest,
    manifest_bytes: u64,
) -> AotResult<()> {
    let entries = storage
        .admit(payload)?
        .checked_add(INSTALLATION_OVERHEAD_ENTRIES)
        .ok_or_else(|| invalid("installation entry demand overflow"))?;
    let files = payload
        .file_count
        .checked_add(INSTALLATION_OVERHEAD_FILES)
        .ok_or_else(|| invalid("installation file demand overflow"))?;
    let bytes = payload
        .total_bytes
        .checked_add(manifest_bytes)
        .and_then(|value| value.checked_add(POINTER_PUBLICATION_BYTES))
        .ok_or_else(|| invalid("installation byte demand overflow"))?;
    storage.metadata(manifest_bytes)?;
    storage.file(manifest_bytes)?;
    storage.admit_total(entries, files, bytes)
}

fn retain_generation_with(
    generation: tempfile::TempDir,
    store: &Path,
    synchronize: impl FnOnce(&Path) -> AotResult<()>,
) -> AotResult<PathBuf> {
    let generation = generation.keep();
    if let Err(source) = synchronize(store) {
        return Err(AotError::InstallationPublication {
            generation,
            source: Box::new(source),
        });
    }
    Ok(generation)
}

/// Reopen using freshly reconstructed, verified expected input. A pointer or
/// manifest obtained from disk alone is never an installation authority.
pub fn open_native_installation(
    bundle: NativeInstallationBundle<'_>,
    request: &NativeInstallationRequest<'_>,
) -> AotResult<InstalledNativeArtifact> {
    let store = store(request)?;
    let path = store.join(request.slot.as_str());
    let bytes = physical::read_regular(&path, POINTER_MAX_BYTES)?;
    #[derive(serde::Deserialize)]
    struct Header {
        schema_version: u16,
    }
    if serde_json::from_slice::<Header>(&bytes)
        .map_err(invalid)?
        .schema_version
        != POINTER_SCHEMA_VERSION
    {
        return Err(invalid("unsupported installation pointer schema"));
    }
    let pointer: InstallationPointer = serde_json::from_slice(&bytes).map_err(invalid)?;
    if pointer.schema_version != POINTER_SCHEMA_VERSION
        || pointer.installation != bundle.identity
        || crate::tree::canonical_json(&pointer, POINTER_MAX_BYTES)? != bytes
    {
        return Err(invalid(
            "installation pointer differs from expected canonical identity",
        ));
    }
    single_locator(&pointer.generation)?;
    receipt(bundle, store.join(pointer.generation.as_str()))
}

fn receipt(
    bundle: NativeInstallationBundle<'_>,
    generation: PathBuf,
) -> AotResult<InstalledNativeArtifact> {
    let installed = InstalledNativeArtifact(Rc::new(InstalledRecord {
        generation,
        manifest: bundle.manifest,
        bytes: bundle.bytes,
        identity: bundle.identity,
        revoked: Cell::new(false),
        source_evaluation_issuer: bundle.source_evaluation_issuer,
    }));
    installed
        .revalidate()
        .map_err(|source| AotError::InstallationPublication {
            generation: installed.0.generation.clone(),
            source: Box::new(source),
        })?;
    Ok(installed)
}

fn store(request: &NativeInstallationRequest<'_>) -> AotResult<PathBuf> {
    single_locator(&request.slot)?;
    let metadata = std::fs::symlink_metadata(request.store).map_err(|error| {
        AotError::io("inspect explicit installation store", request.store, error)
    })?;
    if !metadata.is_dir() || metadata.is_symlink() {
        return Err(invalid(
            "installation store must be an existing physical directory",
        ));
    }
    std::fs::canonicalize(request.store)
        .map_err(|error| AotError::io("resolve explicit installation store", request.store, error))
}

fn single_locator(path: &NormalizedRelativePath) -> AotResult<()> {
    runmat_package_cache::validate_archive(
        [runmat_package_cache::ArchiveEntryHeader {
            path: path.as_str().to_owned(),
            kind: runmat_package_cache::ArchiveEntryKind::File,
            expanded_bytes: 0,
            compressed_bytes: 0,
            executable: false,
            link_target: None,
        }],
        runmat_package_cache::ArchiveLimits::default(),
    )
    .map_err(invalid)?;
    if path.as_str().contains('/') {
        return Err(invalid(
            "installation slot and generation must be single relative locators",
        ));
    }
    Ok(())
}

fn verify_generation(
    generation: &Path,
    manifest: &NativeInstallationManifest,
    expected: &[u8],
) -> AotResult<()> {
    verify_generation_bytes(generation, &manifest.payload, expected)
}

fn verify_generation_bytes(
    generation: &Path,
    payload: &runmat_package_cache::TreeManifest,
    expected: &[u8],
) -> AotResult<()> {
    let metadata = std::fs::symlink_metadata(generation)
        .map_err(|error| AotError::io("inspect installed generation", generation, error))?;
    if !metadata.is_dir() || metadata.is_symlink() {
        return Err(invalid(
            "installation generation is not a physical directory",
        ));
    }
    let mut inventory = Vec::new();
    for entry in std::fs::read_dir(generation)
        .map_err(|error| AotError::io("inspect installation inventory", generation, error))?
    {
        inventory.push(
            entry
                .map_err(|error| AotError::io("inspect installation entry", generation, error))?
                .file_name(),
        );
        if inventory.len() > 2 {
            return Err(invalid("installation generation has unretained entries"));
        }
    }
    inventory.sort();
    if inventory
        != [
            std::ffi::OsString::from("manifest.json"),
            std::ffi::OsString::from("payload"),
        ]
    {
        return Err(invalid("installation generation inventory changed"));
    }
    let path = generation.join("manifest.json");
    if physical::read_regular(&path, expected.len() as u64)? != expected {
        return Err(invalid(
            "installed manifest differs from admitted immutable bytes",
        ));
    }
    physical::verify(&generation.join("payload"), payload)
}

fn publish_pointer(
    store: &Path,
    slot: &NormalizedRelativePath,
    pointer: &[u8],
    overwrite: bool,
) -> AotResult<()> {
    let lease = tempfile::tempdir_in(store)
        .map_err(|error| AotError::io("lease installation publication", store, error))?;
    let source = lease.path().join("pointer");
    physical::write_new(&source, pointer)?;
    crate::link::publication::publish(
        lease,
        &source,
        &store.join(slot.as_str()),
        overwrite,
        false,
    )?;
    sync_directory(store)
}

fn sync_directories(root: &Path) -> AotResult<()> {
    for entry in std::fs::read_dir(root)
        .map_err(|error| AotError::io("inspect installation directories", root, error))?
    {
        let entry =
            entry.map_err(|error| AotError::io("inspect installation directory", root, error))?;
        if entry
            .file_type()
            .map_err(|error| {
                AotError::io("inspect installation directory kind", entry.path(), error)
            })?
            .is_dir()
        {
            sync_directories(&entry.path())?;
        }
    }
    sync_directory(root)
}

fn sync_directory(path: &Path) -> AotResult<()> {
    #[cfg(unix)]
    std::fs::File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|error| AotError::io("synchronize installation directory", path, error))?;
    // Non-Unix durability follows the existing native publication primitive;
    // this layer does not invent a target power-loss guarantee from metadata.
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

#[cfg(test)]
#[path = "tests/materialize.rs"]
mod tests;

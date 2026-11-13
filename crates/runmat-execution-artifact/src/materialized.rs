//! Authenticated Artifact facade over ProcessHost's domain-neutral private
//! byte owner. Logical-object admission and package digests stay here.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::{LogicalObject, ObjectDescriptor, ObjectNamespace};
use runmat_package::ContentDigest;
use runmat_process_host::private_bytes::{make_private, PrivateByteLease};

pub use runmat_process_host::private_bytes::{
    PrivateByteLease as PrivateArtifactLease, SELECTED_GRAPH_ATTEMPT_SCRATCH_PREFIX,
};

/// Private paths are issued only after one exact admitted write. This owner
/// indexes logical artifact members; the lower lease owns their byte lifetime.
pub struct PrivateArtifactRoot {
    root: PrivateByteLease,
    written: BTreeMap<String, PrivateArtifactPath>,
}

#[derive(Clone)]
pub struct PrivateArtifactPath {
    _root: PrivateByteLease,
    path: PathBuf,
    content_digest: ContentDigest,
}

impl PrivateArtifactPath {
    pub fn as_path(&self) -> &Path {
        &self.path
    }

    /// Authenticated digest at issuance, never reopened from ambient paths.
    pub fn content_digest(&self) -> &ContentDigest {
        &self.content_digest
    }
}

impl PrivateArtifactRoot {
    pub fn new(prefix: &str) -> Result<Self, String> {
        Ok(Self {
            root: PrivateByteLease::new_private(prefix)?,
            written: BTreeMap::new(),
        })
    }

    /// Exclusive unit directory under an already-private parent attempt root.
    pub fn in_parent_scratch(parent: &Path, unit_index: usize) -> Result<Self, String> {
        Ok(Self {
            root: PrivateByteLease::in_parent_scratch(parent, unit_index)?,
            written: BTreeMap::new(),
        })
    }

    pub fn path(&self) -> &Path {
        self.root.path()
    }

    pub fn lease(&self) -> PrivateArtifactLease {
        self.root.clone()
    }

    pub fn write_object(&mut self, object: &LogicalObject) -> Result<PrivateArtifactPath, String> {
        object.validate().map_err(|error| error.to_string())?;
        if !matches!(
            object.descriptor.namespace,
            ObjectNamespace::ProgramSource | ObjectNamespace::ForeignArtifact
        ) {
            return Err("private artifact root accepts only source or foreign objects".into());
        }
        self.write_once(&object.descriptor.logical_name, &object.bytes)
    }

    /// The caller has already authenticated the typed native/MEX/Java bundle;
    /// this method validates its logical path and supplies private storage.
    pub fn write_verified(
        &mut self,
        relative: &str,
        bytes: &[u8],
    ) -> Result<PrivateArtifactPath, String> {
        ObjectDescriptor::new(
            ObjectNamespace::ForeignArtifact,
            relative,
            "application/octet-stream",
            bytes,
        )
        .map_err(|error| error.to_string())?;
        self.write_once(relative, bytes)
    }

    pub fn admitted_path(&self, path: &Path) -> Option<&PrivateArtifactPath> {
        let relative = path
            .strip_prefix(self.root.path())
            .ok()?
            .to_str()?
            .replace('\\', "/");
        self.written
            .get(&relative)
            .filter(|issued| issued.as_path() == path)
    }

    fn write_once(&mut self, relative: &str, bytes: &[u8]) -> Result<PrivateArtifactPath, String> {
        if self.written.contains_key(relative) {
            return Err(format!("private artifact path {relative:?} is duplicated"));
        }
        let target = self.root.path().join(relative);
        let parent = target
            .parent()
            .ok_or_else(|| "private artifact has no parent directory".to_owned())?;
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create private artifact directory: {error}"))?;
        make_private(parent)?;
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = options
            .open(&target)
            .map_err(|error| format!("create private artifact {relative:?}: {error}"))?;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("write private artifact {relative:?}: {error}"))?;
        let mut permissions = file
            .metadata()
            .map_err(|error| format!("inspect private artifact {relative:?}: {error}"))?
            .permissions();
        permissions.set_readonly(true);
        std::fs::set_permissions(&target, permissions)
            .map_err(|error| format!("seal private artifact {relative:?}: {error}"))?;
        let issued = PrivateArtifactPath {
            _root: self.root.clone(),
            path: target,
            content_digest: ContentDigest::sha256(bytes),
        };
        self.written.insert(relative.to_owned(), issued.clone());
        Ok(issued)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_root_writes_only_exact_distinct_logical_bytes() {
        let mut root = PrivateArtifactRoot::new("runmat-foreign-root-test-").unwrap();
        let object = LogicalObject::new(
            ObjectNamespace::ForeignArtifact,
            "native/unit/library.bin",
            "application/octet-stream",
            b"exact library".to_vec(),
        )
        .unwrap();
        let path = root.write_object(&object).unwrap();
        assert_eq!(std::fs::read(path.as_path()).unwrap(), object.bytes);
        assert_eq!(path.content_digest(), &ContentDigest::sha256(&object.bytes));
        assert!(path.as_path().metadata().unwrap().permissions().readonly());
        assert!(root.write_object(&object).is_err());
        let mut changed = object;
        changed.bytes.push(b'!');
        assert!(root.write_object(&changed).is_err());
        assert!(root.write_verified("../escape", b"not admitted").is_err());
        assert!(path.as_path().exists());
        assert!(root.admitted_path(path.as_path()).is_some());
        let retained = path.clone();
        drop(root);
        assert!(retained.as_path().exists());
        let location = retained.as_path().to_path_buf();
        drop(path);
        drop(retained);
        assert!(!location.exists());
    }

    #[test]
    fn parent_owned_unit_paths_survive_child_views_until_parent_retires() {
        let parent = tempfile::Builder::new()
            .prefix(SELECTED_GRAPH_ATTEMPT_SCRATCH_PREFIX)
            .tempdir()
            .unwrap();
        make_private(parent.path()).unwrap();
        let mut first = PrivateArtifactRoot::in_parent_scratch(parent.path(), 0).unwrap();
        let private = first
            .write_verified("native/library.bin", b"owned bytes")
            .unwrap();
        let path = private.as_path().to_path_buf();
        assert!(PrivateArtifactRoot::in_parent_scratch(parent.path(), 0).is_err());
        assert!(PrivateArtifactRoot::in_parent_scratch(parent.path(), 1).is_ok());
        drop(private);
        drop(first);
        assert_eq!(std::fs::read(&path).unwrap(), b"owned bytes");
        drop(parent);
        assert!(!path.exists());
    }
}

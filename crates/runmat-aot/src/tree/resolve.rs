use super::{codec::invalid, physical, NativeBuildTree, NativeStorageGrant};
use crate::{AotError, AotResult};
use runmat_execution::Digest;
use runmat_execution_artifact::target_pack::{
    ComponentId, NativeBuildInput, VerifiedTargetPackContents,
};
use runmat_package::ContentDigest;
use runmat_package_cache::{relative_link_target, BlobMetadata, TreeEntryKind, TreeManifest};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Explicit raw hosted bytes. The same digest/extent/coverage checks apply;
/// neither ambient files nor a manifest's declared lengths create permission.
#[derive(Clone, Copy)]
pub struct HostedNativeTree<'a> {
    pub metadata: &'a [u8],
    pub blobs: &'a BTreeMap<ComponentId, Vec<u8>>,
}

pub(crate) enum NativeFile<'a> {
    Bytes(&'a [u8]),
    Open(std::fs::File),
}

pub(crate) struct ResolvedNativeTree<'a> {
    pub manifest: TreeManifest,
    pub files: BTreeMap<ContentDigest, NativeFile<'a>>,
}

pub(crate) fn resolve_build_tree<'a>(
    contents: &'a VerifiedTargetPackContents<'_>,
    input: &NativeBuildInput,
    hosted: Option<HostedNativeTree<'a>>,
    expected: Digest,
    grant: &NativeStorageGrant,
) -> AotResult<ResolvedNativeTree<'a>> {
    match input {
        NativeBuildInput::ContentAddressed { component } => {
            if hosted.is_some() {
                return Err(invalid(
                    "external bytes cannot replace content-addressed tree components",
                ));
            }
            let metadata = contents
                .component_bytes(component)
                .ok_or_else(|| invalid("tree root is not in the verified selected closure"))?;
            let tree = NativeBuildTree::decode(metadata, expected, grant)?;
            let reached = contents
                .component_dependency_closure(component)
                .map_err(invalid)?;
            if tree
                .blobs
                .iter()
                .any(|entry| !reached.contains(&entry.component))
            {
                return Err(invalid(
                    "tree blob is outside its retained dependency closure",
                ));
            }
            resolve(tree, |id| contents.component_bytes(id))
        }
        NativeBuildInput::HostedDeclared { fingerprint } => {
            if fingerprint != &expected {
                return Err(invalid(
                    "hosted tree differs from trusted expected identity",
                ));
            }
            let hosted = hosted
                .ok_or_else(|| invalid("hosted tree requires explicit metadata and raw blobs"))?;
            resolve_hosted(hosted, expected, grant)
        }
    }
}

pub(crate) fn resolve_hosted<'a>(
    hosted: HostedNativeTree<'a>,
    expected: Digest,
    grant: &NativeStorageGrant,
) -> AotResult<ResolvedNativeTree<'a>> {
    let tree = NativeBuildTree::decode(hosted.metadata, expected, grant)?;
    let required = tree
        .blobs
        .iter()
        .map(|entry| &entry.component)
        .collect::<BTreeSet<_>>();
    if !required.iter().copied().eq(hosted.blobs.keys()) {
        return Err(invalid(
            "hosted raw blob inventory differs from exact tree references",
        ));
    }
    resolve(tree, |id| hosted.blobs.get(id).map(Vec::as_slice))
}

fn resolve<'a>(
    tree: NativeBuildTree,
    mut bytes: impl FnMut(&ComponentId) -> Option<&'a [u8]>,
) -> AotResult<ResolvedNativeTree<'a>> {
    let mut lengths = BTreeMap::new();
    for entry in tree
        .tree
        .entries
        .iter()
        .filter(|entry| entry.kind == TreeEntryKind::File)
    {
        let digest = entry.digest.as_ref().expect("validated file");
        if lengths
            .insert(digest.clone(), entry.byte_len)
            .is_some_and(|previous| previous != entry.byte_len)
        {
            return Err(invalid(
                "shared file digest has conflicting physical extents",
            ));
        }
    }
    let mut files = BTreeMap::new();
    if lengths.contains_key(&ContentDigest::sha256([])) {
        files.insert(ContentDigest::sha256([]), NativeFile::Bytes(&[]));
    }
    for reference in &tree.blobs {
        let data = bytes(&reference.component)
            .ok_or_else(|| invalid("referenced raw tree bytes are missing"))?;
        BlobMetadata {
            digest: reference.digest.clone(),
            byte_len: lengths[&reference.digest],
        }
        .verify(data)
        .map_err(invalid)?;
        files.insert(reference.digest.clone(), NativeFile::Bytes(data));
    }
    Ok(ResolvedNativeTree {
        manifest: tree.tree,
        files,
    })
}

impl ResolvedNativeTree<'_> {
    pub fn verify_staged(&self, root: &Path) -> AotResult<()> {
        physical::verify(root, &self.manifest)
    }
    pub fn file_source(&self, path: &str) -> Option<&NativeFile<'_>> {
        let entry = self
            .manifest
            .entries
            .iter()
            .find(|entry| entry.path.as_str() == path && entry.kind == TreeEntryKind::File)?;
        self.files.get(entry.digest.as_ref()?)
    }
    pub fn file(&self, path: &str) -> Option<&[u8]> {
        match self.file_source(path)? {
            NativeFile::Bytes(bytes) => Some(bytes),
            NativeFile::Open(_) => None,
        }
    }
    pub fn executable(&self, path: &str) -> Option<&[u8]> {
        if !self.manifest.entries.iter().any(|entry| {
            entry.path.as_str() == path && entry.executable && entry.kind == TreeEntryKind::File
        }) {
            return None;
        }
        self.file(path)
    }
    pub fn stage(&self, root: &Path, grant: &NativeStorageGrant) -> AotResult<()> {
        use std::io::Seek;
        grant.admit(&self.manifest)?;
        std::fs::create_dir(root)
            .map_err(|error| AotError::io("create private native tree", root, error))?;
        for entry in &self.manifest.entries {
            let path = root.join(entry.path.as_str());
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|error| AotError::io("create native tree parent", parent, error))?;
            }
            match entry.kind {
                TreeEntryKind::Directory => std::fs::create_dir_all(&path)
                    .map_err(|error| AotError::io("create native tree directory", &path, error))?,
                TreeEntryKind::File => match self
                    .file_source(entry.path.as_str())
                    .ok_or_else(|| invalid("tree file source is missing"))?
                {
                    NativeFile::Bytes(bytes) => {
                        physical::write_blob(&path, std::io::Cursor::new(*bytes), entry)?
                    }
                    NativeFile::Open(file) => {
                        let mut reader = file.try_clone().map_err(|error| {
                            AotError::io("retain installation source file", &path, error)
                        })?;
                        reader.rewind().map_err(|error| {
                            AotError::io("rewind installation source file", &path, error)
                        })?;
                        physical::write_blob(&path, reader, entry)?;
                    }
                },
                TreeEntryKind::Symlink => {}
            }
        }
        for entry in self
            .manifest
            .entries
            .iter()
            .filter(|entry| entry.kind == TreeEntryKind::Symlink)
        {
            let path = root.join(entry.path.as_str());
            let destination = entry.link_target.as_ref().expect("validated link");
            let target = relative_link_target(&entry.path, destination);
            #[cfg(unix)]
            std::os::unix::fs::symlink(&target, &path)
                .map_err(|error| AotError::io("stage native tree link", &path, error))?;
            #[cfg(windows)]
            {
                let result = if root.join(destination.as_str()).is_dir() {
                    std::os::windows::fs::symlink_dir(&target, &path)
                } else {
                    std::os::windows::fs::symlink_file(&target, &path)
                };
                result.map_err(|error| AotError::io("stage native tree link", &path, error))?;
            }
        }
        self.verify_staged(root)
    }
}

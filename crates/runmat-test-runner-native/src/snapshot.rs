use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use runmat_test::discovery::{FrozenTestRunSnapshot, SavedRunSource, UnsavedRunBuffer};
use sha2::{Digest, Sha256};

use crate::{NativeRunnerError, NativeRunnerResult};

#[derive(Clone, Debug)]
pub struct NativeSnapshotInput {
    pub project_root: PathBuf,
    pub catalog_roots: Vec<PathBuf>,
    pub graph_digest: String,
    pub base_source_digest: Option<String>,
    pub test_config_digest: String,
    pub unsaved_buffers: Vec<UnsavedRunBuffer>,
}

/// Capture one immutable native source catalog and freeze it through the
/// portable test-domain authority. Symlink entries are deliberately ignored:
/// dependency sources enter through the installed frozen project graph rather
/// than ambient filesystem traversal.
pub fn freeze_native_snapshot(
    input: NativeSnapshotInput,
) -> NativeRunnerResult<FrozenTestRunSnapshot> {
    let sources = collect_saved_sources(&input.project_root, &input.catalog_roots)?;
    if sources.is_empty() {
        return Err(NativeRunnerError::Configuration(
            "no MATLAB source files were found in the selected test inputs".into(),
        ));
    }
    let base_source_digest = input
        .base_source_digest
        .unwrap_or_else(|| source_catalog_digest(&sources));
    FrozenTestRunSnapshot::freeze(
        input.graph_digest,
        base_source_digest,
        1,
        1,
        input.test_config_digest,
        sources,
        input.unsaved_buffers,
    )
    .map_err(|error| NativeRunnerError::Configuration(error.to_string()))
}

pub fn collect_saved_sources(
    project_root: &Path,
    roots: &[PathBuf],
) -> NativeRunnerResult<Vec<SavedRunSource>> {
    let canonical_project_root =
        std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let mut files = BTreeSet::new();
    for root in roots {
        collect_matlab_files(root, &mut files)?;
    }
    files
        .into_iter()
        .map(|path| {
            let content = std::fs::read_to_string(&path)?;
            let (owner_identity, relative_path) = source_identity(&canonical_project_root, &path);
            Ok(SavedRunSource {
                owner_identity,
                relative_path,
                content,
            })
        })
        .collect()
}

pub fn source_catalog_digest(sources: &[SavedRunSource]) -> String {
    let mut hasher = Sha256::new();
    for source in sources {
        for field in [
            source.owner_identity.as_bytes(),
            source.relative_path.as_bytes(),
            source.content.as_bytes(),
        ] {
            hasher.update((field.len() as u64).to_be_bytes());
            hasher.update(field);
        }
    }
    format!("sha256:{:x}", hasher.finalize())
}

pub fn source_prefix(project_root: &Path, target: &Path) -> Option<String> {
    let relative = target.strip_prefix(project_root).ok()?;
    let mut prefix = relative.to_string_lossy().replace('\\', "/");
    if target.is_dir() && !prefix.is_empty() {
        prefix.push('/');
    }
    Some(prefix)
}

fn collect_matlab_files(path: &Path, files: &mut BTreeSet<PathBuf>) -> NativeRunnerResult<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Ok(());
    }
    if metadata.is_file() {
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("m"))
        {
            files.insert(std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()));
        }
        return Ok(());
    }
    let mut entries = std::fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        collect_matlab_files(&entry.path(), files)?;
    }
    Ok(())
}

fn source_identity(project_root: &Path, path: &Path) -> (String, String) {
    if let Ok(relative) = path.strip_prefix(project_root) {
        return (
            "path:workspace".into(),
            relative.to_string_lossy().replace('\\', "/"),
        );
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    (
        format!(
            "path:external:{:x}",
            Sha256::digest(parent.to_string_lossy().as_bytes())
        ),
        path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn freezes_sorted_matlab_sources_and_ignores_other_files() {
        let root = tempdir().unwrap();
        fs::create_dir(root.path().join("tests")).unwrap();
        fs::write(root.path().join("tests/b.m"), "%% b\n").unwrap();
        fs::write(root.path().join("tests/a.m"), "%% a\n").unwrap();
        fs::write(root.path().join("tests/readme.txt"), "ignored").unwrap();
        let snapshot = freeze_native_snapshot(NativeSnapshotInput {
            project_root: root.path().to_path_buf(),
            catalog_roots: vec![root.path().join("tests")],
            graph_digest: "sha256:graph".into(),
            base_source_digest: None,
            test_config_digest: "sha256:config".into(),
            unsaved_buffers: Vec::new(),
        })
        .unwrap();

        assert_eq!(snapshot.sources.len(), 2);
        assert_eq!(snapshot.sources[0].relative_path, "tests/a.m");
        assert_eq!(snapshot.sources[1].relative_path, "tests/b.m");
    }

    #[test]
    fn explicit_project_revision_is_preserved() {
        let root = tempdir().unwrap();
        fs::write(root.path().join("test.m"), "%% test\n").unwrap();
        let snapshot = freeze_native_snapshot(NativeSnapshotInput {
            project_root: root.path().to_path_buf(),
            catalog_roots: vec![root.path().to_path_buf()],
            graph_digest: "sha256:graph".into(),
            base_source_digest: Some("sha256:project-source".into()),
            test_config_digest: "sha256:config".into(),
            unsaved_buffers: Vec::new(),
        })
        .unwrap();

        assert_eq!(snapshot.base_source_digest, "sha256:project-source");
    }
}

use crate::NativeCacheError;
use runmat_package::ContentDigest;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheLayout {
    pub root: PathBuf,
    pub database: PathBuf,
    pub staging: PathBuf,
    pub trees: PathBuf,
    pub locks: PathBuf,
    pub git_repositories: PathBuf,
}

impl CacheLayout {
    pub fn new(root: PathBuf) -> Self {
        Self {
            database: root.join("cache.sqlite3"),
            staging: root.join("staging"),
            trees: root.join("trees"),
            locks: root.join("locks"),
            git_repositories: root.join("git"),
            root,
        }
    }

    pub fn create(&self) -> Result<(), NativeCacheError> {
        for directory in [
            &self.root,
            &self.staging,
            &self.trees,
            &self.locks,
            &self.git_repositories,
        ] {
            std::fs::create_dir_all(directory)
                .map_err(|error| NativeCacheError::io(directory, error))?;
        }
        Ok(())
    }

    pub fn git_repository_path(&self, repository: &runmat_package::GitRepositoryUrl) -> PathBuf {
        self.git_repositories
            .join(storage_digest(&ContentDigest::sha256(repository.as_str())))
    }

    pub fn git_repository_lock(&self, repository: &runmat_package::GitRepositoryUrl) -> PathBuf {
        self.locks.join(format!(
            "git-{}.lock",
            storage_digest(&ContentDigest::sha256(repository.as_str()))
        ))
    }

    pub fn tree_path(&self, digest: &ContentDigest) -> PathBuf {
        self.trees.join(storage_digest(digest))
    }

    pub fn materialization_lock(&self, digest: &ContentDigest) -> PathBuf {
        self.locks
            .join(format!("materialize-{}.lock", storage_digest(digest)))
    }
}

pub(crate) fn storage_digest(digest: &ContentDigest) -> String {
    let mut result = String::with_capacity(64);
    for byte in digest.bytes() {
        use std::fmt::Write as _;
        write!(result, "{byte:02x}").expect("writing to a string cannot fail");
    }
    result
}

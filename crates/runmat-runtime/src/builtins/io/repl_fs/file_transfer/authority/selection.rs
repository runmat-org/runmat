use super::TransferFilesystem;
use std::io;
use std::path::{Path, PathBuf};

impl TransferFilesystem {
    pub(in super::super) async fn same_path(&self, left: &Path, right: &Path) -> bool {
        if self.path(left) == self.path(right) {
            return true;
        }
        match (
            self.canonicalize(left).await,
            self.canonicalize(right).await,
        ) {
            (Ok(left), Ok(right)) => left == right,
            _ => false,
        }
    }

    pub(in super::super) async fn is_descendant(&self, parent: &Path, candidate: &Path) -> bool {
        let parent = self.path(parent);
        let candidate = self.path(candidate);
        if candidate.starts_with(&parent) && candidate != parent {
            return true;
        }
        match (
            self.canonicalize(&parent).await,
            self.canonicalize(&candidate).await,
        ) {
            (Ok(parent), Ok(candidate)) => candidate.starts_with(&parent) && candidate != parent,
            _ => false,
        }
    }

    pub(in super::super) async fn wildcard_matches(
        &self,
        pattern: &str,
    ) -> io::Result<Vec<PathBuf>> {
        let path = Path::new(pattern);
        let name = path.file_name().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "pattern has no file name")
        })?;
        let matcher = glob::Pattern::new(&name.to_string_lossy())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.msg))?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut matches = self
            .read_dir(parent)
            .await?
            .into_iter()
            .filter(|entry| matcher.matches(&entry.file_name().to_string_lossy()))
            .map(|entry| entry.path().to_path_buf())
            .collect::<Vec<_>>();
        matches.sort();
        Ok(matches)
    }

    pub(in super::super) async fn remove_existing(
        &self,
        path: &Path,
        directory: bool,
    ) -> io::Result<()> {
        let result = if directory {
            self.remove_dir_all(path).await
        } else {
            self.remove_file(path).await
        };
        match result {
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            other => other,
        }
    }
}

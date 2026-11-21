//! Moves and removals of the outputs a pinned execution created itself.
//!
//! An execution owns exactly the files and directories it published. It may
//! rename or remove those, and nothing else: inputs, dependency mounts and
//! files other users created stay untouched. A file moves by publishing its
//! bytes at the destination and then removing the exact source version, so the
//! destination becomes visible as one complete file. A directory moves only
//! when the execution owns everything in it.
use super::super::output::Output;
use super::{denied, normalize, PinnedFsProvider};
use crate::FsProvider;
use std::{
    io,
    path::{Path, PathBuf},
};

impl PinnedFsProvider {
    fn owned_output(&self, path: &Path) -> Option<Output> {
        self.outputs
            .lock()
            .expect("output ownership lock")
            .get(path)
            .cloned()
    }

    /// Owned descendants of `root`, parents before children.
    fn owned_descendants(&self, root: &Path) -> Vec<(PathBuf, Output)> {
        let mut descendants: Vec<_> = self
            .outputs
            .lock()
            .expect("output ownership lock")
            .iter()
            .filter(|(path, _)| path.as_path() != root && path.starts_with(root))
            .map(|(path, output)| (path.clone(), output.clone()))
            .collect();
        descendants.sort_by_key(|(path, _)| path.components().count());
        descendants
    }

    /// Refuse a directory that holds an input, a mount or a file this
    /// execution did not create, before changing anything.
    async fn require_wholly_owned(&self, root: &Path) -> io::Result<()> {
        if self.entries.keys().any(|path| path.starts_with(root)) || self.mounts.intersects(root) {
            return Err(denied("execution inputs are immutable"));
        }
        let mut pending = vec![root.to_path_buf()];
        while let Some(directory) = pending.pop() {
            for entry in self.base.read_dir(&directory).await? {
                let path = directory.join(entry.file_name());
                match self.owned_output(&path) {
                    Some(Output::Directory) => pending.push(path),
                    Some(Output::File { .. }) => {}
                    None => {
                        return Err(denied(
                            "the directory holds files this execution did not create",
                        ))
                    }
                }
            }
        }
        Ok(())
    }

    async fn move_owned_file(&self, from: &Path, to: &Path, output: &Output) -> io::Result<()> {
        let bytes = output.read(self.base.as_ref(), from).await?;
        let expected = match self.owned_output(to) {
            Some(Output::File { version, .. }) => Some(version),
            Some(Output::Directory) => {
                return Err(io::Error::new(
                    io::ErrorKind::IsADirectory,
                    "rename destination is a directory",
                ))
            }
            None => None,
        };
        let version = self
            .base
            .publish_output(to, &bytes, expected.as_deref())
            .await?;
        self.outputs.lock().expect("output ownership lock").insert(
            to.to_path_buf(),
            Output::File {
                version,
                size: bytes.len() as u64,
            },
        );
        self.base.remove_output(from, output.version()?).await?;
        self.outputs
            .lock()
            .expect("output ownership lock")
            .remove(from);
        Ok(())
    }

    pub(super) async fn rename_owned(&self, from: &Path, to: &Path) -> io::Result<()> {
        let from = normalize(from)?;
        let to = normalize(to)?;
        self.mutable(&from)?;
        self.mutable(&to)?;
        let Some(output) = self.owned_output(&from) else {
            return Err(denied("rename files this execution did not create"));
        };
        if from == to {
            return Ok(());
        }
        match output {
            Output::File { .. } => self.move_owned_file(&from, &to, &output).await,
            Output::Directory => {
                if to.starts_with(&from) {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "cannot move a directory into itself",
                    ));
                }
                if self.metadata(&to).await.is_ok() {
                    return Err(io::ErrorKind::AlreadyExists.into());
                }
                self.require_wholly_owned(&from).await?;
                let descendants = self.owned_descendants(&from);
                for (path, _) in &descendants {
                    self.mutable(&to.join(path.strip_prefix(&from).expect("descendant")))?;
                }
                self.create_dir(&to).await?;
                for (path, output) in &descendants {
                    let target = to.join(path.strip_prefix(&from).expect("descendant"));
                    match output {
                        Output::Directory => self.create_dir(&target).await?,
                        Output::File { .. } => self.move_owned_file(path, &target, output).await?,
                    }
                }
                self.remove_owned_tree(&from).await
            }
        }
    }

    pub(super) async fn remove_owned_dir(&self, path: &Path) -> io::Result<()> {
        let path = normalize(path)?;
        self.mutable(&path)?;
        match self.owned_output(&path) {
            Some(Output::Directory) => {}
            Some(Output::File { .. }) => {
                return Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    "path is a file",
                ))
            }
            None => return Err(denied("remove directories this execution did not create")),
        }
        self.base.remove_dir(&path).await?;
        self.outputs
            .lock()
            .expect("output ownership lock")
            .remove(&path);
        Ok(())
    }

    pub(super) async fn remove_owned_dir_all(&self, path: &Path) -> io::Result<()> {
        let path = normalize(path)?;
        self.mutable(&path)?;
        if !matches!(self.owned_output(&path), Some(Output::Directory)) {
            return Err(denied("remove directories this execution did not create"));
        }
        self.require_wholly_owned(&path).await?;
        self.remove_owned_tree(&path).await
    }

    /// Remove an owned tree whose contents were already checked, children
    /// before parents.
    async fn remove_owned_tree(&self, root: &Path) -> io::Result<()> {
        for (path, output) in self.owned_descendants(root).into_iter().rev() {
            match output {
                Output::File { ref version, .. } => self.base.remove_output(&path, version).await?,
                Output::Directory => self.base.remove_dir(&path).await?,
            }
            self.outputs
                .lock()
                .expect("output ownership lock")
                .remove(&path);
        }
        self.base.remove_dir(root).await?;
        self.outputs
            .lock()
            .expect("output ownership lock")
            .remove(root);
        Ok(())
    }
}

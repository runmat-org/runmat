//! Shared atomic publication. The old destination never enters a cleanup
//! lease; replacement uses tempfile's existing cross-platform atomic owner.
use std::path::{Path, PathBuf};

use crate::{AotError, AotResult};

#[derive(Clone, Copy)]
enum PublicationPolicy {
    NoClobber,
    Replace,
}

pub(crate) fn publish(
    lease: tempfile::TempDir,
    source: &Path,
    output: &Path,
    overwrite: bool,
    keep_inputs: bool,
) -> AotResult<Option<PathBuf>> {
    let policy = if overwrite {
        PublicationPolicy::Replace
    } else {
        PublicationPolicy::NoClobber
    };
    publish_with(
        lease,
        source,
        output,
        policy,
        keep_inputs,
        |source, output, policy| match policy {
            // Atomic destination insertion, as used by the existing native object
            // stores. Source-name retirement belongs to the enclosing TempDir; it
            // is not part of the no-clobber publication linearization point.
            PublicationPolicy::NoClobber => {
                std::fs::hard_link(&source, output).map_err(|error| tempfile::PathPersistError {
                    error,
                    path: source,
                })
            }
            PublicationPolicy::Replace => source.persist(output),
        },
    )
}

fn publish_with(
    lease: tempfile::TempDir,
    source: &Path,
    output: &Path,
    policy: PublicationPolicy,
    keep_inputs: bool,
    persist: impl FnOnce(
        tempfile::TempPath,
        &Path,
        PublicationPolicy,
    ) -> Result<(), tempfile::PathPersistError>,
) -> AotResult<Option<PathBuf>> {
    let mut owned = match tempfile::TempPath::try_from_path(source) {
        Ok(path) => path,
        Err(source) => {
            return Err(AotError::Publication {
                output: output.to_path_buf(),
                recovery: lease.keep(),
                source,
            })
        }
    };
    // The enclosing lease is the only cleanup owner. In particular, an error's
    // TempPath must not erase the staged image before recovery is retained.
    owned.disable_cleanup(true);
    if let Err(error) = persist(owned, output, policy) {
        let recovery = lease.keep();
        return Err(AotError::Publication {
            output: output.to_path_buf(),
            recovery,
            source: error.error,
        });
    }
    Ok(keep_inputs.then(|| lease.keep()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_publication_creates_without_clobber_and_replaces_explicitly() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("program");
        for (contents, overwrite) in [(b"first".as_slice(), false), (b"second".as_slice(), true)] {
            let lease = tempfile::tempdir_in(root.path()).unwrap();
            let source = lease.path().join("linked");
            std::fs::write(&source, contents).unwrap();
            assert!(publish(lease, &source, &output, overwrite, false)
                .unwrap()
                .is_none());
            assert_eq!(std::fs::read(&output).unwrap(), contents);
        }
    }

    #[test]
    fn failed_publication_preserves_prior_output_and_retains_structured_recovery() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("program");
        std::fs::write(&output, b"prior").unwrap();
        let lease = tempfile::tempdir_in(root.path()).unwrap();
        let source = lease.path().join("linked");
        std::fs::write(&source, b"new").unwrap();
        let error = publish_with(
            lease,
            &source,
            &output,
            PublicationPolicy::Replace,
            false,
            |path, _, _| {
                Err(tempfile::PathPersistError {
                    error: std::io::Error::other("injected publication failure"),
                    path,
                })
            },
        )
        .unwrap_err();
        let AotError::Publication { recovery, .. } = error else {
            panic!("structured recovery required")
        };
        assert_eq!(std::fs::read(&output).unwrap(), b"prior");
        assert_eq!(std::fs::read(recovery.join("linked")).unwrap(), b"new");
        // The outer test-owned TempDir reclaims its retained child afterward.
    }

    #[test]
    fn ordinary_existing_destination_is_not_clobbered() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("program");
        std::fs::write(&output, b"prior").unwrap();
        let lease = tempfile::tempdir_in(root.path()).unwrap();
        let source = lease.path().join("linked");
        std::fs::write(&source, b"new").unwrap();
        let error = publish(lease, &source, &output, false, false).unwrap_err();
        let AotError::Publication {
            recovery, source, ..
        } = error
        else {
            panic!("structured recovery required")
        };
        assert_eq!(source.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&output).unwrap(), b"prior");
        assert_eq!(std::fs::read(recovery.join("linked")).unwrap(), b"new");
    }
}

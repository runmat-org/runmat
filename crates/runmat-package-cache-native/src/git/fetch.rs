use super::credentials::{GitCredential, GitCredentialProvider};
use super::remote::git_error;
use crate::NativeCacheError;
use git2::{Cred, FetchOptions, Oid, RemoteCallbacks, Repository};
use runmat_package::{GitCommitId, GitObjectAlgorithm, GitSelector};

pub(super) fn resolve_commit(
    repository: &Repository,
    selector: &GitSelector,
    allow_network: bool,
    credentials: &dyn GitCredentialProvider,
) -> Result<GitCommitId, NativeCacheError> {
    let exact = match selector {
        GitSelector::Rev { value } => {
            let commit: GitCommitId =
                value
                    .parse()
                    .map_err(|error: runmat_package::IdentityError| {
                        NativeCacheError::Git(error.to_string())
                    })?;
            if commit.algorithm != GitObjectAlgorithm::Sha1 {
                return Err(NativeCacheError::Git(
                    "this libgit2 build does not support SHA-256 repositories".to_string(),
                ));
            }
            Some(commit)
        }
        _ => None,
    };
    let (source_ref, local_ref) = selector_refs(selector)?;
    if allow_network {
        fetch(repository, &source_ref, &local_ref, credentials)?;
    }
    let oid = match selector {
        GitSelector::Rev { .. } => {
            let commit = exact.expect("revision selector parsed before fetch");
            let oid = Oid::from_str(&commit.hex).map_err(git_error)?;
            if repository.find_commit(oid).is_ok() {
                return Ok(commit);
            }
            repository.refname_to_id(&local_ref).map_err(|_| {
                NativeCacheError::Git(format!(
                    "exact commit {} is not available in the local Git cache",
                    commit.hex
                ))
            })?
        }
        _ => repository.refname_to_id(&local_ref).map_err(|_| {
            NativeCacheError::Git(
                "Git selector is not available in the local cache; fetch is required".to_string(),
            )
        })?,
    };
    let commit = repository
        .find_object(oid, None)
        .and_then(|object| object.peel_to_commit())
        .map_err(git_error)?;
    commit
        .id()
        .to_string()
        .parse()
        .map_err(|error: runmat_package::IdentityError| NativeCacheError::Git(error.to_string()))
}

fn selector_refs(selector: &GitSelector) -> Result<(String, String), NativeCacheError> {
    let source = match selector {
        GitSelector::Rev { value } => value.clone(),
        GitSelector::Branch { value } => format!("refs/heads/{value}"),
        GitSelector::Tag { value } => format!("refs/tags/{value}"),
    };
    if !matches!(selector, GitSelector::Rev { .. }) && !git2::Reference::is_valid_name(&source) {
        return Err(NativeCacheError::Git(format!(
            "invalid Git selector reference `{source}`"
        )));
    }
    let canonical = match selector {
        GitSelector::Rev { value } => format!("rev\0{value}"),
        GitSelector::Branch { value } => format!("branch\0{value}"),
        GitSelector::Tag { value } => format!("tag\0{value}"),
    };
    let key = runmat_package::ContentDigest::sha256(canonical);
    let key: String = key
        .bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok((source, format!("refs/runmat/selectors/{key}")))
}

fn fetch(
    repository: &Repository,
    source_ref: &str,
    local_ref: &str,
    credentials: &dyn GitCredentialProvider,
) -> Result<(), NativeCacheError> {
    let mut callbacks = RemoteCallbacks::new();
    callbacks.credentials(|url, username, allowed| {
        let credential = credentials.credential(url, username, allowed);
        match credential {
            Some(GitCredential::UserPassword { username, password }) => {
                Cred::userpass_plaintext(&username, &password)
            }
            Some(GitCredential::SshKey {
                username,
                public_key,
                private_key,
                passphrase,
            }) => Cred::ssh_key(
                &username,
                public_key.as_deref(),
                &private_key,
                passphrase.as_deref(),
            ),
            Some(GitCredential::SshAgent { username }) => Cred::ssh_key_from_agent(&username),
            Some(GitCredential::Default) => Cred::default(),
            None if allowed.contains(git2::CredentialType::DEFAULT) => Cred::default(),
            None => Err(git2::Error::from_str(
                "no credential was provided for the Git remote",
            )),
        }
    });
    let mut options = FetchOptions::new();
    options.remote_callbacks(callbacks);
    let refspec = format!("+{source_ref}:{local_ref}");
    repository
        .find_remote("origin")
        .and_then(|mut remote| remote.fetch(&[&refspec], Some(&mut options), None))
        .map_err(git_error)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::git::NoGitCredentials;

    #[test]
    fn mutable_branch_advances_only_when_network_update_is_explicit() {
        let directory = tempfile::tempdir().unwrap();
        let source_path = directory.path().join("source");
        let source = Repository::init(&source_path).unwrap();
        let first = commit_file(&source, "value.txt", b"one", &[]);
        source.set_head("refs/heads/main").unwrap();

        let cache = Repository::init_bare(directory.path().join("cache")).unwrap();
        cache
            .remote("origin", source_path.to_str().unwrap())
            .unwrap();
        let selector = GitSelector::Branch {
            value: "main".to_string(),
        };
        let credentials = NoGitCredentials;
        let initial = resolve_commit(&cache, &selector, true, &credentials).unwrap();
        assert_eq!(initial.hex, first.to_string());

        let second = commit_file(&source, "value.txt", b"two", &[first]);
        let offline = resolve_commit(&cache, &selector, false, &credentials).unwrap();
        assert_eq!(offline.hex, first.to_string());
        let updated = resolve_commit(&cache, &selector, true, &credentials).unwrap();
        assert_eq!(updated.hex, second.to_string());
    }

    fn commit_file(repository: &Repository, path: &str, bytes: &[u8], parents: &[Oid]) -> Oid {
        let workdir = repository.workdir().unwrap();
        std::fs::write(workdir.join(path), bytes).unwrap();
        let mut index = repository.index().unwrap();
        index.add_path(Path::new(path)).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repository.find_tree(tree_id).unwrap();
        let signature = git2::Signature::now("RunMat Test", "test@runmat.invalid").unwrap();
        let parent_commits = parents
            .iter()
            .map(|oid| repository.find_commit(*oid).unwrap())
            .collect::<Vec<_>>();
        repository
            .commit(
                Some("refs/heads/main"),
                &signature,
                &signature,
                "test",
                &tree,
                &parent_commits.iter().collect::<Vec<_>>(),
            )
            .unwrap()
    }
}

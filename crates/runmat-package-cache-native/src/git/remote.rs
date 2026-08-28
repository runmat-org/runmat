use crate::filesystem::CacheLayout;
use crate::NativeCacheError;
use git2::Repository;
use runmat_package::GitRepositoryUrl;

pub(super) fn open_or_initialize(
    layout: &CacheLayout,
    repository: &GitRepositoryUrl,
) -> Result<Repository, NativeCacheError> {
    let path = layout.git_repository_path(repository);
    let repo = if path.exists() {
        Repository::open_bare(&path).map_err(git_error)?
    } else {
        Repository::init_bare(&path).map_err(git_error)?
    };
    match repo.find_remote("origin") {
        Ok(remote) => {
            if remote.url() != Some(repository.as_str()) {
                return Err(NativeCacheError::Git(
                    "shared repository origin does not match its normalized identity".to_string(),
                ));
            }
        }
        Err(error) if error.code() == git2::ErrorCode::NotFound => {
            repo.remote("origin", repository.as_str())
                .map_err(git_error)?;
        }
        Err(error) => return Err(git_error(error)),
    }
    Ok(repo)
}

pub(super) fn git_error(error: git2::Error) -> NativeCacheError {
    NativeCacheError::Git(error.message().to_string())
}

use futures::executor::block_on;
use runmat_package::{
    GitAcquisitionPlan, GitLockAction, GitPackageProvider, GitRepositoryUrl, GitSelector,
    NormalizedRelativePath,
};
use runmat_package_cache_native::filesystem::CacheLayout;
use runmat_package_cache_native::git::{
    GitAcquireRequest, NativeGitClient, NativeGitPackageProvider,
};
use runmat_package_cache_native::{NativeCacheConfig, SqliteCacheBackend};
use std::sync::Arc;

#[test]
fn exact_commit_and_subdirectory_are_reused_offline() {
    let directory = tempfile::tempdir().unwrap();
    let source = create_source_repository(directory.path().join("source"));
    let layout = CacheLayout::new(directory.path().join("cache"));
    layout.create().unwrap();
    let repository = GitRepositoryUrl::new("https://example.com/acme/project.git").unwrap();
    seed_shared_repository(&layout, &repository, &source);
    let commit = source.head().unwrap().target().unwrap().to_string();
    let request = GitAcquireRequest {
        repository,
        selector: GitSelector::Rev {
            value: commit.clone(),
        },
        subdir: NormalizedRelativePath::new("package").unwrap(),
        allow_network: false,
    };

    let client = NativeGitClient::new(layout.clone());
    let first = client.acquire(&request).unwrap();
    let second = client.acquire(&request).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.source.commit.hex, commit);
    assert_eq!(first.source.subdir.as_str(), "package");
    assert_eq!(
        first
            .tree
            .entries
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>(),
        expected_paths()
    );
    first.validate().unwrap();
}

#[test]
fn concurrent_clients_produce_one_identical_snapshot() {
    let directory = tempfile::tempdir().unwrap();
    let source = create_source_repository(directory.path().join("source"));
    let layout = CacheLayout::new(directory.path().join("cache"));
    layout.create().unwrap();
    let repository = GitRepositoryUrl::new("https://example.com/acme/project.git").unwrap();
    seed_shared_repository(&layout, &repository, &source);
    let request = GitAcquireRequest {
        repository,
        selector: GitSelector::Rev {
            value: source.head().unwrap().target().unwrap().to_string(),
        },
        subdir: NormalizedRelativePath::new("package").unwrap(),
        allow_network: false,
    };
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let layout = layout.clone();
            let request = request.clone();
            std::thread::spawn(move || NativeGitClient::new(layout).acquire(&request).unwrap())
        })
        .collect();
    let snapshots: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(snapshots[0], snapshots[1]);
}

#[test]
fn exact_snapshot_replays_from_transactional_cache_without_git_storage() {
    let directory = tempfile::tempdir().unwrap();
    let source = create_source_repository(directory.path().join("source"));
    let config = NativeCacheConfig {
        root: directory.path().join("cache"),
        quota_bytes: None,
    };
    let layout = config.layout();
    layout.create().unwrap();
    let repository = GitRepositoryUrl::new("https://example.com/acme/project.git").unwrap();
    seed_shared_repository(&layout, &repository, &source);
    let commit = source.head().unwrap().target().unwrap().to_string();
    let backend = Arc::new(SqliteCacheBackend::open(&config).unwrap());
    let provider = NativeGitPackageProvider::new(
        NativeGitClient::new(layout.clone()),
        backend,
        layout.clone(),
    );
    let initial = GitAcquisitionPlan {
        repository: repository.clone(),
        selector: GitSelector::Rev { value: commit },
        subdir: NormalizedRelativePath::new("package").unwrap(),
        allow_network: false,
        expected: None,
        lock_action: GitLockAction::Write,
    };
    let first = block_on(provider.acquire_git(&initial)).unwrap();
    let unavailable = directory.path().join("detached-git-storage");
    std::fs::rename(layout.git_repository_path(&repository), &unavailable).unwrap();
    let replay = GitAcquisitionPlan {
        expected: Some(first.source.clone()),
        lock_action: GitLockAction::Preserve,
        ..initial
    };
    let second = block_on(provider.acquire_git(&replay)).unwrap();
    assert_eq!(second.source, first.source);
    assert_eq!(second.root, first.root);
}

fn create_source_repository(path: std::path::PathBuf) -> git2::Repository {
    std::fs::create_dir_all(path.join("package")).unwrap();
    std::fs::write(path.join("root.txt"), b"outside subdirectory").unwrap();
    std::fs::write(path.join("package/main.m"), b"answer = 42;\n").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink("main.m", path.join("package/link.m")).unwrap();

    let repository = git2::Repository::init(&path).unwrap();
    let mut index = repository.index().unwrap();
    index
        .add_all(["*"], git2::IndexAddOption::DEFAULT, None)
        .unwrap();
    index.write().unwrap();
    let tree_id = index.write_tree().unwrap();
    {
        let tree = repository.find_tree(tree_id).unwrap();
        let signature = git2::Signature::now("RunMat Test", "test@runmat.invalid").unwrap();
        repository
            .commit(
                Some("refs/heads/main"),
                &signature,
                &signature,
                "initial",
                &tree,
                &[],
            )
            .unwrap();
    }
    repository.set_head("refs/heads/main").unwrap();
    repository
}

fn seed_shared_repository(
    layout: &CacheLayout,
    identity: &GitRepositoryUrl,
    source: &git2::Repository,
) {
    let target = git2::Repository::init_bare(layout.git_repository_path(identity)).unwrap();
    target.remote("origin", identity.as_str()).unwrap();
    target
        .remote("seed", source.path().parent().unwrap().to_str().unwrap())
        .unwrap();
    target
        .find_remote("seed")
        .unwrap()
        .fetch(&["+refs/heads/main:refs/heads/seed"], None, None)
        .unwrap();
}

#[cfg(unix)]
fn expected_paths() -> Vec<&'static str> {
    vec!["link.m", "main.m"]
}

#[cfg(not(unix))]
fn expected_paths() -> Vec<&'static str> {
    vec!["main.m"]
}

use runmat_package::{GitRepositoryUrl, GitSelector, NormalizedRelativePath};
use runmat_package_cache_native::filesystem::CacheLayout;
use runmat_package_cache_native::git::{GitAcquireRequest, NativeGitClient};
use std::fs;
use std::process::{Command, Output};
use tempfile::TempDir;

#[test]
fn package_inspect_emits_the_deterministic_release_contract() {
    let temp = TempDir::new().unwrap();
    let project = temp.path().join("project");
    let cache = temp.path().join("cache");
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(
        project.join("runmat.toml"),
        r#"
[package]
name = "tools"
organization = "acme"
version = "1.2.3"

[sources]
roots = ["src"]

[publish]
license = "MIT"
"#,
    )
    .unwrap();
    fs::write(
        project.join("src/tool.m"),
        "function value = tool(); value = 42; end\n",
    )
    .unwrap();

    let first = runmat(&project, &cache, &["package", "inspect", "--json"]);
    let second = runmat(&project, &cache, &["package", "inspect", "--json"]);
    assert!(first.status.success(), "{first:?}");
    assert_eq!(first.stdout, second.stdout);
    let output: serde_json::Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(output["manifest"]["version"], "1.2.3");
    assert_eq!(output["manifest"]["license"], "MIT");
    assert_eq!(output["inventory"]["fileCount"], 2);
    assert!(output["manifest"]["artifactDigest"]
        .as_str()
        .unwrap()
        .starts_with("sha256:"));
}

#[test]
fn offline_git_project_resolves_locks_checks_and_executes_from_shared_cache() {
    let temp = TempDir::new().unwrap();
    let cache = temp.path().join("cache");
    let source = create_source_repository(temp.path().join("source"));
    let repository = GitRepositoryUrl::new("https://example.com/acme/helper.git").unwrap();
    let layout = CacheLayout::new(cache.clone());
    layout.create().unwrap();
    seed_shared_repository(&layout, &repository, &source);
    let commit = source.head().unwrap().target().unwrap().to_string();
    NativeGitClient::new(layout)
        .acquire(&GitAcquireRequest {
            repository,
            selector: GitSelector::Rev {
                value: commit.clone(),
            },
            subdir: NormalizedRelativePath::new(".").unwrap(),
            allow_network: false,
        })
        .unwrap();

    let project = temp.path().join("project");
    fs::create_dir_all(project.join("src")).unwrap();
    fs::write(project.join("src/main.m"), "answer = helper();\n").unwrap();
    fs::write(
        project.join("runmat.toml"),
        format!(
            r#"
[package]
name = "application"
version = "1.0.0"

[sources]
roots = ["src"]

[dependencies]
helper = {{ git = "https://example.com/acme/helper.git", rev = "{commit}", version = "2.0.0" }}
"#
        ),
    )
    .unwrap();

    let resolved = runmat(&project, &cache, &["--offline", "package", "resolve"]);
    assert!(resolved.status.success(), "{resolved:?}");
    assert!(project.join("runmat.lock").is_file());

    let checked = runmat(&project, &cache, &["--offline", "check", "src/main.m"]);
    assert!(checked.status.success(), "{checked:?}");
    assert!(!String::from_utf8_lossy(&checked.stdout).contains("cannot find function"));

    let executed = runmat(&project, &cache, &["--offline", "run", "src/main.m"]);
    assert!(executed.status.success(), "{executed:?}");
}

#[test]
fn frozen_path_project_executes_from_verified_vendor_copy_and_rejects_tampering() {
    let temp = TempDir::new().unwrap();
    let project = temp.path().join("project");
    let cache = temp.path().join("cache");
    fs::create_dir_all(project.join("src")).unwrap();
    fs::create_dir_all(project.join("deps/helper/src")).unwrap();
    fs::write(project.join("src/main.m"), "answer = helper();\n").unwrap();
    fs::write(
        project.join("runmat.toml"),
        r#"
[package]
name = "application"
version = "1.0.0"
[sources]
roots = ["src"]
[dependencies]
helper = { path = "deps/helper", version = "2.0.0" }
"#,
    )
    .unwrap();
    fs::write(
        project.join("deps/helper/runmat.toml"),
        r#"
[package]
name = "helper"
version = "2.0.0"
[sources]
roots = ["src"]
"#,
    )
    .unwrap();
    fs::write(
        project.join("deps/helper/src/helper.m"),
        "function value = helper(); value = 42; end\n",
    )
    .unwrap();

    let resolved = runmat(&project, &cache, &["package", "resolve"]);
    assert!(resolved.status.success(), "{resolved:?}");
    let unvendored = runmat(&project, &cache, &["--frozen", "run", "src/main.m"]);
    assert!(!unvendored.status.success(), "{unvendored:?}");
    assert!(
        String::from_utf8_lossy(&unvendored.stderr)
            .contains(runmat_package::VENDOR_MANIFEST_FILENAME),
        "{unvendored:?}"
    );
    let vendored = runmat(&project, &cache, &["package", "vendor"]);
    assert!(vendored.status.success(), "{vendored:?}");
    let vendor_manifest: runmat_package::VendorManifest = serde_json::from_slice(
        &fs::read(project.join(runmat_package::VENDOR_MANIFEST_FILENAME)).unwrap(),
    )
    .unwrap();
    let helper = vendor_manifest
        .packages
        .iter()
        .find(|package| matches!(package.source, runmat_package::SourceId::Path(_)))
        .unwrap();
    let vendored_helper = project.join(helper.path.as_str()).join("src/helper.m");
    assert!(vendored_helper.is_file());

    fs::remove_dir_all(project.join("deps")).unwrap();
    let frozen = runmat(&project, &cache, &["--frozen", "run", "src/main.m"]);
    assert!(frozen.status.success(), "{frozen:?}");

    fs::write(
        &vendored_helper,
        "function value = helper(); value = 7; end\n",
    )
    .unwrap();
    let tampered = runmat(&project, &cache, &["--frozen", "run", "src/main.m"]);
    assert!(!tampered.status.success(), "{tampered:?}");
    assert!(
        String::from_utf8_lossy(&tampered.stderr)
            .contains("does not match its locked manifest and tree digests"),
        "{tampered:?}"
    );
}

fn runmat(project: &std::path::Path, cache: &std::path::Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_runmat"))
        .current_dir(project)
        .env("RUNMAT_PACKAGE_CACHE_DIR", cache)
        .env_remove("RUNMAT_CONFIG")
        .args(args)
        .output()
        .expect("run package CLI")
}

fn create_source_repository(path: std::path::PathBuf) -> git2::Repository {
    fs::create_dir_all(path.join("src")).unwrap();
    fs::write(
        path.join("src/helper.m"),
        "function value = helper(); value = 42; end\n",
    )
    .unwrap();
    fs::write(
        path.join("runmat.toml"),
        r#"
[package]
name = "helper"
version = "2.0.0"

[sources]
roots = ["src"]
"#,
    )
    .unwrap();
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

use futures::executor::block_on;
use futures::FutureExt;
use runmat_package::{
    build_frozen_project, resolve_project_async, ContentDigest, DependencyGroup,
    GitAcquisitionIntent, GitAcquisitionPlan, GitAcquisitionPolicy, GitCommitId, GitPackageMount,
    GitPackageProvider, GitSourceId, ProjectResolveOptions, SourceId,
};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tempfile::TempDir;

struct FixtureGit {
    root: PathBuf,
    source: GitSourceId,
    plans: Mutex<Vec<GitAcquisitionPlan>>,
}

impl GitPackageProvider for FixtureGit {
    fn acquire<'a>(
        &'a self,
        plan: &'a GitAcquisitionPlan,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<GitPackageMount, String>> + 'a>>
    {
        async move {
            self.plans.lock().unwrap().push(plan.clone());
            let source = plan.expected.clone().unwrap_or_else(|| {
                if plan.subdir == self.source.subdir {
                    self.source.clone()
                } else {
                    GitSourceId {
                        repository: self.source.repository.clone(),
                        commit: self.source.commit.clone(),
                        subdir: plan.subdir.clone(),
                        tree_digest: ContentDigest::sha256(plan.subdir.as_str()),
                    }
                }
            });
            Ok(GitPackageMount {
                source,
                root: self.root.join(plan.subdir.as_str()),
            })
        }
        .boxed_local()
    }
}

fn options() -> ProjectResolveOptions {
    ProjectResolveOptions {
        target: "x86_64-unknown-linux-gnu".to_string(),
        groups: [DependencyGroup::Runtime].into_iter().collect(),
        root_features: BTreeSet::new(),
        host_capabilities: BTreeSet::new(),
        git_intent: GitAcquisitionIntent::Execute,
        git_policy: GitAcquisitionPolicy::default(),
    }
}

fn write_path_fixture(temp: &TempDir) -> PathBuf {
    write_file(temp.path().join("src/main.m"), "result = helper();\n");
    write_file(
        temp.path().join("deps/helper/src/helper.m"),
        "function y = helper(); y = 42; end\n",
    );
    write_file(
        temp.path().join("runmat.toml"),
        r#"
[package]
name = "application"
version = "1.0.0"

[sources]
roots = ["src"]

[dependencies]
helper = { path = "deps/helper", version = "1.0.0" }
"#,
    );
    write_file(
        temp.path().join("deps/helper/runmat.toml"),
        r#"
[package]
name = "helper"
version = "1.0.0"

[sources]
roots = ["src"]
"#,
    );
    temp.path().join("runmat.toml")
}

fn git_fixture(temp: &TempDir) -> (PathBuf, FixtureGit) {
    let mount = temp.path().join("git-mount");
    write_file(temp.path().join("root/src/main.m"), "value = helper();\n");
    write_file(
        temp.path().join("root/runmat.toml"),
        r#"
[package]
name = "application"
version = "1.0.0"

[sources]
roots = ["src"]

[dependencies]
helper = { git = "https://example.com/acme/helper.git", branch = "main", version = "2.0.0" }
"#,
    );
    write_file(
        mount.join("src/helper.m"),
        "function value = helper(); value = 42; end\n",
    );
    write_file(
        mount.join("runmat.toml"),
        r#"
[package]
name = "helper"
version = "2.0.0"

[sources]
roots = ["src"]
"#,
    );
    let source = GitSourceId::new(
        "https://example.com/acme/helper.git",
        "0123456789abcdef0123456789abcdef01234567"
            .parse::<GitCommitId>()
            .unwrap(),
        ".".parse().unwrap(),
        ContentDigest::sha256("git-tree"),
    )
    .unwrap();
    (
        temp.path().join("root/runmat.toml"),
        FixtureGit {
            root: mount,
            source,
            plans: Mutex::new(Vec::new()),
        },
    )
}

#[test]
fn path_dependencies_inside_git_are_exact_subdirectory_git_sources() {
    let temp = TempDir::new().unwrap();
    write_file(temp.path().join("root/src/main.m"), "value = top();\n");
    write_file(
        temp.path().join("root/runmat.toml"),
        r#"
[package]
name = "application"
[sources]
roots = ["src"]
[dependencies]
top = { git = "https://example.com/acme/mono.git", rev = "0123456789abcdef0123456789abcdef01234567", subdir = "packages/top" }
"#,
    );
    let mount = temp.path().join("git-mount");
    write_file(
        mount.join("packages/top/runmat.toml"),
        r#"
[package]
name = "top"
version = "1.0.0"
[sources]
roots = ["src"]
[dependencies]
inner = { path = "deps/inner", version = "1.0.0" }
"#,
    );
    write_file(
        mount.join("packages/top/src/top.m"),
        "function value = top(); value = inner(); end\n",
    );
    write_file(
        mount.join("packages/top/deps/inner/runmat.toml"),
        r#"
[package]
name = "inner"
version = "1.0.0"
[sources]
roots = ["src"]
"#,
    );
    write_file(
        mount.join("packages/top/deps/inner/src/inner.m"),
        "function value = inner(); value = 1; end\n",
    );
    let git = FixtureGit {
        root: mount,
        source: GitSourceId::new(
            "https://example.com/acme/mono.git",
            "0123456789abcdef0123456789abcdef01234567".parse().unwrap(),
            "packages/top".parse().unwrap(),
            ContentDigest::sha256("top"),
        )
        .unwrap(),
        plans: Mutex::new(Vec::new()),
    };
    let resolved = block_on(resolve_project_async(
        &temp.path().join("root/runmat.toml"),
        None,
        options(),
        &git,
    ))
    .unwrap();
    assert_eq!(resolved.frozen.graph.packages.len(), 3);
    let plans = git.plans.lock().unwrap();
    assert_eq!(plans[0].subdir.as_str(), "packages/top");
    assert_eq!(plans[1].subdir.as_str(), "packages/top/deps/inner");
}

#[test]
fn shared_resolver_preserves_existing_path_graph_and_catalog() {
    let temp = TempDir::new().unwrap();
    let manifest = write_path_fixture(&temp);
    let legacy = build_frozen_project(&manifest, BTreeSet::new()).unwrap();
    let git = FixtureGit {
        root: temp.path().to_path_buf(),
        source: GitSourceId::new(
            "https://example.com/unused.git",
            "0123456789abcdef0123456789abcdef01234567".parse().unwrap(),
            ".".parse().unwrap(),
            ContentDigest::sha256("unused"),
        )
        .unwrap(),
        plans: Mutex::new(Vec::new()),
    };
    let resolved = block_on(resolve_project_async(&manifest, None, options(), &git)).unwrap();
    assert_eq!(resolved.frozen.graph, legacy.graph);
    assert_eq!(resolved.frozen.sources, legacy.sources);
    assert!(git.plans.lock().unwrap().is_empty());
}

#[test]
fn git_resolution_locks_exact_identity_and_frozen_replay_uses_it_offline() {
    let temp = TempDir::new().unwrap();
    let (manifest, git) = git_fixture(&temp);
    let first = block_on(resolve_project_async(&manifest, None, options(), &git)).unwrap();
    assert_eq!(first.frozen.graph.packages.len(), 2);
    assert_eq!(first.acquired_git_sources, vec![git.source.clone()]);
    assert_eq!(
        first.lock.packages[0].instance.source,
        SourceId::Git(git.source.clone())
    );
    assert!(git.plans.lock().unwrap()[0].allow_network);

    git.plans.lock().unwrap().clear();
    let mut frozen = options();
    frozen.git_policy.frozen = true;
    let replay = block_on(resolve_project_async(
        &manifest,
        Some(&first.lock),
        frozen,
        &git,
    ))
    .unwrap();
    assert_eq!(replay.lock, first.lock);
    let plan = &git.plans.lock().unwrap()[0];
    assert!(!plan.allow_network);
    assert_eq!(plan.expected.as_ref(), Some(&git.source));
}

#[test]
fn shared_package_features_reach_a_fixed_point_across_diamond_edges() {
    let temp = TempDir::new().unwrap();
    write_file(
        temp.path().join("src/main.m"),
        "value = left() + right();\n",
    );
    write_file(
        temp.path().join("runmat.toml"),
        r#"
[package]
name = "application"
[sources]
roots = ["src"]
[dependencies]
left = { path = "deps/left" }
right = { path = "deps/right" }
"#,
    );
    for branch in ["left", "right"] {
        let feature = if branch == "left" {
            "with-alpha"
        } else {
            "with-beta"
        };
        write_file(
            temp.path().join(format!("deps/{branch}/src/{branch}.m")),
            &format!("function value = {branch}(); value = shared(); end\n"),
        );
        write_file(
            temp.path().join(format!("deps/{branch}/runmat.toml")),
            &format!(
                r#"
[package]
name = "{branch}"
[sources]
roots = ["src"]
[dependencies]
shared = {{ git = "https://example.com/acme/shared.git", rev = "0123456789abcdef0123456789abcdef01234567", default-features = false, features = ["{feature}"] }}
"#
            ),
        );
    }
    let git_mount = temp.path().join("git-mount");
    write_file(
        git_mount.join("src/shared.m"),
        "function value = shared(); value = 1; end\n",
    );
    write_file(
        git_mount.join("runmat.toml"),
        r#"
[package]
name = "shared"
[sources]
roots = ["src"]
[dependencies]
alpha = { path = "alpha", optional = true }
beta = { path = "beta", optional = true }
[features]
with-alpha = ["alpha"]
with-beta = ["beta"]
"#,
    );
    for leaf in ["alpha", "beta"] {
        write_file(
            git_mount.join(format!("{leaf}/src/{leaf}.m")),
            &format!("function value = {leaf}(); value = 1; end\n"),
        );
        write_file(
            git_mount.join(format!("{leaf}/runmat.toml")),
            &format!(
                r#"
[package]
name = "{leaf}"
[sources]
roots = ["src"]
"#
            ),
        );
    }
    let git = FixtureGit {
        root: git_mount,
        source: GitSourceId::new(
            "https://example.com/acme/shared.git",
            "0123456789abcdef0123456789abcdef01234567".parse().unwrap(),
            ".".parse().unwrap(),
            ContentDigest::sha256("shared"),
        )
        .unwrap(),
        plans: Mutex::new(Vec::new()),
    };

    let resolved = block_on(resolve_project_async(
        &temp.path().join("runmat.toml"),
        None,
        options(),
        &git,
    ))
    .unwrap();

    let names = resolved
        .frozen
        .graph
        .packages
        .values()
        .map(|package| package.local_name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        names,
        BTreeSet::from(["alpha", "application", "beta", "left", "right", "shared"])
    );
    let shared = resolved
        .lock
        .packages
        .iter()
        .find(|package| {
            resolved.frozen.graph.packages[&package.instance.identity_digest].local_name == "shared"
        })
        .unwrap();
    assert_eq!(
        shared.features,
        BTreeSet::from(["with-alpha".to_string(), "with-beta".to_string()])
    );
}

fn write_file(path: impl AsRef<Path>, contents: &str) {
    let path = path.as_ref();
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

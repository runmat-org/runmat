use runmat_config::project::{parse_project_manifest_toml, ProjectManifest};
use runmat_package::{
    DependencyGroup, DependencyLocator, GitSelector, HostCapability, PackageManifest,
    TargetPredicate,
};

fn parse(input: &str) -> ProjectManifest {
    parse_project_manifest_toml(input).expect("manifest should parse")
}

#[test]
fn converts_the_complete_portable_manifest_schema() {
    let config = parse(
        r#"
[package]
name = "application"
organization = "acme"
registry = "internal"
version = "1.2.3"

[sources]
roots = ["src"]

[dependencies]
local = { path = "deps/local", features = ["fast"] }
published = { package = "runmat/matrix", registry = "default", version = "^2.1", optional = true, default-features = false }
git-lib = { git = "https://example.com/acme/git-lib.git", rev = "0123456789abcdef0123456789abcdef01234567", subdir = "packages/core" }
server-lib = { project = "project_123", service = "https://projects.runmat.com", snapshot = "snapshot_456" }

[dev-dependencies]
dev-helper = { path = "tools/dev" }

[test-dependencies]
test-helper = { package = "default:runmat/test-helper", version = "=1.0.0" }

[features]
default = ["published"]
accelerated = ["local/fast"]

[capabilities]
required = ["network", "worker"]
optional = ["webgpu"]

[target.'wasm32-unknown-unknown'.dependencies]
web-helper = { package = "runmat/web-helper", version = "3" }

[target.'capability:shared-memory'.test-dependencies]
shared-tests = { path = "tests/shared" }

[registries.default]
index = "https://packages.runmat.com/index"

[registries.internal]
index = "https://packages.acme.test/index"

[source-replacements.default]
replace-with = "internal"

[publish]
registry = "internal"
include = ["src/**", "README.md"]
exclude = ["src/generated/**"]
license = "Apache-2.0"
readme = "README.md"
"#,
    );

    let manifest = PackageManifest::try_from(&config).expect("domain conversion should succeed");
    assert_eq!(
        manifest.canonical_id.unwrap().to_string(),
        "internal:acme/application"
    );
    assert_eq!(manifest.version.unwrap().to_string(), "1.2.3");
    assert_eq!(manifest.dependencies.len(), 8);
    assert_eq!(
        manifest
            .dependencies
            .iter()
            .filter(|dependency| dependency.group == DependencyGroup::Runtime)
            .count(),
        5
    );
    assert!(manifest.dependencies.iter().any(|dependency| {
        matches!(
            &dependency.locator,
            DependencyLocator::Git {
                selector: GitSelector::Rev { value },
                ..
            } if value == "0123456789abcdef0123456789abcdef01234567"
        )
    }));
    assert!(manifest.dependencies.iter().any(|dependency| {
        dependency.target
            == Some(TargetPredicate::Triple(
                "wasm32-unknown-unknown".to_string(),
            ))
    }));
    assert!(manifest
        .required_capabilities
        .contains(&HostCapability::Worker));
    assert_eq!(manifest.registries.len(), 2);
    assert_eq!(manifest.source_replacements.len(), 1);
    assert_eq!(
        manifest
            .publication
            .as_ref()
            .and_then(|publication| publication.registry.as_ref())
            .map(ToString::to_string)
            .as_deref(),
        Some("internal")
    );
}

#[test]
fn rejects_nonexclusive_locators_and_selectors() {
    for dependency in [
        r#"{ path = "dep", package = "runmat/dep", version = "1" }"#,
        r#"{ git = "https://example.com/dep.git", rev = "abc", tag = "v1" }"#,
        r#"{ package = "runmat/dep" }"#,
        r#"{ project = "project_1", service = "http://token@example.com" }"#,
    ] {
        let config = parse(&format!(
            r#"
[package]
name = "application"

[sources]
roots = ["src"]

[dependencies]
dep = {dependency}
"#
        ));
        assert!(
            PackageManifest::try_from(&config).is_err(),
            "dependency should be rejected: {dependency}"
        );
    }
}

#[test]
fn rejects_credentials_in_external_locations() {
    for (table, declaration) in [
        (
            "dependencies",
            r#"dep = { git = "https://token@example.com/dep.git", rev = "0123456789abcdef0123456789abcdef01234567" }"#,
        ),
        (
            "registries.private",
            r#"index = "https://token@example.com/index""#,
        ),
    ] {
        let config = parse(&format!(
            r#"
[package]
name = "application"

[sources]
roots = ["src"]

[{table}]
{declaration}
"#
        ));
        assert!(PackageManifest::try_from(&config).is_err());
    }
}

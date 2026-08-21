use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::manifest::is_relative_without_parent;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectTestConfig {
    pub roots: Vec<PathBuf>,
    pub suites: Vec<String>,
    pub jobs: Option<usize>,
    pub isolation: Option<ProjectTestIsolation>,
    #[serde(rename = "timeout-ms")]
    pub timeout_ms: Option<u64>,
    pub paths: Vec<PathBuf>,
    #[serde(rename = "environment-allowlist")]
    pub environment_allowlist: Vec<String>,
    pub capabilities: Vec<String>,
    pub resources: BTreeMap<String, u32>,
    pub artifacts: ProjectTestArtifacts,
    pub reports: Vec<ProjectTestReport>,
    pub coverage: ProjectTestCoverage,
    pub shard: Option<ProjectTestShard>,
    pub cluster: Option<ProjectTestCluster>,
}

impl ProjectTestConfig {
    pub(crate) fn is_default(&self) -> bool {
        self == &Self::default()
    }

    pub(super) fn validation_messages(&self) -> Vec<String> {
        let mut messages = Vec::new();
        for (field, paths) in [("[test].roots", &self.roots), ("[test].paths", &self.paths)] {
            for path in paths {
                if !is_relative_without_parent(path) {
                    messages.push(format!(
                        "{field} entries must be relative and cannot contain `..`: {}",
                        path.display()
                    ));
                }
            }
        }
        for path in &self.coverage.roots {
            if !is_relative_without_parent(path) {
                messages.push(format!(
                    "[test.coverage].roots entries must be relative and cannot contain `..`: {}",
                    path.display()
                ));
            }
        }
        if self
            .coverage
            .exclude
            .iter()
            .any(|pattern| pattern.trim().is_empty())
        {
            messages.push("[test.coverage].exclude entries must be non-empty".into());
        }
        if self.jobs == Some(0) {
            messages.push("[test].jobs must be greater than zero".into());
        }
        if self.timeout_ms == Some(0) {
            messages.push("[test].timeout-ms must be greater than zero".into());
        }
        if self
            .environment_allowlist
            .iter()
            .any(|name| !valid_environment_name(name))
        {
            messages.push(
                "[test].environment-allowlist entries must be non-empty ASCII environment names"
                    .into(),
            );
        }
        if self.capabilities.iter().any(|name| name.trim().is_empty()) {
            messages.push("[test].capabilities entries must be non-empty".into());
        }
        if self
            .resources
            .iter()
            .any(|(name, amount)| name.trim().is_empty() || *amount == 0)
        {
            messages.push("[test].resources names and quantities must be non-zero".into());
        }
        if let Some(shard) = self.shard {
            if shard.count == 0 || shard.index >= shard.count {
                messages.push("[test.shard] requires count > 0 and index < count".into());
            }
        }
        if self.artifacts.max_runs == Some(0) {
            messages.push("[test.artifacts].max-runs must be greater than zero".into());
        }
        if let Some(cluster) = &self.cluster {
            if cluster.max_workers == Some(0) {
                messages.push("[test.cluster].max-workers must be greater than zero".into());
            }
            for (field, value) in [
                ("profile", cluster.profile.as_deref()),
                ("queue", cluster.queue.as_deref()),
            ] {
                if value.is_some_and(|value| value.trim().is_empty() || value.len() > 256) {
                    messages.push(format!(
                        "[test.cluster].{field} must be a non-empty value of at most 256 bytes"
                    ));
                }
            }
        }
        messages
    }

    pub fn resolved_roots(&self, project_root: &Path) -> Vec<PathBuf> {
        self.roots
            .iter()
            .map(|root| project_root.join(root))
            .collect()
    }
}

fn valid_environment_name(name: &str) -> bool {
    !name.is_empty()
        && name.is_ascii()
        && name
            .bytes()
            .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectTestIsolation {
    Auto,
    Process,
    Worker,
    Session,
    None,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectTestArtifacts {
    #[serde(rename = "keep-successful")]
    pub keep_successful: bool,
    #[serde(rename = "keep-failed")]
    pub keep_failed: bool,
    #[serde(rename = "max-runs")]
    pub max_runs: Option<u32>,
}

impl Default for ProjectTestArtifacts {
    fn default() -> Self {
        Self {
            keep_successful: false,
            keep_failed: true,
            max_runs: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectTestReport {
    Human,
    Json,
    Junit,
    Tap,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectTestCoverage {
    pub enabled: bool,
    pub formats: Vec<ProjectCoverageFormat>,
    pub roots: Vec<PathBuf>,
    pub exclude: Vec<String>,
    #[serde(rename = "include-generated")]
    pub include_generated: bool,
    #[serde(rename = "include-vendor")]
    pub include_vendor: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectCoverageFormat {
    Json,
    Lcov,
    Cobertura,
    Html,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectTestShard {
    pub index: u32,
    pub count: u32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProjectTestCluster {
    pub profile: Option<String>,
    pub queue: Option<String>,
    #[serde(rename = "max-workers")]
    pub max_workers: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_host_independent_test_policy() {
        let config = ProjectTestConfig {
            jobs: Some(0),
            timeout_ms: Some(0),
            shard: Some(ProjectTestShard { index: 2, count: 2 }),
            environment_allowlist: vec!["BAD-NAME".into()],
            ..ProjectTestConfig::default()
        };
        let messages = config.validation_messages();
        assert_eq!(messages.len(), 4, "{messages:#?}");
    }

    #[test]
    fn validates_cluster_selection_and_capacity() {
        let config = ProjectTestConfig {
            cluster: Some(ProjectTestCluster {
                profile: Some(" ".into()),
                queue: Some("q".repeat(257)),
                max_workers: Some(0),
            }),
            ..ProjectTestConfig::default()
        };
        let messages = config.validation_messages();
        assert_eq!(messages.len(), 3, "{messages:#?}");
    }

    #[test]
    fn manifest_round_trips_complete_test_configuration() {
        let manifest = crate::project::parse_project_manifest_toml(
            r#"
[package]
name = "test-project"

[sources]
roots = ["src"]

[test]
roots = ["tests"]
suites = ["unit"]
jobs = 4
isolation = "auto"
timeout-ms = 30000
paths = ["test-support"]
environment-allowlist = ["CI", "RUNMAT_TEST_SEED"]
capabilities = ["gpu"]
reports = ["human", "json", "junit", "tap"]

[test.resources]
gpu = 1

[test.artifacts]
keep-failed = true
max-runs = 20

[test.coverage]
enabled = true
formats = ["json", "lcov", "cobertura", "html"]
roots = ["src"]
exclude = ["vendor/**"]
include-generated = false
include-vendor = false

[test.shard]
index = 1
count = 3

[test.cluster]
profile = "ci"
queue = "tests"
max-workers = 8
"#,
        )
        .unwrap();

        assert_eq!(manifest.test.jobs, Some(4));
        assert_eq!(manifest.test.isolation, Some(ProjectTestIsolation::Auto));
        assert_eq!(manifest.test.reports.len(), 4);
        assert_eq!(manifest.test.resources.get("gpu"), Some(&1));
        assert_eq!(
            manifest.test.coverage.formats,
            vec![
                ProjectCoverageFormat::Json,
                ProjectCoverageFormat::Lcov,
                ProjectCoverageFormat::Cobertura,
                ProjectCoverageFormat::Html,
            ]
        );
        let json = serde_json::to_string(&manifest).unwrap();
        let decoded = crate::project::parse_project_manifest_json(&json).unwrap();
        assert_eq!(decoded, manifest);
    }
}

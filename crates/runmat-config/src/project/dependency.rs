use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ProjectDependency {
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub package: Option<String>,
    #[serde(default)]
    pub registry: Option<String>,
    #[serde(default)]
    pub git: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub service: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub rev: Option<String>,
    #[serde(default)]
    pub tag: Option<String>,
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub snapshot: Option<String>,
    #[serde(default)]
    pub subdir: Option<PathBuf>,
    #[serde(default)]
    pub optional: bool,
    #[serde(default = "default_features_enabled", rename = "default-features")]
    pub default_features: bool,
    #[serde(default)]
    pub features: Vec<String>,
}

const fn default_features_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectDependencyLocator {
    Path,
    Registry,
    Git,
    ServerProject,
}

impl ProjectDependency {
    pub fn locator(&self) -> Result<ProjectDependencyLocator, String> {
        let locator_count = [
            self.path.is_some(),
            self.package.is_some(),
            self.git.is_some(),
            self.project.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count();
        if locator_count != 1 {
            return Err(
                "must select exactly one locator: `path`, registry `package`, `git`, or `project`"
                    .to_string(),
            );
        }
        if self.registry.is_some() && self.package.is_none() {
            return Err("`registry` is valid only with a registry `package` locator".to_string());
        }
        if self.subdir.is_some() && self.git.is_none() {
            return Err("`subdir` is valid only with a `git` locator".to_string());
        }
        let git_selectors = [
            self.rev.is_some(),
            self.tag.is_some(),
            self.branch.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count();
        if self.git.is_some() {
            if git_selectors != 1 {
                return Err(
                    "a `git` locator must select exactly one of `rev`, `tag`, or `branch`"
                        .to_string(),
                );
            }
        } else if git_selectors != 0 {
            return Err("`rev`, `tag`, and `branch` are valid only with `git`".to_string());
        }
        if self.snapshot.is_some() && self.project.is_none() {
            return Err("`snapshot` is valid only with a Server `project` locator".to_string());
        }
        if self.service.is_some() && self.project.is_none() {
            return Err("`service` is valid only with a Server `project` locator".to_string());
        }
        if self.package.is_some() && self.version.as_deref().is_none_or(str::is_empty) {
            return Err("a registry `package` locator must set a non-empty `version`".to_string());
        }
        if self
            .features
            .iter()
            .any(|feature| feature.trim().is_empty())
        {
            return Err("feature requests must be non-empty".to_string());
        }
        Ok(if self.path.is_some() {
            ProjectDependencyLocator::Path
        } else if self.package.is_some() {
            ProjectDependencyLocator::Registry
        } else if self.git.is_some() {
            ProjectDependencyLocator::Git
        } else {
            ProjectDependencyLocator::ServerProject
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ProjectTargetDependencies {
    #[serde(default)]
    pub dependencies: BTreeMap<String, ProjectDependency>,
    #[serde(default, rename = "dev-dependencies")]
    pub dev_dependencies: BTreeMap<String, ProjectDependency>,
    #[serde(default, rename = "test-dependencies")]
    pub test_dependencies: BTreeMap<String, ProjectDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ProjectCapabilities {
    #[serde(default)]
    pub required: Vec<String>,
    #[serde(default)]
    pub optional: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectRegistry {
    pub index: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectSourceReplacement {
    #[serde(rename = "replace-with")]
    pub replace_with: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ProjectPublication {
    #[serde(default)]
    pub registry: Option<String>,
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub exclude: Vec<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub readme: Option<PathBuf>,
}

use super::dependency::{
    ProjectCapabilities, ProjectDependency, ProjectDependencyLocator, ProjectPublication,
    ProjectRegistry, ProjectSourceReplacement, ProjectTargetDependencies,
};
use serde::de::IgnoredAny;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

pub const PROJECT_MANIFEST_FILENAME: &str = "runmat.toml";
pub const PROJECT_MANIFEST_FILENAMES: &[&str] = &["runmat.toml", "runmat.json"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectManifest {
    pub package: ProjectPackage,
    pub sources: ProjectSources,
    pub dependencies: BTreeMap<String, ProjectDependency>,
    pub dev_dependencies: BTreeMap<String, ProjectDependency>,
    pub test_dependencies: BTreeMap<String, ProjectDependency>,
    pub features: BTreeMap<String, Vec<String>>,
    pub capabilities: ProjectCapabilities,
    pub targets: BTreeMap<String, ProjectTargetDependencies>,
    pub registries: BTreeMap<String, ProjectRegistry>,
    pub source_replacements: BTreeMap<String, ProjectSourceReplacement>,
    pub publish: Option<ProjectPublication>,
    pub entrypoints: Vec<ProjectEntrypoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectPackage {
    pub name: String,
    #[serde(default)]
    pub singleton: bool,
    #[serde(default)]
    pub organization: Option<String>,
    #[serde(default)]
    pub registry: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default, rename = "runmat-version")]
    pub runmat_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectSources {
    #[serde(default)]
    pub roots: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectEntrypoint {
    pub name: String,
    #[serde(default)]
    pub path: Option<PathBuf>,
    #[serde(default)]
    pub module: Option<String>,
    #[serde(default)]
    pub function: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProjectManifest {
    package: ProjectPackage,
    sources: ProjectSources,
    #[serde(default)]
    dependencies: BTreeMap<String, ProjectDependency>,
    #[serde(default, rename = "dev-dependencies")]
    dev_dependencies: BTreeMap<String, ProjectDependency>,
    #[serde(default, rename = "test-dependencies")]
    test_dependencies: BTreeMap<String, ProjectDependency>,
    #[serde(default)]
    features: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    capabilities: ProjectCapabilities,
    #[serde(default, rename = "target")]
    targets: BTreeMap<String, ProjectTargetDependencies>,
    #[serde(default)]
    registries: BTreeMap<String, ProjectRegistry>,
    #[serde(default, rename = "source-replacements")]
    source_replacements: BTreeMap<String, ProjectSourceReplacement>,
    #[serde(default)]
    publish: Option<ProjectPublication>,
    #[serde(default)]
    entrypoints: BTreeMap<String, RawProjectEntrypoint>,
    #[serde(default, rename = "runtime")]
    _runtime: Option<IgnoredAny>,
    #[serde(default, rename = "test")]
    _test: Option<IgnoredAny>,
    #[serde(default, rename = "desktop")]
    _desktop: Option<IgnoredAny>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProjectEntrypoint {
    #[serde(default)]
    path: Option<PathBuf>,
    #[serde(default)]
    module: Option<String>,
    #[serde(default)]
    function: Option<String>,
}

impl From<RawProjectManifest> for ProjectManifest {
    fn from(value: RawProjectManifest) -> Self {
        let entrypoints = value
            .entrypoints
            .into_iter()
            .map(|(name, raw)| ProjectEntrypoint {
                name,
                path: raw.path,
                module: raw.module,
                function: raw.function,
            })
            .collect();
        Self {
            package: value.package,
            sources: value.sources,
            dependencies: value.dependencies,
            dev_dependencies: value.dev_dependencies,
            test_dependencies: value.test_dependencies,
            features: value.features,
            capabilities: value.capabilities,
            targets: value.targets,
            registries: value.registries,
            source_replacements: value.source_replacements,
            publish: value.publish,
            entrypoints,
        }
    }
}

impl<'de> Deserialize<'de> for ProjectManifest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        RawProjectManifest::deserialize(deserializer).map(ProjectManifest::from)
    }
}

impl Serialize for ProjectManifest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct CanonicalManifest<'a> {
            package: &'a ProjectPackage,
            sources: &'a ProjectSources,
            dependencies: &'a BTreeMap<String, ProjectDependency>,
            #[serde(rename = "dev-dependencies")]
            dev_dependencies: &'a BTreeMap<String, ProjectDependency>,
            #[serde(rename = "test-dependencies")]
            test_dependencies: &'a BTreeMap<String, ProjectDependency>,
            features: &'a BTreeMap<String, Vec<String>>,
            capabilities: &'a ProjectCapabilities,
            #[serde(rename = "target")]
            targets: &'a BTreeMap<String, ProjectTargetDependencies>,
            registries: &'a BTreeMap<String, ProjectRegistry>,
            #[serde(rename = "source-replacements")]
            source_replacements: &'a BTreeMap<String, ProjectSourceReplacement>,
            #[serde(skip_serializing_if = "Option::is_none")]
            publish: &'a Option<ProjectPublication>,
            entrypoints: BTreeMap<&'a str, CanonicalEntrypoint<'a>>,
        }

        #[derive(Serialize)]
        struct CanonicalEntrypoint<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            path: Option<&'a Path>,
            #[serde(skip_serializing_if = "Option::is_none")]
            module: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            function: Option<&'a str>,
        }

        let entrypoints = self
            .entrypoints
            .iter()
            .map(|entrypoint| {
                (
                    entrypoint.name.as_str(),
                    CanonicalEntrypoint {
                        path: entrypoint.path.as_deref(),
                        module: entrypoint.module.as_deref(),
                        function: entrypoint.function.as_deref(),
                    },
                )
            })
            .collect();
        CanonicalManifest {
            package: &self.package,
            sources: &self.sources,
            dependencies: &self.dependencies,
            dev_dependencies: &self.dev_dependencies,
            test_dependencies: &self.test_dependencies,
            features: &self.features,
            capabilities: &self.capabilities,
            targets: &self.targets,
            registries: &self.registries,
            source_replacements: &self.source_replacements,
            publish: &self.publish,
            entrypoints,
        }
        .serialize(serializer)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectManifestValidationError {
    pub messages: Vec<String>,
}

impl Display for ProjectManifestValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "project manifest validation failed:\n- {}",
            self.messages.join("\n- ")
        )
    }
}

impl std::error::Error for ProjectManifestValidationError {}

#[derive(Debug, Error)]
pub enum ProjectManifestLoadError {
    #[error("failed to read project manifest {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse TOML project manifest {path}: {source}")]
    ParseToml {
        path: PathBuf,
        #[source]
        source: Box<toml::de::Error>,
    },
    #[error("failed to parse JSON project manifest {path}: {source}")]
    ParseJson {
        path: PathBuf,
        #[source]
        source: Box<serde_json::Error>,
    },
    #[error("invalid project manifest {path}: {source}")]
    Validation {
        path: PathBuf,
        #[source]
        source: ProjectManifestValidationError,
    },
}

enum PathRequirement {
    Directory {
        path: PathBuf,
        missing_message: String,
    },
    Entrypoint {
        project_root: PathBuf,
        path: PathBuf,
        missing_message: String,
    },
}

impl ProjectManifest {
    fn validation_plan(&self, project_root: &Path) -> (Vec<String>, Vec<PathRequirement>) {
        let mut messages = Vec::new();
        let mut path_requirements = Vec::new();
        let package_name = self.package.name.trim();
        if package_name.is_empty() {
            messages.push("[package].name is required and must be non-empty".to_string());
        }
        if let Some(requirement) = self.package.runmat_version.as_deref() {
            if let Some(message) = validate_runmat_version_requirement(requirement) {
                messages.push(message);
            }
        }

        if self.sources.roots.is_empty() {
            messages.push("[sources].roots is required and must be non-empty".to_string());
        }
        for root in &self.sources.roots {
            if !is_relative_without_parent(root) {
                messages.push(format!(
                    "source root `{}` must be project-relative without `..` segments",
                    root.display()
                ));
            } else {
                path_requirements.push(PathRequirement::Directory {
                    path: project_root.join(root),
                    missing_message: format!(
                        "source root `{}` does not exist as a directory under project root",
                        root.display()
                    ),
                });
            }
        }

        extend_dependency_validation(
            "dependencies",
            &self.dependencies,
            project_root,
            &mut messages,
            &mut path_requirements,
        );
        extend_dependency_validation(
            "dev-dependencies",
            &self.dev_dependencies,
            project_root,
            &mut messages,
            &mut path_requirements,
        );
        extend_dependency_validation(
            "test-dependencies",
            &self.test_dependencies,
            project_root,
            &mut messages,
            &mut path_requirements,
        );
        for (target, dependencies) in &self.targets {
            if target.trim().is_empty() {
                messages.push("target predicate names must be non-empty".to_string());
            }
            for (group, table) in [
                ("dependencies", &dependencies.dependencies),
                ("dev-dependencies", &dependencies.dev_dependencies),
                ("test-dependencies", &dependencies.test_dependencies),
            ] {
                extend_dependency_validation(
                    &format!("target.{target}.{group}"),
                    table,
                    project_root,
                    &mut messages,
                    &mut path_requirements,
                );
            }
        }

        for (feature, requests) in &self.features {
            if feature.trim().is_empty() {
                messages.push("feature names must be non-empty".to_string());
            }
            if requests.iter().any(|request| request.trim().is_empty()) {
                messages.push(format!("feature `{feature}` contains an empty request"));
            }
        }
        if self
            .capabilities
            .required
            .iter()
            .chain(&self.capabilities.optional)
            .any(|capability| capability.trim().is_empty())
        {
            messages.push("capability names must be non-empty".to_string());
        }
        for (name, registry) in &self.registries {
            if name.trim().is_empty() || registry.index.trim().is_empty() {
                messages.push("registry names and index locations must be non-empty".to_string());
            }
        }
        for (source, replacement) in &self.source_replacements {
            if source.trim().is_empty() || replacement.replace_with.trim().is_empty() {
                messages.push(
                    "source replacement names and `replace-with` targets must be non-empty"
                        .to_string(),
                );
            }
        }

        let mut entrypoint_names = HashSet::new();
        for entrypoint in &self.entrypoints {
            let name = entrypoint.name.trim();
            if name.is_empty() {
                messages.push("entrypoint name must be non-empty".to_string());
                continue;
            }
            if !entrypoint_names.insert(name.to_string()) {
                messages.push(format!("duplicate entrypoint name `{name}`"));
            }

            let has_path = entrypoint.path.is_some();
            let has_module_function = entrypoint
                .module
                .as_ref()
                .is_some_and(|module| !module.trim().is_empty())
                && entrypoint
                    .function
                    .as_ref()
                    .is_some_and(|function| !function.trim().is_empty());
            if has_path == has_module_function {
                messages.push(format!(
                    "entrypoint `{name}` must use exactly one target form: either `path` or (`module` + `function`)"
                ));
                continue;
            }

            if let Some(path) = &entrypoint.path {
                if !is_relative_without_parent(path) {
                    messages.push(format!(
                        "entrypoint `{name}` path `{}` must be project-relative without `..` segments",
                        path.display()
                    ));
                } else {
                    path_requirements.push(PathRequirement::Entrypoint {
                        project_root: project_root.to_path_buf(),
                        path: path.clone(),
                        missing_message: format!(
                            "entrypoint `{name}` path `{}` does not resolve to an existing file (with optional `.m` inference)",
                            path.display()
                        ),
                    });
                }
            } else {
                if entrypoint
                    .module
                    .as_ref()
                    .is_some_and(|module| module.trim().is_empty())
                {
                    messages.push(format!("entrypoint `{name}` has an empty `module`"));
                }
                if entrypoint
                    .function
                    .as_ref()
                    .is_some_and(|function| function.trim().is_empty())
                {
                    messages.push(format!("entrypoint `{name}` has an empty `function`"));
                }
            }
        }
        (messages, path_requirements)
    }

    pub fn validate(&self, project_root: &Path) -> Result<(), ProjectManifestValidationError> {
        let (mut messages, path_requirements) = self.validation_plan(project_root);
        for requirement in path_requirements {
            match requirement {
                PathRequirement::Directory {
                    path,
                    missing_message,
                } if !path.is_dir() => messages.push(missing_message),
                PathRequirement::Entrypoint {
                    project_root,
                    path,
                    missing_message,
                } if resolve_entrypoint_path(&project_root, &path).is_none() => {
                    messages.push(missing_message);
                }
                _ => {}
            }
        }
        validation_result(messages)
    }

    pub async fn validate_async(
        &self,
        project_root: &Path,
    ) -> Result<(), ProjectManifestValidationError> {
        let (mut messages, path_requirements) = self.validation_plan(project_root);
        for requirement in path_requirements {
            match requirement {
                PathRequirement::Directory {
                    path,
                    missing_message,
                } if !path_is_dir_async(&path).await => messages.push(missing_message),
                PathRequirement::Entrypoint {
                    project_root,
                    path,
                    missing_message,
                } if resolve_entrypoint_path_async(&project_root, &path)
                    .await
                    .is_none() =>
                {
                    messages.push(missing_message);
                }
                _ => {}
            }
        }
        validation_result(messages)
    }
}

fn extend_dependency_validation(
    table_name: &str,
    dependencies: &BTreeMap<String, ProjectDependency>,
    project_root: &Path,
    messages: &mut Vec<String>,
    path_requirements: &mut Vec<PathRequirement>,
) {
    for (name, dependency) in dependencies {
        if name.trim().is_empty() {
            messages.push(format!("[{table_name}] dependency names must be non-empty"));
            continue;
        }
        let locator = match dependency.locator() {
            Ok(locator) => locator,
            Err(reason) => {
                messages.push(format!("dependency `{name}` in [{table_name}] {reason}"));
                continue;
            }
        };
        if locator != ProjectDependencyLocator::Path {
            continue;
        }
        let path = dependency
            .path
            .as_ref()
            .expect("path locator has a path after validation");
        if !is_relative_without_parent(path) {
            messages.push(format!(
                "dependency `{name}` path `{}` must be project-relative without `..` segments",
                path.display()
            ));
        } else {
            path_requirements.push(PathRequirement::Directory {
                path: project_root.join(path),
                missing_message: format!(
                    "dependency `{name}` path `{}` does not exist as a directory",
                    path.display()
                ),
            });
        }
    }
}

fn validation_result(messages: Vec<String>) -> Result<(), ProjectManifestValidationError> {
    if messages.is_empty() {
        Ok(())
    } else {
        Err(ProjectManifestValidationError { messages })
    }
}

pub fn parse_project_manifest_toml(input: &str) -> Result<ProjectManifest, toml::de::Error> {
    toml::from_str::<RawProjectManifest>(input).map(ProjectManifest::from)
}

pub fn parse_project_manifest_json(input: &str) -> Result<ProjectManifest, serde_json::Error> {
    serde_json::from_str::<RawProjectManifest>(input).map(ProjectManifest::from)
}

fn parse_project_manifest(
    path: &Path,
    content: &str,
) -> Result<ProjectManifest, ProjectManifestLoadError> {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
    {
        parse_project_manifest_json(content).map_err(|source| ProjectManifestLoadError::ParseJson {
            path: path.to_path_buf(),
            source: Box::new(source),
        })
    } else {
        parse_project_manifest_toml(content).map_err(|source| ProjectManifestLoadError::ParseToml {
            path: path.to_path_buf(),
            source: Box::new(source),
        })
    }
}

pub fn load_project_manifest(path: &Path) -> Result<ProjectManifest, ProjectManifestLoadError> {
    let content = fs::read_to_string(path).map_err(|source| ProjectManifestLoadError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let manifest = parse_project_manifest(path, &content)?;
    let project_root = path.parent().unwrap_or_else(|| Path::new("."));
    manifest
        .validate(project_root)
        .map_err(|source| ProjectManifestLoadError::Validation {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(manifest)
}

pub async fn load_project_manifest_async(
    path: &Path,
) -> Result<ProjectManifest, ProjectManifestLoadError> {
    let content = runmat_filesystem::read_to_string_async(path)
        .await
        .map_err(|source| ProjectManifestLoadError::Read {
            path: path.to_path_buf(),
            source,
        })?;
    let manifest = parse_project_manifest(path, &content)?;
    let project_root = path.parent().unwrap_or_else(|| Path::new("."));
    manifest
        .validate_async(project_root)
        .await
        .map_err(|source| ProjectManifestLoadError::Validation {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(manifest)
}

pub fn discover_project_manifest_from(start: &Path) -> Option<PathBuf> {
    let mut current = if start.is_dir() {
        start.to_path_buf()
    } else {
        start.parent()?.to_path_buf()
    };
    loop {
        for filename in PROJECT_MANIFEST_FILENAMES {
            let candidate = current.join(filename);
            if candidate.is_file() && file_declares_project_manifest(&candidate) {
                return Some(candidate);
            }
        }
        if !current.pop() {
            return None;
        }
    }
}

pub async fn discover_project_manifest_from_async(start: &Path) -> Option<PathBuf> {
    let mut current = if path_is_dir_async(start).await {
        start.to_path_buf()
    } else {
        start.parent()?.to_path_buf()
    };
    loop {
        for filename in PROJECT_MANIFEST_FILENAMES {
            let candidate = current.join(filename);
            if path_is_file_async(&candidate).await
                && file_declares_project_manifest_async(&candidate).await
            {
                return Some(candidate);
            }
        }
        if !current.pop() {
            return None;
        }
    }
}

fn file_declares_project_manifest(path: &Path) -> bool {
    fs::read_to_string(path)
        .map(|content| content_declares_project_manifest(path, &content))
        .unwrap_or(true)
}

async fn file_declares_project_manifest_async(path: &Path) -> bool {
    runmat_filesystem::read_to_string_async(path)
        .await
        .map(|content| content_declares_project_manifest(path, &content))
        .unwrap_or(true)
}

fn content_declares_project_manifest(path: &Path, content: &str) -> bool {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
    {
        return serde_json::from_str::<serde_json::Value>(content)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .map(|object| object.contains_key("package") || object.contains_key("sources"))
            .unwrap_or(true);
    }
    toml::from_str::<toml::Value>(content)
        .ok()
        .and_then(|value| value.as_table().cloned())
        .map(|table| table.contains_key("package") || table.contains_key("sources"))
        .unwrap_or(true)
}

pub(super) fn is_relative_without_parent(path: &Path) -> bool {
    !path.is_absolute()
        && path
            .components()
            .all(|component| !matches!(component, Component::ParentDir | Component::RootDir))
}

pub(super) fn resolve_entrypoint_path(project_root: &Path, path: &Path) -> Option<PathBuf> {
    let direct = project_root.join(path);
    if direct.is_file() {
        return Some(direct);
    }
    if path.extension().is_none() {
        let inferred = direct.with_extension("m");
        if inferred.is_file() {
            return Some(inferred);
        }
    }
    None
}

pub(super) async fn resolve_entrypoint_path_async(
    project_root: &Path,
    path: &Path,
) -> Option<PathBuf> {
    let direct = project_root.join(path);
    if path_is_file_async(&direct).await {
        return Some(direct);
    }
    if path.extension().is_none() {
        let inferred = direct.with_extension("m");
        if path_is_file_async(&inferred).await {
            return Some(inferred);
        }
    }
    None
}

fn validate_runmat_version_requirement(requirement: &str) -> Option<String> {
    let trimmed = requirement.trim();
    if trimmed.is_empty() {
        return Some("[package].runmat-version must be non-empty when set".to_string());
    }
    let target = trimmed.strip_prefix(">=").unwrap_or(trimmed).trim();
    let required = match parse_semver_triplet(target) {
        Ok(version) => version,
        Err(reason) => {
            return Some(format!(
                "[package].runmat-version `{trimmed}` is invalid: {reason}"
            ));
        }
    };
    let current = match parse_semver_triplet(env!("CARGO_PKG_VERSION")) {
        Ok(version) => version,
        Err(_) => return None,
    };
    if current < required {
        return Some(format!(
            "[package].runmat-version requires {trimmed}, but current runtime is {}",
            env!("CARGO_PKG_VERSION")
        ));
    }
    None
}

fn parse_semver_triplet(input: &str) -> Result<(u64, u64, u64), String> {
    let base = input.split(['-', '+']).next().unwrap_or(input);
    let mut parts = base.split('.');
    let major = parts
        .next()
        .ok_or_else(|| "missing major".to_string())?
        .parse::<u64>()
        .map_err(|_| "invalid major".to_string())?;
    let minor = parts
        .next()
        .ok_or_else(|| "missing minor".to_string())?
        .parse::<u64>()
        .map_err(|_| "invalid minor".to_string())?;
    let patch = parts
        .next()
        .ok_or_else(|| "missing patch".to_string())?
        .parse::<u64>()
        .map_err(|_| "invalid patch".to_string())?;
    Ok((major, minor, patch))
}

pub(super) async fn path_is_file_async(path: &Path) -> bool {
    runmat_filesystem::metadata_async(path)
        .await
        .map(|metadata| metadata.is_file())
        .unwrap_or(false)
}

pub(super) async fn path_is_dir_async(path: &Path) -> bool {
    runmat_filesystem::metadata_async(path)
        .await
        .map(|metadata| metadata.is_dir())
        .unwrap_or(false)
}

use super::TargetPredicate;
use crate::{
    CanonicalPackageId, GitRepositoryUrl, IdentityError, ManifestError, NormalizedRelativePath,
    PackageAlias, RegistryId,
};
use semver::VersionReq;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DependencyGroup {
    Runtime,
    Development,
    Test,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum GitSelector {
    Rev { value: String },
    Tag { value: String },
    Branch { value: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum DependencyLocator {
    Path {
        path: NormalizedRelativePath,
    },
    Registry {
        package: CanonicalPackageId,
    },
    Git {
        repository: GitRepositoryUrl,
        selector: GitSelector,
        subdir: NormalizedRelativePath,
    },
    ServerProject {
        project: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        service: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        snapshot: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencySpec {
    pub alias: PackageAlias,
    pub group: DependencyGroup,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<TargetPredicate>,
    pub locator: DependencyLocator,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<VersionReq>,
    pub optional: bool,
    pub default_features: bool,
    pub features: BTreeSet<String>,
}

impl DependencySpec {
    pub(crate) fn from_config(
        table: &str,
        alias: &str,
        dependency: &runmat_config::project::ProjectDependency,
        group: DependencyGroup,
        target: Option<TargetPredicate>,
    ) -> Result<Self, ManifestError> {
        dependency
            .locator()
            .map_err(|reason| invalid_dependency(table, alias, reason))?;
        let raw_alias = alias;
        let alias: PackageAlias = raw_alias
            .parse()
            .map_err(|error: IdentityError| invalid_dependency(table, raw_alias, error))?;
        let version = dependency
            .version
            .as_deref()
            .map(VersionReq::parse)
            .transpose()
            .map_err(|error| invalid_dependency(table, alias.as_str(), error))?;
        let locator = if let Some(path) = &dependency.path {
            DependencyLocator::Path {
                path: NormalizedRelativePath::new(path)
                    .map_err(|error| invalid_dependency(table, alias.as_str(), error))?,
            }
        } else if let Some(package) = &dependency.package {
            DependencyLocator::Registry {
                package: registry_package_id(package, dependency.registry.as_deref())
                    .map_err(|error| invalid_dependency(table, alias.as_str(), error))?,
            }
        } else if let Some(repository) = &dependency.git {
            DependencyLocator::Git {
                repository: GitRepositoryUrl::new(repository)
                    .map_err(|error| invalid_dependency(table, alias.as_str(), error))?,
                selector: git_selector(dependency)
                    .map_err(|reason| invalid_dependency(table, alias.as_str(), reason))?,
                subdir: dependency
                    .subdir
                    .as_deref()
                    .map(NormalizedRelativePath::new)
                    .transpose()
                    .map_err(|error| invalid_dependency(table, alias.as_str(), error))?
                    .unwrap_or_else(|| ".".parse().expect("dot is a valid relative path")),
            }
        } else {
            let project = dependency
                .project
                .as_deref()
                .expect("validated project locator")
                .trim();
            if project.is_empty() {
                return Err(invalid_dependency(
                    table,
                    alias.as_str(),
                    "Server project ID must be non-empty",
                ));
            }
            if let Some(service) = dependency.service.as_deref() {
                validate_service_url(service)
                    .map_err(|reason| invalid_dependency(table, alias.as_str(), reason))?;
            }
            DependencyLocator::ServerProject {
                project: project.to_string(),
                service: dependency.service.clone(),
                snapshot: dependency.snapshot.clone(),
            }
        };
        let features = dependency
            .features
            .iter()
            .map(|feature| feature.trim().to_string())
            .collect();
        Ok(Self {
            alias,
            group,
            target,
            locator,
            version,
            optional: dependency.optional,
            default_features: dependency.default_features,
            features,
        })
    }
}

fn registry_package_id(
    package: &str,
    registry: Option<&str>,
) -> Result<CanonicalPackageId, IdentityError> {
    if package.contains(':') {
        let canonical: CanonicalPackageId = package.parse()?;
        if registry.is_some_and(|registry| registry != canonical.registry().as_str()) {
            return Err(IdentityError::InvalidName {
                kind: "registry package locator",
                value: package.to_string(),
                reason: "embedded and explicit registry names disagree",
            });
        }
        return Ok(canonical);
    }
    let Some((organization, name)) = package.split_once('/') else {
        return Err(IdentityError::InvalidName {
            kind: "registry package locator",
            value: package.to_string(),
            reason: "must use `organization/name`",
        });
    };
    CanonicalPackageId::new(
        registry.unwrap_or("default").parse::<RegistryId>()?,
        organization,
        name,
    )
}

fn git_selector(
    dependency: &runmat_config::project::ProjectDependency,
) -> Result<GitSelector, &'static str> {
    if let Some(value) = dependency.rev.as_deref() {
        nonempty_selector(value).map(|value| GitSelector::Rev { value })
    } else if let Some(value) = dependency.tag.as_deref() {
        nonempty_selector(value).map(|value| GitSelector::Tag { value })
    } else if let Some(value) = dependency.branch.as_deref() {
        nonempty_selector(value).map(|value| GitSelector::Branch { value })
    } else {
        Err("Git selector is missing")
    }
}

fn nonempty_selector(value: &str) -> Result<String, &'static str> {
    let value = value.trim();
    if value.is_empty() {
        Err("Git selectors must be non-empty")
    } else {
        Ok(value.to_string())
    }
}

fn validate_service_url(value: &str) -> Result<(), &'static str> {
    let url = url::Url::parse(value).map_err(|_| "Server service URL must be an absolute URL")?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(
            "Server service URL must use HTTPS and cannot contain credentials, query, or fragment",
        );
    }
    Ok(())
}

fn invalid_dependency(table: &str, alias: &str, reason: impl ToString) -> ManifestError {
    ManifestError::InvalidDependency {
        table: table.to_string(),
        alias: alias.to_string(),
        reason: reason.to_string(),
    }
}

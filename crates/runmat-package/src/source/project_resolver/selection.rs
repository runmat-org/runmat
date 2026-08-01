use super::loader::LoadedPackage;
use super::ProjectResolveError;
use crate::{
    DependencySpec, GitSourceId, NormalizedRelativePath, PackageInstanceId, PackageLock,
    PackageManifest, SourceId,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn join_relative(
    parent: &NormalizedRelativePath,
    child: &NormalizedRelativePath,
) -> Result<NormalizedRelativePath, ProjectResolveError> {
    let joined = match (parent.as_str(), child.as_str()) {
        (".", child) => child.to_string(),
        (parent, ".") => parent.to_string(),
        (parent, child) => format!("{parent}/{child}"),
    };
    NormalizedRelativePath::new(joined)
        .map_err(|error| ProjectResolveError::Invalid(error.to_string()))
}

pub(super) fn feature_activation(
    manifest: &PackageManifest,
    enabled: &BTreeSet<String>,
) -> Result<BTreeMap<String, BTreeSet<String>>, ProjectResolveError> {
    let aliases = manifest
        .dependencies
        .iter()
        .map(|dependency| dependency.alias.as_str())
        .collect::<BTreeSet<_>>();
    let mut activation = BTreeMap::<String, BTreeSet<String>>::new();
    for feature in enabled {
        let Some(requests) = manifest.features.get(feature) else {
            if feature == "default" {
                continue;
            }
            return Err(ProjectResolveError::Invalid(format!(
                "package `{}` does not declare requested feature `{feature}`",
                manifest.local_name
            )));
        };
        for request in requests {
            let (alias, child_feature) = request
                .split_once('/')
                .map_or((request.as_str(), None), |(alias, feature)| {
                    (alias, Some(feature))
                });
            if !aliases.contains(alias) {
                return Err(ProjectResolveError::Invalid(format!(
                    "feature `{feature}` in package `{}` activates unknown dependency `{alias}`",
                    manifest.local_name
                )));
            }
            if let Some(child_feature) = child_feature {
                activation
                    .entry(alias.to_string())
                    .or_default()
                    .insert(child_feature.to_string());
            } else {
                activation.entry(alias.to_string()).or_default();
            }
        }
    }
    Ok(activation)
}

pub(super) fn locked_git_source<'a>(
    lock: Option<&'a PackageLock>,
    from_root: bool,
    from: &PackageInstanceId,
    dependency: &DependencySpec,
) -> Option<&'a GitSourceId> {
    let lock = lock?;
    let edge = lock.edges.iter().find(|edge| {
        edge.from.as_ref()
            == if from_root {
                None
            } else {
                Some(&from.identity_digest)
            }
            && edge.alias == dependency.alias
            && edge.group == dependency.group
            && edge.target == dependency.target
    })?;
    lock.packages
        .iter()
        .find(|package| package.instance.identity_digest == edge.to)
        .and_then(|package| match &package.instance.source {
            SourceId::Git(source) => Some(source),
            _ => None,
        })
}

pub(super) fn validate_version(
    package: &str,
    dependency: &DependencySpec,
    target: &LoadedPackage,
) -> Result<(), ProjectResolveError> {
    let Some(requirement) = &dependency.version else {
        return Ok(());
    };
    let Some(version) = &target.domain.version else {
        return Err(ProjectResolveError::Version {
            package: package.to_string(),
            dependency: dependency.alias.to_string(),
            requirement: requirement.to_string(),
            target: target.domain.local_name.clone(),
            actual: "does not declare a version".to_string(),
        });
    };
    if !requirement.matches(version.as_semver()) {
        return Err(ProjectResolveError::Version {
            package: package.to_string(),
            dependency: dependency.alias.to_string(),
            requirement: requirement.to_string(),
            target: target.domain.local_name.clone(),
            actual: format!("declares {version}"),
        });
    }
    Ok(())
}

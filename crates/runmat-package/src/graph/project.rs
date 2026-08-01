use super::{build_path_graph, PackageGraph, PathGraphInput, PathPackageInput};
use crate::{
    CanonicalPackageId, ContentDigest, GraphError, HostCapability, NormalizedRelativePath,
    PackageAlias, PackageManifest, RegistryId,
};
use runmat_config::project::{ProjectCompositionGraph, ProjectCompositionPackage};
use std::collections::BTreeMap;
use std::path::Path;

pub fn build_project_path_graph(
    composition: &ProjectCompositionGraph,
    workspace_root: &Path,
    host_capabilities: std::collections::BTreeSet<HostCapability>,
) -> Result<PackageGraph, GraphError> {
    validate_path_versions(composition)?;
    let workspace_root =
        std::fs::canonicalize(workspace_root).unwrap_or_else(|_| workspace_root.to_path_buf());
    let mut packages = BTreeMap::new();
    for (key, package) in &composition.packages {
        packages.insert(
            key.clone(),
            project_package_input(package, &workspace_root, |path| {
                std::fs::read(path).map_err(|error| {
                    GraphError::Invalid(format!(
                        "failed to read package source {}: {error}",
                        path.display()
                    ))
                })
            })?,
        );
    }
    build_path_graph(PathGraphInput {
        root: composition.root_package.clone(),
        packages,
        host_capabilities,
    })
}

pub async fn build_project_path_graph_async(
    composition: &ProjectCompositionGraph,
    workspace_root: &Path,
    host_capabilities: std::collections::BTreeSet<HostCapability>,
) -> Result<PackageGraph, GraphError> {
    validate_path_versions(composition)?;
    let workspace_root = runmat_filesystem::canonicalize_async(workspace_root)
        .await
        .unwrap_or_else(|_| workspace_root.to_path_buf());
    let mut packages = BTreeMap::new();
    for (key, package) in &composition.packages {
        let domain = package_domain(package)?;
        let (workspace_path, manifest_digest, dependencies) =
            project_metadata(package, &workspace_root)?;
        let mut tree_input = Vec::new();
        for source in &package.source_index.files {
            let path = package
                .project_root
                .join(&source.source_root)
                .join(&source.relative_path);
            let bytes = runmat_filesystem::read_async(&path)
                .await
                .map_err(|error| {
                    GraphError::Invalid(format!(
                        "failed to read package source {}: {error}",
                        path.display()
                    ))
                })?;
            append_tree_entry(&mut tree_input, source, &bytes)?;
        }
        packages.insert(
            key.clone(),
            PathPackageInput {
                package: canonical_package(package)?,
                local_name: package.package_name.clone(),
                workspace_path,
                manifest_digest,
                tree_digest: ContentDigest::sha256(tree_input),
                version: domain.version,
                dependencies,
                required_capabilities: domain.required_capabilities,
                singleton: domain.singleton,
            },
        );
    }
    build_path_graph(PathGraphInput {
        root: composition.root_package.clone(),
        packages,
        host_capabilities,
    })
}

fn project_package_input(
    package: &ProjectCompositionPackage,
    workspace_root: &Path,
    mut read: impl FnMut(&Path) -> Result<Vec<u8>, GraphError>,
) -> Result<PathPackageInput, GraphError> {
    let domain = package_domain(package)?;
    let (workspace_path, manifest_digest, dependencies) =
        project_metadata(package, workspace_root)?;
    let mut tree_input = Vec::new();
    for source in &package.source_index.files {
        let path = package
            .project_root
            .join(&source.source_root)
            .join(&source.relative_path);
        append_tree_entry(&mut tree_input, source, &read(&path)?)?;
    }
    Ok(PathPackageInput {
        package: canonical_package(package)?,
        local_name: package.package_name.clone(),
        workspace_path,
        manifest_digest,
        tree_digest: ContentDigest::sha256(tree_input),
        version: domain.version,
        dependencies,
        required_capabilities: domain.required_capabilities,
        singleton: domain.singleton,
    })
}

fn project_metadata(
    package: &ProjectCompositionPackage,
    workspace_root: &Path,
) -> Result<
    (
        NormalizedRelativePath,
        ContentDigest,
        BTreeMap<PackageAlias, String>,
    ),
    GraphError,
> {
    let relative = package
        .project_root
        .strip_prefix(workspace_root)
        .map_err(|_| {
            GraphError::Invalid(format!(
                "package root {} is outside workspace {}",
                package.project_root.display(),
                workspace_root.display()
            ))
        })?;
    let workspace_path = NormalizedRelativePath::new(relative)
        .map_err(|error| GraphError::Invalid(error.to_string()))?;
    let canonical_manifest = toml::to_string(&package.manifest)
        .map_err(|error| GraphError::Invalid(format!("cannot encode package manifest: {error}")))?;
    let dependencies = package
        .dependencies
        .iter()
        .map(|(alias, target)| {
            alias
                .parse()
                .map(|alias| (alias, target.clone()))
                .map_err(|error| GraphError::Invalid(format!("invalid dependency alias: {error}")))
        })
        .collect::<Result<_, _>>()?;
    Ok((
        workspace_path,
        ContentDigest::sha256(canonical_manifest),
        dependencies,
    ))
}

fn package_domain(package: &ProjectCompositionPackage) -> Result<PackageManifest, GraphError> {
    PackageManifest::try_from(&package.manifest)
        .map_err(|error| GraphError::Invalid(error.to_string()))
}

fn canonical_package(
    package: &ProjectCompositionPackage,
) -> Result<CanonicalPackageId, GraphError> {
    if let Some(organization) = package.manifest.package.organization.as_deref() {
        return CanonicalPackageId::new(
            package
                .manifest
                .package
                .registry
                .as_deref()
                .unwrap_or("default")
                .parse::<RegistryId>()
                .map_err(|error| GraphError::Invalid(error.to_string()))?,
            organization,
            &package.package_name,
        )
        .map_err(|error| GraphError::Invalid(error.to_string()));
    }
    CanonicalPackageId::new("workspace".parse().unwrap(), "local", &package.package_name)
        .map_err(|error| GraphError::Invalid(error.to_string()))
}

fn append_tree_entry(
    output: &mut Vec<u8>,
    source: &runmat_config::project::ProjectSourceFile,
    bytes: &[u8],
) -> Result<(), GraphError> {
    let path = source.source_root.join(&source.relative_path);
    let path = NormalizedRelativePath::new(path)
        .map_err(|error| GraphError::Invalid(error.to_string()))?;
    output.extend_from_slice(path.as_str().as_bytes());
    output.push(0);
    output.extend_from_slice(bytes.len().to_string().as_bytes());
    output.push(0);
    output.extend_from_slice(bytes);
    output.push(0);
    Ok(())
}

fn validate_path_versions(composition: &ProjectCompositionGraph) -> Result<(), GraphError> {
    for package in composition.packages.values() {
        for (alias, target_key) in &package.dependencies {
            let dependency = package.manifest.dependencies.get(alias).ok_or_else(|| {
                GraphError::Invalid(format!(
                    "composition edge `{alias}` is absent from package `{}` manifest",
                    package.package_name
                ))
            })?;
            let Some(requirement) = dependency.version.as_deref() else {
                continue;
            };
            let requirement = semver::VersionReq::parse(requirement).map_err(|error| {
                GraphError::Invalid(format!(
                    "dependency `{alias}` of `{}` has invalid version requirement `{requirement}`: {error}",
                    package.package_name
                ))
            })?;
            let target = composition.packages.get(target_key).ok_or_else(|| {
                GraphError::Invalid(format!(
                    "composition edge `{alias}` references missing package `{target_key}`"
                ))
            })?;
            let version = target
                .manifest
                .package
                .version
                .as_deref()
                .ok_or_else(|| {
                    GraphError::Invalid(format!(
                        "path dependency `{alias}` requires {requirement}, but package `{target_key}` has no version"
                    ))
                })?
                .parse::<semver::Version>()
                .map_err(|error| {
                    GraphError::Invalid(format!(
                        "path dependency package `{target_key}` has invalid version: {error}"
                    ))
                })?;
            if !requirement.matches(&version) {
                return Err(GraphError::Invalid(format!(
                    "path dependency `{alias}` of `{}` requires {requirement}, but `{target_key}` is {version}",
                    package.package_name
                )));
            }
        }
    }
    Ok(())
}

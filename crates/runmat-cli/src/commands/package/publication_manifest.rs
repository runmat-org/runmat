use super::publication_artifact::collect_entries;
use crate::cli::PackageInspectArgs;
use anyhow::{bail, Context, Result};
use runmat_package::{DependencyGroup, DependencyLocator, PackageManifest};
use runmat_package_cache::{
    PublicationPolicy, ReleaseArtifactBuilder, ReleaseArtifactBundle, ReleaseManifest,
};
use runmat_server_client::packages::{PublicationDependencyRequest, PublicationMetadataRequest};
use std::path::PathBuf;

pub(super) struct PreparedPublication {
    pub(super) root: PathBuf,
    pub(super) manifest: PackageManifest,
    pub(super) bundle: ReleaseArtifactBundle,
    pub(super) release_manifest: ReleaseManifest,
}

impl PreparedPublication {
    pub(super) fn build(args: &PackageInspectArgs) -> Result<Self> {
        let manifest_path = std::fs::canonicalize(&args.manifest_path).with_context(|| {
            format!(
                "failed to locate project manifest {}",
                args.manifest_path.display()
            )
        })?;
        let root = manifest_path
            .parent()
            .context("project manifest has no parent directory")?
            .to_path_buf();
        let config = runmat_config::project::load_project_manifest(&manifest_path)
            .context("failed to load project manifest")?;
        let manifest =
            PackageManifest::try_from(&config).context("project manifest is not publishable")?;
        let package = manifest
            .canonical_id
            .clone()
            .context("publishing requires package.organization")?;
        let version = manifest
            .version
            .clone()
            .context("publishing requires package.version")?;
        let declaration = manifest.publication.as_ref();
        let include = declaration
            .map(|value| value.include.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let exclude = declaration
            .map(|value| value.exclude.iter().cloned().collect::<Vec<_>>())
            .unwrap_or_default();
        let policy = PublicationPolicy::new(&include, &exclude, args.allow_native)
            .context("publication policy is invalid")?;
        let bundle = ReleaseArtifactBuilder::build(collect_entries(&root, &policy)?, &policy)
            .context("failed to build deterministic release artifact")?;
        let readme = declaration
            .and_then(|value| value.readme.as_deref())
            .map(str::parse)
            .transpose()
            .context("publication README path is invalid")?;
        let release_manifest = ReleaseManifest::new(
            package,
            version,
            manifest
                .runmat_requirement
                .as_ref()
                .map(ToString::to_string),
            &bundle,
            declaration.and_then(|value| value.license.clone()),
            readme,
        )
        .context("failed to build release manifest")?;
        Ok(Self {
            root,
            manifest,
            bundle,
            release_manifest,
        })
    }
}

pub(super) fn publication_metadata(
    manifest: &PackageManifest,
    bundle: &ReleaseArtifactBundle,
) -> Result<PublicationMetadataRequest> {
    let dependencies = manifest
        .dependencies
        .iter()
        .map(|dependency| {
            let DependencyLocator::Registry { package } = &dependency.locator else {
                bail!(
                    "published dependency `{}` must use a registry locator",
                    dependency.alias
                );
            };
            let requirement = dependency.version.as_ref().with_context(|| {
                format!(
                    "published dependency `{}` requires an explicit version requirement",
                    dependency.alias
                )
            })?;
            Ok(PublicationDependencyRequest {
                alias: dependency.alias.to_string(),
                registry: package.registry().to_string(),
                namespace: package.organization().to_string(),
                name: package.name().to_string(),
                requirement: requirement.to_string(),
                group: match dependency.group {
                    DependencyGroup::Runtime => "runtime",
                    DependencyGroup::Development => "development",
                    DependencyGroup::Test => "test",
                }
                .to_string(),
                target: dependency.target.as_ref().map(ToString::to_string),
                optional: dependency.optional,
                default_features: dependency.default_features,
                features: dependency.features.iter().cloned().collect(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let readme_digest = manifest
        .publication
        .as_ref()
        .and_then(|value| value.readme.as_deref())
        .and_then(|readme| {
            bundle
                .inventory
                .entries
                .iter()
                .find(|entry| entry.path.as_str() == readme)
                .and_then(|entry| entry.digest.as_ref())
                .map(ToString::to_string)
        });
    Ok(PublicationMetadataRequest {
        singleton: manifest.singleton,
        runmat_requirement: manifest
            .runmat_requirement
            .as_ref()
            .map(ToString::to_string),
        dependencies,
        features: manifest
            .features
            .iter()
            .map(|(name, values)| (name.clone(), values.iter().cloned().collect()))
            .collect(),
        required_capabilities: manifest
            .required_capabilities
            .iter()
            .map(ToString::to_string)
            .collect(),
        optional_capabilities: manifest
            .optional_capabilities
            .iter()
            .map(ToString::to_string)
            .collect(),
        readme_digest,
        license: manifest
            .publication
            .as_ref()
            .and_then(|value| value.license.clone()),
    })
}

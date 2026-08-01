use super::{DependencyGroup, DependencySpec, HostCapability, TargetPredicate};
use crate::{CanonicalPackageId, ManifestError, PackageVersion, RegistryId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryDeclaration {
    pub name: RegistryId,
    pub index: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceReplacement {
    pub source: RegistryId,
    pub replace_with: RegistryId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicationDeclaration {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registry: Option<RegistryId>,
    pub include: BTreeSet<String>,
    pub exclude: BTreeSet<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub readme: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackageManifest {
    pub local_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_id: Option<CanonicalPackageId>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<PackageVersion>,
    pub dependencies: Vec<DependencySpec>,
    pub features: BTreeMap<String, BTreeSet<String>>,
    pub required_capabilities: BTreeSet<HostCapability>,
    pub optional_capabilities: BTreeSet<HostCapability>,
    pub registries: Vec<RegistryDeclaration>,
    pub source_replacements: Vec<SourceReplacement>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub publication: Option<PublicationDeclaration>,
}

impl TryFrom<&runmat_config::project::ProjectManifest> for PackageManifest {
    type Error = ManifestError;

    fn try_from(manifest: &runmat_config::project::ProjectManifest) -> Result<Self, Self::Error> {
        let canonical_id = match manifest.package.organization.as_deref() {
            Some(organization) => Some(
                CanonicalPackageId::new(
                    manifest
                        .package
                        .registry
                        .as_deref()
                        .unwrap_or("default")
                        .parse::<RegistryId>()
                        .map_err(|error| ManifestError::InvalidPackage(error.to_string()))?,
                    organization,
                    &manifest.package.name,
                )
                .map_err(|error| ManifestError::InvalidPackage(error.to_string()))?,
            ),
            None if manifest.package.registry.is_some() => {
                return Err(ManifestError::InvalidPackage(
                    "`registry` requires `organization`".to_string(),
                ));
            }
            None => None,
        };
        let version = manifest
            .package
            .version
            .as_deref()
            .map(str::parse::<PackageVersion>)
            .transpose()
            .map_err(|error| ManifestError::InvalidPackage(error.to_string()))?;

        let mut dependencies = Vec::new();
        append_dependencies(
            &mut dependencies,
            "dependencies",
            &manifest.dependencies,
            DependencyGroup::Runtime,
            None,
        )?;
        append_dependencies(
            &mut dependencies,
            "dev-dependencies",
            &manifest.dev_dependencies,
            DependencyGroup::Development,
            None,
        )?;
        append_dependencies(
            &mut dependencies,
            "test-dependencies",
            &manifest.test_dependencies,
            DependencyGroup::Test,
            None,
        )?;
        for (target, groups) in &manifest.targets {
            let predicate: TargetPredicate = target.parse()?;
            append_dependencies(
                &mut dependencies,
                &format!("target.{target}.dependencies"),
                &groups.dependencies,
                DependencyGroup::Runtime,
                Some(predicate.clone()),
            )?;
            append_dependencies(
                &mut dependencies,
                &format!("target.{target}.dev-dependencies"),
                &groups.dev_dependencies,
                DependencyGroup::Development,
                Some(predicate.clone()),
            )?;
            append_dependencies(
                &mut dependencies,
                &format!("target.{target}.test-dependencies"),
                &groups.test_dependencies,
                DependencyGroup::Test,
                Some(predicate),
            )?;
        }

        let features = manifest
            .features
            .iter()
            .map(|(feature, requests)| {
                if feature.trim().is_empty() {
                    return Err(ManifestError::InvalidFeature {
                        feature: feature.clone(),
                        reason: "name must be non-empty".to_string(),
                    });
                }
                Ok((
                    feature.clone(),
                    requests.iter().map(|request| request.to_string()).collect(),
                ))
            })
            .collect::<Result<_, _>>()?;
        let required_capabilities = parse_capabilities(&manifest.capabilities.required)?;
        let optional_capabilities = parse_capabilities(&manifest.capabilities.optional)?;
        let registries = manifest
            .registries
            .iter()
            .map(|(name, registry)| {
                validate_public_url(name, &registry.index)?;
                Ok(RegistryDeclaration {
                    name: name.parse::<RegistryId>().map_err(|error| {
                        ManifestError::InvalidRegistry {
                            registry: name.clone(),
                            reason: error.to_string(),
                        }
                    })?,
                    index: registry.index.clone(),
                })
            })
            .collect::<Result<_, ManifestError>>()?;
        let source_replacements = manifest
            .source_replacements
            .iter()
            .map(|(source, replacement)| {
                Ok(SourceReplacement {
                    source: source.parse::<RegistryId>().map_err(|error| {
                        ManifestError::InvalidSourceReplacement {
                            registry: source.clone(),
                            reason: error.to_string(),
                        }
                    })?,
                    replace_with: replacement.replace_with.parse::<RegistryId>().map_err(
                        |error| ManifestError::InvalidSourceReplacement {
                            registry: source.clone(),
                            reason: error.to_string(),
                        },
                    )?,
                })
            })
            .collect::<Result<_, ManifestError>>()?;
        let publication = manifest
            .publish
            .as_ref()
            .map(|publication| {
                let registry = publication
                    .registry
                    .as_deref()
                    .map(str::parse::<RegistryId>)
                    .transpose()
                    .map_err(|error| ManifestError::InvalidPackage(error.to_string()))?;
                let readme = publication
                    .readme
                    .as_deref()
                    .map(|path| {
                        path.to_str().map(str::to_string).ok_or_else(|| {
                            ManifestError::InvalidPackage(
                                "publication readme path must be valid UTF-8".to_string(),
                            )
                        })
                    })
                    .transpose()?;
                Ok(PublicationDeclaration {
                    registry,
                    include: publication.include.iter().cloned().collect(),
                    exclude: publication.exclude.iter().cloned().collect(),
                    license: publication.license.clone(),
                    readme,
                })
            })
            .transpose()?;
        Ok(Self {
            local_name: manifest.package.name.clone(),
            canonical_id,
            version,
            dependencies,
            features,
            required_capabilities,
            optional_capabilities,
            registries,
            source_replacements,
            publication,
        })
    }
}

fn append_dependencies(
    output: &mut Vec<DependencySpec>,
    table: &str,
    dependencies: &BTreeMap<String, runmat_config::project::ProjectDependency>,
    group: DependencyGroup,
    target: Option<TargetPredicate>,
) -> Result<(), ManifestError> {
    for (alias, dependency) in dependencies {
        output.push(DependencySpec::from_config(
            table,
            alias,
            dependency,
            group,
            target.clone(),
        )?);
    }
    Ok(())
}

fn parse_capabilities(values: &[String]) -> Result<BTreeSet<HostCapability>, ManifestError> {
    values.iter().map(|value| value.parse()).collect()
}

fn validate_public_url(registry: &str, value: &str) -> Result<(), ManifestError> {
    let url = Url::parse(value).map_err(|error| ManifestError::InvalidRegistry {
        registry: registry.to_string(),
        reason: error.to_string(),
    })?;
    if url.scheme() != "https"
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(ManifestError::InvalidRegistry {
            registry: registry.to_string(),
            reason: "index must be a credential-free HTTPS URL without query or fragment"
                .to_string(),
        });
    }
    Ok(())
}

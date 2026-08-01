use crate::{ContentDigest, DependencyGroup, RegistryOrigin, RegistrySourceId};
use semver::VersionReq;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryPackageReference {
    pub registry: RegistryOrigin,
    pub namespace: String,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryReleaseDependency {
    pub alias: String,
    pub package: RegistryPackageReference,
    pub requirement: String,
    pub group: DependencyGroup,
    pub target: Option<String>,
    pub optional: bool,
    pub default_features: bool,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryReleaseMetadata {
    pub singleton: bool,
    pub runmat_requirement: Option<String>,
    pub dependencies: Vec<RegistryReleaseDependency>,
    pub features: BTreeMap<String, Vec<String>>,
    pub required_capabilities: Vec<String>,
    pub optional_capabilities: Vec<String>,
    pub readme_digest: Option<String>,
    pub license: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RegistryCandidateRecord {
    pub source: RegistrySourceId,
    pub metadata: RegistryReleaseMetadata,
    pub yanked: bool,
}

impl RegistryReleaseMetadata {
    pub fn compute_digest(&self, source: &RegistrySourceId) -> Result<ContentDigest, String> {
        self.validate_semantics()?;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct CanonicalPackage<'a> {
            registry: &'a str,
            namespace: &'a str,
            name: &'a str,
        }
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Canonical<'a> {
            format: &'static str,
            package: CanonicalPackage<'a>,
            version: String,
            artifact_digest: String,
            tree_digest: String,
            metadata: &'a RegistryReleaseMetadata,
        }
        let bytes = serde_json::to_vec(&Canonical {
            format: "runmat-registry-release-v1",
            package: CanonicalPackage {
                registry: source.registry_origin.as_str(),
                namespace: source.package.organization(),
                name: source.package.name(),
            },
            version: source.version.to_string(),
            artifact_digest: source.artifact_digest.to_string(),
            tree_digest: source.tree_digest.to_string(),
            metadata: self,
        })
        .map_err(|error| error.to_string())?;
        Ok(ContentDigest::sha256(bytes))
    }

    pub fn validate_source(&self, source: &RegistrySourceId) -> Result<(), String> {
        source.validate().map_err(|error| error.to_string())?;
        if self.compute_digest(source)? != source.release_digest {
            return Err(
                "registry release metadata digest differs from the exact source".to_string(),
            );
        }
        Ok(())
    }

    fn validate_semantics(&self) -> Result<(), String> {
        if self.dependencies.len() > 512
            || self.features.len() > 512
            || self.required_capabilities.len() > 128
            || self.optional_capabilities.len() > 128
        {
            return Err("registry release metadata exceeds semantic limits".to_string());
        }
        if self
            .runmat_requirement
            .as_deref()
            .map(VersionReq::parse)
            .transpose()
            .is_err()
            || self.dependencies.iter().any(|dependency| {
                dependency.alias.is_empty()
                    || dependency.alias.len() > 64
                    || VersionReq::parse(&dependency.requirement).is_err()
                    || dependency
                        .target
                        .as_ref()
                        .is_some_and(|target| target.is_empty() || target.len() > 256)
                    || dependency.features.len() > 128
            })
            || self.features.iter().any(|(feature, requests)| {
                feature.is_empty()
                    || feature.len() > 64
                    || requests.len() > 512
                    || requests
                        .iter()
                        .any(|request| request.is_empty() || request.len() > 129)
            })
            || self
                .required_capabilities
                .iter()
                .chain(&self.optional_capabilities)
                .any(|capability| capability.is_empty() || capability.len() > 64)
        {
            return Err("registry release metadata contains invalid semantic fields".to_string());
        }
        Ok(())
    }
}

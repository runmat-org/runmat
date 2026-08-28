use super::loader::{LoadedDependency, Loader, PackageOrigin};
use super::source::{find_manifest, is_file};
use super::{PackageSourceProvider, ProjectResolveError};
use crate::{
    CandidateMetadata, CandidateProvider, CandidateQuery, CanonicalPackageId, DependencyLocator,
    DependencySpec, HostCapability, PackageInstanceId, PackageManifest, RegistryCandidateRecord,
    RegistryId, RegistryOrigin, ResolutionRequirement, ResolveError, SourceAcquisitionIntent,
    SourceAcquisitionPolicy, SourceId, SourceSelectionPolicy, TargetEnvironment, TargetPredicate,
};
use runmat_config::project::PROJECT_MANIFEST_FILENAME;
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub(super) struct RegistrySettings {
    indexes: BTreeMap<RegistryId, String>,
    replacements: BTreeMap<RegistryId, RegistryId>,
    origins: BTreeMap<RegistryOrigin, RegistryId>,
}

impl RegistrySettings {
    pub(super) fn new(
        manifest: &PackageManifest,
        default_index: &str,
    ) -> Result<Self, ProjectResolveError> {
        let mut indexes = BTreeMap::from([(RegistryId::default(), default_index.to_string())]);
        for declaration in &manifest.registries {
            indexes.insert(declaration.name.clone(), declaration.index.clone());
        }
        let replacements = manifest
            .source_replacements
            .iter()
            .map(|replacement| (replacement.source.clone(), replacement.replace_with.clone()))
            .collect::<BTreeMap<_, _>>();
        for source in replacements.keys() {
            selected_registry(source, &replacements)?;
        }
        let origins = indexes
            .iter()
            .filter_map(|(name, index)| {
                RegistryOrigin::new(index)
                    .ok()
                    .map(|origin| (origin, name.clone()))
            })
            .collect();
        Ok(Self {
            indexes,
            replacements,
            origins,
        })
    }

    pub(super) fn source(
        &self,
        registry: &RegistryId,
    ) -> Result<(RegistryId, &str), ProjectResolveError> {
        let selected = selected_registry(registry, &self.replacements)?;
        let index = self.indexes.get(&selected).ok_or_else(|| {
            ProjectResolveError::Invalid(format!(
                "registry `{selected}` has no [registries.{selected}] index declaration"
            ))
        })?;
        Ok((selected, index))
    }

    pub(super) fn selection_policy(&self, offline: bool) -> SourceSelectionPolicy {
        SourceSelectionPolicy {
            replacements: self.replacements.clone(),
            offline,
        }
    }

    fn logical_registry(
        &self,
        authority: &RegistryOrigin,
        current: &RegistryCandidateRecord,
    ) -> Result<RegistryId, String> {
        if authority == &current.source.registry_origin {
            return Ok(current.source.package.registry().clone());
        }
        self.origins.get(authority).cloned().ok_or_else(|| {
            format!(
                "registry dependency authority `{authority}` has no matching root registry declaration"
            )
        })
    }
}

pub(super) struct ProjectRegistryCandidateProvider<'a> {
    pub(super) sources: &'a dyn PackageSourceProvider,
    pub(super) settings: &'a RegistrySettings,
    pub(super) policy: SourceAcquisitionPolicy,
}

impl CandidateProvider for ProjectRegistryCandidateProvider<'_> {
    fn candidates<'a>(
        &'a self,
        query: &'a CandidateQuery,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<CandidateMetadata>, ResolveError>> + 'a>> {
        Box::pin(async move {
            let index = self
                .settings
                .indexes
                .get(&query.source_registry)
                .ok_or_else(|| {
                    ResolveError::Provider(format!(
                        "registry `{}` has no index declaration",
                        query.source_registry
                    ))
                })?;
            let plan = crate::plan_registry_candidates(
                query.source_registry.clone(),
                index,
                query.package.clone(),
                self.policy,
            )
            .map_err(|error| ResolveError::Provider(error.to_string()))?;
            if !plan.allow_network {
                return Err(ResolveError::Provider(
                    "registry candidate metadata is unavailable offline without a lock".to_string(),
                ));
            }
            self.sources
                .registry_candidates(&plan)
                .await
                .map_err(ResolveError::Provider)?
                .into_iter()
                .map(|record| record_to_candidate(query, record, self.settings))
                .collect()
        })
    }
}

pub(super) async fn resolve_dependencies(
    loader: &mut Loader<'_>,
    root: &str,
) -> Result<(), ProjectResolveError> {
    if loader.pending_registry.is_empty()
        && !loader.existing_lock.is_some_and(|lock| {
            lock.packages
                .iter()
                .any(|package| matches!(package.instance.source, SourceId::Registry(_)))
        })
    {
        return Ok(());
    }
    let root_manifest = &loader.packages[root].domain;
    let settings = RegistrySettings::new(root_manifest, &loader.options.default_registry_index)?;
    if loader.options.source_intent != SourceAcquisitionIntent::Update {
        if let Some(lock) = loader.existing_lock {
            if lock
                .packages
                .iter()
                .any(|package| matches!(package.instance.source, SourceId::Registry(_)))
            {
                return resolve_locked(loader, root, &settings).await;
            }
        }
    }
    resolve_fresh(loader, root, &settings).await
}

async fn resolve_fresh(
    loader: &mut Loader<'_>,
    root: &str,
    settings: &RegistrySettings,
) -> Result<(), ProjectResolveError> {
    let requirements = loader
        .pending_registry
        .iter()
        .map(|pending| requirement(&pending.dependency, pending.features.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    let roots = requirements
        .iter()
        .map(|requirement| requirement.package.clone())
        .collect::<BTreeSet<_>>();
    let provider = ProjectRegistryCandidateProvider {
        sources: loader.sources,
        settings,
        policy: loader.options.source_policy,
    };
    let candidates = crate::acquire_candidates_with_policy(
        &provider,
        roots,
        &settings.selection_policy(loader.options.source_policy.offline),
    )
    .await
    .map_err(|error| ProjectResolveError::Invalid(error.to_string()))?;
    let request = crate::ResolutionRequest {
        root: loader.packages[root].domain.local_name.clone(),
        requirements,
        groups: loader.options.groups.clone(),
        root_features: loader.options.root_features.clone(),
        environment: TargetEnvironment {
            triple: loader.options.target.clone(),
            capabilities: loader.options.host_capabilities.clone(),
        },
        runmat_version: semver::Version::parse(env!("CARGO_PKG_VERSION"))
            .expect("crate version is semver"),
        offline: loader.options.source_policy.offline,
        locked_instances: loader
            .existing_lock
            .into_iter()
            .flat_map(|lock| lock.packages.iter())
            .map(|package| package.instance.identity_digest.clone())
            .collect(),
        update_packages: None,
    };
    let resolution = crate::resolve(&request, &candidates)
        .map_err(|error| ProjectResolveError::Invalid(error.to_string()))?;
    let selected = resolution
        .packages
        .iter()
        .map(|(identity, package)| (identity.clone(), package.clone()))
        .collect::<Vec<_>>();
    let mut loaded = BTreeMap::new();
    for (identity, package) in &selected {
        let source = match &package.candidate.instance.source {
            SourceId::Registry(source) => source.clone(),
            _ => {
                return Err(ProjectResolveError::Invalid(
                    "registry solver selected a non-registry source".to_string(),
                ));
            }
        };
        let metadata = package.candidate.registry_metadata.clone().ok_or_else(|| {
            ProjectResolveError::Invalid(
                "registry candidate is missing signed release metadata".to_string(),
            )
        })?;
        let (source_registry, index) = settings.source(source.package.registry())?;
        let plan = crate::plan_selected_registry_acquisition(
            source_registry,
            index,
            source.clone(),
            loader.options.source_intent,
            loader.options.source_policy,
        )?;
        let mount = loader
            .sources
            .acquire_registry(&plan)
            .await
            .map_err(|reason| ProjectResolveError::RegistryAcquire {
                package: source.package.to_string(),
                reason,
            })?;
        if mount.source != source {
            return Err(ProjectResolveError::Invalid(
                "registry provider returned a different selected release".to_string(),
            ));
        }
        if mount
            .metadata
            .as_ref()
            .is_some_and(|received| received != &metadata)
        {
            return Err(ProjectResolveError::Invalid(
                "exact registry metadata differs from candidate metadata".to_string(),
            ));
        }
        let manifest_path = registry_manifest(&mount.root, &source.package).await?;
        let key = loader
            .load(
                manifest_path,
                PackageOrigin::Registry(source.clone()),
                package.enabled_features.clone(),
                false,
                &mut Vec::new(),
            )
            .await?;
        validate_manifest(
            &loader.packages[&key].domain,
            &RegistryCandidateRecord {
                source: source.clone(),
                metadata,
                yanked: package.candidate.yanked,
            },
            settings,
        )?;
        loader.acquired_registry_sources.insert(source);
        loaded.insert(identity.clone(), key);
    }
    wire_fresh(loader, &resolution, &loaded)?;
    loader.pending_registry.clear();
    Ok(())
}

async fn resolve_locked(
    loader: &mut Loader<'_>,
    root: &str,
    settings: &RegistrySettings,
) -> Result<(), ProjectResolveError> {
    let lock = loader
        .existing_lock
        .expect("locked registry resolution requires a lock");
    let selected = lock
        .packages
        .iter()
        .filter_map(|package| match &package.instance.source {
            SourceId::Registry(source) => Some((
                package.instance.identity_digest.clone(),
                source.clone(),
                package.features.clone(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut loaded = BTreeMap::new();
    for (identity, source, features) in selected {
        let (source_registry, index) = settings.source(source.package.registry())?;
        let plan = crate::plan_registry_acquisition(
            source_registry,
            index,
            source.package.clone(),
            semver::VersionReq::parse(&format!("={}", source.version))
                .expect("package versions form exact requirements"),
            Some(&source),
            loader.options.source_intent,
            loader.options.source_policy,
        )?;
        let mount = loader
            .sources
            .acquire_registry(&plan)
            .await
            .map_err(|reason| ProjectResolveError::RegistryAcquire {
                package: source.package.to_string(),
                reason,
            })?;
        let manifest_path = registry_manifest(&mount.root, &source.package).await?;
        let key = loader
            .load(
                manifest_path,
                PackageOrigin::Registry(source.clone()),
                features,
                false,
                &mut Vec::new(),
            )
            .await?;
        validate_locked_manifest(&loader.packages[&key].domain, &source)?;
        loader.acquired_registry_sources.insert(source);
        loaded.insert(identity, key);
    }
    for edge in &lock.edges {
        let Some(target) = loaded.get(&edge.to) else {
            continue;
        };
        let owner = match &edge.from {
            None => root.to_string(),
            Some(identity) => loader
                .packages
                .iter()
                .find(|(_, package)| package.instance.identity_digest == *identity)
                .map(|(key, _)| key.clone())
                .or_else(|| loaded.get(identity).cloned())
                .ok_or_else(|| {
                    ProjectResolveError::Invalid(format!(
                        "locked registry edge owner {identity} is not materialized"
                    ))
                })?,
        };
        let target_instance = loader.packages[target].instance.clone();
        loader
            .packages
            .get_mut(&owner)
            .ok_or_else(|| {
                ProjectResolveError::Invalid("locked registry edge owner is missing".to_string())
            })?
            .dependencies
            .push(LoadedDependency {
                spec: DependencySpec {
                    alias: edge.alias.clone(),
                    group: edge.group,
                    target: edge.target.clone(),
                    locator: DependencyLocator::Registry {
                        package: target_instance.package.clone(),
                    },
                    version: target_instance.version.as_ref().map(|version| {
                        semver::VersionReq::parse(&format!("={version}"))
                            .expect("package versions form exact requirements")
                    }),
                    optional: edge.optional,
                    default_features: false,
                    features: BTreeSet::new(),
                },
                target: target.clone(),
            });
    }
    loader.pending_registry.clear();
    Ok(())
}

fn wire_fresh(
    loader: &mut Loader<'_>,
    resolution: &crate::Resolution,
    loaded: &BTreeMap<crate::ContentDigest, String>,
) -> Result<(), ProjectResolveError> {
    for pending in &loader.pending_registry {
        let target = resolution
            .edges
            .iter()
            .filter(|edge| edge.from.is_none())
            .filter(|edge| {
                edge.alias == pending.dependency.alias && edge.group == pending.dependency.group
            })
            .filter_map(|edge| {
                let selected = &resolution.packages[&edge.to].candidate;
                let version = selected.instance.version.as_ref()?;
                let DependencyLocator::Registry { package } = &pending.dependency.locator else {
                    return None;
                };
                (selected.instance.package == *package
                    && pending
                        .dependency
                        .version
                        .as_ref()
                        .is_some_and(|requirement| requirement.matches(version.as_semver())))
                .then_some(&edge.to)
            })
            .min()
            .and_then(|identity| loaded.get(identity))
            .ok_or_else(|| {
                ProjectResolveError::Invalid(format!(
                    "solver did not return a target for registry dependency `{}`",
                    pending.dependency.alias
                ))
            })?
            .clone();
        loader
            .packages
            .get_mut(&pending.owner)
            .expect("pending registry owner remains loaded")
            .dependencies
            .push(LoadedDependency {
                spec: pending.dependency.clone(),
                target,
            });
    }
    for edge in resolution.edges.iter().filter(|edge| edge.from.is_some()) {
        let from = edge.from.as_ref().expect("filtered");
        let owner = loaded.get(from).ok_or_else(|| {
            ProjectResolveError::Invalid("solver registry edge owner is not loaded".to_string())
        })?;
        let target = loaded.get(&edge.to).ok_or_else(|| {
            ProjectResolveError::Invalid("solver registry edge target is not loaded".to_string())
        })?;
        let requirement = resolution.packages[from]
            .candidate
            .dependencies
            .iter()
            .find(|dependency| {
                dependency.alias == edge.alias
                    && dependency.group == edge.group
                    && dependency.package
                        == resolution.packages[&edge.to].candidate.instance.package
            })
            .ok_or_else(|| {
                ProjectResolveError::Invalid(
                    "solver registry edge has no signed dependency metadata".to_string(),
                )
            })?;
        loader
            .packages
            .get_mut(owner)
            .expect("registry owner remains loaded")
            .dependencies
            .push(LoadedDependency {
                spec: DependencySpec {
                    alias: requirement.alias.clone(),
                    group: requirement.group,
                    target: requirement.target.clone(),
                    locator: DependencyLocator::Registry {
                        package: requirement.package.clone(),
                    },
                    version: Some(requirement.version.clone()),
                    optional: requirement.optional,
                    default_features: requirement.default_features,
                    features: requirement.features.clone(),
                },
                target: target.clone(),
            });
    }
    Ok(())
}

async fn registry_manifest(
    root: &std::path::Path,
    package: &CanonicalPackageId,
) -> Result<std::path::PathBuf, ProjectResolveError> {
    let manifest = find_manifest(root)
        .await
        .unwrap_or_else(|| root.join(PROJECT_MANIFEST_FILENAME));
    if !is_file(&manifest).await {
        return Err(ProjectResolveError::MissingManifest {
            package: package.to_string(),
            dependency: package.to_string(),
            path: manifest,
        });
    }
    Ok(manifest)
}

fn validate_locked_manifest(
    manifest: &PackageManifest,
    source: &crate::RegistrySourceId,
) -> Result<(), ProjectResolveError> {
    if manifest.canonical_id.as_ref() != Some(&source.package)
        || manifest.version.as_ref() != Some(&source.version)
    {
        return Err(ProjectResolveError::Invalid(
            "locked registry artifact identity or version differs from runmat.lock".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn requirement(
    dependency: &DependencySpec,
    features: BTreeSet<String>,
) -> Result<ResolutionRequirement, ProjectResolveError> {
    let DependencyLocator::Registry { package } = &dependency.locator else {
        return Err(ProjectResolveError::Invalid(
            "registry solver received a non-registry dependency".to_string(),
        ));
    };
    Ok(ResolutionRequirement {
        alias: dependency.alias.clone(),
        package: package.clone(),
        version: dependency.version.clone().ok_or_else(|| {
            ProjectResolveError::Invalid(format!(
                "registry dependency `{}` has no version requirement",
                dependency.alias
            ))
        })?,
        group: dependency.group,
        optional: dependency.optional,
        default_features: false,
        features,
        target: dependency.target.clone(),
    })
}

pub(super) fn validate_manifest(
    manifest: &PackageManifest,
    record: &RegistryCandidateRecord,
    settings: &RegistrySettings,
) -> Result<(), ProjectResolveError> {
    if manifest.canonical_id.as_ref() != Some(&record.source.package)
        || manifest.version.as_ref() != Some(&record.source.version)
        || manifest.singleton != record.metadata.singleton
        || manifest
            .runmat_requirement
            .as_ref()
            .map(ToString::to_string)
            != record.metadata.runmat_requirement
    {
        return Err(ProjectResolveError::Invalid(
            "registry artifact package identity, version, singleton, or RunMat requirement differs from signed metadata".to_string(),
        ));
    }
    let manifest_features = manifest
        .features
        .iter()
        .map(|(name, requests)| (name.clone(), requests.clone()))
        .collect::<BTreeMap<_, _>>();
    let signed_features = record
        .metadata
        .features
        .iter()
        .map(|(name, requests)| (name.clone(), requests.iter().cloned().collect()))
        .collect::<BTreeMap<_, BTreeSet<_>>>();
    let required = parse_capabilities(&record.metadata.required_capabilities)?;
    let optional = parse_capabilities(&record.metadata.optional_capabilities)?;
    if manifest_features != signed_features
        || manifest.required_capabilities != required
        || manifest.optional_capabilities != optional
    {
        return Err(ProjectResolveError::Invalid(
            "registry artifact features or capabilities differ from signed metadata".to_string(),
        ));
    }
    let manifest_dependencies = manifest
        .dependencies
        .iter()
        .map(manifest_dependency)
        .collect::<Result<BTreeSet<_>, _>>()?;
    let signed_dependencies = record
        .metadata
        .dependencies
        .iter()
        .map(|dependency| {
            let registry = settings
                .logical_registry(&dependency.package.registry, record)
                .map_err(ProjectResolveError::Invalid)?;
            Ok((
                dependency.alias.clone(),
                CanonicalPackageId::new(
                    registry,
                    &dependency.package.namespace,
                    &dependency.package.name,
                )
                .map_err(|error| ProjectResolveError::Invalid(error.to_string()))?,
                dependency.requirement.clone(),
                dependency.group,
                dependency.target.clone(),
                dependency.optional,
                dependency.default_features,
                dependency.features.iter().cloned().collect(),
            ))
        })
        .collect::<Result<BTreeSet<_>, ProjectResolveError>>()?;
    if manifest_dependencies != signed_dependencies {
        return Err(ProjectResolveError::Invalid(
            "registry artifact dependencies differ from signed metadata".to_string(),
        ));
    }
    Ok(())
}

fn record_to_candidate(
    query: &CandidateQuery,
    record: RegistryCandidateRecord,
    settings: &RegistrySettings,
) -> Result<CandidateMetadata, ResolveError> {
    if record.source.package != query.package {
        return Err(ResolveError::Provider(
            "registry returned a candidate for a different package".to_string(),
        ));
    }
    record
        .metadata
        .validate_source(&record.source)
        .map_err(ResolveError::Provider)?;
    let dependencies = metadata_requirements(&record, settings).map_err(ResolveError::Provider)?;
    let features = record
        .metadata
        .features
        .iter()
        .map(|(name, requests)| (name.clone(), requests.iter().cloned().collect()))
        .collect();
    let required_capabilities = record
        .metadata
        .required_capabilities
        .iter()
        .map(|capability| capability.parse::<HostCapability>())
        .collect::<Result<_, _>>()
        .map_err(|error| ResolveError::Provider(error.to_string()))?;
    let runmat_version = record
        .metadata
        .runmat_requirement
        .as_deref()
        .map(semver::VersionReq::parse)
        .transpose()
        .map_err(|error| ResolveError::Provider(error.to_string()))?;
    let source = record.source.clone();
    let instance = PackageInstanceId::new(
        query.package.clone(),
        SourceId::Registry(source.clone()),
        Some(source.version.clone()),
        source.tree_digest.clone(),
    );
    Ok(CandidateMetadata {
        instance,
        dependencies,
        features,
        required_capabilities,
        runmat_version,
        singleton: record.metadata.singleton,
        yanked: record.yanked,
        available_offline: false,
        target_artifacts: BTreeSet::new(),
        registry_metadata: Some(record.metadata),
    })
}

fn metadata_requirements(
    record: &RegistryCandidateRecord,
    settings: &RegistrySettings,
) -> Result<Vec<ResolutionRequirement>, String> {
    record
        .metadata
        .dependencies
        .iter()
        .map(|dependency| {
            let registry = settings.logical_registry(&dependency.package.registry, record)?;
            Ok(ResolutionRequirement {
                alias: dependency
                    .alias
                    .parse()
                    .map_err(|error: crate::IdentityError| error.to_string())?,
                package: CanonicalPackageId::new(
                    registry,
                    &dependency.package.namespace,
                    &dependency.package.name,
                )
                .map_err(|error| error.to_string())?,
                version: semver::VersionReq::parse(&dependency.requirement)
                    .map_err(|error| error.to_string())?,
                group: dependency.group,
                optional: dependency.optional,
                default_features: dependency.default_features,
                features: dependency.features.iter().cloned().collect(),
                target: dependency
                    .target
                    .as_deref()
                    .map(str::parse::<TargetPredicate>)
                    .transpose()
                    .map_err(|error| error.to_string())?,
            })
        })
        .collect()
}

type ManifestDependencyIdentity = (
    String,
    CanonicalPackageId,
    String,
    crate::DependencyGroup,
    Option<String>,
    bool,
    bool,
    BTreeSet<String>,
);

fn manifest_dependency(
    dependency: &DependencySpec,
) -> Result<ManifestDependencyIdentity, ProjectResolveError> {
    let DependencyLocator::Registry { package } = &dependency.locator else {
        return Err(ProjectResolveError::Invalid(
            "published registry artifacts may contain only registry dependencies".to_string(),
        ));
    };
    Ok((
        dependency.alias.to_string(),
        package.clone(),
        dependency
            .version
            .as_ref()
            .ok_or_else(|| {
                ProjectResolveError::Invalid(
                    "published registry dependency has no version requirement".to_string(),
                )
            })?
            .to_string(),
        dependency.group,
        dependency.target.as_ref().map(ToString::to_string),
        dependency.optional,
        dependency.default_features,
        dependency.features.clone(),
    ))
}

fn parse_capabilities(values: &[String]) -> Result<BTreeSet<HostCapability>, ProjectResolveError> {
    values
        .iter()
        .map(|value| {
            value.parse().map_err(|error: crate::ManifestError| {
                ProjectResolveError::Invalid(error.to_string())
            })
        })
        .collect()
}

fn selected_registry(
    source: &RegistryId,
    replacements: &BTreeMap<RegistryId, RegistryId>,
) -> Result<RegistryId, ProjectResolveError> {
    let mut current = source.clone();
    let mut visited = BTreeSet::new();
    while let Some(replacement) = replacements.get(&current) {
        if !visited.insert(current.clone()) {
            return Err(ProjectResolveError::Invalid(format!(
                "registry source replacement cycle begins at `{source}`"
            )));
        }
        current = replacement.clone();
    }
    Ok(current)
}

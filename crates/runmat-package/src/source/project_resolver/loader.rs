use super::selection::{
    feature_activation, join_relative, locked_dependency, locked_git_source, locked_server_source,
    validate_version,
};
use super::source::{canonical_path, find_manifest, is_file, load_sources, source_identity};
use super::{PackageSourceProvider, ProjectResolveError, ProjectResolveOptions};
use crate::{
    plan_git_acquisition, plan_server_project_acquisition, CanonicalPackageId, DependencyLocator,
    DependencySpec, GitSourceId, PackageInstanceId, PackageLock, PackageManifest, RegistryId,
    ServerProjectSourceId, ServerSnapshotSelector, SourceId, TargetEnvironment,
};
use runmat_config::project::{
    load_project_manifest_async_with_options, ProjectManifestValidationOptions, ProjectSourceFile,
    PROJECT_MANIFEST_FILENAME,
};
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;

pub(super) struct Loader<'a> {
    pub(super) workspace_root: PathBuf,
    pub(super) existing_lock: Option<&'a PackageLock>,
    pub(super) options: &'a ProjectResolveOptions,
    pub(super) sources: &'a dyn PackageSourceProvider,
    pub(super) packages: BTreeMap<String, LoadedPackage>,
    pub(super) acquired_git_sources: BTreeSet<GitSourceId>,
    pub(super) acquired_server_sources: BTreeSet<ServerProjectSourceId>,
    pub(super) acquired_registry_sources: BTreeSet<crate::RegistrySourceId>,
    pub(super) pending_registry: Vec<PendingRegistryDependency>,
    pub(super) vendor: Option<&'a crate::VendorManifest>,
}

#[derive(Clone)]
pub(super) enum PackageOrigin {
    Workspace,
    Git(GitSourceId),
    ServerProject(ServerProjectSourceId),
    Registry(crate::RegistrySourceId),
    Vendor(SourceId),
}

pub(super) struct LoadedPackage {
    pub(super) root: PathBuf,
    pub(super) domain: PackageManifest,
    pub(super) instance: PackageInstanceId,
    pub(super) sources: Vec<LoadedSource>,
    pub(super) enabled_features: BTreeSet<String>,
    pub(super) dependencies: Vec<LoadedDependency>,
    pub(super) inventory: crate::SourceInventory,
}

pub(super) struct LoadedSource {
    pub(super) descriptor: ProjectSourceFile,
    pub(super) bytes: Vec<u8>,
}

pub(super) struct LoadedDependency {
    pub(super) spec: DependencySpec,
    pub(super) target: String,
}

#[derive(Clone)]
pub(super) struct PendingRegistryDependency {
    pub(super) owner: String,
    pub(super) dependency: DependencySpec,
    pub(super) features: BTreeSet<String>,
}

type LoadFuture<'a> = Pin<Box<dyn Future<Output = Result<String, ProjectResolveError>> + 'a>>;

impl Loader<'_> {
    pub(super) fn load<'a>(
        &'a mut self,
        manifest_path: PathBuf,
        origin: PackageOrigin,
        enabled_features: BTreeSet<String>,
        is_root: bool,
        active: &'a mut Vec<(String, String)>,
    ) -> LoadFuture<'a> {
        Box::pin(async move {
            let manifest_path = canonical_path(&manifest_path).await;
            let key = package_key(&manifest_path, &origin);
            if let Some(index) = active.iter().position(|(candidate, _)| candidate == &key) {
                let mut cycle = active[index..]
                    .iter()
                    .map(|(_, name)| name.clone())
                    .collect::<Vec<_>>();
                cycle.push(
                    active
                        .get(index)
                        .map(|(_, name)| name.clone())
                        .unwrap_or_else(|| key.clone()),
                );
                return Err(ProjectResolveError::Cycle {
                    cycle: cycle.join(" -> "),
                });
            }
            if self.packages.contains_key(&key) {
                let changed = {
                    let existing = self.packages.get_mut(&key).expect("package exists");
                    let before = existing.enabled_features.len();
                    existing.enabled_features.extend(enabled_features);
                    before != existing.enabled_features.len()
                };
                if changed {
                    self.expand(key.clone(), origin, is_root, active).await?;
                }
                return Ok(key);
            }

            let config = load_project_manifest_async_with_options(
                &manifest_path,
                ProjectManifestValidationOptions {
                    check_dependency_paths: !self.options.source_policy.frozen,
                },
            )
            .await
            .map_err(|error| ProjectResolveError::Manifest {
                path: manifest_path.clone(),
                reason: error.to_string(),
            })?;
            let domain = PackageManifest::try_from(&config)
                .map_err(|error| ProjectResolveError::Invalid(error.to_string()))?;
            let root = manifest_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            let (sources, source_index) = load_sources(&root, &config).await?;
            let source = source_identity(
                &self.workspace_root,
                &manifest_path,
                &root,
                &config,
                &sources,
                &origin,
            )?;
            let package = domain.canonical_id.clone().unwrap_or_else(|| {
                CanonicalPackageId::new(
                    "workspace".parse::<RegistryId>().expect("static registry"),
                    "local",
                    &domain.local_name,
                )
                .expect("validated manifest package name")
            });
            let tree_digest = match &source {
                SourceId::Path(source) => source.tree_digest.clone(),
                SourceId::Git(source) => source.tree_digest.clone(),
                SourceId::ServerProject(source) => source.tree_digest.clone(),
                SourceId::Registry(source) => source.tree_digest.clone(),
            };
            let inventory =
                crate::SourceInventory::from_project_index(tree_digest.clone(), source_index)
                    .map_err(|error| ProjectResolveError::Invalid(error.to_string()))?;
            let instance =
                PackageInstanceId::new(package, source, domain.version.clone(), tree_digest);
            let active_features = enabled_features;
            self.packages.insert(
                key.clone(),
                LoadedPackage {
                    root: root.clone(),
                    domain: domain.clone(),
                    instance: instance.clone(),
                    sources,
                    enabled_features: active_features.clone(),
                    dependencies: Vec::new(),
                    inventory,
                },
            );

            self.expand(key.clone(), origin, is_root, active).await?;
            Ok(key)
        })
    }

    fn expand<'a>(
        &'a mut self,
        key: String,
        origin: PackageOrigin,
        is_root: bool,
        active: &'a mut Vec<(String, String)>,
    ) -> Pin<Box<dyn Future<Output = Result<(), ProjectResolveError>> + 'a>> {
        Box::pin(async move {
            let (root, domain, instance, active_features) = {
                let package = self.packages.get(&key).expect("loaded package");
                (
                    package.root.clone(),
                    package.domain.clone(),
                    package.instance.clone(),
                    package.enabled_features.clone(),
                )
            };
            active.push((key.clone(), domain.local_name.clone()));
            let activation = feature_activation(&domain, &active_features)?;
            let environment = TargetEnvironment {
                triple: self.options.target.clone(),
                capabilities: self.options.host_capabilities.clone(),
            };
            let dependencies = domain
                .dependencies
                .iter()
                .filter(|dependency| self.options.groups.contains(&dependency.group))
                .filter(|dependency| {
                    dependency
                        .target
                        .as_ref()
                        .is_none_or(|target| environment.supports(target))
                })
                .filter(|dependency| {
                    !dependency.optional || activation.contains_key(dependency.alias.as_str())
                })
                .cloned()
                .collect::<Vec<_>>();
            for dependency in dependencies {
                let mut child_features = dependency.features.clone();
                if dependency.default_features {
                    child_features.insert("default".to_string());
                }
                if let Some(activated) = activation.get(dependency.alias.as_str()) {
                    child_features.extend(activated.iter().cloned());
                }
                if matches!(dependency.locator, DependencyLocator::Registry { .. }) {
                    if !matches!(origin, PackageOrigin::Registry(_)) {
                        self.pending_registry.push(PendingRegistryDependency {
                            owner: key.clone(),
                            dependency,
                            features: child_features,
                        });
                    }
                    continue;
                }
                if matches!(origin, PackageOrigin::Registry(_)) {
                    return Err(ProjectResolveError::Invalid(format!(
                        "published registry package `{}` contains a non-registry dependency `{}`",
                        domain.local_name, dependency.alias
                    )));
                }
                let (child_manifest, child_origin) = match &dependency.locator {
                    DependencyLocator::Path { path } => {
                        if let PackageOrigin::Git(parent) = &origin {
                            let plan = plan_git_acquisition(
                                parent.repository.clone(),
                                crate::GitSelector::Rev {
                                    value: parent.commit.hex.clone(),
                                },
                                join_relative(&parent.subdir, path)?,
                                locked_git_source(
                                    self.existing_lock,
                                    is_root,
                                    &instance,
                                    &dependency,
                                ),
                                self.options.source_intent,
                                self.options.source_policy,
                            )?;
                            let mount =
                                self.sources.acquire_git(&plan).await.map_err(|reason| {
                                    ProjectResolveError::GitAcquire {
                                        package: domain.local_name.clone(),
                                        dependency: dependency.alias.to_string(),
                                        reason,
                                    }
                                })?;
                            crate::validate_git_acquisition(&plan, &mount.source)?;
                            self.acquired_git_sources.insert(mount.source.clone());
                            let child_manifest = find_manifest(&mount.root)
                                .await
                                .unwrap_or_else(|| mount.root.join(PROJECT_MANIFEST_FILENAME));
                            if !is_file(&child_manifest).await {
                                return Err(ProjectResolveError::MissingManifest {
                                    package: domain.local_name.clone(),
                                    dependency: dependency.alias.to_string(),
                                    path: child_manifest,
                                });
                            }
                            (child_manifest, PackageOrigin::Git(mount.source))
                        } else if self.options.source_policy.frozen {
                            let locked = locked_dependency(
                                self.existing_lock,
                                is_root,
                                &instance,
                                &dependency,
                            )
                            .ok_or_else(|| {
                                ProjectResolveError::Invalid(format!(
                                    "frozen path dependency `{}` in package `{}` requires an exact lock entry",
                                    dependency.alias, domain.local_name
                                ))
                            })?;
                            if !matches!(locked.instance.source, SourceId::Path(_)) {
                                return Err(ProjectResolveError::Invalid(format!(
                                    "frozen path dependency `{}` in package `{}` does not resolve to a locked path source",
                                    dependency.alias, domain.local_name
                                )));
                            }
                            let vendor = self.vendor.ok_or_else(|| {
                                ProjectResolveError::Invalid(format!(
                                    "frozen path dependency `{}` requires {} at the workspace root; run `runmat package vendor` first",
                                    dependency.alias,
                                    crate::VENDOR_MANIFEST_FILENAME
                                ))
                            })?;
                            let vendored = vendor
                                .package(&locked.instance.identity_digest)
                                .ok_or_else(|| {
                                    ProjectResolveError::Invalid(format!(
                                        "vendor manifest does not contain locked path package {}",
                                        locked.instance.identity_digest
                                    ))
                                })?;
                            if vendored.source != locked.instance.source {
                                return Err(ProjectResolveError::Invalid(format!(
                                    "vendored source for package {} does not match runmat.lock",
                                    locked.instance.identity_digest
                                )));
                            }
                            let dependency_root = self.workspace_root.join(vendored.path.as_str());
                            let child_manifest = find_manifest(&dependency_root)
                                .await
                                .unwrap_or_else(|| dependency_root.join(PROJECT_MANIFEST_FILENAME));
                            if !is_file(&child_manifest).await {
                                return Err(ProjectResolveError::MissingManifest {
                                    package: domain.local_name.clone(),
                                    dependency: dependency.alias.to_string(),
                                    path: child_manifest,
                                });
                            }
                            (
                                child_manifest,
                                PackageOrigin::Vendor(vendored.source.clone()),
                            )
                        } else {
                            let dependency_root = root.join(path.as_str());
                            let child_manifest = find_manifest(&dependency_root)
                                .await
                                .unwrap_or_else(|| dependency_root.join(PROJECT_MANIFEST_FILENAME));
                            if !is_file(&child_manifest).await {
                                return Err(ProjectResolveError::MissingManifest {
                                    package: domain.local_name.clone(),
                                    dependency: dependency.alias.to_string(),
                                    path: child_manifest,
                                });
                            }
                            (child_manifest, PackageOrigin::Workspace)
                        }
                    }
                    DependencyLocator::Git {
                        repository,
                        selector,
                        subdir,
                    } => {
                        let locked =
                            locked_git_source(self.existing_lock, is_root, &instance, &dependency);
                        let plan = plan_git_acquisition(
                            repository.clone(),
                            selector.clone(),
                            subdir.clone(),
                            locked,
                            self.options.source_intent,
                            self.options.source_policy,
                        )?;
                        let mount = self.sources.acquire_git(&plan).await.map_err(|reason| {
                            ProjectResolveError::GitAcquire {
                                package: domain.local_name.clone(),
                                dependency: dependency.alias.to_string(),
                                reason,
                            }
                        })?;
                        crate::validate_git_acquisition(&plan, &mount.source)?;
                        self.acquired_git_sources.insert(mount.source.clone());
                        let child_manifest = find_manifest(&mount.root)
                            .await
                            .unwrap_or_else(|| mount.root.join(PROJECT_MANIFEST_FILENAME));
                        if !is_file(&child_manifest).await {
                            return Err(ProjectResolveError::MissingManifest {
                                package: domain.local_name.clone(),
                                dependency: dependency.alias.to_string(),
                                path: child_manifest,
                            });
                        }
                        (child_manifest, PackageOrigin::Git(mount.source))
                    }
                    DependencyLocator::Registry { .. } => unreachable!("handled above"),
                    DependencyLocator::ServerProject {
                        project,
                        service,
                        snapshot,
                    } => {
                        let locked = locked_server_source(
                            self.existing_lock,
                            is_root,
                            &instance,
                            &dependency,
                        );
                        let selector = ServerSnapshotSelector::from_manifest(snapshot.as_deref())?;
                        let plan = plan_server_project_acquisition(
                            service
                                .as_deref()
                                .unwrap_or(&self.options.default_server_origin),
                            project,
                            selector,
                            locked,
                            self.options.source_intent,
                            self.options.source_policy,
                        )?;
                        let mount =
                            self.sources
                                .acquire_server_project(&plan)
                                .await
                                .map_err(|reason| ProjectResolveError::ServerAcquire {
                                    package: domain.local_name.clone(),
                                    dependency: dependency.alias.to_string(),
                                    reason,
                                })?;
                        crate::validate_server_project_acquisition(&plan, &mount.source)?;
                        self.acquired_server_sources.insert(mount.source.clone());
                        let child_manifest = find_manifest(&mount.root)
                            .await
                            .unwrap_or_else(|| mount.root.join(PROJECT_MANIFEST_FILENAME));
                        if !is_file(&child_manifest).await {
                            return Err(ProjectResolveError::MissingManifest {
                                package: domain.local_name.clone(),
                                dependency: dependency.alias.to_string(),
                                path: child_manifest,
                            });
                        }
                        (child_manifest, PackageOrigin::ServerProject(mount.source))
                    }
                };
                let child = self
                    .load(child_manifest, child_origin, child_features, false, active)
                    .await?;
                validate_version(&domain.local_name, &dependency, &self.packages[&child])?;
                let loaded = LoadedDependency {
                    spec: dependency,
                    target: child,
                };
                let package = self
                    .packages
                    .get_mut(&key)
                    .expect("current package remains loaded");
                if !package.dependencies.iter().any(|existing| {
                    existing.spec.alias == loaded.spec.alias
                        && existing.spec.group == loaded.spec.group
                        && existing.spec.target == loaded.spec.target
                        && existing.target == loaded.target
                }) {
                    package.dependencies.push(loaded);
                }
            }
            active.pop();
            Ok(())
        })
    }
}

fn package_key(manifest: &Path, origin: &PackageOrigin) -> String {
    match origin {
        PackageOrigin::Workspace => format!("path:{}", manifest.display()),
        PackageOrigin::Git(source) => format!("git:{}:{}", source.tree_digest, manifest.display()),
        PackageOrigin::ServerProject(source) => {
            format!("server:{}:{}", source.tree_digest, manifest.display())
        }
        PackageOrigin::Registry(source) => {
            format!("registry:{}:{}", source.tree_digest, manifest.display())
        }
        PackageOrigin::Vendor(source) => format!("vendor:{source:?}:{}", manifest.display()),
    }
}

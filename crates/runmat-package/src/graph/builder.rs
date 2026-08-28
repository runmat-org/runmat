use super::{DependencyPath, GraphEdge, GraphPackage, PackageGraph};
use crate::{
    CanonicalPackageId, ContentDigest, DependencyGroup, GraphError, HostCapability,
    NormalizedRelativePath, PackageAlias, PackageInstanceId, PackageVersion, PathSourceId,
    SourceId, TargetPredicate,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathPackageInput {
    pub package: CanonicalPackageId,
    pub local_name: String,
    pub workspace_path: NormalizedRelativePath,
    pub manifest_digest: ContentDigest,
    pub tree_digest: ContentDigest,
    pub version: Option<PackageVersion>,
    pub dependencies: BTreeMap<PackageAlias, String>,
    pub required_capabilities: BTreeSet<HostCapability>,
    pub singleton: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathGraphInput {
    pub root: String,
    pub packages: BTreeMap<String, PathPackageInput>,
    pub host_capabilities: BTreeSet<HostCapability>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDependencyInput {
    pub alias: PackageAlias,
    pub target: String,
    pub group: DependencyGroup,
    pub optional: bool,
    pub target_predicate: Option<TargetPredicate>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPackageInput {
    pub instance: PackageInstanceId,
    pub local_name: String,
    pub dependencies: Vec<ResolvedDependencyInput>,
    pub required_capabilities: BTreeSet<HostCapability>,
    pub singleton: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedGraphInput {
    pub root: String,
    pub packages: BTreeMap<String, ResolvedPackageInput>,
    pub host_capabilities: BTreeSet<HostCapability>,
}

pub fn build_path_graph(input: PathGraphInput) -> Result<PackageGraph, GraphError> {
    if !input.packages.contains_key(&input.root) {
        return Err(GraphError::Invalid(format!(
            "root package key `{}` does not exist",
            input.root
        )));
    }
    let mut resolved = BTreeMap::new();
    for (key, package) in &input.packages {
        let instance = PackageInstanceId::new(
            package.package.clone(),
            SourceId::Path(PathSourceId {
                workspace_path: package.workspace_path.clone(),
                manifest_digest: package.manifest_digest.clone(),
                tree_digest: package.tree_digest.clone(),
            }),
            package.version.clone(),
            package.tree_digest.clone(),
        );
        resolved.insert(
            key.clone(),
            ResolvedPackageInput {
                instance,
                local_name: package.local_name.clone(),
                dependencies: package
                    .dependencies
                    .iter()
                    .map(|(alias, target)| ResolvedDependencyInput {
                        alias: alias.clone(),
                        target: target.clone(),
                        group: DependencyGroup::Runtime,
                        optional: false,
                        target_predicate: None,
                    })
                    .collect(),
                required_capabilities: package.required_capabilities.clone(),
                singleton: package.singleton,
            },
        );
    }
    build_resolved_graph(ResolvedGraphInput {
        root: input.root,
        packages: resolved,
        host_capabilities: input.host_capabilities,
    })
}

pub fn build_resolved_graph(input: ResolvedGraphInput) -> Result<PackageGraph, GraphError> {
    if !input.packages.contains_key(&input.root) {
        return Err(GraphError::Invalid(format!(
            "root package key `{}` does not exist",
            input.root
        )));
    }
    let mut packages = BTreeMap::new();
    let mut instances_by_key = BTreeMap::new();
    for (key, package) in &input.packages {
        let missing = package
            .required_capabilities
            .difference(&input.host_capabilities)
            .copied()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            let path = resolved_dependency_path(&input, key);
            return Err(GraphError::UnavailableCapabilities {
                dependency_path: path.to_string(),
                capabilities: missing
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        instances_by_key.insert(key.clone(), package.instance.identity_digest.clone());
        if packages
            .insert(
                package.instance.identity_digest.clone(),
                GraphPackage {
                    instance: package.instance.clone(),
                    local_name: package.local_name.clone(),
                    required_capabilities: package.required_capabilities.clone(),
                    singleton: package.singleton,
                },
            )
            .is_some()
        {
            return Err(GraphError::Invalid(format!(
                "package key `{key}` resolves to a duplicate instance"
            )));
        }
    }
    validate_singletons(&packages)?;
    let mut edges = Vec::new();
    for (key, package) in &input.packages {
        let from = instances_by_key[key].clone();
        for dependency in &package.dependencies {
            let Some(to) = instances_by_key.get(&dependency.target) else {
                return Err(GraphError::Invalid(format!(
                    "dependency `{}` of `{key}` references missing package key `{}`",
                    dependency.alias, dependency.target
                )));
            };
            edges.push(GraphEdge {
                from: from.clone(),
                alias: dependency.alias.clone(),
                to: to.clone(),
                group: dependency.group,
                optional: dependency.optional,
                target: dependency.target_predicate.clone(),
            });
        }
    }
    PackageGraph::finish(instances_by_key[&input.root].clone(), packages, edges)
}

fn resolved_dependency_path(input: &ResolvedGraphInput, target: &str) -> DependencyPath {
    if target == input.root {
        return DependencyPath {
            root: input.root.clone(),
            aliases: Vec::new(),
        };
    }
    let mut queue = vec![(input.root.clone(), Vec::new())];
    let mut visited = BTreeSet::new();
    while let Some((key, path)) = queue.pop() {
        if !visited.insert(key.clone()) {
            continue;
        }
        if let Some(package) = input.packages.get(&key) {
            for dependency in &package.dependencies {
                let mut next = path.clone();
                next.push(dependency.alias.clone());
                if dependency.target == target {
                    return DependencyPath {
                        root: input.root.clone(),
                        aliases: next,
                    };
                }
                queue.push((dependency.target.clone(), next));
            }
        }
    }
    DependencyPath {
        root: input.root.clone(),
        aliases: Vec::new(),
    }
}

fn validate_singletons(packages: &BTreeMap<ContentDigest, GraphPackage>) -> Result<(), GraphError> {
    let mut counts = BTreeMap::new();
    let mut singletons = BTreeSet::new();
    for package in packages.values() {
        *counts.entry(package.instance.package.clone()).or_insert(0) += 1;
        if package.singleton {
            singletons.insert(package.instance.package.clone());
        }
    }
    for singleton in singletons {
        if counts[&singleton] > 1 {
            return Err(GraphError::Invalid(format!(
                "singleton package {singleton} resolves to multiple instances"
            )));
        }
    }
    Ok(())
}

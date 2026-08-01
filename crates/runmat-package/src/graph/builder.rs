use super::{DependencyPath, GraphEdge, GraphPackage, PackageGraph};
use crate::{
    CanonicalPackageId, ContentDigest, DependencyGroup, GraphError, HostCapability,
    NormalizedRelativePath, PackageAlias, PackageInstanceId, PackageVersion, PathSourceId,
    SourceId,
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

pub fn build_path_graph(input: PathGraphInput) -> Result<PackageGraph, GraphError> {
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
            let path = dependency_path(&input, key);
            return Err(GraphError::UnavailableCapabilities {
                dependency_path: path.to_string(),
                capabilities: missing
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
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
        instances_by_key.insert(key.clone(), instance.identity_digest.clone());
        if packages
            .insert(
                instance.identity_digest.clone(),
                GraphPackage {
                    instance,
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
        for (alias, target_key) in &package.dependencies {
            let Some(to) = instances_by_key.get(target_key) else {
                return Err(GraphError::Invalid(format!(
                    "dependency `{alias}` of `{key}` references missing package key `{target_key}`"
                )));
            };
            edges.push(GraphEdge {
                from: from.clone(),
                alias: alias.clone(),
                to: to.clone(),
                group: DependencyGroup::Runtime,
                optional: false,
                target: None,
            });
        }
    }
    PackageGraph::finish(instances_by_key[&input.root].clone(), packages, edges)
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

fn dependency_path(input: &PathGraphInput, target: &str) -> DependencyPath {
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
            for (alias, dependency) in &package.dependencies {
                let mut next = path.clone();
                next.push(alias.clone());
                if dependency == target {
                    return DependencyPath {
                        root: input.root.clone(),
                        aliases: next,
                    };
                }
                queue.push((dependency.clone(), next));
            }
        }
    }
    DependencyPath {
        root: input.root.clone(),
        aliases: Vec::new(),
    }
}

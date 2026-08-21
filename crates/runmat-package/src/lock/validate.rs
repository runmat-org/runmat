use super::model::{PackageLock, LOCK_SCHEMA_VERSION, RESOLVER_FORMAT_VERSION};
use crate::{ContentDigest, LockError, PackageInstanceId};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn canonicalized(mut lock: PackageLock) -> PackageLock {
    lock.packages.sort_by(|left, right| {
        left.instance
            .identity_digest
            .cmp(&right.instance.identity_digest)
    });
    lock.edges.sort_by(|left, right| {
        (
            &left.from,
            &left.alias,
            &left.to,
            left.group,
            left.optional,
            &left.target,
        )
            .cmp(&(
                &right.from,
                &right.alias,
                &right.to,
                right.group,
                right.optional,
                &right.target,
            ))
    });
    lock
}

pub(crate) fn validate_lock(lock: &PackageLock) -> Result<(), LockError> {
    if lock.schema_version != LOCK_SCHEMA_VERSION {
        return Err(LockError::Incompatible(format!(
            "schema version {} is unsupported; expected {LOCK_SCHEMA_VERSION}",
            lock.schema_version
        )));
    }
    if lock.resolver_version != RESOLVER_FORMAT_VERSION {
        return Err(LockError::Incompatible(format!(
            "resolver format {} is unsupported; expected {RESOLVER_FORMAT_VERSION}",
            lock.resolver_version
        )));
    }
    if lock.selection.target.trim().is_empty() {
        return Err(invalid("selected target must be non-empty"));
    }

    let mut instances = BTreeMap::new();
    let mut package_counts = BTreeMap::new();
    let mut singleton_packages = BTreeSet::new();
    for package in &lock.packages {
        package
            .instance
            .source
            .validate()
            .map_err(|error| invalid(error.to_string()))?;
        let expected = PackageInstanceId::new(
            package.instance.package.clone(),
            package.instance.source.clone(),
            package.instance.version.clone(),
            package.instance.tree_digest.clone(),
        );
        if expected.identity_digest != package.instance.identity_digest {
            return Err(invalid(format!(
                "package instance {} has an invalid identity digest",
                package.instance.package
            )));
        }
        if package.instance.source.tree_digest() != &package.instance.tree_digest {
            return Err(invalid(format!(
                "package instance {} has conflicting source and instance tree digests",
                package.instance.package
            )));
        }
        if instances
            .insert(
                package.instance.identity_digest.clone(),
                &package.instance.package,
            )
            .is_some()
        {
            return Err(invalid(format!(
                "duplicate package instance {}",
                package.instance.identity_digest
            )));
        }
        if !package
            .required_capabilities
            .is_subset(&lock.selection.host_capabilities)
        {
            return Err(invalid(format!(
                "package {} requires unavailable host capabilities",
                package.instance.package
            )));
        }
        *package_counts
            .entry(package.instance.package.clone())
            .or_insert(0_usize) += 1;
        if package.singleton {
            singleton_packages.insert(package.instance.package.clone());
        }
    }
    for package in singleton_packages {
        if package_counts.get(&package).copied().unwrap_or_default() > 1 {
            return Err(invalid(format!(
                "singleton package {package} appears more than once"
            )));
        }
    }

    let mut aliases = BTreeSet::new();
    for edge in &lock.edges {
        if let Some(from) = &edge.from {
            if !instances.contains_key(from) {
                return Err(invalid(format!(
                    "edge source instance {from} does not exist"
                )));
            }
        }
        if !instances.contains_key(&edge.to) {
            return Err(invalid(format!(
                "edge target instance {} does not exist",
                edge.to
            )));
        }
        if !lock.selection.groups.contains(&edge.group) {
            return Err(invalid(format!(
                "edge alias `{}` belongs to unselected group {:?}",
                edge.alias, edge.group
            )));
        }
        if !aliases.insert((edge.from.clone(), edge.alias.clone())) {
            return Err(invalid(format!(
                "dependency alias `{}` is repeated for one package",
                edge.alias
            )));
        }
    }

    let expected_digest = compute_graph_digest(lock)?;
    if expected_digest != lock.graph_digest {
        return Err(invalid(format!(
            "graph digest mismatch: expected {expected_digest}, found {}",
            lock.graph_digest
        )));
    }
    Ok(())
}

pub(crate) fn compute_graph_digest(lock: &PackageLock) -> Result<ContentDigest, LockError> {
    #[derive(Serialize)]
    struct GraphDigestInput<'a> {
        format: &'static str,
        schema_version: u32,
        resolver_version: &'a str,
        root: &'a super::model::RootLock,
        selection: &'a super::model::LockSelection,
        packages: &'a [super::model::LockedPackage],
        edges: &'a [super::model::LockedEdge],
    }

    let canonical = canonicalized(lock.clone());
    let input = GraphDigestInput {
        format: "runmat-package-graph-v1",
        schema_version: canonical.schema_version,
        resolver_version: &canonical.resolver_version,
        root: &canonical.root,
        selection: &canonical.selection,
        packages: &canonical.packages,
        edges: &canonical.edges,
    };
    let bytes = serde_json::to_vec(&input)
        .map_err(|error| invalid(format!("cannot encode graph digest input: {error}")))?;
    Ok(ContentDigest::sha256(bytes))
}

fn invalid(reason: impl Into<String>) -> LockError {
    LockError::Invalid(reason.into())
}

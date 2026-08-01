use super::Resolution;
use crate::{CanonicalPackageId, ContentDigest, ResolveError};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdatePolicy {
    Full,
    Packages {
        packages: BTreeSet<CanonicalPackageId>,
        recursive: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdatePlan {
    pub added: BTreeSet<ContentDigest>,
    pub removed: BTreeSet<ContentDigest>,
    pub retained: BTreeSet<ContentDigest>,
    pub changed_packages: BTreeSet<CanonicalPackageId>,
}

pub fn plan_update(
    current: &Resolution,
    proposed: &Resolution,
    policy: &UpdatePolicy,
) -> Result<UpdatePlan, ResolveError> {
    let current_instances = current.packages.keys().cloned().collect::<BTreeSet<_>>();
    let proposed_instances = proposed.packages.keys().cloned().collect::<BTreeSet<_>>();
    let current_by_package = instances_by_package(current);
    let proposed_by_package = instances_by_package(proposed);
    let package_ids = current_by_package
        .keys()
        .chain(proposed_by_package.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let changed_packages = package_ids
        .into_iter()
        .filter(|package| current_by_package.get(package) != proposed_by_package.get(package))
        .collect::<BTreeSet<_>>();
    let allowed = allowed_packages(current, proposed, policy);
    let prohibited = changed_packages
        .difference(&allowed)
        .cloned()
        .collect::<Vec<_>>();
    if !prohibited.is_empty() {
        return Err(ResolveError::Conflict(format!(
            "constrained update would change frozen packages: {}",
            prohibited
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    Ok(UpdatePlan {
        added: proposed_instances
            .difference(&current_instances)
            .cloned()
            .collect(),
        removed: current_instances
            .difference(&proposed_instances)
            .cloned()
            .collect(),
        retained: current_instances
            .intersection(&proposed_instances)
            .cloned()
            .collect(),
        changed_packages,
    })
}

fn instances_by_package(
    resolution: &Resolution,
) -> BTreeMap<CanonicalPackageId, BTreeSet<ContentDigest>> {
    let mut result = BTreeMap::<CanonicalPackageId, BTreeSet<ContentDigest>>::new();
    for (identity, package) in &resolution.packages {
        result
            .entry(package.candidate.instance.package.clone())
            .or_default()
            .insert(identity.clone());
    }
    result
}

fn allowed_packages(
    current: &Resolution,
    proposed: &Resolution,
    policy: &UpdatePolicy,
) -> BTreeSet<CanonicalPackageId> {
    match policy {
        UpdatePolicy::Full => current
            .packages
            .values()
            .chain(proposed.packages.values())
            .map(|package| package.candidate.instance.package.clone())
            .collect(),
        UpdatePolicy::Packages {
            packages,
            recursive: false,
        } => packages.clone(),
        UpdatePolicy::Packages {
            packages,
            recursive: true,
        } => {
            let mut allowed = packages.clone();
            extend_private_descendants(current, &mut allowed);
            extend_private_descendants(proposed, &mut allowed);
            allowed
        }
    }
}

fn extend_private_descendants(resolution: &Resolution, allowed: &mut BTreeSet<CanonicalPackageId>) {
    loop {
        let mut changed = false;
        for edge in &resolution.edges {
            let Some(from) = &edge.from else {
                continue;
            };
            let from_package = &resolution.packages[from].candidate.instance.package;
            let to_package = &resolution.packages[&edge.to].candidate.instance.package;
            if !allowed.contains(from_package) || allowed.contains(to_package) {
                continue;
            }
            let externally_shared = resolution.edges.iter().any(|incoming| {
                incoming.to == edge.to
                    && incoming.from.as_ref().is_none_or(|owner| {
                        !allowed.contains(&resolution.packages[owner].candidate.instance.package)
                    })
            });
            if !externally_shared {
                changed |= allowed.insert(to_package.clone());
            }
        }
        if !changed {
            break;
        }
    }
}

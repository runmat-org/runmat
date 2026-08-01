use super::{
    CandidateIndex, CandidateMetadata, Incompatibility, RequirementPath, ResolutionRequest,
};
use crate::{ContentDigest, DependencyGroup, PackageAlias, ResolutionRequirement};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolutionPackage {
    pub candidate: CandidateMetadata,
    pub enabled_features: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ResolutionEdge {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from: Option<ContentDigest>,
    pub alias: PackageAlias,
    pub to: ContentDigest,
    pub group: DependencyGroup,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resolution {
    pub packages: BTreeMap<ContentDigest, ResolutionPackage>,
    pub edges: Vec<ResolutionEdge>,
}

#[derive(Debug, Clone, Default)]
struct SolverState {
    packages: BTreeMap<ContentDigest, ResolutionPackage>,
    by_package: BTreeMap<crate::CanonicalPackageId, Vec<ContentDigest>>,
    edges: BTreeSet<ResolutionEdge>,
    paths: BTreeMap<ContentDigest, BTreeSet<RequirementPath>>,
}

pub fn resolve(
    request: &ResolutionRequest,
    candidates: &CandidateIndex,
) -> Result<Resolution, Incompatibility> {
    let mut state = SolverState::default();
    let mut requirements = request.requirements.clone();
    requirements.sort_by(|left, right| left.alias.cmp(&right.alias));
    for requirement in requirements {
        if !requirement_applies(&requirement, request) {
            continue;
        }
        let path = RequirementPath {
            root: request.root.clone(),
            aliases: vec![requirement.alias.clone()],
        };
        resolve_requirement(
            &mut state,
            request,
            candidates,
            &requirement,
            None,
            path,
            &mut Vec::new(),
        )?;
    }
    Ok(Resolution {
        packages: state.packages,
        edges: state.edges.into_iter().collect(),
    })
}

fn resolve_requirement(
    state: &mut SolverState,
    request: &ResolutionRequest,
    index: &CandidateIndex,
    requirement: &ResolutionRequirement,
    from: Option<ContentDigest>,
    path: RequirementPath,
    active: &mut Vec<ContentDigest>,
) -> Result<ContentDigest, Incompatibility> {
    let requested_features = requested_features(requirement);
    if let Some(existing) = state
        .by_package
        .get(&requirement.package)
        .into_iter()
        .flatten()
        .filter_map(|identity| state.packages.get(identity))
        .find(|package| candidate_matches(&package.candidate, requirement, request))
        .map(|package| package.candidate.instance.identity_digest.clone())
    {
        add_edge(state, from, requirement, existing.clone(), &path)?;
        state
            .paths
            .entry(existing.clone())
            .or_default()
            .insert(path.clone());
        let changed = {
            let package = state.packages.get_mut(&existing).expect("indexed package");
            let before = package.enabled_features.len();
            package
                .enabled_features
                .extend(requested_features.iter().cloned());
            before != package.enabled_features.len()
        };
        if changed {
            expand_candidate(state, request, index, &existing, path, active)?;
        }
        return Ok(existing);
    }

    let mut eligible = index
        .candidates(&requirement.package)
        .iter()
        .filter(|candidate| candidate_matches(candidate, requirement, request))
        .cloned()
        .collect::<Vec<_>>();
    let locked_for_package = index
        .candidates(&requirement.package)
        .iter()
        .filter(|candidate| {
            request
                .locked_instances
                .contains(&candidate.instance.identity_digest)
        })
        .map(|candidate| candidate.instance.identity_digest.clone())
        .collect::<BTreeSet<_>>();
    if request.update_packages.as_ref().is_some_and(|packages| {
        !packages.contains(&requirement.package) && !locked_for_package.is_empty()
    }) {
        eligible
            .retain(|candidate| locked_for_package.contains(&candidate.instance.identity_digest));
    }
    eligible.sort_by(|left, right| {
        right
            .instance
            .version
            .cmp(&left.instance.version)
            .then_with(|| {
                left.instance
                    .identity_digest
                    .cmp(&right.instance.identity_digest)
            })
    });
    let mut last_conflict = None;
    for candidate in eligible {
        if let Some(mut paths) = singleton_conflict_paths(state, &candidate) {
            paths.push(path.clone());
            paths.sort();
            paths.dedup();
            last_conflict = Some(Incompatibility {
                package: Box::new(requirement.package.clone()),
                requirement: Box::new(requirement.version.clone()),
                paths,
                reason: "a singleton/native package would require multiple instances".to_string(),
            });
            continue;
        }
        let mut trial = state.clone();
        let identity = candidate.instance.identity_digest.clone();
        trial
            .by_package
            .entry(candidate.instance.package.clone())
            .or_default()
            .push(identity.clone());
        trial.packages.insert(
            identity.clone(),
            ResolutionPackage {
                candidate,
                enabled_features: requested_features.clone(),
            },
        );
        trial
            .paths
            .entry(identity.clone())
            .or_default()
            .insert(path.clone());
        if let Err(error) = add_edge(
            &mut trial,
            from.clone(),
            requirement,
            identity.clone(),
            &path,
        )
        .and_then(|_| expand_candidate(&mut trial, request, index, &identity, path.clone(), active))
        {
            last_conflict = Some(error);
            continue;
        }
        *state = trial;
        return Ok(identity);
    }
    Err(last_conflict.unwrap_or_else(|| {
        conflict(
            requirement,
            path,
            "no eligible candidate satisfies version, target, capability, RunMat, yank, and offline policy",
        )
    }))
}

fn expand_candidate(
    state: &mut SolverState,
    request: &ResolutionRequest,
    index: &CandidateIndex,
    identity: &ContentDigest,
    path: RequirementPath,
    active: &mut Vec<ContentDigest>,
) -> Result<(), Incompatibility> {
    if active.contains(identity) {
        let package = &state.packages[identity].candidate.instance.package;
        return Err(Incompatibility {
            package: Box::new(package.clone()),
            requirement: Box::new(semver::VersionReq::STAR),
            paths: vec![path],
            reason: "dependency cycle detected".to_string(),
        });
    }
    active.push(identity.clone());
    let package = state.packages[identity].clone();
    let activation = feature_activation(&package, &path)?;
    let mut dependencies = package.candidate.dependencies.clone();
    dependencies.sort_by(|left, right| left.alias.cmp(&right.alias));
    for mut dependency in dependencies {
        if !requirement_applies(&dependency, request) {
            continue;
        }
        if dependency.optional && !activation.contains_key(dependency.alias.as_str()) {
            continue;
        }
        if let Some(features) = activation.get(dependency.alias.as_str()) {
            dependency.features.extend(features.iter().cloned());
        }
        let mut child_path = path.clone();
        child_path.aliases.push(dependency.alias.clone());
        resolve_requirement(
            state,
            request,
            index,
            &dependency,
            Some(identity.clone()),
            child_path,
            active,
        )?;
    }
    active.pop();
    Ok(())
}

fn feature_activation(
    package: &ResolutionPackage,
    path: &RequirementPath,
) -> Result<BTreeMap<String, BTreeSet<String>>, Incompatibility> {
    let aliases = package
        .candidate
        .dependencies
        .iter()
        .map(|dependency| dependency.alias.as_str())
        .collect::<BTreeSet<_>>();
    let mut activation = BTreeMap::<String, BTreeSet<String>>::new();
    for feature in &package.enabled_features {
        let Some(requests) = package.candidate.features.get(feature) else {
            if feature == "default" {
                continue;
            }
            return Err(Incompatibility {
                package: Box::new(package.candidate.instance.package.clone()),
                requirement: Box::new(semver::VersionReq::STAR),
                paths: vec![path.clone()],
                reason: format!("requested feature `{feature}` is not declared"),
            });
        };
        for request in requests {
            let (alias, dependency_feature) = request
                .split_once('/')
                .map_or((request.as_str(), None), |(alias, feature)| {
                    (alias, Some(feature))
                });
            if !aliases.contains(alias) {
                return Err(Incompatibility {
                    package: Box::new(package.candidate.instance.package.clone()),
                    requirement: Box::new(semver::VersionReq::STAR),
                    paths: vec![path.clone()],
                    reason: format!(
                        "feature `{feature}` activates unknown dependency alias `{alias}`"
                    ),
                });
            }
            let features = activation.entry(alias.to_string()).or_default();
            if let Some(dependency_feature) = dependency_feature {
                features.insert(dependency_feature.to_string());
            }
        }
    }
    Ok(activation)
}

fn requested_features(requirement: &ResolutionRequirement) -> BTreeSet<String> {
    let mut features = requirement.features.clone();
    if requirement.default_features {
        features.insert("default".to_string());
    }
    features
}

fn requirement_applies(requirement: &ResolutionRequirement, request: &ResolutionRequest) -> bool {
    request.groups.contains(&requirement.group)
        && requirement
            .target
            .as_ref()
            .is_none_or(|target| request.environment.supports(target))
}

fn candidate_matches(
    candidate: &CandidateMetadata,
    requirement: &ResolutionRequirement,
    request: &ResolutionRequest,
) -> bool {
    let Some(version) = candidate.instance.version.as_ref() else {
        return false;
    };
    requirement.version.matches(version.as_semver())
        && (!candidate.yanked)
        && (!request.offline || candidate.available_offline)
        && candidate
            .runmat_version
            .as_ref()
            .is_none_or(|required| required.matches(&request.runmat_version))
        && candidate
            .required_capabilities
            .is_subset(&request.environment.capabilities)
        && (candidate.target_artifacts.is_empty()
            || candidate
                .target_artifacts
                .iter()
                .any(|target| request.environment.supports(target)))
}

fn singleton_conflict_paths(
    state: &SolverState,
    candidate: &CandidateMetadata,
) -> Option<Vec<RequirementPath>> {
    let conflicting = state
        .by_package
        .get(&candidate.instance.package)
        .into_iter()
        .flatten()
        .filter_map(|identity| state.packages.get(identity))
        .any(|selected| selected.candidate.singleton || candidate.singleton);
    conflicting.then(|| {
        state
            .by_package
            .get(&candidate.instance.package)
            .into_iter()
            .flatten()
            .flat_map(|identity| state.paths.get(identity).into_iter().flatten().cloned())
            .collect()
    })
}

fn add_edge(
    state: &mut SolverState,
    from: Option<ContentDigest>,
    requirement: &ResolutionRequirement,
    to: ContentDigest,
    path: &RequirementPath,
) -> Result<(), Incompatibility> {
    if state
        .edges
        .iter()
        .any(|edge| edge.from == from && edge.alias == requirement.alias && edge.to != to)
    {
        return Err(conflict(
            requirement,
            path.clone(),
            "one package cannot bind the same edge-local alias to two instances",
        ));
    }
    state.edges.insert(ResolutionEdge {
        from,
        alias: requirement.alias.clone(),
        to,
        group: requirement.group,
    });
    Ok(())
}

fn conflict(
    requirement: &ResolutionRequirement,
    path: RequirementPath,
    reason: impl Into<String>,
) -> Incompatibility {
    Incompatibility {
        package: Box::new(requirement.package.clone()),
        requirement: Box::new(requirement.version.clone()),
        paths: vec![path],
        reason: reason.into(),
    }
}

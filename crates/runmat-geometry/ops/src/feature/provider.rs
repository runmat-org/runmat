use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;

use super::{
    ExecutedGeometryFeaturePlan, GeometryEntityLineage, GeometryFeatureCapabilities,
    GeometryFeatureError, GeometryFeaturePlan, GeometryFeatureProviderOutput,
    GeometryFeatureProviderSelection, GeometryTopologyHistory, ResolvedGeometryFeatureOperation,
};

pub trait GeometryFeatureProvider {
    type Geometry;
    type Error: Display;

    fn capabilities(&self) -> GeometryFeatureCapabilities;

    fn entity_inventory(
        &self,
        geometry: &Self::Geometry,
    ) -> Result<
        BTreeMap<
            runmat_geometry_core::GeometryRegionRole,
            Vec<runmat_geometry_core::PersistentEntityId>,
        >,
        Self::Error,
    >;

    fn execute(
        &self,
        feature_id: &str,
        operation: &ResolvedGeometryFeatureOperation,
        selection: &GeometryFeatureProviderSelection,
        dependencies: &[(&str, &Self::Geometry)],
        source: Option<&Self::Geometry>,
    ) -> Result<GeometryFeatureProviderOutput<Self::Geometry>, Self::Error>;
}

pub fn execute_feature_plan<P: GeometryFeatureProvider>(
    provider: &P,
    plan: &GeometryFeaturePlan,
    source: Option<&P::Geometry>,
) -> Result<ExecutedGeometryFeaturePlan<P::Geometry>, GeometryFeatureError> {
    let capabilities = provider.capabilities();
    let mut values = BTreeMap::<String, P::Geometry>::new();
    let mut lineage_by_value = BTreeMap::<String, BTreeMap<String, GeometryTopologyHistory>>::new();

    for step in &plan.steps {
        if !capabilities.supports(step.operation.kind()) {
            return Err(GeometryFeatureError::UnsupportedOperation {
                feature_id: step.feature_id.clone(),
                kind: step.operation.kind(),
            });
        }
        let dependencies = step
            .dependency_ids
            .iter()
            .map(|reference| {
                if reference.0 == "source.result" {
                    source
                        .map(|value| (reference.0.as_str(), value))
                        .ok_or_else(|| {
                            GeometryFeatureError::InvalidPlan(format!(
                                "feature `{}` requires source geometry",
                                step.feature_id
                            ))
                        })
                } else {
                    values
                        .get(&reference.0)
                        .map(|value| (reference.0.as_str(), value))
                        .ok_or_else(|| {
                            GeometryFeatureError::InvalidPlan(format!(
                                "feature `{}` references unavailable dependency `{}`",
                                step.feature_id, reference.0
                            ))
                        })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let selection = resolve_provider_selection(
            provider,
            &step.feature_id,
            &step.operation,
            &dependencies,
            &lineage_by_value,
        )?;
        let output = provider
            .execute(
                &step.feature_id,
                &step.operation,
                &selection,
                &dependencies,
                source,
            )
            .map_err(|error| GeometryFeatureError::Provider {
                feature_id: step.feature_id.clone(),
                message: error.to_string(),
            })?;
        output.history.validate()?;
        let mut projected = BTreeMap::new();
        for dependency in &step.dependency_ids {
            if dependency.0 == "source.result" {
                continue;
            }
            let dependency_lineage = lineage_by_value.get(&dependency.0).ok_or_else(|| {
                GeometryFeatureError::InvalidPlan(format!(
                    "feature `{}` has no lineage for dependency `{}`",
                    step.feature_id, dependency.0
                ))
            })?;
            for (origin, history) in dependency_lineage {
                let history = project_history(history, &output.history)?;
                match projected.entry(origin.clone()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(history);
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        merge_projected_history(entry.get_mut(), history)?;
                    }
                }
            }
        }
        if projected
            .insert(step.feature_id.clone(), output.history)
            .is_some()
        {
            return Err(GeometryFeatureError::InvalidPlan(format!(
                "feature `{}` depends on an identically named prior feature",
                step.feature_id
            )));
        }
        if values
            .insert(step.feature_id.clone(), output.geometry)
            .is_some()
        {
            return Err(GeometryFeatureError::InvalidPlan(format!(
                "duplicate feature `{}`",
                step.feature_id
            )));
        }
        lineage_by_value.insert(step.feature_id.clone(), projected);
    }

    let geometry = values.remove(&plan.result.0).ok_or_else(|| {
        GeometryFeatureError::InvalidPlan(format!(
            "result feature `{}` was not produced",
            plan.result.0
        ))
    })?;
    let history_by_feature = lineage_by_value.remove(&plan.result.0).ok_or_else(|| {
        GeometryFeatureError::InvalidPlan(format!(
            "result feature `{}` has no projected topology lineage",
            plan.result.0
        ))
    })?;
    Ok(ExecutedGeometryFeaturePlan {
        geometry,
        history_by_feature,
    })
}

pub(super) fn resolve_provider_selection<P: GeometryFeatureProvider>(
    provider: &P,
    feature_id: &str,
    operation: &ResolvedGeometryFeatureOperation,
    dependencies: &[(&str, &P::Geometry)],
    lineage_by_value: &BTreeMap<String, BTreeMap<String, GeometryTopologyHistory>>,
) -> Result<GeometryFeatureProviderSelection, GeometryFeatureError> {
    let (target, selection) = match operation {
        ResolvedGeometryFeatureOperation::Fillet { target, edges, .. }
        | ResolvedGeometryFeatureOperation::Chamfer { target, edges, .. } => (target, edges),
        _ => return Ok(GeometryFeatureProviderSelection::None),
    };
    if selection.role != runmat_geometry_core::GeometryRegionRole::Edge {
        return Err(GeometryFeatureError::InvalidPlan(format!(
            "feature `{feature_id}` requires an edge selection"
        )));
    }
    let target_geometry = dependencies
        .iter()
        .find_map(|(name, geometry)| (*name == target.0).then_some(*geometry))
        .ok_or_else(|| {
            GeometryFeatureError::InvalidPlan(format!(
                "feature `{feature_id}` is missing selection target `{}`",
                target.0
            ))
        })?;
    let inventory = provider
        .entity_inventory(target_geometry)
        .map_err(|error| GeometryFeatureError::Provider {
            feature_id: feature_id.to_string(),
            message: error.to_string(),
        })?;
    let available = inventory
        .get(&runmat_geometry_core::GeometryRegionRole::Edge)
        .cloned()
        .unwrap_or_default();
    let available_set = available.iter().cloned().collect::<BTreeSet<_>>();
    let lineage = lineage_by_value.get(&target.0);
    let mut edges = match &selection.selector {
        runmat_geometry_core::GeometryRegionSelector::Result => available,
        runmat_geometry_core::GeometryRegionSelector::PersistentId(value) => available
            .into_iter()
            .filter(|entity| entity.matches_analysis_selection(value.as_str()))
            .collect(),
        runmat_geometry_core::GeometryRegionSelector::FeatureResult(origin) => lineage
            .and_then(|history| history.get(&origin.0))
            .ok_or_else(|| selection_history_error(feature_id, &target.0, &origin.0))?
            .result_entities
            .get(&runmat_geometry_core::GeometryRegionRole::Edge)
            .cloned()
            .unwrap_or_default(),
        runmat_geometry_core::GeometryRegionSelector::BoundaryIntroducedBy(origin) => {
            let history = lineage
                .and_then(|history| history.get(&origin.0))
                .ok_or_else(|| selection_history_error(feature_id, &target.0, &origin.0))?;
            if !history.inventory_complete {
                return Err(GeometryFeatureError::InvalidPlan(format!(
                    "feature `{feature_id}` requires complete lineage for edge selection `{}`",
                    origin.0
                )));
            }
            history
                .introduced
                .iter()
                .filter(|entity| entity.kind == runmat_geometry_core::PersistentEntityKind::Edge)
                .cloned()
                .collect()
        }
        runmat_geometry_core::GeometryRegionSelector::Name(_)
        | runmat_geometry_core::GeometryRegionSelector::Tag(_)
        | runmat_geometry_core::GeometryRegionSelector::GeneratedOutput(_) => {
            return Err(GeometryFeatureError::InvalidPlan(format!(
                "feature `{feature_id}` uses an edge selector unavailable during feature execution"
            )))
        }
    };
    edges.sort();
    edges.dedup();
    if edges.iter().any(|edge| !available_set.contains(edge)) {
        return Err(GeometryFeatureError::InvalidPlan(format!(
            "feature `{feature_id}` edge selection escapes target `{}`",
            target.0
        )));
    }
    let valid_cardinality = match selection.cardinality {
        runmat_geometry_core::GeometryRegionCardinality::ExactlyOne => edges.len() == 1,
        runmat_geometry_core::GeometryRegionCardinality::OneOrMore => !edges.is_empty(),
    };
    if !valid_cardinality {
        return Err(GeometryFeatureError::InvalidPlan(format!(
            "feature `{feature_id}` edge selection resolved {} entities, violating {:?}",
            edges.len(),
            selection.cardinality
        )));
    }
    Ok(GeometryFeatureProviderSelection::Edges(edges))
}

fn selection_history_error(feature_id: &str, target: &str, origin: &str) -> GeometryFeatureError {
    GeometryFeatureError::InvalidPlan(format!(
        "feature `{feature_id}` cannot resolve `{origin}` against target `{target}`"
    ))
}

fn project_history(
    origin: &GeometryTopologyHistory,
    next: &GeometryTopologyHistory,
) -> Result<GeometryTopologyHistory, GeometryFeatureError> {
    if !origin.inventory_complete || !next.inventory_complete {
        return Ok(GeometryTopologyHistory {
            inventory_complete: false,
            result_entities: BTreeMap::new(),
            introduced: Vec::new(),
            source_lineage: Vec::new(),
        });
    }
    let result_roles = next
        .result_entities
        .iter()
        .flat_map(|(role, entities)| entities.iter().map(move |entity| (entity.clone(), *role)))
        .collect::<BTreeMap<_, _>>();
    let next_lineage = next
        .source_lineage
        .iter()
        .map(|lineage| (&lineage.source, &lineage.descendants))
        .collect::<BTreeMap<_, _>>();
    let project = |entities: &[runmat_geometry_core::PersistentEntityId]| {
        let mut projected = BTreeSet::new();
        for entity in entities {
            let descendants = next_lineage.get(entity).ok_or_else(|| {
                GeometryFeatureError::InvalidHistory(format!(
                    "complete downstream history omits source entity `{}`",
                    entity.source_topology_id
                ))
            })?;
            projected.extend(descendants.iter().cloned());
        }
        Ok::<Vec<_>, GeometryFeatureError>(projected.into_iter().collect())
    };

    let mut result_entities = BTreeMap::<_, Vec<_>>::new();
    for entities in origin.result_entities.values() {
        for entity in project(entities)? {
            let role = result_roles.get(&entity).ok_or_else(|| {
                GeometryFeatureError::InvalidHistory(format!(
                    "projected entity `{}` is absent from downstream result roles",
                    entity.source_topology_id
                ))
            })?;
            result_entities.entry(*role).or_default().push(entity);
        }
    }
    canonicalize_role_inventory(&mut result_entities);
    let introduced = project(&origin.introduced)?;
    let source_lineage = origin
        .source_lineage
        .iter()
        .map(|lineage| {
            Ok(GeometryEntityLineage {
                source: lineage.source.clone(),
                descendants: project(&lineage.descendants)?,
            })
        })
        .collect::<Result<_, GeometryFeatureError>>()?;
    let projected = GeometryTopologyHistory {
        inventory_complete: true,
        result_entities,
        introduced,
        source_lineage,
    };
    projected.validate()?;
    Ok(projected)
}

fn merge_projected_history(
    target: &mut GeometryTopologyHistory,
    incoming: GeometryTopologyHistory,
) -> Result<(), GeometryFeatureError> {
    if !target.inventory_complete || !incoming.inventory_complete {
        target.inventory_complete = false;
        target.result_entities.clear();
        target.introduced.clear();
        target.source_lineage.clear();
        return Ok(());
    }
    for (role, entities) in incoming.result_entities {
        target
            .result_entities
            .entry(role)
            .or_default()
            .extend(entities);
    }
    target.introduced.extend(incoming.introduced);
    let mut source_lineage = target
        .source_lineage
        .drain(..)
        .chain(incoming.source_lineage)
        .fold(BTreeMap::<_, Vec<_>>::new(), |mut merged, lineage| {
            merged
                .entry(lineage.source)
                .or_default()
                .extend(lineage.descendants);
            merged
        });
    canonicalize_role_inventory(&mut target.result_entities);
    target.introduced.sort();
    target.introduced.dedup();
    for descendants in source_lineage.values_mut() {
        descendants.sort();
        descendants.dedup();
    }
    target.source_lineage = source_lineage
        .into_iter()
        .map(|(source, descendants)| GeometryEntityLineage {
            source,
            descendants,
        })
        .collect();
    target.validate()
}

fn canonicalize_role_inventory(
    inventory: &mut BTreeMap<
        runmat_geometry_core::GeometryRegionRole,
        Vec<runmat_geometry_core::PersistentEntityId>,
    >,
) {
    for entities in inventory.values_mut() {
        entities.sort();
        entities.dedup();
    }
}

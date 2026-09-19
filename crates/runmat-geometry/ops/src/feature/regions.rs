use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;

use runmat_geometry_core::{
    ExactBRepTopology, GeneratedGeometryRegionOutput, GeometryDefinition,
    GeometryRegionCardinality, GeometryRegionRole, GeometryRegionSelector, PersistentEntityId,
};

pub use runmat_geometry_core::{ResolvedGeometryRegion, ResolvedGeometryRegions};

use super::{GeometryFeatureError, GeometryTopologyHistory};

pub trait GeometryRegionEvidence {
    type Error: Display;

    fn by_name(
        &self,
        role: GeometryRegionRole,
        name: &str,
    ) -> Result<Vec<PersistentEntityId>, Self::Error>;

    fn by_tag(
        &self,
        role: GeometryRegionRole,
        tag: &str,
    ) -> Result<Vec<PersistentEntityId>, Self::Error>;

    fn generated_output(
        &self,
        role: GeometryRegionRole,
        output: &str,
    ) -> Result<Vec<PersistentEntityId>, Self::Error>;
}

pub fn resolve_geometry_regions<E: GeometryRegionEvidence>(
    definition: &GeometryDefinition,
    topology: &ExactBRepTopology,
    history_by_feature: &BTreeMap<String, GeometryTopologyHistory>,
    evidence: &E,
) -> Result<ResolvedGeometryRegions, GeometryFeatureError> {
    resolve_geometry_regions_with_generated_outputs(
        definition,
        topology,
        history_by_feature,
        &BTreeMap::new(),
        evidence,
    )
}

pub fn resolve_geometry_regions_with_generated_outputs<E: GeometryRegionEvidence>(
    definition: &GeometryDefinition,
    topology: &ExactBRepTopology,
    history_by_feature: &BTreeMap<String, GeometryTopologyHistory>,
    generated_outputs: &BTreeMap<String, GeneratedGeometryRegionOutput>,
    evidence: &E,
) -> Result<ResolvedGeometryRegions, GeometryFeatureError> {
    let mut definition = definition.clone();
    for region in &mut definition.regions {
        let GeometryRegionSelector::GeneratedOutput(output_name) = &region.selector else {
            continue;
        };
        let output = generated_outputs.get(output_name).ok_or_else(|| {
            GeometryFeatureError::InvalidPlan(format!(
                "region `{}` requires missing generated output `{output_name}`",
                region.region_id
            ))
        })?;
        if region.role != output.role || region.cardinality != output.cardinality {
            return Err(GeometryFeatureError::InvalidPlan(format!(
                "region `{}` differs from generated output `{output_name}` role or cardinality",
                region.region_id
            )));
        }
        region.selector = output.selector.clone();
    }
    resolve_geometry_regions_inner(&definition, topology, history_by_feature, evidence)
}

fn resolve_geometry_regions_inner<E: GeometryRegionEvidence>(
    definition: &GeometryDefinition,
    topology: &ExactBRepTopology,
    history_by_feature: &BTreeMap<String, GeometryTopologyHistory>,
    evidence: &E,
) -> Result<ResolvedGeometryRegions, GeometryFeatureError> {
    let mut resolved = BTreeMap::new();
    for region in &definition.regions {
        let available = role_entities(topology, region.role);
        let available_set = available.iter().cloned().collect::<BTreeSet<_>>();
        let mut entity_ids = match &region.selector {
            GeometryRegionSelector::Name(name) => evidence
                .by_name(region.role, name)
                .map_err(|error| evidence_error(&region.region_id, error))?,
            GeometryRegionSelector::Tag(tag) => evidence
                .by_tag(region.role, tag)
                .map_err(|error| evidence_error(&region.region_id, error))?,
            GeometryRegionSelector::PersistentId(selection) => available
                .into_iter()
                .filter(|entity| entity.matches_analysis_selection(selection))
                .collect(),
            GeometryRegionSelector::Result => available,
            GeometryRegionSelector::FeatureResult(feature) => history_by_feature
                .get(&feature.0)
                .ok_or_else(|| missing_history(&region.region_id, &feature.0))?
                .result_entities
                .get(&region.role)
                .into_iter()
                .flatten()
                .filter(|entity| available_set.contains(*entity))
                .cloned()
                .collect(),
            GeometryRegionSelector::BoundaryIntroducedBy(feature) => {
                let history = history_by_feature
                    .get(&feature.0)
                    .ok_or_else(|| missing_history(&region.region_id, &feature.0))?;
                if !history.inventory_complete {
                    return Err(GeometryFeatureError::InvalidPlan(format!(
                        "region `{}` requires complete topology inventory for feature `{}`",
                        region.region_id, feature.0
                    )));
                }
                history
                    .introduced
                    .iter()
                    .filter(|entity| available_set.contains(*entity))
                    .cloned()
                    .collect()
            }
            GeometryRegionSelector::GeneratedOutput(output) => evidence
                .generated_output(region.role, output)
                .map_err(|error| evidence_error(&region.region_id, error))?,
        };
        entity_ids.sort();
        entity_ids.dedup();
        if entity_ids
            .iter()
            .any(|entity| !available_set.contains(entity))
        {
            return Err(GeometryFeatureError::InvalidPlan(format!(
                "region `{}` evidence names an entity outside its {:?} result topology",
                region.region_id, region.role
            )));
        }
        validate_cardinality(&region.region_id, region.cardinality, entity_ids.len())?;
        resolved.insert(
            region.region_id.clone(),
            ResolvedGeometryRegion {
                role: region.role,
                entity_ids,
            },
        );
    }
    Ok(resolved)
}

fn role_entities(
    topology: &ExactBRepTopology,
    role: GeometryRegionRole,
) -> Vec<PersistentEntityId> {
    match role {
        GeometryRegionRole::Vertex => topology
            .vertices
            .iter()
            .map(|entity| entity.id.clone())
            .collect(),
        GeometryRegionRole::Edge => topology
            .edges
            .iter()
            .map(|entity| entity.id.clone())
            .collect(),
        GeometryRegionRole::Face => topology
            .faces
            .iter()
            .map(|entity| entity.id.clone())
            .collect(),
        GeometryRegionRole::Solid => topology
            .solids
            .iter()
            .map(|entity| entity.id.clone())
            .collect(),
        GeometryRegionRole::Cell => topology
            .regions
            .iter()
            .map(|entity| entity.id.clone())
            .collect(),
    }
}

fn validate_cardinality(
    region_id: &str,
    cardinality: GeometryRegionCardinality,
    count: usize,
) -> Result<(), GeometryFeatureError> {
    let valid = match cardinality {
        GeometryRegionCardinality::ExactlyOne => count == 1,
        GeometryRegionCardinality::OneOrMore => count > 0,
    };
    if valid {
        Ok(())
    } else {
        Err(GeometryFeatureError::InvalidPlan(format!(
            "region `{region_id}` resolved {count} entities, violating {cardinality:?}"
        )))
    }
}

fn missing_history(region_id: &str, feature_id: &str) -> GeometryFeatureError {
    GeometryFeatureError::InvalidPlan(format!(
        "region `{region_id}` requires missing feature history `{feature_id}`"
    ))
}

fn evidence_error(region_id: &str, error: impl Display) -> GeometryFeatureError {
    GeometryFeatureError::InvalidPlan(format!(
        "region `{region_id}` evidence resolution failed: {error}"
    ))
}

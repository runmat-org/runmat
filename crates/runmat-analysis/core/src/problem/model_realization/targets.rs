use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{GeometryDefinitionRef, GeometryRegionRole, ResolvedGeometryRegions};

use crate::{
    AuthoredMaterialRef, AuthoredMaterialSource, AuthoredModel, AuthoredPropertyTarget,
    AuthoredRegionSource, EntityDimension, EvidenceConfidence, ExpectedTopology,
    MaterialAssignment, MaterialModel, PropertyRegionTarget, RegionTarget, TargetCardinality,
};

use super::AnalysisModelRealizationError;

pub(super) struct MaterialResolution {
    pub materials: Vec<MaterialModel>,
    pub assignments: Vec<MaterialAssignment>,
}

impl MaterialResolution {
    pub fn new(
        catalog: &BTreeMap<AuthoredMaterialRef, MaterialModel>,
        models: &[&AuthoredModel],
        targets: &TargetResolver<'_>,
    ) -> Result<Self, AnalysisModelRealizationError> {
        let mut used = BTreeSet::new();
        let mut assignments = Vec::new();
        for model in models {
            for assignment in &model.material_assignments {
                let AuthoredMaterialSource::Declaration(reference) = &assignment.material else {
                    return Err(AnalysisModelRealizationError::UnboundModel(
                        model.model_id.clone(),
                    ));
                };
                if !catalog.contains_key(reference) {
                    return Err(AnalysisModelRealizationError::MissingMaterial {
                        module_path: reference.module_path.clone(),
                        material_id: reference.material_id.clone(),
                    });
                }
                used.insert(reference.clone());
                let material_id = reference.qualified_id();
                assignments.push(MaterialAssignment {
                    target: targets.property_target(&assignment.target)?,
                    expected_material_id: material_id.clone(),
                    assigned_material_id: material_id,
                    confidence: EvidenceConfidence::Verified,
                });
            }
            for layup in &model.layups {
                for ply in &layup.plies {
                    let AuthoredMaterialSource::Declaration(reference) = &ply.material else {
                        return Err(AnalysisModelRealizationError::UnboundModel(
                            model.model_id.clone(),
                        ));
                    };
                    if !catalog.contains_key(reference) {
                        return Err(AnalysisModelRealizationError::MissingMaterial {
                            module_path: reference.module_path.clone(),
                            material_id: reference.material_id.clone(),
                        });
                    }
                    used.insert(reference.clone());
                }
            }
        }
        let materials = used
            .into_iter()
            .map(|reference| {
                let mut material = catalog[&reference].clone();
                material.material_id = reference.qualified_id();
                material
            })
            .collect();
        Ok(Self {
            materials,
            assignments,
        })
    }
}

pub(super) struct TargetResolver<'a> {
    geometry: &'a GeometryDefinitionRef,
    regions: &'a ResolvedGeometryRegions,
}

impl<'a> TargetResolver<'a> {
    pub fn new(geometry: &'a GeometryDefinitionRef, regions: &'a ResolvedGeometryRegions) -> Self {
        Self { geometry, regions }
    }

    pub(super) fn property_target(
        &self,
        target: &AuthoredPropertyTarget,
    ) -> Result<PropertyRegionTarget, AnalysisModelRealizationError> {
        let expected_role = match target {
            AuthoredPropertyTarget::Beam(_) => GeometryRegionRole::Edge,
            AuthoredPropertyTarget::Shell(_) => GeometryRegionRole::Face,
            AuthoredPropertyTarget::Solid(_) => GeometryRegionRole::Solid,
            AuthoredPropertyTarget::Cell(_) => GeometryRegionRole::Cell,
        };
        let mut target = match target {
            AuthoredPropertyTarget::Beam(source) => {
                PropertyRegionTarget::beam(self.region_id(source)?)
            }
            AuthoredPropertyTarget::Shell(source) => {
                PropertyRegionTarget::shell(self.region_id(source)?)
            }
            AuthoredPropertyTarget::Solid(source) => {
                PropertyRegionTarget::solid(self.region_id(source)?)
            }
            AuthoredPropertyTarget::Cell(source) => {
                PropertyRegionTarget::cell(self.region_id(source)?)
            }
        };
        let region_id = target
            .selected_region_id()
            .expect("authored property targets select one named region")
            .to_owned();
        let entities = self
            .resolved_region(&region_id, expected_role)?
            .entity_ids
            .clone();
        target.bind_resolved_entities(entities).map_err(|error| {
            AnalysisModelRealizationError::InvalidRegionResolution(error.to_string())
        })?;
        Ok(target)
    }

    pub(super) fn shell_property_target(
        &self,
        target: &AuthoredRegionSource,
    ) -> Result<PropertyRegionTarget, AnalysisModelRealizationError> {
        let mut target = PropertyRegionTarget::shell(self.region_id(target)?);
        let region_id = target
            .selected_region_id()
            .expect("authored layup targets select one named region")
            .to_owned();
        let entities = self
            .resolved_region(&region_id, GeometryRegionRole::Face)?
            .entity_ids
            .clone();
        target.bind_resolved_entities(entities).map_err(|error| {
            AnalysisModelRealizationError::InvalidRegionResolution(error.to_string())
        })?;
        Ok(target)
    }

    fn region_id(
        &self,
        source: &AuthoredRegionSource,
    ) -> Result<String, AnalysisModelRealizationError> {
        let AuthoredRegionSource::Geometry(region) = source else {
            return Err(AnalysisModelRealizationError::UnboundRegion);
        };
        if region.module_path != self.geometry.module_path
            || region.geometry_id != self.geometry.geometry_id
        {
            return Err(AnalysisModelRealizationError::RegionFromWrongGeometry {
                module_path: region.module_path.clone(),
                geometry_id: region.geometry_id.clone(),
            });
        }
        Ok(region.region_id.clone())
    }

    pub fn region_target(
        &self,
        source: &AuthoredRegionSource,
        cardinality: TargetCardinality,
    ) -> Result<RegionTarget, AnalysisModelRealizationError> {
        let AuthoredRegionSource::Geometry(region) = source else {
            return Err(AnalysisModelRealizationError::UnboundRegion);
        };
        let region_id = self.region_id(source)?;
        let (entity_dimension, expected_topology) = match region.role {
            GeometryRegionRole::Vertex => (EntityDimension::Point, ExpectedTopology::MeshNode),
            GeometryRegionRole::Edge => (EntityDimension::Edge, ExpectedTopology::GeometryEntity),
            GeometryRegionRole::Face => (EntityDimension::Face, ExpectedTopology::BoundaryFace),
            GeometryRegionRole::Solid => (EntityDimension::Volume, ExpectedTopology::MeshElement),
            GeometryRegionRole::Cell => (EntityDimension::Volume, ExpectedTopology::Cell),
        };
        let mut target =
            RegionTarget::new(region_id, entity_dimension, expected_topology, cardinality);
        target
            .bind_resolved_entities(
                self.resolved_region(&target.region_id, region.role)?
                    .entity_ids
                    .clone(),
            )
            .map_err(|error| {
                AnalysisModelRealizationError::InvalidRegionResolution(error.to_string())
            })?;
        Ok(target)
    }

    pub fn bind_existing_region_target(
        &self,
        target: &mut RegionTarget,
    ) -> Result<(), AnalysisModelRealizationError> {
        let expected_role = match (target.entity_dimension, target.expected_topology) {
            (EntityDimension::Point, ExpectedTopology::MeshNode) => GeometryRegionRole::Vertex,
            (EntityDimension::Edge, ExpectedTopology::GeometryEntity) => GeometryRegionRole::Edge,
            (EntityDimension::Face, ExpectedTopology::BoundaryFace) => GeometryRegionRole::Face,
            (EntityDimension::Volume, ExpectedTopology::MeshElement) => GeometryRegionRole::Solid,
            (EntityDimension::Volume, ExpectedTopology::Cell) => GeometryRegionRole::Cell,
            _ => {
                return Err(AnalysisModelRealizationError::InvalidRegionResolution(
                    format!(
                        "region `{}` has incompatible dimension/topology intent",
                        target.region_id
                    ),
                ));
            }
        };
        target
            .bind_resolved_entities(
                self.resolved_region(&target.region_id, expected_role)?
                    .entity_ids
                    .clone(),
            )
            .map_err(|error| {
                AnalysisModelRealizationError::InvalidRegionResolution(error.to_string())
            })
    }

    fn resolved_region(
        &self,
        region_id: &str,
        expected_role: GeometryRegionRole,
    ) -> Result<&runmat_geometry_core::ResolvedGeometryRegion, AnalysisModelRealizationError> {
        let resolved = self.regions.get(region_id).ok_or_else(|| {
            AnalysisModelRealizationError::MissingResolvedRegion(region_id.to_owned())
        })?;
        if resolved.role != expected_role {
            return Err(AnalysisModelRealizationError::InvalidRegionResolution(
                format!(
                    "region `{region_id}` was authored as {expected_role:?} but materialized as {:?}",
                    resolved.role
                ),
            ));
        }
        Ok(resolved)
    }
}

use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::{GeometryDefinitionRef, GeometryRegionRole};

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
                let material_id = qualified_material_id(reference);
                assignments.push(MaterialAssignment {
                    target: targets.property_target(&assignment.target)?,
                    expected_material_id: material_id.clone(),
                    assigned_material_id: material_id,
                    confidence: EvidenceConfidence::Verified,
                });
            }
        }
        let materials = used
            .into_iter()
            .map(|reference| {
                let mut material = catalog[&reference].clone();
                material.material_id = qualified_material_id(&reference);
                material
            })
            .collect();
        Ok(Self {
            materials,
            assignments,
        })
    }
}

fn qualified_material_id(reference: &AuthoredMaterialRef) -> String {
    format!("{}::{}", reference.module_path, reference.material_id)
}

pub(super) struct TargetResolver<'a> {
    geometry: &'a GeometryDefinitionRef,
}

impl<'a> TargetResolver<'a> {
    pub fn new(geometry: &'a GeometryDefinitionRef) -> Self {
        Self { geometry }
    }

    fn property_target(
        &self,
        target: &AuthoredPropertyTarget,
    ) -> Result<PropertyRegionTarget, AnalysisModelRealizationError> {
        Ok(match target {
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
        })
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
        Ok(RegionTarget {
            region_id,
            entity_dimension,
            expected_topology,
            cardinality,
        })
    }
}

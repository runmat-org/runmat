//! Borrowed outputs of the single property-owner selector. Profiling and materialization consume
//! these same records; neither consumer is allowed to select a different constitutive owner.
use std::{collections::BTreeMap, sync::Arc};

use runmat_analysis_core::{
    BeamSectionModel, HomogeneousShellSectionModel, LaminatePlyModel, LaminateShellSectionModel,
    MaterialAssignment, MaterialModel,
};
use runmat_meshing_core::StableDigest;
use serde::{ser::SerializeStruct, Serialize, Serializer};

use super::{
    CompiledBeamProperty, CompiledElementMaterial, CompiledLaminatePly, CompiledPropertyAudit,
    CompiledPropertyAuditOwner, CompiledShellProperty, CompiledShellPropertyOwner,
    PropertyCompilationError,
};

#[derive(Clone, Copy)]
pub(super) enum SelectedProperty<'a> {
    Cell {
        element_id: u64,
        identity: StableDigest,
        material: &'a MaterialModel,
        assignment_index: usize,
        assignment: &'a MaterialAssignment,
    },
    Solid {
        element_id: u64,
        identity: StableDigest,
        material: &'a MaterialModel,
        assignment_index: usize,
        assignment: &'a MaterialAssignment,
    },
    Beam {
        element_id: &'a str,
        identity: StableDigest,
        material: &'a MaterialModel,
        assignment_index: usize,
        assignment: &'a MaterialAssignment,
        section: &'a BeamSectionModel,
        reference_axis: [f64; 3],
    },
    Shell {
        element_id: &'a str,
        identity: StableDigest,
        material: &'a MaterialModel,
        assignment_index: usize,
        assignment: &'a MaterialAssignment,
        section: &'a HomogeneousShellSectionModel,
        reference_axis: [f64; 3],
    },
    Laminate {
        element_id: &'a str,
        identity: StableDigest,
        section: &'a LaminateShellSectionModel,
        reference_axis: [f64; 3],
        materials: &'a BTreeMap<&'a str, &'a MaterialModel>,
    },
}

// Resource projections include each owned occurrence, including the duplicated section/compiled
// ply definitions and audit strings. These are not new canonical property-identity contracts.
impl Serialize for SelectedProperty<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let fields = match self {
            Self::Solid { .. } | Self::Cell { .. } => 4,
            Self::Beam { .. } => 6,
            _ => 7,
        };
        let mut record = serializer.serialize_struct("SelectedProperty", fields)?;
        match *self {
            Self::Solid {
                element_id,
                identity,
                material,
                assignment_index,
                assignment,
            }
            | Self::Cell {
                element_id,
                identity,
                material,
                assignment_index,
                assignment,
            } => {
                record.serialize_field("element_id", &element_id)?;
                record.serialize_field("element_identity", &identity)?;
                record.serialize_field("material", material)?;
                record.serialize_field(
                    "audit",
                    &AssignmentAudit {
                        identity,
                        assignment_index,
                        assignment,
                    },
                )?;
            }
            Self::Beam {
                element_id,
                identity,
                material,
                assignment_index,
                assignment,
                section,
                reference_axis,
            } => {
                record.serialize_field("element_id", element_id)?;
                record.serialize_field("element_identity", &identity)?;
                record.serialize_field("material", material)?;
                record.serialize_field("section", section)?;
                record.serialize_field("reference_axis", &reference_axis)?;
                record.serialize_field(
                    "audit",
                    &AssignmentAudit {
                        identity,
                        assignment_index,
                        assignment,
                    },
                )?;
            }
            Self::Shell {
                element_id,
                identity,
                material,
                assignment_index,
                assignment,
                section,
                reference_axis,
            } => {
                record.serialize_field("element_id", element_id)?;
                record.serialize_field("element_identity", &identity)?;
                record.serialize_field("owner_kind", "homogeneous")?;
                record.serialize_field("material", material)?;
                record.serialize_field("section", section)?;
                record.serialize_field("reference_axis", &reference_axis)?;
                record.serialize_field(
                    "audit",
                    &AssignmentAudit {
                        identity,
                        assignment_index,
                        assignment,
                    },
                )?;
            }
            Self::Laminate {
                element_id,
                identity,
                section,
                reference_axis,
                materials,
            } => {
                record.serialize_field("element_id", element_id)?;
                record.serialize_field("element_identity", &identity)?;
                record.serialize_field("owner_kind", "laminate")?;
                record.serialize_field("section", section)?;
                record.serialize_field("reference_axis", &reference_axis)?;
                record.serialize_field("plies", &PlyViews { section, materials })?;
                record.serialize_field("audit", &LaminateAudit { identity, section })?;
            }
        }
        record.end()
    }
}

struct AssignmentAudit<'a> {
    identity: StableDigest,
    assignment_index: usize,
    assignment: &'a MaterialAssignment,
}
impl Serialize for AssignmentAudit<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut record = serializer.serialize_struct("AssignmentAudit", 6)?;
        record.serialize_field("element_identity", &self.identity)?;
        record.serialize_field("kind", "material_assignment")?;
        record.serialize_field("assignment_index", &self.assignment_index)?;
        record.serialize_field(
            "expected_material_id",
            &self.assignment.expected_material_id,
        )?;
        record.serialize_field(
            "assigned_material_id",
            &self.assignment.assigned_material_id,
        )?;
        record.serialize_field("confidence", &self.assignment.confidence)?;
        record.end()
    }
}

#[derive(Serialize)]
struct PlyView<'a> {
    definition: &'a LaminatePlyModel,
    material: &'a MaterialModel,
}
struct PlyViews<'a> {
    section: &'a LaminateShellSectionModel,
    materials: &'a BTreeMap<&'a str, &'a MaterialModel>,
}
impl Serialize for PlyViews<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::{Error, SerializeSeq};
        let mut sequence = serializer.serialize_seq(Some(self.section.plies.len()))?;
        for definition in &self.section.plies {
            let material = self
                .materials
                .get(definition.material_id.as_str())
                .ok_or_else(|| S::Error::custom("validated laminate material is absent"))?;
            sequence.serialize_element(&PlyView {
                definition,
                material,
            })?;
        }
        sequence.end()
    }
}
struct LaminateAudit<'a> {
    identity: StableDigest,
    section: &'a LaminateShellSectionModel,
}
impl Serialize for LaminateAudit<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut record = serializer.serialize_struct("LaminateAudit", 4)?;
        record.serialize_field("element_identity", &self.identity)?;
        record.serialize_field("kind", "laminate")?;
        record.serialize_field("section_id", &self.section.section_id)?;
        record.serialize_field("ordered_ply_material_ids", &PlyIds(&self.section.plies))?;
        record.end()
    }
}
struct PlyIds<'a>(&'a [LaminatePlyModel]);
impl Serialize for PlyIds<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(|ply| &ply.material_id))
    }
}

#[derive(Default)]
pub(super) struct OwnedProperties {
    pub cells: BTreeMap<StableDigest, CompiledElementMaterial>,
    pub solids: BTreeMap<StableDigest, CompiledElementMaterial>,
    pub beams: BTreeMap<StableDigest, CompiledBeamProperty>,
    pub shells: BTreeMap<StableDigest, CompiledShellProperty>,
    pub audit: Vec<CompiledPropertyAudit>,
    /// One shared owner per selected source material, keyed by its address in the model.
    shared_materials: std::collections::HashMap<usize, Arc<MaterialModel>>,
    /// One shared pair of material-id owners per source assignment, keyed by its index.
    shared_assignment_ids: std::collections::HashMap<usize, (Arc<str>, Arc<str>)>,
}

impl OwnedProperties {
    fn shared_material(&mut self, material: &MaterialModel) -> Arc<MaterialModel> {
        self.shared_materials
            .entry(material as *const MaterialModel as usize)
            .or_insert_with(|| Arc::new(material.clone()))
            .clone()
    }
}

impl SelectedProperty<'_> {
    pub(super) fn materialize(
        self,
        output: &mut OwnedProperties,
    ) -> Result<(), PropertyCompilationError> {
        let is_cell = matches!(self, Self::Cell { .. });
        match self {
            Self::Solid {
                element_id,
                identity,
                material,
                assignment_index,
                assignment,
            }
            | Self::Cell {
                element_id,
                identity,
                material,
                assignment_index,
                assignment,
            } => {
                let material = output.shared_material(material);
                let owners = if is_cell {
                    &mut output.cells
                } else {
                    &mut output.solids
                };
                owners.insert(
                    identity,
                    CompiledElementMaterial {
                        element_id,
                        element_identity: identity,
                        material,
                    },
                );
                let audit = output.assignment_audit(identity, assignment_index, assignment)?;
                output.audit.push(audit);
            }
            Self::Beam {
                element_id,
                identity,
                material,
                assignment_index,
                assignment,
                section,
                reference_axis,
            } => {
                output.beams.insert(
                    identity,
                    CompiledBeamProperty {
                        element_id: element_id.into(),
                        element_identity: identity,
                        material: material.clone(),
                        section: section.clone(),
                        reference_axis,
                    },
                );
                let audit = output.assignment_audit(identity, assignment_index, assignment)?;
                output.audit.push(audit);
            }
            Self::Shell {
                element_id,
                identity,
                material,
                assignment_index,
                assignment,
                section,
                reference_axis,
            } => {
                output.shells.insert(
                    identity,
                    CompiledShellProperty {
                        element_id: element_id.into(),
                        element_identity: identity,
                        reference_axis,
                        owner: CompiledShellPropertyOwner::Homogeneous {
                            material: Box::new(material.clone()),
                            section: section.clone(),
                        },
                    },
                );
                let audit = output.assignment_audit(identity, assignment_index, assignment)?;
                output.audit.push(audit);
            }
            Self::Laminate {
                element_id,
                identity,
                section,
                reference_axis,
                materials,
            } => {
                let mut plies = Vec::new();
                plies
                    .try_reserve_exact(section.plies.len())
                    .map_err(|error| PropertyCompilationError::Resource(error.to_string()))?;
                for definition in &section.plies {
                    let material =
                        materials
                            .get(definition.material_id.as_str())
                            .ok_or_else(|| PropertyCompilationError::UnknownPlyMaterial {
                                section_id: section.section_id.clone(),
                                material_id: definition.material_id.clone(),
                            })?;
                    plies.push(CompiledLaminatePly {
                        definition: definition.clone(),
                        material: (*material).clone(),
                    });
                }
                output.shells.insert(
                    identity,
                    CompiledShellProperty {
                        element_id: element_id.into(),
                        element_identity: identity,
                        reference_axis,
                        owner: CompiledShellPropertyOwner::Laminate {
                            section: section.clone(),
                            plies,
                        },
                    },
                );
                output.audit.push(CompiledPropertyAudit {
                    element_identity: identity,
                    owner: CompiledPropertyAuditOwner::Laminate {
                        section_id: section.section_id.clone(),
                        ordered_ply_material_ids: section
                            .plies
                            .iter()
                            .map(|ply| ply.material_id.clone())
                            .collect(),
                    },
                });
            }
        }
        Ok(())
    }
}

impl OwnedProperties {
    fn assignment_audit(
        &mut self,
        identity: StableDigest,
        index: usize,
        assignment: &MaterialAssignment,
    ) -> Result<CompiledPropertyAudit, PropertyCompilationError> {
        let (expected_material_id, assigned_material_id) = self
            .shared_assignment_ids
            .entry(index)
            .or_insert_with(|| {
                (
                    Arc::from(assignment.expected_material_id.as_str()),
                    Arc::from(assignment.assigned_material_id.as_str()),
                )
            })
            .clone();
        assignment_audit(
            identity,
            index,
            expected_material_id,
            assigned_material_id,
            assignment,
        )
    }
}

fn assignment_audit(
    identity: StableDigest,
    index: usize,
    expected_material_id: Arc<str>,
    assigned_material_id: Arc<str>,
    assignment: &MaterialAssignment,
) -> Result<CompiledPropertyAudit, PropertyCompilationError> {
    Ok(CompiledPropertyAudit {
        element_identity: identity,
        owner: CompiledPropertyAuditOwner::MaterialAssignment {
            assignment_index: u64::try_from(index).map_err(|_| {
                PropertyCompilationError::Identity("assignment index exceeds u64".into())
            })?,
            expected_material_id,
            assigned_material_id,
            confidence: assignment.confidence,
        },
    })
}

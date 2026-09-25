use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AnalysisModel, MaterialAssignment, MaterialCapability, MaterialModel, PropertyRegionScope,
    PropertyRegionTarget, ShellSectionModel, StructuralElementKind,
};
use runmat_execution::Digest;
use runmat_meshing_core::{
    CanonicalMeshingContract, SolverMeshArtifact, SolverVolumeElement, StableDigest,
};
use thiserror::Error;

use crate::physics::structural::topology::{explicit_element_identity, explicit_topology_digest};
use crate::properties::contract::CompiledPropertySource;

use super::{
    CompiledBeamProperty, CompiledElementMaterial, CompiledLaminatePly, CompiledPropertyAudit,
    CompiledPropertyAuditOwner, CompiledPropertyMap, CompiledShellProperty,
    CompiledShellPropertyOwner,
};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PropertyCompilationError {
    #[error("compiled property map received an invalid solver mesh: {0}")]
    InvalidMesh(String),
    #[error("model geometry revision {model_revision} does not match mesh geometry revision {mesh_revision}")]
    GeometryRevisionMismatch {
        model_revision: u32,
        mesh_revision: u64,
    },
    #[error("material id `{material_id}` is duplicated")]
    DuplicateMaterial { material_id: String },
    #[error("material assignment {assignment_index} references absent assigned material `{material_id}`")]
    UnknownAssignedMaterial {
        assignment_index: usize,
        material_id: String,
    },
    #[error("{role} element `{element_id}` in region `{region_id}` requires exactly one compatible material assignment; found {assignment_count}")]
    InvalidCoverage {
        role: &'static str,
        element_id: String,
        region_id: String,
        assignment_count: usize,
    },
    #[error(
        "{role} material assignment {assignment_index} for `{scope}` selects no compatible element"
    )]
    UnusedAssignment {
        role: &'static str,
        assignment_index: usize,
        scope: String,
    },
    #[error("explicit structural topology is invalid: {0}")]
    InvalidExplicitTopology(String),
    #[error(
        "{role} element `{element_id}` references missing or duplicate section `{section_id}`"
    )]
    InvalidSection {
        role: &'static str,
        element_id: String,
        section_id: String,
    },
    #[error("laminate section `{section_id}` references absent ply material `{material_id}`")]
    UnknownPlyMaterial {
        section_id: String,
        material_id: String,
    },
    #[error("laminate shell element `{element_id}` must not also have a homogeneous material assignment; found {assignment_count}")]
    LaminateMaterialOverlap {
        element_id: String,
        assignment_count: usize,
    },
    #[error("{role} element `{element_id}` uses material `{material_id}` without required `{capability:?}` capability")]
    MissingCapability {
        role: &'static str,
        element_id: String,
        material_id: String,
        capability: MaterialCapability,
    },
    #[error("compiled property identity encoding failed: {0}")]
    Identity(String),
}

type Assignment<'a> = (usize, &'a MaterialAssignment, &'a MaterialModel);

pub fn compile_solid_property_map(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<CompiledPropertyMap, PropertyCompilationError> {
    compile_property_map(model, Some(mesh))
}

/// Compile the shared property map and prove that every solid owner is usable by structural
/// operators before any element contribution is assembled. The generic map remains usable by
/// thermal and other formulations with different constitutive requirements.
pub fn compile_structural_property_map(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<CompiledPropertyMap, PropertyCompilationError> {
    let compiled = compile_property_map(model, mesh)?;
    compiled.validate_structural_solid_capabilities()?;
    Ok(compiled)
}

/// Compile ownership and require the constitutive inputs used by thermal solid operators before
/// assembly starts. This does not impose thermal requirements on other physics families.
pub fn compile_thermal_property_map(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<CompiledPropertyMap, PropertyCompilationError> {
    let compiled = compile_property_map(model, mesh)?;
    compiled.validate_thermal_solid_capabilities()?;
    Ok(compiled)
}

impl CompiledPropertyMap {
    /// Validate the constitutive view needed by structural solid operators without selecting
    /// owners again. Study identity checks use this on already-compiled property maps.
    pub fn validate_structural_solid_capabilities(&self) -> Result<(), PropertyCompilationError> {
        for entry in self.solid_materials() {
            require_capabilities(
                "structural solid",
                &entry.element_id.to_string(),
                &entry.material,
                &[
                    MaterialCapability::IsotropicLinearElasticity,
                    MaterialCapability::MassDensity,
                ],
            )?;
        }
        Ok(())
    }

    /// Validate an already-compiled map without reselecting material owners.
    pub fn validate_thermal_solid_capabilities(&self) -> Result<(), PropertyCompilationError> {
        for entry in self.solid_materials() {
            require_capabilities(
                "thermal solid",
                &entry.element_id.to_string(),
                &entry.material,
                &[MaterialCapability::Thermal, MaterialCapability::MassDensity],
            )?;
        }
        Ok(())
    }
}

/// Compile every material-bearing formulation whose topology is present.
pub fn compile_property_map(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<CompiledPropertyMap, PropertyCompilationError> {
    if let Some(mesh) = mesh {
        mesh.validate_canonical()
            .map_err(|error| PropertyCompilationError::InvalidMesh(error.to_string()))?;
        if u64::from(model.geometry_revision) != mesh.geometry.geometry_revision {
            return Err(PropertyCompilationError::GeometryRevisionMismatch {
                model_revision: model.geometry_revision,
                mesh_revision: mesh.geometry.geometry_revision,
            });
        }
    }
    let materials = material_inventory(model)?;
    let mut audit = Vec::new();
    let solid_materials = if let Some(mesh) = mesh {
        compile_solids(model, mesh, &materials, &mut audit)?
    } else {
        BTreeMap::new()
    };
    let (explicit_topology_digest, beam_properties, shell_properties) =
        compile_explicit(model, &materials, &mut audit)?;

    CompiledPropertyMap::new(
        CompiledPropertySource {
            geometry_id: model.geometry_id.clone(),
            geometry_revision: model.geometry_revision,
            mesh_digest: mesh.map(|mesh| mesh.canonical_digest),
            explicit_topology_digest,
        },
        solid_materials,
        beam_properties,
        shell_properties,
        audit,
    )
    .map_err(PropertyCompilationError::Identity)
}

fn compile_solids(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    materials: &BTreeMap<&str, &MaterialModel>,
    audit: &mut Vec<CompiledPropertyAudit>,
) -> Result<BTreeMap<StableDigest, CompiledElementMaterial>, PropertyCompilationError> {
    let assignments = assignments_for(model, materials, |target| {
        matches!(target, PropertyRegionTarget::Solid(_))
    })?;
    let mut matched = BTreeSet::new();
    let mut compiled = BTreeMap::new();
    for element in &mesh.topology.volume_elements {
        let matching = assignments
            .iter()
            .filter(|(_, assignment, _)| solid_scope_matches(assignment.target.scope(), element))
            .collect::<Vec<_>>();
        let (assignment_index, assignment, material) = require_one(
            "solid",
            element.element_id.to_string(),
            &element.region_id.source_topology_id,
            matching,
        )?;
        matched.insert(*assignment_index);
        compiled.insert(
            element.stable_identity,
            CompiledElementMaterial {
                element_id: element.element_id,
                element_identity: element.stable_identity,
                material: (*material).clone(),
            },
        );
        audit.push(audit_entry(
            element.stable_identity,
            *assignment_index,
            assignment,
        )?);
    }
    reject_unmatched("solid", &assignments, &matched)?;
    Ok(compiled)
}

type ExplicitMaps = (
    Option<Digest>,
    BTreeMap<StableDigest, CompiledBeamProperty>,
    BTreeMap<StableDigest, CompiledShellProperty>,
);

fn compile_explicit(
    model: &AnalysisModel,
    materials: &BTreeMap<&str, &MaterialModel>,
    audit: &mut Vec<CompiledPropertyAudit>,
) -> Result<ExplicitMaps, PropertyCompilationError> {
    let Some(structural) = model.structural.as_ref() else {
        return Ok((None, BTreeMap::new(), BTreeMap::new()));
    };
    let topology_digest = explicit_topology_digest(structural)
        .map_err(|error| PropertyCompilationError::InvalidExplicitTopology(error.to_string()))?;
    let beams = assignments_for(model, materials, |target| {
        matches!(target, PropertyRegionTarget::Beam(_))
    })?;
    let shells = assignments_for(model, materials, |target| {
        matches!(target, PropertyRegionTarget::Shell(_))
    })?;
    let mut matched_beams = BTreeSet::new();
    let mut matched_shells = BTreeSet::new();
    let mut beam_properties = BTreeMap::new();
    let mut shell_properties = BTreeMap::new();

    for element in &structural.elements {
        let identity = explicit_element_identity(
            topology_digest,
            &element.element_id,
            &element.region_id,
            &element.kind,
        )
        .map_err(|error| PropertyCompilationError::InvalidExplicitTopology(error.to_string()))?;
        match &element.kind {
            StructuralElementKind::Beam(beam) => {
                let matching = beams
                    .iter()
                    .filter(|(_, assignment, _)| {
                        explicit_scope_matches(assignment.target.scope(), &element.region_id)
                    })
                    .collect::<Vec<_>>();
                let (assignment_index, assignment, material) = require_one(
                    "beam",
                    element.element_id.clone(),
                    &element.region_id,
                    matching,
                )?;
                require_capabilities(
                    "beam",
                    &element.element_id,
                    material,
                    &[
                        MaterialCapability::IsotropicLinearElasticity,
                        MaterialCapability::MassDensity,
                    ],
                )?;
                let sections = structural
                    .beam_sections
                    .iter()
                    .filter(|section| section.section_id == beam.section_id)
                    .collect::<Vec<_>>();
                let [section] = sections.as_slice() else {
                    return Err(PropertyCompilationError::InvalidSection {
                        role: "beam",
                        element_id: element.element_id.clone(),
                        section_id: beam.section_id.clone(),
                    });
                };
                matched_beams.insert(*assignment_index);
                beam_properties.insert(
                    identity,
                    CompiledBeamProperty {
                        element_id: element.element_id.clone(),
                        element_identity: identity,
                        material: (*material).clone(),
                        section: (*section).clone(),
                        reference_axis: beam.reference_axis,
                    },
                );
                audit.push(audit_entry(identity, *assignment_index, assignment)?);
            }
            StructuralElementKind::Shell(shell) => {
                let sections = structural
                    .shell_sections
                    .iter()
                    .filter(|section| section.section_id() == shell.section_id)
                    .collect::<Vec<_>>();
                let [section] = sections.as_slice() else {
                    return Err(PropertyCompilationError::InvalidSection {
                        role: "shell",
                        element_id: element.element_id.clone(),
                        section_id: shell.section_id.clone(),
                    });
                };
                let matching = shells
                    .iter()
                    .filter(|(_, assignment, _)| {
                        explicit_scope_matches(assignment.target.scope(), &element.region_id)
                    })
                    .collect::<Vec<_>>();
                let owner = match section {
                    ShellSectionModel::Homogeneous(section) => {
                        let (assignment_index, assignment, material) = require_one(
                            "shell",
                            element.element_id.clone(),
                            &element.region_id,
                            matching,
                        )?;
                        require_capabilities(
                            "homogeneous shell",
                            &element.element_id,
                            material,
                            &[
                                MaterialCapability::IsotropicLinearElasticity,
                                MaterialCapability::MassDensity,
                            ],
                        )?;
                        matched_shells.insert(*assignment_index);
                        audit.push(audit_entry(identity, *assignment_index, assignment)?);
                        CompiledShellPropertyOwner::Homogeneous {
                            material: Box::new((*material).clone()),
                            section: section.clone(),
                        }
                    }
                    ShellSectionModel::Laminate(section) => {
                        if !matching.is_empty() {
                            return Err(PropertyCompilationError::LaminateMaterialOverlap {
                                element_id: element.element_id.clone(),
                                assignment_count: matching.len(),
                            });
                        }
                        let plies = section
                            .plies
                            .iter()
                            .map(|definition| {
                                let material = materials
                                    .get(definition.material_id.as_str())
                                    .copied()
                                    .ok_or_else(|| {
                                        PropertyCompilationError::UnknownPlyMaterial {
                                            section_id: section.section_id.clone(),
                                            material_id: definition.material_id.clone(),
                                        }
                                    })?;
                                require_capabilities(
                                    "laminate ply",
                                    &element.element_id,
                                    material,
                                    &[
                                        MaterialCapability::LaminaElasticity,
                                        MaterialCapability::MassDensity,
                                    ],
                                )?;
                                Ok(CompiledLaminatePly {
                                    definition: definition.clone(),
                                    material: material.clone(),
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        audit.push(CompiledPropertyAudit {
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
                        CompiledShellPropertyOwner::Laminate {
                            section: section.clone(),
                            plies,
                        }
                    }
                };
                shell_properties.insert(
                    identity,
                    CompiledShellProperty {
                        element_id: element.element_id.clone(),
                        element_identity: identity,
                        reference_axis: shell.reference_axis,
                        owner,
                    },
                );
            }
        }
    }
    reject_unmatched("beam", &beams, &matched_beams)?;
    reject_unmatched("shell", &shells, &matched_shells)?;
    Ok((Some(topology_digest), beam_properties, shell_properties))
}

fn require_capabilities(
    role: &'static str,
    element_id: &str,
    material: &MaterialModel,
    capabilities: &[MaterialCapability],
) -> Result<(), PropertyCompilationError> {
    if let Some(capability) = capabilities
        .iter()
        .copied()
        .find(|capability| !material.has_capability(*capability))
    {
        return Err(PropertyCompilationError::MissingCapability {
            role,
            element_id: element_id.into(),
            material_id: material.material_id.clone(),
            capability,
        });
    }
    Ok(())
}

fn material_inventory(
    model: &AnalysisModel,
) -> Result<BTreeMap<&str, &MaterialModel>, PropertyCompilationError> {
    let mut materials = BTreeMap::new();
    for material in &model.materials {
        if materials
            .insert(material.material_id.as_str(), material)
            .is_some()
        {
            return Err(PropertyCompilationError::DuplicateMaterial {
                material_id: material.material_id.clone(),
            });
        }
    }
    Ok(materials)
}

fn assignments_for<'a>(
    model: &'a AnalysisModel,
    materials: &BTreeMap<&'a str, &'a MaterialModel>,
    include: impl Fn(&PropertyRegionTarget) -> bool,
) -> Result<Vec<Assignment<'a>>, PropertyCompilationError> {
    model
        .material_assignments
        .iter()
        .enumerate()
        .filter(|(_, assignment)| include(&assignment.target))
        .map(|(index, assignment)| {
            materials
                .get(assignment.assigned_material_id.as_str())
                .copied()
                .map(|material| (index, assignment, material))
                .ok_or_else(|| PropertyCompilationError::UnknownAssignedMaterial {
                    assignment_index: index,
                    material_id: assignment.assigned_material_id.clone(),
                })
        })
        .collect()
}

fn require_one<'a>(
    role: &'static str,
    element_id: String,
    region_id: &str,
    matching: Vec<&'a Assignment<'a>>,
) -> Result<&'a Assignment<'a>, PropertyCompilationError> {
    if matching.len() != 1 {
        return Err(PropertyCompilationError::InvalidCoverage {
            role,
            element_id,
            region_id: region_id.to_string(),
            assignment_count: matching.len(),
        });
    }
    Ok(matching[0])
}

fn reject_unmatched(
    role: &'static str,
    assignments: &[Assignment<'_>],
    matched: &BTreeSet<usize>,
) -> Result<(), PropertyCompilationError> {
    if let Some((index, assignment, _)) = assignments
        .iter()
        .find(|(index, _, _)| !matched.contains(index))
    {
        return Err(PropertyCompilationError::UnusedAssignment {
            role,
            assignment_index: *index,
            scope: scope_label(assignment.target.scope()).to_string(),
        });
    }
    Ok(())
}

fn audit_entry(
    identity: StableDigest,
    assignment_index: usize,
    assignment: &MaterialAssignment,
) -> Result<CompiledPropertyAudit, PropertyCompilationError> {
    Ok(CompiledPropertyAudit {
        element_identity: identity,
        owner: CompiledPropertyAuditOwner::MaterialAssignment {
            assignment_index: u64::try_from(assignment_index).map_err(|_| {
                PropertyCompilationError::Identity("assignment index exceeds u64".into())
            })?,
            expected_material_id: assignment.expected_material_id.clone(),
            assigned_material_id: assignment.assigned_material_id.clone(),
            confidence: assignment.confidence,
        },
    })
}

fn solid_scope_matches(scope: &PropertyRegionScope, element: &SolverVolumeElement) -> bool {
    match scope {
        PropertyRegionScope::All => true,
        PropertyRegionScope::Region(selection) => {
            selection.matches_persistent_entity(&element.region_id)
                || element
                    .provenance
                    .iter()
                    .any(|entity| selection.matches_persistent_entity(entity))
        }
    }
}

fn explicit_scope_matches(scope: &PropertyRegionScope, region_id: &str) -> bool {
    match scope {
        PropertyRegionScope::All => true,
        PropertyRegionScope::Region(selection) => selection.region_id() == region_id,
    }
}

fn scope_label(scope: &PropertyRegionScope) -> &str {
    scope.selected_region_id().unwrap_or("<all>")
}

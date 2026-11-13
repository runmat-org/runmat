use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AnalysisModel, MaterialAssignment, MaterialCapability, MaterialModel, PropertyRegionScope,
    PropertyRegionTarget, ShellSectionModel, StructuralElementKind,
};
use runmat_execution::Digest;
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact, SolverVolumeElement};
use thiserror::Error;

use super::{
    preparation::explicit::PreparedExplicitTopology, selection::SelectedProperty,
    CompiledPropertyMap,
};

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PropertyCompilationError {
    #[error(
        "combined Cell and Solid volume material roles are outside the admitted property envelope"
    )]
    UnsupportedMixedVolumeRoles,
    #[error("Cell volume ownership cannot satisfy solid constitutive capability requirements")]
    CellVolumeForSolidPhysics,
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
    #[error("compiled property resources require {required} bytes, exceeding the {maximum} byte ceiling")]
    ResourceCeiling { required: u64, maximum: u64 },
    #[error("compiled property resource planning failed: {0}")]
    Resource(String),
}

pub(super) type Assignment<'a> = (usize, &'a MaterialAssignment, &'a MaterialModel);

pub fn compile_solid_property_map(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<CompiledPropertyMap, PropertyCompilationError> {
    let compiled = compile_property_map(model, Some(mesh))?;
    // A generic Cell map must not silently satisfy the explicitly Solid API.
    for element in &mesh.topology.volume_elements {
        if compiled.solid_material(element.stable_identity).is_none() {
            return Err(PropertyCompilationError::InvalidCoverage {
                role: "solid",
                element_id: element.element_id.to_string(),
                region_id: element.region_id.source_topology_id.clone(),
                assignment_count: 0,
            });
        }
    }
    Ok(compiled)
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
        if self.cell_materials().len() != 0 {
            return Err(PropertyCompilationError::CellVolumeForSolidPhysics);
        }
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
        if self.cell_materials().len() != 0 {
            return Err(PropertyCompilationError::CellVolumeForSolidPhysics);
        }
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
    super::preparation::PreparedProperties::prepare(model, mesh, u64::MAX)?.materialize()
}

pub(super) fn validate_mesh(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
) -> Result<(), PropertyCompilationError> {
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
    Ok(())
}

/// Resolve ownership once per traversal for both profiling and owned materialization.
/// Selection emits borrowed metadata only; the caller owns allocation and admission policy.
pub(super) fn visit<'a>(
    model: &'a AnalysisModel,
    mesh: Option<&'a SolverMeshArtifact>,
    materials: &'a BTreeMap<&'a str, &'a MaterialModel>,
    explicit: Option<&PreparedExplicitTopology<'a>>,
    visitor: &mut impl FnMut(SelectedProperty<'a>) -> Result<(), PropertyCompilationError>,
) -> Result<Option<Digest>, PropertyCompilationError> {
    match (model.structural.as_ref(), explicit) {
        (None, None) => {}
        (Some(structural), Some(prepared)) if std::ptr::eq(structural, prepared.structural()) => {}
        _ => {
            return Err(PropertyCompilationError::InvalidExplicitTopology(
                "prepared explicit identities do not belong to this structural model".into(),
            ));
        }
    }
    if let Some(mesh) = mesh {
        let fluid_step = model
            .sequence
            .steps
            .iter()
            .any(|step| step.family == runmat_analysis_core::PhysicsFamily::Fluid);
        let has_cell = model
            .material_assignments
            .iter()
            .any(|assignment| matches!(assignment.target, PropertyRegionTarget::Cell(_)));
        let has_solid = model
            .material_assignments
            .iter()
            .any(|assignment| matches!(assignment.target, PropertyRegionTarget::Solid(_)));
        if fluid_step && has_solid {
            return Err(PropertyCompilationError::UnsupportedMixedVolumeRoles);
        }
        if has_cell && has_solid {
            visit_partitioned_volumes(model, mesh, materials, visitor)?;
        } else if fluid_step || has_cell {
            visit_cells(model, mesh, materials, visitor)?;
        } else {
            visit_solids(model, mesh, materials, visitor)?;
        }
    }
    visit_explicit(model, materials, explicit, visitor)
}

fn visit_solids<'a>(
    model: &'a AnalysisModel,
    mesh: &'a SolverMeshArtifact,
    materials: &'a BTreeMap<&'a str, &'a MaterialModel>,
    visitor: &mut impl FnMut(SelectedProperty<'a>) -> Result<(), PropertyCompilationError>,
) -> Result<(), PropertyCompilationError> {
    let assignments = assignments_for(model, materials, |target| {
        matches!(target, PropertyRegionTarget::Solid(_))
    })?;
    let mut matched = BTreeSet::new();
    for element in &mesh.topology.volume_elements {
        let matching = assignments
            .iter()
            .filter(|(_, assignment, _)| solid_scope_matches(assignment.target.scope(), element));
        let (assignment_index, assignment, material) = require_one(
            "solid",
            || element.element_id.to_string(),
            &element.region_id.source_topology_id,
            matching,
        )?;
        matched.insert(*assignment_index);
        visitor(SelectedProperty::Solid {
            element_id: element.element_id,
            identity: element.stable_identity,
            material,
            assignment_index: *assignment_index,
            assignment,
        })?;
    }
    reject_unmatched("solid", &assignments, &matched)?;
    Ok(())
}

/// Cell ownership is explicit authored finite-volume intent, not an alias for Solid.
fn visit_cells<'a>(
    model: &'a AnalysisModel,
    mesh: &'a SolverMeshArtifact,
    materials: &'a BTreeMap<&'a str, &'a MaterialModel>,
    visitor: &mut impl FnMut(SelectedProperty<'a>) -> Result<(), PropertyCompilationError>,
) -> Result<(), PropertyCompilationError> {
    let assignments = assignments_for(model, materials, |target| {
        matches!(target, PropertyRegionTarget::Cell(_))
    })?;
    let mut matched = BTreeSet::new();
    for element in &mesh.topology.volume_elements {
        let matching = assignments
            .iter()
            .filter(|(_, assignment, _)| solid_scope_matches(assignment.target.scope(), element));
        let (assignment_index, assignment, material) = require_one(
            "cell",
            || element.element_id.to_string(),
            &element.region_id.source_topology_id,
            matching,
        )?;
        matched.insert(*assignment_index);
        visitor(SelectedProperty::Cell {
            element_id: element.element_id,
            identity: element.stable_identity,
            material,
            assignment_index: *assignment_index,
            assignment,
        })?;
    }
    reject_unmatched("cell", &assignments, &matched)
}

/// A coupled fluid-solid model partitions the volume: every element is owned by exactly one
/// cell (fluid) or solid assignment, and every assignment owns at least one element.
fn visit_partitioned_volumes<'a>(
    model: &'a AnalysisModel,
    mesh: &'a SolverMeshArtifact,
    materials: &'a BTreeMap<&'a str, &'a MaterialModel>,
    visitor: &mut impl FnMut(SelectedProperty<'a>) -> Result<(), PropertyCompilationError>,
) -> Result<(), PropertyCompilationError> {
    let cells = assignments_for(model, materials, |target| {
        matches!(target, PropertyRegionTarget::Cell(_))
    })?;
    let solids = assignments_for(model, materials, |target| {
        matches!(target, PropertyRegionTarget::Solid(_))
    })?;
    let mut matched_cells = BTreeSet::new();
    let mut matched_solids = BTreeSet::new();
    for element in &mesh.topology.volume_elements {
        let cell_matches = cells
            .iter()
            .filter(|(_, assignment, _)| solid_scope_matches(assignment.target.scope(), element))
            .count();
        let solid_matches = solids
            .iter()
            .filter(|(_, assignment, _)| solid_scope_matches(assignment.target.scope(), element))
            .count();
        if cell_matches + solid_matches != 1 {
            return Err(PropertyCompilationError::InvalidCoverage {
                role: "volume",
                element_id: element.element_id.to_string(),
                region_id: element.region_id.source_topology_id.clone(),
                assignment_count: cell_matches + solid_matches,
            });
        }
        if cell_matches == 1 {
            let (assignment_index, assignment, material) = cells
                .iter()
                .find(|(_, assignment, _)| solid_scope_matches(assignment.target.scope(), element))
                .expect("counted one cell assignment");
            matched_cells.insert(*assignment_index);
            visitor(SelectedProperty::Cell {
                element_id: element.element_id,
                identity: element.stable_identity,
                material,
                assignment_index: *assignment_index,
                assignment,
            })?;
        } else {
            let (assignment_index, assignment, material) = solids
                .iter()
                .find(|(_, assignment, _)| solid_scope_matches(assignment.target.scope(), element))
                .expect("counted one solid assignment");
            matched_solids.insert(*assignment_index);
            visitor(SelectedProperty::Solid {
                element_id: element.element_id,
                identity: element.stable_identity,
                material,
                assignment_index: *assignment_index,
                assignment,
            })?;
        }
    }
    reject_unmatched("cell", &cells, &matched_cells)?;
    reject_unmatched("solid", &solids, &matched_solids)
}

fn visit_explicit<'a>(
    model: &'a AnalysisModel,
    materials: &'a BTreeMap<&'a str, &'a MaterialModel>,
    explicit: Option<&PreparedExplicitTopology<'a>>,
    visitor: &mut impl FnMut(SelectedProperty<'a>) -> Result<(), PropertyCompilationError>,
) -> Result<Option<Digest>, PropertyCompilationError> {
    let Some(explicit) = explicit else {
        return Ok(None);
    };
    let structural = explicit.structural();
    let topology_digest = explicit.topology_digest();
    let beams = assignments_for(model, materials, |target| {
        matches!(target, PropertyRegionTarget::Beam(_))
    })?;
    let shells = assignments_for(model, materials, |target| {
        matches!(target, PropertyRegionTarget::Shell(_))
    })?;
    let mut matched_beams = BTreeSet::new();
    let mut matched_shells = BTreeSet::new();

    for (element, &identity) in structural
        .elements
        .iter()
        .zip(explicit.element_identities())
    {
        match &element.kind {
            StructuralElementKind::Beam(beam) => {
                let matching = beams.iter().filter(|(_, assignment, _)| {
                    explicit_scope_matches(assignment.target.scope(), &element.region_id)
                });
                let (assignment_index, assignment, material) = require_one(
                    "beam",
                    || element.element_id.clone(),
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
                let mut sections = structural
                    .beam_sections
                    .iter()
                    .filter(|section| section.section_id == beam.section_id);
                let section = sections.next();
                let Some(section) = section.filter(|_| sections.next().is_none()) else {
                    return Err(PropertyCompilationError::InvalidSection {
                        role: "beam",
                        element_id: element.element_id.clone(),
                        section_id: beam.section_id.clone(),
                    });
                };
                matched_beams.insert(*assignment_index);
                visitor(SelectedProperty::Beam {
                    element_id: &element.element_id,
                    identity,
                    material,
                    assignment_index: *assignment_index,
                    assignment,
                    section,
                    reference_axis: beam.reference_axis,
                })?;
            }
            StructuralElementKind::Shell(shell) => {
                let mut sections = structural
                    .shell_sections
                    .iter()
                    .filter(|section| section.section_id() == shell.section_id);
                let section = sections.next();
                let Some(section) = section.filter(|_| sections.next().is_none()) else {
                    return Err(PropertyCompilationError::InvalidSection {
                        role: "shell",
                        element_id: element.element_id.clone(),
                        section_id: shell.section_id.clone(),
                    });
                };
                let matching = shells.iter().filter(|(_, assignment, _)| {
                    explicit_scope_matches(assignment.target.scope(), &element.region_id)
                });
                match section {
                    ShellSectionModel::Homogeneous(section) => {
                        let (assignment_index, assignment, material) = require_one(
                            "shell",
                            || element.element_id.clone(),
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
                        visitor(SelectedProperty::Shell {
                            element_id: &element.element_id,
                            identity,
                            material,
                            assignment_index: *assignment_index,
                            assignment,
                            section,
                            reference_axis: shell.reference_axis,
                        })?;
                    }
                    ShellSectionModel::Laminate(section) => {
                        let assignment_count = matching.count();
                        if assignment_count != 0 {
                            return Err(PropertyCompilationError::LaminateMaterialOverlap {
                                element_id: element.element_id.clone(),
                                assignment_count,
                            });
                        }
                        // Validate the complete ply list before exposing an owner to either
                        // sink. Profiling and materialization therefore share every check.
                        for definition in &section.plies {
                            let material = materials
                                .get(definition.material_id.as_str())
                                .copied()
                                .ok_or_else(|| PropertyCompilationError::UnknownPlyMaterial {
                                    section_id: section.section_id.clone(),
                                    material_id: definition.material_id.clone(),
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
                        }
                        visitor(SelectedProperty::Laminate {
                            element_id: &element.element_id,
                            identity,
                            section,
                            reference_axis: shell.reference_axis,
                            materials,
                        })?;
                    }
                }
            }
        }
    }
    reject_unmatched("beam", &beams, &matched_beams)?;
    reject_unmatched("shell", &shells, &matched_shells)?;
    Ok(Some(topology_digest))
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

pub(super) fn material_inventory(
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

fn require_one<'a, 'b>(
    role: &'static str,
    element_id: impl FnOnce() -> String,
    region_id: &str,
    mut matching: impl Iterator<Item = &'b Assignment<'a>>,
) -> Result<&'b Assignment<'a>, PropertyCompilationError> {
    let first = matching.next();
    let remaining = matching.count();
    if remaining == 0 {
        if let Some(assignment) = first {
            return Ok(assignment);
        }
    }
    Err(PropertyCompilationError::InvalidCoverage {
        role,
        element_id: element_id(),
        region_id: region_id.to_string(),
        assignment_count: usize::from(first.is_some()) + remaining,
    })
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

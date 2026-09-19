use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AnalysisModel, MaterialAssignment, MaterialModel, PropertyRegionScope, PropertyRegionTarget,
    StructuralElementKind,
};
use runmat_canonical_codec::{encode_contract, CanonicalLimits};
use runmat_execution::Digest;
use runmat_meshing_core::{
    CanonicalMeshingContract, SolverMeshArtifact, SolverVolumeElement, StableDigest,
};
use serde::Serialize;
use thiserror::Error;

use crate::physics::structural::topology::{explicit_element_identity, explicit_topology_digest};

use super::{
    CompiledElementMaterial, CompiledExplicitElementMaterial, CompiledPropertyAudit,
    CompiledPropertyMap, CompiledPropertyMapIdentity,
};

const SEMANTIC_PREFIX: &[u8] = b"runmat-fea-compiled-properties/v1\0";
const AUDIT_PREFIX: &[u8] = b"runmat-fea-compiled-property-audit/v1\0";
const LIMITS: CanonicalLimits = CanonicalLimits::new(512 * 1024 * 1024, 8_000_000, 4096, 64);

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
    #[error("compiled property identity encoding failed: {0}")]
    Identity(String),
}

#[derive(Serialize)]
struct SemanticContract<'a> {
    geometry_id: &'a str,
    geometry_revision: u32,
    mesh_digest: Option<StableDigest>,
    explicit_topology_digest: Option<Digest>,
    solids: Vec<SolidSemanticElement<'a>>,
    beams: Vec<ExplicitSemanticElement<'a>>,
    shells: Vec<ExplicitSemanticElement<'a>>,
}

#[derive(Serialize)]
struct SolidSemanticElement<'a> {
    element_id: u64,
    element_identity: StableDigest,
    material: &'a MaterialModel,
}

#[derive(Serialize)]
struct ExplicitSemanticElement<'a> {
    element_id: &'a str,
    element_identity: StableDigest,
    material: &'a MaterialModel,
}

type Assignment<'a> = (usize, &'a MaterialAssignment, &'a MaterialModel);

pub fn compile_solid_property_map(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> Result<CompiledPropertyMap, PropertyCompilationError> {
    compile_property_map(model, Some(mesh))
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
    let (explicit_topology_digest, beam_materials, shell_materials) =
        compile_explicit(model, &materials, &mut audit)?;

    let semantic = SemanticContract {
        geometry_id: &model.geometry_id,
        geometry_revision: model.geometry_revision,
        mesh_digest: mesh.map(|mesh| mesh.canonical_digest),
        explicit_topology_digest,
        solids: solid_materials
            .values()
            .map(|entry| SolidSemanticElement {
                element_id: entry.element_id,
                element_identity: entry.element_identity,
                material: &entry.material,
            })
            .collect(),
        beams: beam_materials
            .values()
            .map(|entry| ExplicitSemanticElement {
                element_id: &entry.element_id,
                element_identity: entry.element_identity,
                material: &entry.material,
            })
            .collect(),
        shells: shell_materials
            .values()
            .map(|entry| ExplicitSemanticElement {
                element_id: &entry.element_id,
                element_identity: entry.element_identity,
                material: &entry.material,
            })
            .collect(),
    };
    let semantic_digest = encode_contract(
        SEMANTIC_PREFIX,
        "compiled-property-semantics",
        &semantic,
        LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| PropertyCompilationError::Identity(error.to_string()))?;
    let audit_digest = encode_contract(
        AUDIT_PREFIX,
        "compiled-property-audit",
        &(semantic_digest, &audit),
        LIMITS,
    )
    .map(Digest::sha256)
    .map_err(|error| PropertyCompilationError::Identity(error.to_string()))?;
    Ok(CompiledPropertyMap::new(
        CompiledPropertyMapIdentity {
            geometry_id: model.geometry_id.clone(),
            geometry_revision: model.geometry_revision,
            mesh_digest: mesh.map(|mesh| mesh.canonical_digest),
            explicit_topology_digest,
            semantic_digest,
        },
        audit_digest,
        solid_materials,
        beam_materials,
        shell_materials,
        audit,
    ))
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
    BTreeMap<StableDigest, CompiledExplicitElementMaterial>,
    BTreeMap<StableDigest, CompiledExplicitElementMaterial>,
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
    let mut beam_materials = BTreeMap::new();
    let mut shell_materials = BTreeMap::new();

    for element in &structural.elements {
        let (role, assignments, matched, output) = match &element.kind {
            StructuralElementKind::Beam(_) => {
                ("beam", &beams, &mut matched_beams, &mut beam_materials)
            }
            StructuralElementKind::Shell(_) => {
                ("shell", &shells, &mut matched_shells, &mut shell_materials)
            }
        };
        let matching = assignments
            .iter()
            .filter(|(_, assignment, _)| {
                explicit_scope_matches(assignment.target.scope(), &element.region_id)
            })
            .collect::<Vec<_>>();
        let (assignment_index, assignment, material) = require_one(
            role,
            element.element_id.clone(),
            &element.region_id,
            matching,
        )?;
        matched.insert(*assignment_index);
        let identity = explicit_element_identity(
            topology_digest,
            &element.element_id,
            &element.region_id,
            &element.kind,
        )
        .map_err(|error| PropertyCompilationError::InvalidExplicitTopology(error.to_string()))?;
        output.insert(
            identity,
            CompiledExplicitElementMaterial {
                element_id: element.element_id.clone(),
                element_identity: identity,
                material: (*material).clone(),
            },
        );
        audit.push(audit_entry(identity, *assignment_index, assignment)?);
    }
    reject_unmatched("beam", &beams, &matched_beams)?;
    reject_unmatched("shell", &shells, &matched_shells)?;
    Ok((Some(topology_digest), beam_materials, shell_materials))
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
        assignment_index: u64::try_from(assignment_index).map_err(|_| {
            PropertyCompilationError::Identity("assignment index exceeds u64".into())
        })?,
        expected_material_id: assignment.expected_material_id.clone(),
        assigned_material_id: assignment.assigned_material_id.clone(),
        confidence: assignment.confidence,
    })
}

fn solid_scope_matches(scope: &PropertyRegionScope, element: &SolverVolumeElement) -> bool {
    match scope {
        PropertyRegionScope::All => true,
        PropertyRegionScope::Region(selection) => {
            selection.matches_persistent_entity(&element.region_id)
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

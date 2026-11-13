use runmat_analysis_core::{
    BeamElementModel, MechanicalDampingModel, StructuralElement, StructuralModel,
};

use super::{beam_element_dof_indices, section_from_model, structural_node_index};
use crate::{
    assembly::{
        elements::beam::{
            global_stiffness_matrix, transformation_matrix, BeamElementGeometry, BeamMaterial,
            BEAM_ELEMENT_DOF_COUNT,
        },
        BeamRecoveryElementSummary, StructuralAssemblyError, StructuralDofLayout,
    },
    physics::structural::{
        material::StructuralElementMaterial, topology::explicit_element_identity,
    },
    properties::CompiledPropertyMap,
};

use super::beam_inertia::lumped_beam_mass;

pub(super) struct BeamElementContribution {
    pub dofs: [usize; BEAM_ELEMENT_DOF_COUNT],
    pub stiffness: [[f64; BEAM_ELEMENT_DOF_COUNT]; BEAM_ELEMENT_DOF_COUNT],
    pub mass: [f64; BEAM_ELEMENT_DOF_COUNT],
    pub damping: MechanicalDampingModel,
    pub recovery: BeamRecoveryElementSummary,
}

pub(super) fn compute(
    structural: &StructuralModel,
    nodes: &super::ExplicitNodeIndex,
    element: &StructuralElement,
    beam: &BeamElementModel,
    properties: &CompiledPropertyMap,
    layout: &StructuralDofLayout,
    canonical_node_indices: &[usize],
    topology_digest: runmat_execution::Digest,
) -> Result<BeamElementContribution, StructuralAssemblyError> {
    let explicit_i = structural_node_index(nodes, beam.node_ids[0], element)?;
    let explicit_j = structural_node_index(nodes, beam.node_ids[1], element)?;
    let node_i = canonical_node_indices[explicit_i];
    let node_j = canonical_node_indices[explicit_j];
    let element_identity = explicit_element_identity(
        topology_digest,
        &element.element_id,
        &element.region_id,
        &element.kind,
    )
    .map_err(|error| StructuralAssemblyError::InvalidMaterialCoverage(error.to_string()))?;
    let property = properties.beam_property(element_identity).ok_or_else(|| {
        StructuralAssemblyError::UnassignedExplicitRegion {
            element_id: element.element_id.clone(),
            region_id: element.region_id.clone(),
        }
    })?;
    let section = section_from_model(&property.section);
    let material = StructuralElementMaterial::from_model(&property.material)
        .map_err(|error| StructuralAssemblyError::InvalidMaterialCoverage(error.to_string()))?;
    let beam_material = BeamMaterial {
        youngs_modulus_pa: material.youngs_modulus_pa,
        shear_modulus_pa: material.shear_modulus_pa(),
    };
    let geometry = BeamElementGeometry {
        node_i_m: structural.nodes[explicit_i].coordinates_m,
        node_j_m: structural.nodes[explicit_j].coordinates_m,
        reference_axis: beam.reference_axis,
    };
    let frame = geometry
        .local_frame()
        .map_err(|error| StructuralAssemblyError::BeamElement {
            element_id: element.element_id.clone(),
            error,
        })?;
    let transform_global_to_local = transformation_matrix(frame);
    let stiffness = global_stiffness_matrix(section, beam_material, geometry).map_err(|error| {
        StructuralAssemblyError::BeamElement {
            element_id: element.element_id.clone(),
            error,
        }
    })?;
    let dofs = beam_element_dof_indices(layout, node_i, node_j).ok_or_else(|| {
        StructuralAssemblyError::MissingRotationalDof {
            element_id: element.element_id.clone(),
        }
    })?;
    let mass = lumped_beam_mass(
        section,
        material.density_kg_per_m3.canonical_value(),
        frame.length_m.canonical_value(),
        &transform_global_to_local,
    );
    Ok(BeamElementContribution {
        dofs,
        stiffness,
        mass,
        damping: material.damping,
        recovery: BeamRecoveryElementSummary {
            element_id: element.element_id.clone(),
            region_id: element.region_id.clone(),
            node_i_index: node_i,
            node_j_index: node_j,
            length_m: runmat_types::physical::Length::new(frame.length_m.canonical_value()),
            section,
            material: beam_material,
            transform_global_to_local,
        },
    })
}

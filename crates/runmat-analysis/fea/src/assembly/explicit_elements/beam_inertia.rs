use runmat_analysis_core::MechanicalDampingModel;

use crate::{
    algebra::RealSparseAssembler,
    assembly::elements::beam::{BeamSection, BeamTransform12, BEAM_ELEMENT_DOF_COUNT},
};

pub(super) fn lumped_beam_mass(
    section: BeamSection,
    density_kg_per_m3: f64,
    length_m: f64,
    transform_global_to_local: &BeamTransform12,
) -> [f64; BEAM_ELEMENT_DOF_COUNT] {
    let nodal_mass = density_kg_per_m3 * section.area_m2 * length_m / 2.0;
    let local_rotational_inertia = [
        density_kg_per_m3 * section.torsion_j_m4 * length_m / 2.0,
        density_kg_per_m3 * section.iy_m4 * length_m / 2.0,
        density_kg_per_m3 * section.iz_m4 * length_m / 2.0,
    ];
    let mut element_mass = [0.0; BEAM_ELEMENT_DOF_COUNT];
    for node_offset in [0usize, 6] {
        for component in 0..3 {
            element_mass[node_offset + component] = nodal_mass;
        }
        for global_component in 0..3 {
            let local_col = node_offset + 3 + global_component;
            element_mass[node_offset + 3 + global_component] = local_rotational_inertia
                .iter()
                .enumerate()
                .map(|(local_component, local_inertia)| {
                    let row = node_offset + 3 + local_component;
                    local_inertia * transform_global_to_local[row][local_col].powi(2)
                })
                .sum();
        }
    }
    element_mass
}

pub(super) fn add_rayleigh_element_damping<const D: usize>(
    damping: &mut RealSparseAssembler,
    element_dofs: &[usize; D],
    stiffness: &[[f64; D]; D],
    lumped_mass: &[f64; D],
    law: MechanicalDampingModel,
) -> bool {
    let MechanicalDampingModel::Rayleigh {
        mass_coefficient,
        stiffness_coefficient,
    } = law
    else {
        return false;
    };
    for (local_row, global_row) in element_dofs.iter().copied().enumerate() {
        for (local_column, global_column) in element_dofs.iter().copied().enumerate() {
            let mass = if local_row == local_column {
                lumped_mass[local_row]
            } else {
                0.0
            };
            damping.add(
                global_row,
                global_column,
                mass_coefficient.per_second * mass
                    + stiffness_coefficient.seconds * stiffness[local_row][local_column],
            );
        }
    }
    true
}

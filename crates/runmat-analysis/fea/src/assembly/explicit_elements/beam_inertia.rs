use crate::assembly::elements::beam::{BeamSection, BeamTransform12, BEAM_ELEMENT_DOF_COUNT};

pub(super) fn lumped_beam_mass(
    section: BeamSection,
    density_kg_per_m3: f64,
    length_m: f64,
    transform_global_to_local: &BeamTransform12,
) -> [f64; BEAM_ELEMENT_DOF_COUNT] {
    let nodal_mass = density_kg_per_m3 * section.area_m2.canonical_value() * length_m / 2.0;
    let local_rotational_inertia = [
        density_kg_per_m3 * section.torsion_j_m4.canonical_value() * length_m / 2.0,
        density_kg_per_m3 * section.iy_m4.canonical_value() * length_m / 2.0,
        density_kg_per_m3 * section.iz_m4.canonical_value() * length_m / 2.0,
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

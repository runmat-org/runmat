use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const SHELL_NODE_DOF_COUNT: usize = 6;
pub const SHELL_ELEMENT_NODE_COUNT: usize = 3;
pub const SHELL_ELEMENT_DOF_COUNT: usize = SHELL_NODE_DOF_COUNT * SHELL_ELEMENT_NODE_COUNT;

pub type ShellMatrix18 = [[f64; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_DOF_COUNT];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShellSection {
    pub thickness_m: f64,
    pub shear_correction: f64,
    pub drilling_stiffness_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShellMaterial {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub shear_modulus_pa: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShellElementGeometry {
    pub nodes_m: [[f64; 3]; SHELL_ELEMENT_NODE_COUNT],
    pub reference_axis: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellLocalFrame {
    pub x: [f64; 3],
    pub y: [f64; 3],
    pub z: [f64; 3],
    pub area_m2: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellResultants {
    pub membrane_force_n_per_m: [f64; 3],
    pub bending_moment_n: [f64; 3],
    pub transverse_shear_n_per_m: [f64; 2],
    pub surface_von_mises_pa: f64,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ShellElementError {
    #[error("shell element area must be positive and finite")]
    DegenerateArea,
    #[error("shell reference axis must be finite and non-parallel to the shell normal")]
    DegenerateReferenceAxis,
    #[error("shell thickness, shear correction, and drilling scale must be positive and finite")]
    InvalidSection,
    #[error("shell material properties must be positive and finite")]
    InvalidMaterial,
}

impl ShellSection {
    pub fn validate(self) -> Result<(), ShellElementError> {
        if positive_finite(self.thickness_m)
            && positive_finite(self.shear_correction)
            && positive_finite(self.drilling_stiffness_scale)
        {
            Ok(())
        } else {
            Err(ShellElementError::InvalidSection)
        }
    }
}

impl ShellMaterial {
    pub fn validate(self) -> Result<(), ShellElementError> {
        if positive_finite(self.youngs_modulus_pa)
            && positive_finite(self.shear_modulus_pa)
            && self.poisson_ratio.is_finite()
            && self.poisson_ratio > -1.0
            && self.poisson_ratio < 0.5
        {
            Ok(())
        } else {
            Err(ShellElementError::InvalidMaterial)
        }
    }
}

impl ShellElementGeometry {
    pub fn local_frame(self) -> Result<ShellLocalFrame, ShellElementError> {
        let edge_a = sub(self.nodes_m[1], self.nodes_m[0]);
        let edge_b = sub(self.nodes_m[2], self.nodes_m[0]);
        let normal_area = cross(edge_a, edge_b);
        let normal_norm = norm(normal_area);
        if !positive_finite(normal_norm) {
            return Err(ShellElementError::DegenerateArea);
        }
        let z = scale(normal_area, 1.0 / normal_norm);
        if !self.reference_axis.iter().all(|value| value.is_finite()) {
            return Err(ShellElementError::DegenerateReferenceAxis);
        }
        let reference_projection = sub(self.reference_axis, scale(z, dot(self.reference_axis, z)));
        let projection_norm = norm(reference_projection);
        let x = if projection_norm > 1.0e-12 && projection_norm.is_finite() {
            scale(reference_projection, 1.0 / projection_norm)
        } else {
            let edge_norm = norm(edge_a);
            if !positive_finite(edge_norm) {
                return Err(ShellElementError::DegenerateReferenceAxis);
            }
            scale(edge_a, 1.0 / edge_norm)
        };
        let y = cross(z, x);
        Ok(ShellLocalFrame {
            x,
            y,
            z,
            area_m2: 0.5 * normal_norm,
        })
    }
}

pub fn global_stiffness_matrix(
    section: ShellSection,
    material: ShellMaterial,
    geometry: ShellElementGeometry,
) -> Result<ShellMatrix18, ShellElementError> {
    section.validate()?;
    material.validate()?;
    let frame = geometry.local_frame()?;
    let operators = local_operators(section, material, geometry, frame)?;
    let mut local = [[0.0; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_DOF_COUNT];
    add_bdb3(
        &mut local,
        &operators.membrane,
        membrane_constitutive(section, material),
        frame.area_m2,
    );
    add_bdb3(
        &mut local,
        &operators.bending,
        bending_constitutive(section, material),
        frame.area_m2,
    );
    for shape in [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ] {
        add_bdb2(
            &mut local,
            &shear_operator(operators.dx, operators.dy, shape),
            shear_constitutive(section, material),
            frame.area_m2 / 3.0,
        );
    }
    let drilling_stiffness = section.drilling_stiffness_scale
        * material.youngs_modulus_pa
        * section.thickness_m
        * frame.area_m2;
    for drilling in &operators.drilling {
        add_outer(
            &mut local,
            drilling,
            drilling_stiffness / SHELL_ELEMENT_NODE_COUNT as f64,
        );
    }
    Ok(transform_matrix_to_global(local, frame))
}

pub fn recover_resultants(
    section: ShellSection,
    material: ShellMaterial,
    geometry: ShellElementGeometry,
    global_displacement: [f64; SHELL_ELEMENT_DOF_COUNT],
) -> Result<ShellResultants, ShellElementError> {
    section.validate()?;
    material.validate()?;
    let frame = geometry.local_frame()?;
    let operators = local_operators(section, material, geometry, frame)?;
    let local_displacement = transform_vector_to_local(global_displacement, frame);
    let membrane_strain = apply_b3(&operators.membrane, &local_displacement);
    let curvature = apply_b3(&operators.bending, &local_displacement);
    let transverse_shear_strain = apply_b2(&operators.shear, &local_displacement);
    let membrane_force = apply_d3(membrane_constitutive(section, material), membrane_strain);
    let bending_moment = apply_d3(bending_constitutive(section, material), curvature);
    let transverse_shear = apply_d2(
        shear_constitutive(section, material),
        transverse_shear_strain,
    );
    let surface_von_mises_pa = [-1.0, 1.0]
        .into_iter()
        .map(|surface| {
            let stress = std::array::from_fn::<_, 3, _>(|component| {
                membrane_force[component] / section.thickness_m
                    + surface * 6.0 * bending_moment[component] / section.thickness_m.powi(2)
            });
            (stress[0].powi(2) - stress[0] * stress[1]
                + stress[1].powi(2)
                + 3.0 * stress[2].powi(2))
            .max(0.0)
            .sqrt()
        })
        .fold(0.0, f64::max);
    Ok(ShellResultants {
        membrane_force_n_per_m: membrane_force,
        bending_moment_n: bending_moment,
        transverse_shear_n_per_m: transverse_shear,
        surface_von_mises_pa,
    })
}

struct LocalOperators {
    membrane: [[f64; SHELL_ELEMENT_DOF_COUNT]; 3],
    bending: [[f64; SHELL_ELEMENT_DOF_COUNT]; 3],
    shear: [[f64; SHELL_ELEMENT_DOF_COUNT]; 2],
    drilling: [[f64; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_NODE_COUNT],
    dx: [f64; SHELL_ELEMENT_NODE_COUNT],
    dy: [f64; SHELL_ELEMENT_NODE_COUNT],
}

fn local_operators(
    _section: ShellSection,
    _material: ShellMaterial,
    geometry: ShellElementGeometry,
    frame: ShellLocalFrame,
) -> Result<LocalOperators, ShellElementError> {
    let local = geometry.nodes_m.map(|node| {
        let offset = sub(node, geometry.nodes_m[0]);
        [dot(offset, frame.x), dot(offset, frame.y)]
    });
    let twice_area = local[1][0] * local[2][1] - local[2][0] * local[1][1];
    if !positive_finite(twice_area) {
        return Err(ShellElementError::DegenerateArea);
    }
    let dx = [
        (local[1][1] - local[2][1]) / twice_area,
        local[2][1] / twice_area,
        -local[1][1] / twice_area,
    ];
    let dy = [
        (local[2][0] - local[1][0]) / twice_area,
        -local[2][0] / twice_area,
        local[1][0] / twice_area,
    ];
    let mut membrane = [[0.0; SHELL_ELEMENT_DOF_COUNT]; 3];
    let mut bending = [[0.0; SHELL_ELEMENT_DOF_COUNT]; 3];
    let mut shear = [[0.0; SHELL_ELEMENT_DOF_COUNT]; 2];
    let mut drilling = [[0.0; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_NODE_COUNT];
    for node in 0..SHELL_ELEMENT_NODE_COUNT {
        let base = node * SHELL_NODE_DOF_COUNT;
        membrane[0][base] = dx[node];
        membrane[1][base + 1] = dy[node];
        membrane[2][base] = dy[node];
        membrane[2][base + 1] = dx[node];

        bending[0][base + 4] = dx[node];
        bending[1][base + 3] = -dy[node];
        bending[2][base + 4] = dy[node];
        bending[2][base + 3] = -dx[node];

        shear[0][base + 2] = dx[node];
        shear[0][base + 4] = 1.0 / 3.0;
        shear[1][base + 2] = dy[node];
        shear[1][base + 3] = -1.0 / 3.0;

        for drilling_row in &mut drilling {
            drilling_row[base] = 0.5 * dy[node];
            drilling_row[base + 1] = -0.5 * dx[node];
        }
        drilling[node][base + 5] = 1.0;
    }
    Ok(LocalOperators {
        membrane,
        bending,
        shear,
        drilling,
        dx,
        dy,
    })
}

fn shear_operator(
    dx: [f64; SHELL_ELEMENT_NODE_COUNT],
    dy: [f64; SHELL_ELEMENT_NODE_COUNT],
    shape: [f64; SHELL_ELEMENT_NODE_COUNT],
) -> [[f64; SHELL_ELEMENT_DOF_COUNT]; 2] {
    let mut shear = [[0.0; SHELL_ELEMENT_DOF_COUNT]; 2];
    for node in 0..SHELL_ELEMENT_NODE_COUNT {
        let base = node * SHELL_NODE_DOF_COUNT;
        shear[0][base + 2] = dx[node];
        shear[0][base + 4] = shape[node];
        shear[1][base + 2] = dy[node];
        shear[1][base + 3] = -shape[node];
    }
    shear
}

fn membrane_constitutive(section: ShellSection, material: ShellMaterial) -> [[f64; 3]; 3] {
    plane_stress_constitutive(
        material.youngs_modulus_pa * section.thickness_m,
        material.poisson_ratio,
    )
}

fn bending_constitutive(section: ShellSection, material: ShellMaterial) -> [[f64; 3]; 3] {
    plane_stress_constitutive(
        material.youngs_modulus_pa * section.thickness_m.powi(3) / 12.0,
        material.poisson_ratio,
    )
}

fn plane_stress_constitutive(scale: f64, poisson_ratio: f64) -> [[f64; 3]; 3] {
    let factor = scale / (1.0 - poisson_ratio.powi(2));
    [
        [factor, factor * poisson_ratio, 0.0],
        [factor * poisson_ratio, factor, 0.0],
        [0.0, 0.0, factor * (1.0 - poisson_ratio) / 2.0],
    ]
}

fn shear_constitutive(section: ShellSection, material: ShellMaterial) -> [[f64; 2]; 2] {
    let value = material.shear_modulus_pa * section.thickness_m * section.shear_correction;
    [[value, 0.0], [0.0, value]]
}

fn add_bdb3(
    matrix: &mut ShellMatrix18,
    b: &[[f64; SHELL_ELEMENT_DOF_COUNT]; 3],
    d: [[f64; 3]; 3],
    measure: f64,
) {
    for (row, matrix_row) in matrix.iter_mut().enumerate() {
        for (column, value) in matrix_row.iter_mut().enumerate() {
            for alpha in 0..3 {
                for beta in 0..3 {
                    *value += measure * b[alpha][row] * d[alpha][beta] * b[beta][column];
                }
            }
        }
    }
}

fn add_bdb2(
    matrix: &mut ShellMatrix18,
    b: &[[f64; SHELL_ELEMENT_DOF_COUNT]; 2],
    d: [[f64; 2]; 2],
    measure: f64,
) {
    for (row, matrix_row) in matrix.iter_mut().enumerate() {
        for (column, value) in matrix_row.iter_mut().enumerate() {
            for alpha in 0..2 {
                for beta in 0..2 {
                    *value += measure * b[alpha][row] * d[alpha][beta] * b[beta][column];
                }
            }
        }
    }
}

fn add_outer(matrix: &mut ShellMatrix18, vector: &[f64; SHELL_ELEMENT_DOF_COUNT], scale: f64) {
    for (row, matrix_row) in matrix.iter_mut().enumerate() {
        for (column, value) in matrix_row.iter_mut().enumerate() {
            *value += scale * vector[row] * vector[column];
        }
    }
}

fn transform_vector_to_local(
    global: [f64; SHELL_ELEMENT_DOF_COUNT],
    frame: ShellLocalFrame,
) -> [f64; SHELL_ELEMENT_DOF_COUNT] {
    let rotation = [frame.x, frame.y, frame.z];
    let mut local = [0.0; SHELL_ELEMENT_DOF_COUNT];
    for node in 0..SHELL_ELEMENT_NODE_COUNT {
        for block in 0..2 {
            let offset = node * SHELL_NODE_DOF_COUNT + block * 3;
            for row in 0..3 {
                for column in 0..3 {
                    local[offset + row] += rotation[row][column] * global[offset + column];
                }
            }
        }
    }
    local
}

fn transform_matrix_to_global(local: ShellMatrix18, frame: ShellLocalFrame) -> ShellMatrix18 {
    let rotation = [frame.x, frame.y, frame.z];
    let mut global = [[0.0; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_DOF_COUNT];
    for (global_row, global_row_values) in global.iter_mut().enumerate() {
        let row_block = global_row / 3;
        let row_component = global_row % 3;
        for (global_column, global_value) in global_row_values.iter_mut().enumerate() {
            let column_block = global_column / 3;
            let column_component = global_column % 3;
            for local_row_component in 0..3 {
                let local_row = row_block * 3 + local_row_component;
                for local_column_component in 0..3 {
                    let local_column = column_block * 3 + local_column_component;
                    *global_value += rotation[local_row_component][row_component]
                        * local[local_row][local_column]
                        * rotation[local_column_component][column_component];
                }
            }
        }
    }
    global
}

fn apply_b3(
    b: &[[f64; SHELL_ELEMENT_DOF_COUNT]; 3],
    value: &[f64; SHELL_ELEMENT_DOF_COUNT],
) -> [f64; 3] {
    std::array::from_fn(|row| b[row].iter().zip(value).map(|(a, b)| a * b).sum())
}

fn apply_b2(
    b: &[[f64; SHELL_ELEMENT_DOF_COUNT]; 2],
    value: &[f64; SHELL_ELEMENT_DOF_COUNT],
) -> [f64; 2] {
    std::array::from_fn(|row| b[row].iter().zip(value).map(|(a, b)| a * b).sum())
}

fn apply_d3(d: [[f64; 3]; 3], value: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| (0..3).map(|column| d[row][column] * value[column]).sum())
}

fn apply_d2(d: [[f64; 2]; 2], value: [f64; 2]) -> [f64; 2] {
    std::array::from_fn(|row| (0..2).map(|column| d[row][column] * value[column]).sum())
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f64; 3], factor: f64) -> [f64; 3] {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section() -> ShellSection {
        ShellSection {
            thickness_m: 0.002,
            shear_correction: 5.0 / 6.0,
            drilling_stiffness_scale: 1.0e-4,
        }
    }

    fn material() -> ShellMaterial {
        ShellMaterial {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.3,
            shear_modulus_pa: 76.9e9,
        }
    }

    fn geometry() -> ShellElementGeometry {
        ShellElementGeometry {
            nodes_m: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            reference_axis: [1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn shell_local_frame_is_orthonormal() {
        let frame = geometry().local_frame().expect("frame should build");
        assert_close(frame.area_m2, 0.5, 1.0e-12);
        assert_close(dot(frame.x, frame.y), 0.0, 1.0e-12);
        assert_close(dot(frame.x, frame.z), 0.0, 1.0e-12);
        assert_close(dot(frame.y, frame.z), 0.0, 1.0e-12);
        assert_close(norm(frame.x), 1.0, 1.0e-12);
        assert_close(norm(frame.y), 1.0, 1.0e-12);
        assert_close(norm(frame.z), 1.0, 1.0e-12);
    }

    #[test]
    fn shell_global_stiffness_is_symmetric() {
        let k = global_stiffness_matrix(section(), material(), geometry())
            .expect("stiffness should build");
        for row in 0..SHELL_ELEMENT_DOF_COUNT {
            for col in 0..SHELL_ELEMENT_DOF_COUNT {
                assert_close(k[row][col], k[col][row], 1.0e-6);
            }
        }
    }

    #[test]
    fn shell_has_exact_six_rigid_body_modes() {
        let stiffness = global_stiffness_matrix(section(), material(), geometry()).unwrap();
        for mode_index in 0..6 {
            let mut mode = [0.0; SHELL_ELEMENT_DOF_COUNT];
            for node in 0..SHELL_ELEMENT_NODE_COUNT {
                let base = node * SHELL_NODE_DOF_COUNT;
                if mode_index < 3 {
                    mode[base + mode_index] = 1.0;
                } else {
                    let mut rotation = [0.0; 3];
                    rotation[mode_index - 3] = 1.0;
                    let translation = cross(rotation, geometry().nodes_m[node]);
                    mode[base..base + 3].copy_from_slice(&translation);
                    mode[base + 3..base + 6].copy_from_slice(&rotation);
                }
            }
            for row in &stiffness {
                let residual = row
                    .iter()
                    .zip(mode)
                    .map(|(value, mode)| value * mode)
                    .sum::<f64>();
                assert_close(residual, 0.0, 1.0e-5);
            }
        }
    }

    #[test]
    fn shell_constant_membrane_strain_matches_plane_stress_resultants() {
        let geometry = geometry();
        let epsilon_x = 2.0e-4;
        let mut displacement = [0.0; SHELL_ELEMENT_DOF_COUNT];
        for (node, position) in geometry.nodes_m.iter().enumerate() {
            displacement[node * SHELL_NODE_DOF_COUNT] = epsilon_x * position[0];
        }
        let result = recover_resultants(section(), material(), geometry, displacement).unwrap();
        let factor = material().youngs_modulus_pa * section().thickness_m
            / (1.0 - material().poisson_ratio.powi(2));
        assert_close(
            result.membrane_force_n_per_m[0],
            factor * epsilon_x,
            1.0e-8 * factor,
        );
        assert_close(
            result.membrane_force_n_per_m[1],
            factor * material().poisson_ratio * epsilon_x,
            1.0e-8 * factor,
        );
        assert_close(result.membrane_force_n_per_m[2], 0.0, 1.0e-8 * factor);
        assert!(result.bending_moment_n.iter().all(|value| *value == 0.0));
        assert!(result
            .transverse_shear_n_per_m
            .iter()
            .all(|value| *value == 0.0));
    }

    #[test]
    fn shell_energy_is_invariant_under_rigid_frame_rotation() {
        let original_geometry = geometry();
        let rotated_geometry = ShellElementGeometry {
            nodes_m: original_geometry.nodes_m.map(cyclic_rotation),
            reference_axis: cyclic_rotation(original_geometry.reference_axis),
        };
        let original_displacement = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0e-4, -1.0e-4, 4.0e-5, 3.0e-4, -2.0e-4, 1.0e-4,
            -1.0e-4, 3.0e-4, -2.0e-5, -4.0e-4, 1.0e-4, 2.0e-4,
        ];
        let mut rotated_displacement = [0.0; SHELL_ELEMENT_DOF_COUNT];
        for node in 0..SHELL_ELEMENT_NODE_COUNT {
            let base = node * SHELL_NODE_DOF_COUNT;
            for block in 0..2 {
                let offset = base + block * 3;
                rotated_displacement[offset..offset + 3].copy_from_slice(&cyclic_rotation([
                    original_displacement[offset],
                    original_displacement[offset + 1],
                    original_displacement[offset + 2],
                ]));
            }
        }
        let original_stiffness =
            global_stiffness_matrix(section(), material(), original_geometry).unwrap();
        let rotated_stiffness =
            global_stiffness_matrix(section(), material(), rotated_geometry).unwrap();
        assert_close(
            quadratic_energy(&original_stiffness, &original_displacement),
            quadratic_energy(&rotated_stiffness, &rotated_displacement),
            1.0e-10,
        );
        assert_eq!(
            recover_resultants(
                section(),
                material(),
                original_geometry,
                original_displacement,
            )
            .unwrap(),
            recover_resultants(
                section(),
                material(),
                rotated_geometry,
                rotated_displacement,
            )
            .unwrap()
        );
    }

    #[test]
    fn shell_resultant_thickness_scaling_separates_membrane_bending_and_shear() {
        let mut displacement = [0.0; SHELL_ELEMENT_DOF_COUNT];
        for (node, position) in geometry().nodes_m.iter().enumerate() {
            let base = node * SHELL_NODE_DOF_COUNT;
            displacement[base] = 2.0e-4 * position[0];
            displacement[base + 4] = 3.0e-4 * position[0];
        }
        let thin = recover_resultants(section(), material(), geometry(), displacement).unwrap();
        let thick = recover_resultants(
            ShellSection {
                thickness_m: section().thickness_m * 2.0,
                ..section()
            },
            material(),
            geometry(),
            displacement,
        )
        .unwrap();
        for component in 0..3 {
            if thin.membrane_force_n_per_m[component].abs() > 0.0 {
                assert_close(
                    thick.membrane_force_n_per_m[component]
                        / thin.membrane_force_n_per_m[component],
                    2.0,
                    1.0e-12,
                );
            }
            if thin.bending_moment_n[component].abs() > 0.0 {
                assert_close(
                    thick.bending_moment_n[component] / thin.bending_moment_n[component],
                    8.0,
                    1.0e-12,
                );
            }
        }
        for component in 0..2 {
            if thin.transverse_shear_n_per_m[component].abs() > 0.0 {
                assert_close(
                    thick.transverse_shear_n_per_m[component]
                        / thin.transverse_shear_n_per_m[component],
                    2.0,
                    1.0e-12,
                );
            }
        }
    }

    #[test]
    fn shell_frame_rejects_degenerate_inputs() {
        assert_eq!(
            ShellElementGeometry {
                nodes_m: [[0.0, 0.0, 0.0]; 3],
                reference_axis: [1.0, 0.0, 0.0],
            }
            .local_frame()
            .expect_err("zero-area shell should fail"),
            ShellElementError::DegenerateArea
        );
    }

    fn cyclic_rotation(vector: [f64; 3]) -> [f64; 3] {
        [vector[2], vector[0], vector[1]]
    }

    fn quadratic_energy(
        stiffness: &ShellMatrix18,
        displacement: &[f64; SHELL_ELEMENT_DOF_COUNT],
    ) -> f64 {
        0.5 * stiffness
            .iter()
            .zip(displacement)
            .map(|(row, value)| {
                value
                    * row
                        .iter()
                        .zip(displacement)
                        .map(|(coefficient, displacement)| coefficient * displacement)
                        .sum::<f64>()
            })
            .sum::<f64>()
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} tolerance={tolerance}",
        );
    }
}

use thiserror::Error;

pub const BEAM_NODE_DOF_COUNT: usize = 6;
pub const BEAM_ELEMENT_DOF_COUNT: usize = 12;

pub type BeamMatrix12 = [[f64; BEAM_ELEMENT_DOF_COUNT]; BEAM_ELEMENT_DOF_COUNT];
pub type BeamTransform12 = [[f64; BEAM_ELEMENT_DOF_COUNT]; BEAM_ELEMENT_DOF_COUNT];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamSection {
    pub area_m2: f64,
    pub iy_m4: f64,
    pub iz_m4: f64,
    pub torsion_j_m4: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamMaterial {
    pub youngs_modulus_pa: f64,
    pub shear_modulus_pa: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamElementGeometry {
    pub node_i_m: [f64; 3],
    pub node_j_m: [f64; 3],
    pub reference_axis: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamLocalFrame {
    pub x: [f64; 3],
    pub y: [f64; 3],
    pub z: [f64; 3],
    pub length_m: f64,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum BeamElementError {
    #[error("beam element length must be positive and finite")]
    DegenerateLength,
    #[error("beam reference axis must be finite and non-parallel to the beam axis")]
    DegenerateReferenceAxis,
    #[error("beam section area, second moments, and torsion constant must be positive and finite")]
    InvalidSection,
    #[error("beam Young's modulus and shear modulus must be positive and finite")]
    InvalidMaterial,
}

impl BeamSection {
    pub fn validate(self) -> Result<(), BeamElementError> {
        if positive_finite(self.area_m2)
            && positive_finite(self.iy_m4)
            && positive_finite(self.iz_m4)
            && positive_finite(self.torsion_j_m4)
        {
            Ok(())
        } else {
            Err(BeamElementError::InvalidSection)
        }
    }
}

impl BeamMaterial {
    pub fn validate(self) -> Result<(), BeamElementError> {
        if positive_finite(self.youngs_modulus_pa) && positive_finite(self.shear_modulus_pa) {
            Ok(())
        } else {
            Err(BeamElementError::InvalidMaterial)
        }
    }
}

impl BeamElementGeometry {
    pub fn local_frame(self) -> Result<BeamLocalFrame, BeamElementError> {
        let axis = sub(self.node_j_m, self.node_i_m);
        let length_m = norm(axis);
        if !positive_finite(length_m) {
            return Err(BeamElementError::DegenerateLength);
        }
        let x = scale(axis, 1.0 / length_m);
        if !self.reference_axis.iter().all(|value| value.is_finite()) {
            return Err(BeamElementError::DegenerateReferenceAxis);
        }
        let reference_projection = sub(self.reference_axis, scale(x, dot(self.reference_axis, x)));
        let projection_norm = norm(reference_projection);
        if projection_norm <= 1.0e-12 || !projection_norm.is_finite() {
            return Err(BeamElementError::DegenerateReferenceAxis);
        }
        let y = scale(reference_projection, 1.0 / projection_norm);
        let z = cross(x, y);
        Ok(BeamLocalFrame { x, y, z, length_m })
    }
}

pub fn local_stiffness_matrix(
    section: BeamSection,
    material: BeamMaterial,
    length_m: f64,
) -> Result<BeamMatrix12, BeamElementError> {
    section.validate()?;
    material.validate()?;
    if !positive_finite(length_m) {
        return Err(BeamElementError::DegenerateLength);
    }

    let l = length_m;
    let l2 = l * l;
    let l3 = l2 * l;
    let ea_l = material.youngs_modulus_pa * section.area_m2 / l;
    let gj_l = material.shear_modulus_pa * section.torsion_j_m4 / l;
    let eiy = material.youngs_modulus_pa * section.iy_m4;
    let eiz = material.youngs_modulus_pa * section.iz_m4;

    let mut k = [[0.0; BEAM_ELEMENT_DOF_COUNT]; BEAM_ELEMENT_DOF_COUNT];

    add_symmetric(&mut k, 0, 0, ea_l);
    add_symmetric(&mut k, 0, 6, -ea_l);
    add_symmetric(&mut k, 6, 6, ea_l);

    add_symmetric(&mut k, 3, 3, gj_l);
    add_symmetric(&mut k, 3, 9, -gj_l);
    add_symmetric(&mut k, 9, 9, gj_l);

    add_bending_z(&mut k, eiz, l, l2, l3);
    add_bending_y(&mut k, eiy, l, l2, l3);

    Ok(k)
}

pub fn transformation_matrix(frame: BeamLocalFrame) -> BeamTransform12 {
    let rotation = [frame.x, frame.y, frame.z];
    let mut transform = [[0.0; BEAM_ELEMENT_DOF_COUNT]; BEAM_ELEMENT_DOF_COUNT];
    for block in 0..4 {
        let offset = block * 3;
        for row in 0..3 {
            for col in 0..3 {
                transform[offset + row][offset + col] = rotation[row][col];
            }
        }
    }
    transform
}

pub fn global_stiffness_matrix(
    section: BeamSection,
    material: BeamMaterial,
    geometry: BeamElementGeometry,
) -> Result<BeamMatrix12, BeamElementError> {
    let frame = geometry.local_frame()?;
    let local = local_stiffness_matrix(section, material, frame.length_m)?;
    let transform = transformation_matrix(frame);
    Ok(transform_transpose_multiply(&transform, &local))
}

fn add_bending_z(k: &mut BeamMatrix12, ei: f64, l: f64, l2: f64, l3: f64) {
    let c12 = 12.0 * ei / l3;
    let c6 = 6.0 * ei / l2;
    let c4 = 4.0 * ei / l;
    let c2 = 2.0 * ei / l;
    add_symmetric(k, 1, 1, c12);
    add_symmetric(k, 1, 5, c6);
    add_symmetric(k, 1, 7, -c12);
    add_symmetric(k, 1, 11, c6);
    add_symmetric(k, 5, 5, c4);
    add_symmetric(k, 5, 7, -c6);
    add_symmetric(k, 5, 11, c2);
    add_symmetric(k, 7, 7, c12);
    add_symmetric(k, 7, 11, -c6);
    add_symmetric(k, 11, 11, c4);
}

fn add_bending_y(k: &mut BeamMatrix12, ei: f64, l: f64, l2: f64, l3: f64) {
    let c12 = 12.0 * ei / l3;
    let c6 = 6.0 * ei / l2;
    let c4 = 4.0 * ei / l;
    let c2 = 2.0 * ei / l;
    add_symmetric(k, 2, 2, c12);
    add_symmetric(k, 2, 4, -c6);
    add_symmetric(k, 2, 8, -c12);
    add_symmetric(k, 2, 10, -c6);
    add_symmetric(k, 4, 4, c4);
    add_symmetric(k, 4, 8, c6);
    add_symmetric(k, 4, 10, c2);
    add_symmetric(k, 8, 8, c12);
    add_symmetric(k, 8, 10, c6);
    add_symmetric(k, 10, 10, c4);
}

fn add_symmetric(matrix: &mut BeamMatrix12, row: usize, col: usize, value: f64) {
    matrix[row][col] += value;
    if row != col {
        matrix[col][row] += value;
    }
}

fn transform_transpose_multiply(transform: &BeamTransform12, local: &BeamMatrix12) -> BeamMatrix12 {
    let mut temp = [[0.0; BEAM_ELEMENT_DOF_COUNT]; BEAM_ELEMENT_DOF_COUNT];
    for row in 0..BEAM_ELEMENT_DOF_COUNT {
        for col in 0..BEAM_ELEMENT_DOF_COUNT {
            temp[row][col] = (0..BEAM_ELEMENT_DOF_COUNT)
                .map(|idx| local[row][idx] * transform[idx][col])
                .sum();
        }
    }

    let mut global = [[0.0; BEAM_ELEMENT_DOF_COUNT]; BEAM_ELEMENT_DOF_COUNT];
    for row in 0..BEAM_ELEMENT_DOF_COUNT {
        for col in 0..BEAM_ELEMENT_DOF_COUNT {
            global[row][col] = (0..BEAM_ELEMENT_DOF_COUNT)
                .map(|idx| transform[idx][row] * temp[idx][col])
                .sum();
        }
    }
    global
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

    fn section() -> BeamSection {
        BeamSection {
            area_m2: 2.0e-4,
            iy_m4: 1.6e-9,
            iz_m4: 6.4e-9,
            torsion_j_m4: 2.4e-9,
        }
    }

    fn material() -> BeamMaterial {
        BeamMaterial {
            youngs_modulus_pa: 200.0e9,
            shear_modulus_pa: 79.3e9,
        }
    }

    #[test]
    fn beam_local_frame_is_orthonormal() {
        let frame = BeamElementGeometry {
            node_i_m: [0.0, 0.0, 0.0],
            node_j_m: [2.0, 0.0, 0.0],
            reference_axis: [0.0, 0.0, 1.0],
        }
        .local_frame()
        .expect("frame should build");

        assert_close(frame.length_m, 2.0, 1.0e-12);
        assert_close(dot(frame.x, frame.y), 0.0, 1.0e-12);
        assert_close(dot(frame.x, frame.z), 0.0, 1.0e-12);
        assert_close(dot(frame.y, frame.z), 0.0, 1.0e-12);
        assert_close(norm(frame.x), 1.0, 1.0e-12);
        assert_close(norm(frame.y), 1.0, 1.0e-12);
        assert_close(norm(frame.z), 1.0, 1.0e-12);
    }

    #[test]
    fn beam_local_stiffness_matches_closed_form_terms() {
        let l = 2.5;
        let k = local_stiffness_matrix(section(), material(), l).expect("matrix should build");
        let ea_l = material().youngs_modulus_pa * section().area_m2 / l;
        let gj_l = material().shear_modulus_pa * section().torsion_j_m4 / l;
        let eiy = material().youngs_modulus_pa * section().iy_m4;
        let eiz = material().youngs_modulus_pa * section().iz_m4;

        assert_close(k[0][0], ea_l, 1.0e-6);
        assert_close(k[0][6], -ea_l, 1.0e-6);
        assert_close(k[3][3], gj_l, 1.0e-9);
        assert_close(k[3][9], -gj_l, 1.0e-9);
        assert_close(k[1][1], 12.0 * eiz / l.powi(3), 1.0e-6);
        assert_close(k[1][5], 6.0 * eiz / l.powi(2), 1.0e-6);
        assert_close(k[5][11], 2.0 * eiz / l, 1.0e-6);
        assert_close(k[2][2], 12.0 * eiy / l.powi(3), 1.0e-6);
        assert_close(k[2][4], -6.0 * eiy / l.powi(2), 1.0e-6);
        assert_close(k[4][10], 2.0 * eiy / l, 1.0e-6);
    }

    #[test]
    fn beam_local_stiffness_is_symmetric() {
        let k = local_stiffness_matrix(section(), material(), 3.0).expect("matrix should build");
        for row in 0..BEAM_ELEMENT_DOF_COUNT {
            for col in 0..BEAM_ELEMENT_DOF_COUNT {
                assert_close(k[row][col], k[col][row], 1.0e-9);
            }
        }
    }

    #[test]
    fn beam_transformation_is_block_orthonormal() {
        let frame = BeamElementGeometry {
            node_i_m: [0.0, 0.0, 0.0],
            node_j_m: [1.0, 1.0, 0.0],
            reference_axis: [0.0, 0.0, 1.0],
        }
        .local_frame()
        .expect("frame should build");
        let transform = transformation_matrix(frame);

        for row in 0..3 {
            for col in 0..3 {
                let value: f64 = (0..3)
                    .map(|idx| transform[row][idx] * transform[col][idx])
                    .sum();
                assert_close(value, if row == col { 1.0 } else { 0.0 }, 1.0e-12);
            }
        }
    }

    #[test]
    fn beam_global_stiffness_is_symmetric() {
        let k = global_stiffness_matrix(
            section(),
            material(),
            BeamElementGeometry {
                node_i_m: [0.0, 0.0, 0.0],
                node_j_m: [1.0, 1.0, 0.25],
                reference_axis: [0.0, 0.0, 1.0],
            },
        )
        .expect("global matrix should build");

        for row in 0..BEAM_ELEMENT_DOF_COUNT {
            for col in 0..BEAM_ELEMENT_DOF_COUNT {
                assert_close(k[row][col], k[col][row], 1.0e-6);
            }
        }
    }

    #[test]
    fn beam_frame_rejects_degenerate_inputs() {
        assert_eq!(
            BeamElementGeometry {
                node_i_m: [0.0, 0.0, 0.0],
                node_j_m: [0.0, 0.0, 0.0],
                reference_axis: [0.0, 1.0, 0.0],
            }
            .local_frame()
            .expect_err("zero-length beam should fail"),
            BeamElementError::DegenerateLength
        );
        assert_eq!(
            BeamElementGeometry {
                node_i_m: [0.0, 0.0, 0.0],
                node_j_m: [1.0, 0.0, 0.0],
                reference_axis: [2.0, 0.0, 0.0],
            }
            .local_frame()
            .expect_err("parallel reference should fail"),
            BeamElementError::DegenerateReferenceAxis
        );
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} tolerance={tolerance}",
        );
    }
}

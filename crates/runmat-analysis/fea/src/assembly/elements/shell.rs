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

#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Error, Clone, PartialEq)]
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
    let mut k = [[0.0; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_DOF_COUNT];

    let membrane = material.youngs_modulus_pa * section.thickness_m * frame.area_m2;
    let bending = material.youngs_modulus_pa * section.thickness_m.powi(3) * frame.area_m2
        / (12.0 * (1.0 - material.poisson_ratio.powi(2)).max(1.0e-9));
    let shear =
        material.shear_modulus_pa * section.thickness_m * section.shear_correction * frame.area_m2;
    let drilling = membrane * section.drilling_stiffness_scale;

    for (a, b) in [(0usize, 1usize), (1, 2), (2, 0)] {
        let length = distance(geometry.nodes_m[a], geometry.nodes_m[b]).max(1.0e-12);
        let membrane_k = membrane / length.powi(2);
        let shear_k = shear / length.powi(2);
        let bending_k = bending / length.powi(2);
        let drilling_k = drilling / length.powi(2);
        add_pair_spring(&mut k, a, b, 0, membrane_k);
        add_pair_spring(&mut k, a, b, 1, membrane_k);
        add_pair_spring(&mut k, a, b, 2, shear_k);
        add_pair_spring(&mut k, a, b, 3, bending_k);
        add_pair_spring(&mut k, a, b, 4, bending_k);
        add_pair_spring(&mut k, a, b, 5, drilling_k);
    }

    let stabilization = (membrane + shear + bending + drilling).max(1.0) * 1.0e-12;
    for (index, row) in k.iter_mut().enumerate() {
        row[index] += stabilization;
    }
    Ok(k)
}

fn add_pair_spring(
    matrix: &mut ShellMatrix18,
    node_a: usize,
    node_b: usize,
    component: usize,
    stiffness: f64,
) {
    let row = node_a * SHELL_NODE_DOF_COUNT + component;
    let col = node_b * SHELL_NODE_DOF_COUNT + component;
    matrix[row][row] += stiffness;
    matrix[col][col] += stiffness;
    matrix[row][col] -= stiffness;
    matrix[col][row] -= stiffness;
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

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
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

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual} expected={expected} tolerance={tolerance}",
        );
    }
}

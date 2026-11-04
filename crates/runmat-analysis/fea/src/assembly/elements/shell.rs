use runmat_simulation_usage::{LedgerError, OperationCount, UsageCategory, UsageContext};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod constitutive;
mod frame_math;
mod inertia;
pub(crate) mod laminate;

use frame_math::{cross, dot, norm, positive_finite, scale, sub};

pub(crate) use constitutive::admitted_homogeneous_constitutive;
pub use constitutive::{ShellConstitutive, ShellConstitutiveMatrices};
pub(crate) use inertia::{global_lumped_mass_matrix, homogeneous_mass_moments, ShellMassMoments};

pub const SHELL_NODE_DOF_COUNT: usize = 6;
pub const SHELL_ELEMENT_NODE_COUNT: usize = 3;
pub const SHELL_ELEMENT_DOF_COUNT: usize = SHELL_NODE_DOF_COUNT * SHELL_ELEMENT_NODE_COUNT;

pub type ShellMatrix18 = [[f64; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_DOF_COUNT];

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShellSection {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub thickness_m: runmat_types::physical::Length,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub reference_offset_m: runmat_types::physical::Length,
    pub shear_correction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ShellMaterial {
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub youngs_modulus_pa: runmat_types::physical::YoungsModulus,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub poisson_ratio: runmat_types::physical::PoissonRatio,
    #[serde(with = "runmat_types::physical::canonical_scalar")]
    pub shear_modulus_pa: runmat_types::physical::ShearModulus,
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
    pub area_m2: runmat_types::physical::Area,
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
    #[error("shell thickness and shear correction must be positive and finite, and reference offset finite")]
    InvalidSection,
    #[error("shell material properties must be positive and finite")]
    InvalidMaterial,
    #[error("shell constitutive matrices, recovery layers, or mass properties are invalid")]
    InvalidConstitutive,
    #[error("shell numerical accounting failed: {0}")]
    Accounting(#[from] LedgerError),
}

impl ShellSection {
    pub fn validate(self) -> Result<(), ShellElementError> {
        if positive_finite(self.thickness_m.canonical_value())
            && self.reference_offset_m.canonical_value().is_finite()
            && positive_finite(self.shear_correction)
        {
            Ok(())
        } else {
            Err(ShellElementError::InvalidSection)
        }
    }
}

impl ShellMaterial {
    pub fn validate(self) -> Result<(), ShellElementError> {
        if positive_finite(self.youngs_modulus_pa.canonical_value())
            && positive_finite(self.shear_modulus_pa.canonical_value())
            && self.poisson_ratio.canonical_value().is_finite()
            && self.poisson_ratio.canonical_value() > -1.0
            && self.poisson_ratio.canonical_value() < 0.5
        {
            Ok(())
        } else {
            Err(ShellElementError::InvalidMaterial)
        }
    }
}

impl ShellElementGeometry {
    pub fn local_frame(self) -> Result<ShellLocalFrame, ShellElementError> {
        // Maximum is the fallback-axis path: edges (6), normal (9), norm (6),
        // normalized z (4), projection (11), projection norm (6), fallback norm
        // (6), normalized x (4), y cross (9), and area scaling (1).
        let usage = UsageContext::capture();
        let mut batch = usage
            .as_ref()
            .map(|usage| {
                usage
                    .ledger()
                    .reserve(UsageCategory::Assembly, OperationCount::from_operations(62))
            })
            .transpose()?;
        let edge_a = sub(self.nodes_m[1], self.nodes_m[0]);
        let edge_b = sub(self.nodes_m[2], self.nodes_m[0]);
        let normal_area = cross(edge_a, edge_b);
        let normal_norm = norm(normal_area);
        if !positive_finite(normal_norm) {
            if let Some(batch) = &mut batch {
                batch.complete(OperationCount::from_operations(21))?;
            }
            return Err(ShellElementError::DegenerateArea);
        }
        let z = scale(normal_area, 1.0 / normal_norm);
        if !self.reference_axis.iter().all(|value| value.is_finite()) {
            if let Some(batch) = &mut batch {
                batch.complete(OperationCount::from_operations(25))?;
            }
            return Err(ShellElementError::DegenerateReferenceAxis);
        }
        let reference_projection = sub(self.reference_axis, scale(z, dot(self.reference_axis, z)));
        let projection_norm = norm(reference_projection);
        let x = if projection_norm > 1.0e-12 && projection_norm.is_finite() {
            scale(reference_projection, 1.0 / projection_norm)
        } else {
            let edge_norm = norm(edge_a);
            if !positive_finite(edge_norm) {
                if let Some(batch) = &mut batch {
                    batch.complete(OperationCount::from_operations(48))?;
                }
                return Err(ShellElementError::DegenerateReferenceAxis);
            }
            scale(edge_a, 1.0 / edge_norm)
        };
        let y = cross(z, x);
        let area = 0.5 * normal_norm;
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(
                if projection_norm > 1.0e-12 && projection_norm.is_finite() {
                    56
                } else {
                    62
                },
            ))?;
        }
        Ok(ShellLocalFrame {
            x,
            y,
            z,
            area_m2: runmat_types::physical::Area::new(area),
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
    global_stiffness_matrix_for_constitutive(
        &admitted_homogeneous_constitutive(section, material)?,
        geometry,
    )
}

pub(crate) fn global_stiffness_matrix_for_constitutive(
    constitutive: &ShellConstitutive,
    geometry: ShellElementGeometry,
) -> Result<ShellMatrix18, ShellElementError> {
    constitutive.validate()?;
    let frame = geometry.local_frame()?;
    let operators = local_operators(geometry, frame)?;
    global_stiffness_from_operators(&constitutive.matrices, &operators, frame)
}

pub(crate) fn global_matrix_for_constitutive_matrices(
    matrices: &ShellConstitutiveMatrices,
    geometry: ShellElementGeometry,
) -> Result<ShellMatrix18, ShellElementError> {
    matrices.validate_damping()?;
    let frame = geometry.local_frame()?;
    let operators = local_operators(geometry, frame)?;
    global_stiffness_from_operators(matrices, &operators, frame)
}

fn global_stiffness_from_operators(
    matrices: &ShellConstitutiveMatrices,
    operators: &LocalOperators,
    frame: ShellLocalFrame,
) -> Result<ShellMatrix18, ShellElementError> {
    // Four 3x3 BᵀDB contractions: 4 * 18² * 9 * (3 products + 1 add).
    // Three 2x2 shear contractions: 3 * 18² * 4 * (3 products + 1 add).
    // Three drilling outers: 3 * 18² * (2 products + 1 add).
    // Three shear measure divisions, one drilling scale product, three
    // drilling divisions, two 153-pair symmetrizations (add + product),
    // and one 18² * 9-term global transform (2 products + 1 add).
    // Local and assumed-shear operator construction are measured separately.
    const CONTRACT_OPS: u64 = 74_491;
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Assembly,
                OperationCount::from_operations(CONTRACT_OPS),
            )
        })
        .transpose()?;
    let mut local = [[0.0; SHELL_ELEMENT_DOF_COUNT]; SHELL_ELEMENT_DOF_COUNT];
    add_bdb3(
        &mut local,
        &operators.membrane,
        matrices.extensional,
        frame.area_m2.canonical_value(),
    );
    add_bdb3_pair(
        &mut local,
        &operators.membrane,
        matrices.coupling,
        &operators.bending,
        frame.area_m2.canonical_value(),
    );
    add_bdb3_pair(
        &mut local,
        &operators.bending,
        matrices.coupling,
        &operators.membrane,
        frame.area_m2.canonical_value(),
    );
    add_bdb3(
        &mut local,
        &operators.bending,
        matrices.bending,
        frame.area_m2.canonical_value(),
    );
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(46_656))?;
    }
    for shape in [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ] {
        let shear =
            admitted_assumed_shear_operator(operators.dx, operators.dy, operators.jacobian, shape)?;
        add_bdb2(
            &mut local,
            &shear,
            matrices.transverse_shear,
            frame.area_m2.canonical_value() / 3.0,
        );
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(5_185))?;
        }
    }
    let drilling_stiffness = matrices.drilling_stiffness_per_area * frame.area_m2.canonical_value();
    for drilling in &operators.drilling {
        add_outer(
            &mut local,
            drilling,
            drilling_stiffness / SHELL_ELEMENT_NODE_COUNT as f64,
        );
    }
    symmetrize(&mut local);
    let mut global = transform_matrix_to_global(local, frame);
    symmetrize(&mut global);
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(12_280))?;
    }
    Ok(global)
}

pub fn recover_resultants(
    section: ShellSection,
    material: ShellMaterial,
    geometry: ShellElementGeometry,
    global_displacement: [f64; SHELL_ELEMENT_DOF_COUNT],
) -> Result<ShellResultants, ShellElementError> {
    section.validate()?;
    material.validate()?;
    recover_resultants_for_constitutive(
        &admitted_homogeneous_constitutive(section, material)?,
        geometry,
        global_displacement,
    )
}

pub(crate) fn recover_resultants_for_constitutive(
    constitutive: &ShellConstitutive,
    geometry: ShellElementGeometry,
    global_displacement: [f64; SHELL_ELEMENT_DOF_COUNT],
) -> Result<ShellResultants, ShellElementError> {
    constitutive.validate()?;
    let frame = geometry.local_frame()?;
    let operators = local_operators(geometry, frame)?;
    let local_displacement = transform_vector_to_local(global_displacement, frame);
    let membrane_strain = apply_b3(&operators.membrane, &local_displacement);
    let curvature = apply_b3(&operators.bending, &local_displacement);
    let transverse_shear_strain = apply_b2(&operators.shear, &local_displacement);
    let membrane_force = add3(
        apply_d3(constitutive.matrices.extensional, membrane_strain),
        apply_d3(constitutive.matrices.coupling, curvature),
    );
    let bending_moment = add3(
        apply_d3(constitutive.matrices.coupling, membrane_strain),
        apply_d3(constitutive.matrices.bending, curvature),
    );
    let transverse_shear = apply_d2(
        constitutive.matrices.transverse_shear,
        transverse_shear_strain,
    );
    let surface_von_mises_pa = constitutive
        .recovery_layers
        .iter()
        .flat_map(|layer| [layer.bottom_m, layer.top_m].map(|z| (layer, z)))
        .map(|(layer, z)| {
            let strain = add3(membrane_strain, scale3(curvature, z));
            plane_stress_von_mises(apply_d3(layer.plane_stress, strain))
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
    jacobian: [[f64; 2]; 2],
}

fn local_operators(
    geometry: ShellElementGeometry,
    frame: ShellLocalFrame,
) -> Result<LocalOperators, ShellElementError> {
    // Three projected nodes (39), twice-area determinant (3), six shape
    // derivatives (8), and three drilling rows at each node (18).
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage
                .ledger()
                .reserve(UsageCategory::Assembly, OperationCount::from_operations(68))
        })
        .transpose()?;
    let local = geometry.nodes_m.map(|node| {
        let offset = sub(node, geometry.nodes_m[0]);
        [dot(offset, frame.x), dot(offset, frame.y)]
    });
    let twice_area = local[1][0] * local[2][1] - local[2][0] * local[1][1];
    if !positive_finite(twice_area) {
        if let Some(batch) = &mut batch {
            batch.complete(OperationCount::from_operations(42))?;
        }
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

        for drilling_row in &mut drilling {
            drilling_row[base] = 0.5 * dy[node];
            drilling_row[base + 1] = -0.5 * dx[node];
        }
        drilling[node][base + 5] = 1.0;
    }
    let jacobian = [local[1], local[2]];
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(68))?;
    }
    let shear = admitted_assumed_shear_operator(dx, dy, jacobian, [1.0 / 3.0; 3])?;
    Ok(LocalOperators {
        membrane,
        bending,
        shear,
        drilling,
        dx,
        dy,
        jacobian,
    })
}

fn displacement_shear_operator(
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

/// Builds the MITC3 assumed transverse-shear field in natural coordinates.
///
/// The displacement shear is sampled at the midpoints of the three edges. The covariant
/// components are constant on their corresponding edges and are interpolated across the
/// triangle before being transformed back to the element frame. This is the tying construction
/// that removes the parasitic shear energy of a pure-bending patch.
fn assumed_shear_operator(
    dx: [f64; SHELL_ELEMENT_NODE_COUNT],
    dy: [f64; SHELL_ELEMENT_NODE_COUNT],
    jacobian: [[f64; 2]; 2],
    shape: [f64; SHELL_ELEMENT_NODE_COUNT],
) -> [[f64; SHELL_ELEMENT_DOF_COUNT]; 2] {
    let tying_shapes = [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]];
    let tying = tying_shapes.map(|tying_shape| {
        covariant_shear_operator(displacement_shear_operator(dx, dy, tying_shape), jacobian)
    });
    let correction = std::array::from_fn::<_, SHELL_ELEMENT_DOF_COUNT, _>(|column| {
        tying[2][0][column] - tying[0][0][column] - tying[2][1][column] + tying[1][1][column]
    });
    let r = shape[1];
    let s = shape[2];
    let covariant = std::array::from_fn(|row| {
        std::array::from_fn(|column| match row {
            0 => tying[0][0][column] + s * correction[column],
            1 => tying[1][1][column] - r * correction[column],
            _ => unreachable!("transverse shear has exactly two components"),
        })
    });
    physical_shear_operator(covariant, jacobian)
}

fn admitted_assumed_shear_operator(
    dx: [f64; SHELL_ELEMENT_NODE_COUNT],
    dy: [f64; SHELL_ELEMENT_NODE_COUNT],
    jacobian: [[f64; 2]; 2],
    shape: [f64; SHELL_ELEMENT_NODE_COUNT],
) -> Result<[[f64; SHELL_ELEMENT_DOF_COUNT]; 2], ShellElementError> {
    // Three tying covariant transforms (324), correction (54), interpolated
    // covariant field (72), determinant (3), physical inverse transform (144).
    let usage = UsageContext::capture();
    let mut batch = usage
        .as_ref()
        .map(|usage| {
            usage.ledger().reserve(
                UsageCategory::Assembly,
                OperationCount::from_operations(597),
            )
        })
        .transpose()?;
    let shear = assumed_shear_operator(dx, dy, jacobian, shape);
    if let Some(batch) = &mut batch {
        batch.complete(OperationCount::from_operations(597))?;
    }
    Ok(shear)
}

fn covariant_shear_operator(
    physical: [[f64; SHELL_ELEMENT_DOF_COUNT]; 2],
    jacobian: [[f64; 2]; 2],
) -> [[f64; SHELL_ELEMENT_DOF_COUNT]; 2] {
    std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            jacobian[row][0] * physical[0][column] + jacobian[row][1] * physical[1][column]
        })
    })
}

fn physical_shear_operator(
    covariant: [[f64; SHELL_ELEMENT_DOF_COUNT]; 2],
    jacobian: [[f64; 2]; 2],
) -> [[f64; SHELL_ELEMENT_DOF_COUNT]; 2] {
    let determinant = jacobian[0][0] * jacobian[1][1] - jacobian[1][0] * jacobian[0][1];
    std::array::from_fn(|row| {
        std::array::from_fn(|column| match row {
            0 => {
                (jacobian[1][1] * covariant[0][column] - jacobian[0][1] * covariant[1][column])
                    / determinant
            }
            1 => {
                (-jacobian[1][0] * covariant[0][column] + jacobian[0][0] * covariant[1][column])
                    / determinant
            }
            _ => unreachable!("transverse shear has exactly two components"),
        })
    })
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

fn add_bdb3_pair(
    matrix: &mut ShellMatrix18,
    left: &[[f64; SHELL_ELEMENT_DOF_COUNT]; 3],
    d: [[f64; 3]; 3],
    right: &[[f64; SHELL_ELEMENT_DOF_COUNT]; 3],
    measure: f64,
) {
    for (row, matrix_row) in matrix.iter_mut().enumerate() {
        for (column, value) in matrix_row.iter_mut().enumerate() {
            for (alpha, left_values) in left.iter().enumerate() {
                for (beta, right_values) in right.iter().enumerate() {
                    *value += measure * left_values[row] * d[alpha][beta] * right_values[column];
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

fn symmetrize(matrix: &mut ShellMatrix18) {
    for row in 0..SHELL_ELEMENT_DOF_COUNT {
        for column in row + 1..SHELL_ELEMENT_DOF_COUNT {
            let value = 0.5 * (matrix[row][column] + matrix[column][row]);
            matrix[row][column] = value;
            matrix[column][row] = value;
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

pub(super) fn transform_matrix_to_global(
    local: ShellMatrix18,
    frame: ShellLocalFrame,
) -> ShellMatrix18 {
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

fn add3(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|index| left[index] + right[index])
}

fn scale3(value: [f64; 3], factor: f64) -> [f64; 3] {
    value.map(|component| component * factor)
}

fn plane_stress_von_mises(stress: [f64; 3]) -> f64 {
    (stress[0].powi(2) - stress[0] * stress[1] + stress[1].powi(2) + 3.0 * stress[2].powi(2))
        .max(0.0)
        .sqrt()
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod qualification_tests;

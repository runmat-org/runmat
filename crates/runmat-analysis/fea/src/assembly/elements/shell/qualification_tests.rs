use super::{global_stiffness_matrix, ShellElementGeometry, ShellMaterial, ShellSection};

const BENDING_DOF_PER_NODE: usize = 3;
const SHELL_BENDING_LOCAL_DOF: [usize; BENDING_DOF_PER_NODE] = [2, 3, 4];

// witness: FEA-STRUCTURAL-SHELL-001/physical_shell_operator @
//   linear_structural/shell/numerical_kernel;
//   check: navier-plate; boundary: Dense test assembly of transverse simply-supported triangular plate; center
//   displacement decreases toward independent Navier value and fine error below8%. No membrane,
//   mixed-shell, native or complete locking qualification.
#[test]
fn simply_supported_sinusoidal_plate_converges_to_the_navier_solution() {
    let reference = navier_center_deflection();
    let coarse = solve_plate(4, 0.0);
    let medium = solve_plate(8, 0.0);
    let fine = solve_plate(12, 0.0);
    let coarse_error = relative_error(coarse, reference);
    let medium_error = relative_error(medium, reference);
    let fine_error = relative_error(fine, reference);

    assert!(
        medium_error < coarse_error,
        "coarse_error={coarse_error} medium_error={medium_error}"
    );
    assert!(
        fine_error < medium_error,
        "medium_error={medium_error} fine_error={fine_error}"
    );
    assert!(
        fine_error < 0.08,
        "fine={fine} reference={reference} fine_error={fine_error}"
    );
}

#[test]
fn distorted_plate_mesh_retains_convergence_and_the_correct_response_sign() {
    let reference = navier_center_deflection();
    let coarse = solve_plate(4, 0.18);
    let fine = solve_plate(10, 0.18);
    let coarse_error = relative_error(coarse, reference);
    let fine_error = relative_error(fine, reference);

    assert!(coarse.is_sign_positive());
    assert!(fine.is_sign_positive());
    assert!(
        fine_error < coarse_error,
        "coarse_error={coarse_error} fine_error={fine_error}"
    );
    assert!(
        fine_error < 0.12,
        "fine={fine} reference={reference} fine_error={fine_error}"
    );
}

fn solve_plate(divisions: usize, distortion: f64) -> f64 {
    assert!(divisions > 1 && divisions.is_multiple_of(2));
    let node_width = divisions + 1;
    let node_count = node_width * node_width;
    let dof_count = node_count * BENDING_DOF_PER_NODE;
    let spacing = 1.0 / divisions as f64;
    let nodes = (0..node_width)
        .flat_map(|j| {
            (0..node_width).map(move |i| {
                let boundary = i == 0 || j == 0 || i == divisions || j == divisions;
                let offset = if boundary {
                    [0.0, 0.0]
                } else {
                    let parity = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                    [
                        distortion * spacing * parity,
                        distortion * spacing * 0.5 * parity,
                    ]
                };
                [
                    i as f64 * spacing + offset[0],
                    j as f64 * spacing + offset[1],
                    0.0,
                ]
            })
        })
        .collect::<Vec<_>>();
    let mut stiffness = vec![vec![0.0; dof_count]; dof_count];
    let mut rhs = vec![0.0; dof_count];
    for j in 0..divisions {
        for i in 0..divisions {
            let lower_left = j * node_width + i;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + node_width;
            let upper_right = upper_left + 1;
            for triangle in [
                [lower_left, lower_right, upper_right],
                [lower_left, upper_right, upper_left],
            ] {
                assemble_triangle(&nodes, triangle, &mut stiffness, &mut rhs);
            }
        }
    }

    let free = (0..dof_count)
        .filter(|dof| {
            let node = dof / BENDING_DOF_PER_NODE;
            let component = dof % BENDING_DOF_PER_NODE;
            let i = node % node_width;
            let j = node / node_width;
            component != 0 || !(i == 0 || j == 0 || i == divisions || j == divisions)
        })
        .collect::<Vec<_>>();
    let reduced = free
        .iter()
        .map(|row| free.iter().map(|column| stiffness[*row][*column]).collect())
        .collect::<Vec<Vec<f64>>>();
    let reduced_rhs = free.iter().map(|dof| rhs[*dof]).collect::<Vec<_>>();
    let solution = solve_dense(reduced, reduced_rhs);
    let center_node = (divisions / 2) * node_width + divisions / 2;
    let center_dof = center_node * BENDING_DOF_PER_NODE;
    let reduced_center = free
        .iter()
        .position(|dof| *dof == center_dof)
        .expect("plate center transverse displacement is free");
    solution[reduced_center]
}

fn assemble_triangle(
    nodes: &[[f64; 3]],
    triangle: [usize; 3],
    stiffness: &mut [Vec<f64>],
    rhs: &mut [f64],
) {
    let geometry = ShellElementGeometry {
        nodes_m: triangle.map(|node| nodes[node]),
        reference_axis: [1.0, 0.0, 0.0],
    };
    let local = global_stiffness_matrix(section(), material(), geometry).expect("valid triangle");
    let area = geometry
        .local_frame()
        .expect("valid triangle frame")
        .area_m2
        .canonical_value();
    let centroid = [
        geometry.nodes_m.iter().map(|node| node[0]).sum::<f64>() / 3.0,
        geometry.nodes_m.iter().map(|node| node[1]).sum::<f64>() / 3.0,
    ];
    let pressure =
        (std::f64::consts::PI * centroid[0]).sin() * (std::f64::consts::PI * centroid[1]).sin();
    for (local_node, global_node) in triangle.into_iter().enumerate() {
        rhs[global_node * BENDING_DOF_PER_NODE] += pressure * area / 3.0;
        for (local_component, shell_local_dof) in SHELL_BENDING_LOCAL_DOF.into_iter().enumerate() {
            let local_row = local_node * 6 + shell_local_dof;
            let global_row = global_node * BENDING_DOF_PER_NODE + local_component;
            for (other_local_node, other_global_node) in triangle.into_iter().enumerate() {
                for (other_component, other_shell_local_dof) in
                    SHELL_BENDING_LOCAL_DOF.into_iter().enumerate()
                {
                    let local_column = other_local_node * 6 + other_shell_local_dof;
                    let global_column = other_global_node * BENDING_DOF_PER_NODE + other_component;
                    stiffness[global_row][global_column] += local[local_row][local_column];
                }
            }
        }
    }
}

fn navier_center_deflection() -> f64 {
    let bending_rigidity = material().youngs_modulus_pa.canonical_value()
        * section().thickness_m.canonical_value().powi(3)
        / (12.0 * (1.0 - material().poisson_ratio.canonical_value().powi(2)));
    1.0 / (4.0 * std::f64::consts::PI.powi(4) * bending_rigidity)
}

fn section() -> ShellSection {
    ShellSection {
        thickness_m: runmat_types::physical::Length::new(0.01),
        reference_offset_m: runmat_types::physical::Length::new(0.0),
        shear_correction: 5.0 / 6.0,
    }
}

fn material() -> ShellMaterial {
    ShellMaterial {
        youngs_modulus_pa: runmat_types::physical::YoungsModulus::new(200.0e9),
        poisson_ratio: runmat_types::physical::PoissonRatio::new(0.3),
        shear_modulus_pa: runmat_types::physical::ShearModulus::new(200.0e9 / (2.0 * 1.3)),
    }
}

fn solve_dense(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    for pivot in 0..rhs.len() {
        let pivot_row = (pivot..rhs.len())
            .max_by(|left, right| {
                matrix[*left][pivot]
                    .abs()
                    .total_cmp(&matrix[*right][pivot].abs())
            })
            .expect("remaining pivot row");
        assert!(
            matrix[pivot_row][pivot].abs() > 1.0e-20,
            "singular plate system"
        );
        matrix.swap(pivot, pivot_row);
        rhs.swap(pivot, pivot_row);
        for row in pivot + 1..rhs.len() {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for column in pivot + 1..rhs.len() {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = vec![0.0; rhs.len()];
    for row in (0..rhs.len()).rev() {
        let remainder = (row + 1..rhs.len())
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        solution[row] = (rhs[row] - remainder) / matrix[row][row];
    }
    solution
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

//! Decomposition correctness against exact solutions and the single-rank solve.

use crate::partition::tests::box_mesh;
use crate::*;

/// Linear tetrahedral conduction, unit conductivity per material index + 1.
struct Conduction;

/// Linear tetrahedral isotropic elasticity, E = 1000, nu = 0.25.
struct Elastic;

fn gradients(coordinates: &[[f64; 3]]) -> ([[f64; 3]; 4], f64) {
    let [p0, p1, p2, p3] = [
        coordinates[0],
        coordinates[1],
        coordinates[2],
        coordinates[3],
    ];
    let a = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let b = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let c = [p3[0] - p0[0], p3[1] - p0[1], p3[2] - p0[2]];
    let cross = |u: [f64; 3], v: [f64; 3]| {
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    };
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let det = dot(a, cross(b, c));
    let g1 = cross(b, c).map(|value| value / det);
    let g2 = cross(c, a).map(|value| value / det);
    let g3 = cross(a, b).map(|value| value / det);
    let g0 = [
        -(g1[0] + g2[0] + g3[0]),
        -(g1[1] + g2[1] + g3[1]),
        -(g1[2] + g2[2] + g3[2]),
    ];
    ([g0, g1, g2, g3], det.abs() / 6.0)
}

impl ElementKernel for Conduction {
    fn field(&self) -> FieldKind {
        FieldKind::Scalar
    }
    fn nodes_per_element(&self) -> usize {
        4
    }
    fn element_matrix(
        &self,
        material: u32,
        coordinates: &[[f64; 3]],
        matrix: &mut [f64],
    ) -> Result<(), crate::ElementKernelError> {
        let (g, volume) = gradients(coordinates);
        let k = 1.0 + material as f64;
        for i in 0..4 {
            for j in 0..4 {
                matrix[i * 4 + j] =
                    k * volume * (g[i][0] * g[j][0] + g[i][1] * g[j][1] + g[i][2] * g[j][2]);
            }
        }
        Ok(())
    }
    fn result_width(&self) -> usize {
        3
    }
    fn element_result(
        &self,
        material: u32,
        coordinates: &[[f64; 3]],
        solution: &[f64],
        result: &mut [f64],
    ) -> Result<(), crate::ElementKernelError> {
        let (g, _) = gradients(coordinates);
        let k = 1.0 + material as f64;
        for (axis, value) in result.iter_mut().enumerate().take(3) {
            *value = -k
                * (0..4)
                    .map(|node| g[node][axis] * solution[node])
                    .sum::<f64>();
        }
        Ok(())
    }
}

impl ElementKernel for Elastic {
    fn field(&self) -> FieldKind {
        FieldKind::Elastic
    }
    fn nodes_per_element(&self) -> usize {
        4
    }
    fn element_matrix(
        &self,
        _material: u32,
        coordinates: &[[f64; 3]],
        matrix: &mut [f64],
    ) -> Result<(), crate::ElementKernelError> {
        let (g, volume) = gradients(coordinates);
        let (e, nu) = (1000.0, 0.25);
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let mu = e / (2.0 * (1.0 + nu));
        for a in 0..4 {
            for b in 0..4 {
                for i in 0..3 {
                    for j in 0..3 {
                        let mut value = lambda * g[a][i] * g[b][j] + mu * g[a][j] * g[b][i];
                        if i == j {
                            value += mu * (0..3).map(|k| g[a][k] * g[b][k]).sum::<f64>();
                        }
                        matrix[(a * 3 + i) * 12 + b * 3 + j] = volume * value;
                    }
                }
            }
        }
        Ok(())
    }
    fn result_width(&self) -> usize {
        1
    }
    fn element_result(
        &self,
        _material: u32,
        coordinates: &[[f64; 3]],
        solution: &[f64],
        result: &mut [f64],
    ) -> Result<(), crate::ElementKernelError> {
        let (g, _) = gradients(coordinates);
        // Volumetric strain.
        result[0] = (0..4)
            .map(|node| {
                (0..3)
                    .map(|axis| g[node][axis] * solution[node * 3 + axis])
                    .sum::<f64>()
            })
            .sum();
        Ok(())
    }
}

fn conduction_problem(n: usize) -> PartitionedProblem {
    let mesh = box_mesh(n, 1.0);
    let mut prescribed = Vec::new();
    for (node, point) in mesh.coordinates.iter().enumerate() {
        if point[0] == 0.0 {
            prescribed.push(PrescribedDof {
                dof: node as u64,
                value: 300.0,
            });
        } else if point[0] == 1.0 {
            prescribed.push(PrescribedDof {
                dof: node as u64,
                value: 350.0,
            });
        }
    }
    PartitionedProblem {
        element_materials: vec![0; mesh.element_count()],
        mesh,
        field: FieldKind::Scalar,
        prescribed,
        loads: Vec::new(),
    }
}

fn cantilever_problem(n: usize) -> PartitionedProblem {
    let mesh = box_mesh(n, 1.0);
    let mut prescribed = Vec::new();
    let mut loads = Vec::new();
    let loaded = mesh
        .coordinates
        .iter()
        .filter(|point| point[0] == 1.0)
        .count() as f64;
    for (node, point) in mesh.coordinates.iter().enumerate() {
        if point[0] == 0.0 {
            for component in 0..3 {
                prescribed.push(PrescribedDof {
                    dof: node as u64 * 3 + component,
                    value: 0.0,
                });
            }
        } else if point[0] == 1.0 {
            loads.push(PointLoad {
                dof: node as u64 * 3 + 2,
                value: -1.0 / loaded,
            });
        }
    }
    PartitionedProblem {
        element_materials: vec![0; mesh.element_count()],
        mesh,
        field: FieldKind::Elastic,
        prescribed,
        loads,
    }
}

#[test]
fn conduction_reproduces_the_exact_linear_field_for_every_rank_count() {
    let problem = conduction_problem(8);
    // Recovering a derivative amplifies solution error. Request sufficient
    // solve accuracy for the absolute heat-flux bound checked below.
    let settings = KrylovSettings {
        relative_tolerance: 1e-12,
        ..KrylovSettings::default()
    };
    for ranks in [1, 2, 3, 4, 7] {
        let (solution, quality, layout) = solve_in_threads(
            &problem,
            PartitionOptions::new(ranks),
            settings,
            ranks,
            &Conduction,
            &FaerCholeskyFactory,
        )
        .unwrap();
        assert_eq!(layout.ranks, ranks);
        assert!(quality.smallest_part > 0);
        for (node, point) in problem.mesh.coordinates.iter().enumerate() {
            let exact = 300.0 + 50.0 * point[0];
            assert!(
                (solution.solution[node] - exact).abs() < 1e-6,
                "{ranks}: {} vs {exact}",
                solution.solution[node]
            );
        }
        let exact_flux = [-50.0, 0.0, 0.0];
        assert_eq!(solution.result_width, exact_flux.len());
        assert_eq!(
            solution.element_results.len(),
            problem.mesh.element_count() * exact_flux.len()
        );
        for (element, flux) in solution
            .element_results
            .chunks_exact(exact_flux.len())
            .enumerate()
        {
            for (axis, (&actual, expected)) in flux.iter().zip(exact_flux).enumerate() {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "ranks={ranks}, element={element}, axis={axis}, flux={actual:.16e}, expected={expected}, iterations={}, recursive_residual={:.16e}, true_residual={:.16e}",
                    solution.iterations,
                    solution.relative_residual,
                    solution.true_relative_residual,
                );
            }
        }
        assert!(solution.true_relative_residual < 10.0 * settings.relative_tolerance);
    }
}

#[test]
fn elastic_decomposition_matches_the_single_rank_solution_with_bounded_iterations() {
    let problem = cantilever_problem(8);
    let reference = solve_in_threads(
        &problem,
        PartitionOptions::new(1),
        KrylovSettings::default(),
        1,
        &Elastic,
        &FaerCholeskyFactory,
    )
    .unwrap()
    .0;
    assert!(reference.iterations <= 2);
    let scale = reference
        .solution
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    assert!(scale > 0.0);
    let mut iterations = Vec::new();
    for ranks in [2, 4, 8] {
        let (solution, _, _) = solve_in_threads(
            &problem,
            PartitionOptions::new(ranks),
            KrylovSettings::default(),
            ranks,
            &Elastic,
            &FaerCholeskyFactory,
        )
        .unwrap();
        let difference = solution
            .solution
            .iter()
            .zip(&reference.solution)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            difference <= 1e-7 * scale,
            "{ranks}: {difference} vs {scale}"
        );
        assert!(solution.coarse_dimension == 6 * ranks as u64);
        iterations.push(solution.iterations);
    }
    // The rigid-body coarse space keeps the iteration count bounded as ranks grow.
    assert!(
        iterations.iter().all(|&count| count < 120),
        "{iterations:?}"
    );
}

#[test]
fn coarse_space_reduces_iterations() {
    let problem = cantilever_problem(8);
    let run = |coarse_space| {
        solve_in_threads(
            &problem,
            PartitionOptions::new(8),
            KrylovSettings {
                coarse_space,
                ..KrylovSettings::default()
            },
            4,
            &Elastic,
            &FaerCholeskyFactory,
        )
        .unwrap()
        .0
        .iterations
    };
    let with = run(true);
    let without = run(false);
    assert!(with < without, "coarse {with} vs one-level {without}");
}

#[test]
fn rank_frames_round_trip() {
    let problem = cantilever_problem(3);
    let (parts, _) = partition_elements(&problem.mesh, PartitionOptions::new(3)).unwrap();
    let (inputs, _) = extract_rank_inputs(&problem, &parts, 3, KrylovSettings::default()).unwrap();
    for input in inputs {
        assert_eq!(SubdomainInput::decode(&input.encode()).unwrap(), input);
    }
    let plan = ProcessPlan::blocked(1, 2, 3, 4);
    assert_eq!(ProcessPlan::decode(&plan.encode()).unwrap(), plan);
}

#[test]
fn results_are_bitwise_identical_for_every_process_count() {
    let problem = cantilever_problem(6);
    let solve = |processes| {
        solve_in_threads(
            &problem,
            PartitionOptions::new(6),
            KrylovSettings::default(),
            processes,
            &Elastic,
            &FaerCholeskyFactory,
        )
        .unwrap()
        .0
    };
    let single = solve(1);
    for processes in [2, 3, 6] {
        let placed = solve(processes);
        assert_eq!(placed.iterations, single.iterations);
        assert_eq!(placed.solution, single.solution, "{processes} processes");
        assert_eq!(placed.element_results, single.element_results);
    }
}

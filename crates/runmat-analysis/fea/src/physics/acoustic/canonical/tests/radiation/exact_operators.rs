//! Independent affine P2 simplex algebra: no production shape/quadrature/matrix oracle.
//! These checks certify discrete entries and lifting, not continuum convergence.
use super::*;
use runmat_meshing_core::CanonicalMeshingContract;

type Polynomial = Vec<([usize; 4], f64)>;
type Matrix = [[f64; 10]; 10];

fn factorial(n: usize) -> f64 {
    (1..=n).fold(1.0, |value, next| value * next as f64)
}

// ∫_tet λ^α = 6 V ∏α!/(3+|α|)!; ∫_face λ^α = 2 A ∏α!/(2+|α|)!.
// The omitted vertex has λ=0 identically on the selected planar face.
fn integral(poly: &Polynomial, measure: f64, omitted: Option<usize>) -> f64 {
    let dimension = if omitted.is_some() { 2 } else { 3 };
    poly.iter()
        .filter(|(powers, _)| omitted.is_none_or(|i| powers[i] == 0))
        .map(|(powers, coefficient)| {
            coefficient
                * factorial(dimension)
                * measure
                * powers
                    .iter()
                    .map(|&power| factorial(power))
                    .product::<f64>()
                / factorial(dimension + powers.iter().sum::<usize>())
        })
        .sum()
}

fn multiply_poly(a: &Polynomial, b: &Polynomial) -> Polynomial {
    a.iter()
        .flat_map(|(pa, ca)| {
            b.iter()
                .map(move |(pb, cb)| (std::array::from_fn(|i| pa[i] + pb[i]), ca * cb))
        })
        .collect()
}

fn shapes() -> [Polynomial; 10] {
    // Literal canonical Tet10 edge order, independently written rather than imported.
    let edges = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]];
    std::array::from_fn(|i| {
        let mut powers = [0; 4];
        if i < 4 {
            powers[i] = 1;
            let linear = powers;
            powers[i] = 2;
            vec![(powers, 2.0), (linear, -1.0)]
        } else {
            for vertex in edges[i - 4] {
                powers[vertex] = 1;
            }
            vec![(powers, 4.0)]
        }
    })
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}

struct ExactOperators {
    stiffness: Matrix,
    mass: Matrix,
    radiation: Matrix,
}

fn exact(corners: [[f64; 3]; 4], omitted: usize, rho: f64, c: f64) -> ExactOperators {
    let edges = [1, 2, 3].map(|i| subtract(corners[i], corners[0]));
    let det = dot(edges[0], cross(edges[1], edges[2]));
    assert!(det > 0.0);
    // Reciprocal edge vectors derive physical barycentric gradients directly.
    let mut gradients = [[0.0; 3]; 4];
    gradients[1] = cross(edges[1], edges[2]).map(|value| value / det);
    gradients[2] = cross(edges[2], edges[0]).map(|value| value / det);
    gradients[3] = cross(edges[0], edges[1]).map(|value| value / det);
    gradients[0] =
        std::array::from_fn(|axis| -(gradients[1][axis] + gradients[2][axis] + gradients[3][axis]));
    let shapes = shapes();
    let derivatives: [[Polynomial; 3]; 10] = std::array::from_fn(|i| {
        std::array::from_fn(|axis| {
            shapes[i]
                .iter()
                .flat_map(|(powers, coefficient)| {
                    (0..4)
                        .filter(|&vertex| powers[vertex] > 0)
                        .map(move |vertex| {
                            let mut reduced = *powers;
                            reduced[vertex] -= 1;
                            (
                                reduced,
                                coefficient * powers[vertex] as f64 * gradients[vertex][axis],
                            )
                        })
                })
                .collect()
        })
    });
    let face = (0..4).filter(|&i| i != omitted).collect::<Vec<_>>();
    let area_vector = cross(
        subtract(corners[face[1]], corners[face[0]]),
        subtract(corners[face[2]], corners[face[0]]),
    );
    let area = dot(area_vector, area_vector).sqrt() / 2.0;
    ExactOperators {
        stiffness: std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                (0..3)
                    .map(|axis| {
                        integral(
                            &multiply_poly(&derivatives[i][axis], &derivatives[j][axis]),
                            det / 6.0,
                            None,
                        ) / rho
                    })
                    .sum()
            })
        }),
        mass: std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                integral(&multiply_poly(&shapes[i], &shapes[j]), det / 6.0, None) / (rho * c * c)
            })
        }),
        radiation: std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                integral(&multiply_poly(&shapes[i], &shapes[j]), area, Some(omitted)) / (rho * c)
            })
        }),
    }
}

fn fixture(oblique: bool) -> (AnalysisModel, SolverMeshArtifact, AnalysisProjection) {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    if oblique {
        // Genuinely oblique affine geometry (not merely axis-aligned scale).
        let transform = [[1.3, 0.2, 0.1], [0.1, 1.1, 0.3], [0.2, 0.1, 0.9]];
        for node in &mut mesh.topology.nodes {
            node.coordinates_m =
                runmat_types::physical::Quantity::<runmat_types::physical::kind::Position, _>::new(
                    transform.map(|row| dot(row, node.coordinates_m.canonical_value())),
                );
        }
        mesh.seal_canonical_digest().unwrap();
    }
    mesh.validate_canonical().unwrap();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let mut model = model();
    model.sequence.steps[0].independent_domain = IndependentDomain::FrequencyHertz(vec![0.2]);
    (model, mesh, projection)
}

fn boundary_setup(
    model: &mut AnalysisModel,
    projection: &AnalysisProjection,
    face: usize,
    pressure: Option<(usize, [f64; 2])>,
) -> usize {
    let target = |ordinal: usize| {
        RegionTarget::boundary(
            projection.boundary_faces[ordinal].provenance[0]
                .analysis_selection_id()
                .unwrap(),
        )
    };
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "radiation".into(),
        target: target(face),
        operator: BoundaryOperator::AcousticRadiation,
    });
    if let Some((pressure_face, amplitude)) = pressure {
        model.signals.push(PhysicalSignal::Pressure(complex_signal(
            Pressure::new(amplitude[0]),
            Pressure::new(amplitude[1]),
        )));
        model
            .constraints
            .push(AnalysisConstraint::Acoustic(AcousticConstraint {
                constraint_id: "pressure".into(),
                kind: AcousticConstraintKind::PrescribedPressure {
                    target: target(pressure_face),
                    value: PrescribedValue::Signal(SignalId("drive".into())),
                },
            }));
    }
    let element = &projection.elements[0];
    (0..4)
        .find(|&i| {
            !projection.boundary_faces[face]
                .node_indices
                .contains(&element.node_indices[i])
        })
        .unwrap()
}

fn assemble(
    model: &AnalysisModel,
    mesh: &SolverMeshArtifact,
    projection: AnalysisProjection,
) -> AcousticAssembly {
    let properties = crate::properties::compile_solid_property_map(model, mesh).unwrap();
    let step = &model.sequence.steps[0];
    let sample = ExcitationSamplePoint::frequency(0.2);
    let sources = realize_step_excitations(model, step, sample).unwrap();
    AcousticAssembly::assemble(
        projection,
        &properties,
        runmat_types::physical::AngularFrequency::new(std::f64::consts::TAU * 0.2),
    )
    .unwrap()
    .with_boundaries(model, step, &sources, sample)
    .unwrap()
}

#[test]
fn simplex_oracle_matches_literal_unit_tet10_and_tri6_entries() {
    let operators = exact(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        3,
        3.0,
        2.0,
    );
    // ∫ N0² = 1/420, ∫ |∇N0|² = 3/10 on this unit tetrahedron.
    assert!((operators.mass[0][0] - 1.0 / 5040.0).abs() < 1e-15);
    assert!((operators.stiffness[0][0] - 0.1).abs() < 1e-15);
    // On the z=0 face, A=1/2: ∫ N0²=A/30, ∫ N0 N1=-A/180.
    assert!((operators.radiation[0][0] - 1.0 / 360.0).abs() < 1e-15);
    assert!((operators.radiation[0][1] + 1.0 / 2160.0).abs() < 1e-15);
    assert_eq!(operators.radiation[3], [0.0; 10]);
}

// witness: FEA-ACOUSTIC-OPERATOR-001/integrals @ acoustic_harmonic/tet10/numerical_kernel;
//   boundary: Every Tet10/Tri6 matrix entry on all faces; independent simplex moments. No Tet4 or
//   public topology audit credit.
#[test]
fn tet10_and_tri6_all_entries_match_independent_simplex_moments_on_every_face() {
    for oblique in [false, true] {
        for face in 0..4 {
            let (mut model, mesh, projection) = fixture(oblique);
            let omitted = boundary_setup(&mut model, &projection, face, None);
            let element = &projection.elements[0];
            let nodes = element.node_indices.clone();
            let corners =
                std::array::from_fn(|i| projection.nodes[nodes[i]].coordinates_m.canonical_value());
            let expected = exact(corners, omitted, 3.0, 2.0);
            let assembly = assemble(&model, &mesh, projection);
            for (i, &row) in nodes.iter().enumerate() {
                for (j, &column) in nodes.iter().enumerate() {
                    for (name, actual, reference) in [
                        (
                            "K",
                            assembly
                                .stiffness
                                .coefficient(row, column)
                                .unwrap()
                                .canonical_value(),
                            expected.stiffness[i][j],
                        ),
                        (
                            "M",
                            assembly
                                .mass
                                .coefficient(row, column)
                                .unwrap()
                                .canonical_value(),
                            expected.mass[i][j],
                        ),
                        (
                            "B",
                            assembly
                                .impedance
                                .coefficient(row, column)
                                .unwrap()
                                .canonical_value(),
                            expected.radiation[i][j],
                        ),
                    ] {
                        assert!((actual-reference).abs() < 2e-12, "{name}[{i},{j}] oblique={oblique} face={face}: actual={actual} exact={reference}");
                    }
                }
            }
        }
    }
}

// Independent tiny dense elimination with row pivoting; never calls shared sparse/Krylov solve.
fn dense_solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    for column in 0..rhs.len() {
        let pivot = (column..rhs.len())
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        assert!(matrix[column][column].abs() > 1e-12);
        let pivot_rhs = rhs[column];
        let (fixed, lower) = matrix.split_at_mut(column + 1);
        let pivot_row = &fixed[column];
        for (row, row_rhs) in lower.iter_mut().zip(rhs.iter_mut().skip(column + 1)) {
            let factor = row[column] / pivot_row[column];
            for (entry, &pivot_entry) in row.iter_mut().zip(pivot_row).skip(column) {
                *entry -= factor * pivot_entry;
            }
            *row_rhs -= factor * pivot_rhs;
        }
    }
    let mut answer = vec![0.0; rhs.len()];
    for row in (0..rhs.len()).rev() {
        answer[row] = (rhs[row]
            - (row + 1..rhs.len())
                .map(|j| matrix[row][j] * answer[j])
                .sum::<f64>())
            / matrix[row][row];
    }
    answer
}

// witness: FEA-ACOUSTIC-BOUNDARY-001/lifting @ acoustic_harmonic/tet10/numerical_kernel;
//   boundary: Tet10 lifted complex system checked against independent dense oracle; no generic
//   solve or conditioning claim.
#[test]
fn complex_prescribed_pressure_lifting_matches_independent_reduced_block_system() {
    let omega = std::f64::consts::TAU * 0.2;
    for oblique in [false, true] {
        for radiation_face in 0..4 {
            let pressure_face = (radiation_face + 1) % 4;
            for amplitude in [[1.0, 0.4], [-0.4, 1.0]] {
                let (mut model, mesh, projection) = fixture(oblique);
                let omitted = boundary_setup(
                    &mut model,
                    &projection,
                    radiation_face,
                    Some((pressure_face, amplitude)),
                );
                let nodes = projection.elements[0].node_indices.clone();
                let corners = std::array::from_fn(|i| {
                    projection.nodes[nodes[i]].coordinates_m.canonical_value()
                });
                let expected = exact(corners, omitted, 3.0, 2.0);
                let constrained = (0..10)
                    .filter(|&i| {
                        projection.boundary_faces[pressure_face]
                            .node_indices
                            .contains(&nodes[i])
                    })
                    .collect::<Vec<_>>();
                assert_eq!(constrained.len(), 6);
                let free = (0..10)
                    .filter(|i| !constrained.contains(i))
                    .collect::<Vec<_>>();
                let h: Matrix = std::array::from_fn(|i| {
                    std::array::from_fn(|j| {
                        expected.stiffness[i][j] - omega * omega * expected.mass[i][j]
                    })
                });
                let n = free.len();
                let mut block = vec![vec![0.0; 2 * n]; 2 * n];
                let mut rhs = vec![0.0; 2 * n];
                for (r, &i) in free.iter().enumerate() {
                    for &j in &constrained {
                        rhs[r] -= h[i][j] * amplitude[0]
                            - omega * expected.radiation[i][j] * amplitude[1];
                        rhs[r + n] -= h[i][j] * amplitude[1]
                            + omega * expected.radiation[i][j] * amplitude[0];
                    }
                    for (s, &j) in free.iter().enumerate() {
                        block[r][s] = h[i][j];
                        block[r + n][s + n] = h[i][j];
                        block[r][s + n] = -omega * expected.radiation[i][j];
                        block[r + n][s] = omega * expected.radiation[i][j];
                    }
                }
                let reference = dense_solve(block, rhs);
                let assembly = assemble(&model, &mesh, projection);
                for (i, &node) in nodes.iter().enumerate() {
                    assert_eq!(
                        assembly.prescribed.canonical_values()[node].is_some(),
                        constrained.contains(&i)
                    );
                }
                let actual = super::super::super::response::solve(
                    assembly,
                    0.2,
                    AcousticHarmonicPolicy {
                        relative_tolerance: 1e-12,
                        ..Default::default()
                    },
                    1e-8,
                )
                .unwrap();
                for (r, &i) in free.iter().enumerate() {
                    let p = actual.pressure_pa.canonical_values()[nodes[i]];
                    assert!((p.real-reference[r]).abs()<1e-9 && (p.imaginary-reference[r+n]).abs()<1e-9,
                        "free local node{i} oblique={oblique} radiation_face={radiation_face} phase={amplitude:?}");
                }
                for &i in &constrained {
                    assert_eq!(
                        [
                            actual.pressure_pa.canonical_values()[nodes[i]].real,
                            actual.pressure_pa.canonical_values()[nodes[i]].imaginary
                        ],
                        amplitude
                    );
                }
            }
        }
    }
}

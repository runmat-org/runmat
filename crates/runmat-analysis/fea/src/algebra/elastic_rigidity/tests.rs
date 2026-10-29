use super::*;

/// Linear-elastic Tet4 stiffness (12 × 12, node-major) from first principles.
fn tet4_stiffness(points: &[[f64; 3]; 4], young: f64, poisson: f64) -> Vec<f64> {
    let m = |r: usize, c: usize| points[r][c] - points[0][c];
    let j = [
        [m(1, 0), m(2, 0), m(3, 0)],
        [m(1, 1), m(2, 1), m(3, 1)],
        [m(1, 2), m(2, 2), m(3, 2)],
    ];
    let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
    // Inverse transpose rows give the gradients of the barycentric coordinates 1..3.
    let inv = |r: usize, c: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
        (j[c1][r1] * j[c2][r2] - j[c1][r2] * j[c2][r1]) / det
    };
    let mut grad = [[0.0; 3]; 4];
    for node in 1..4 {
        for axis in 0..3 {
            grad[node][axis] = inv(node - 1, axis);
            grad[0][axis] -= grad[node][axis];
        }
    }
    let lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu = young / (2.0 * (1.0 + poisson));
    let volume = det.abs() / 6.0;
    let mut k = vec![0.0; 144];
    for a in 0..4 {
        for b in 0..4 {
            let dot = (0..3).map(|i| grad[a][i] * grad[b][i]).sum::<f64>();
            for i in 0..3 {
                for jx in 0..3 {
                    let mut value =
                        lambda * grad[a][i] * grad[b][jx] + mu * grad[a][jx] * grad[b][i];
                    if i == jx {
                        value += mu * dot;
                    }
                    k[(3 * a + i) * 12 + 3 * b + jx] = volume * value;
                }
            }
        }
    }
    k
}

fn dofs(nodes: &[u32]) -> Vec<u32> {
    nodes
        .iter()
        .flat_map(|&n| (0..3).map(move |a| 3 * n + a))
        .collect()
}

/// Streams the given tetrahedra (node lists into `points`) through a capture and finishes it.
fn prove(
    points: &[[f64; 3]],
    tets: &[[u32; 4]],
    fixed_nodes: &[u32],
    mutate: impl Fn(usize, &mut Vec<f64>),
) -> Result<ElasticRigidityProof, ElasticRigidityError> {
    let mut fixed = vec![false; points.len() * 3];
    for &node in fixed_nodes {
        fixed[3 * node as usize..3 * node as usize + 3].fill(true);
    }
    let mut capture = ElasticRigidityCapture::new(
        points.iter().enumerate().map(|(i, p)| (100 + i as u64, *p)),
        tets.iter()
            .enumerate()
            .map(|(e, t)| (10 + e as u64, t.to_vec())),
        fixed,
    )?;
    for (e, tet) in tets.iter().enumerate() {
        let corners = tet.map(|n| points[n as usize]);
        let mut k = tet4_stiffness(&corners, 200.0e9, 0.3);
        mutate(e, &mut k);
        capture.capture(&dofs(tet), &k)?;
    }
    capture.finish()
}

const POINTS: [[f64; 3]; 6] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, -0.5, 0.7],
];

// witness: FEA-SHARED-OPERATORS-001/compatible_solver_and_preconditioner_selection @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Face-glued elastic tetrahedra with a three-node anchor are proved positive definite without a global factor.
#[test]
fn face_glued_and_anchored_tetrahedra_are_proved() {
    let proof = prove(
        &POINTS[..5],
        &[[0, 1, 2, 3], [1, 2, 3, 4]],
        &[0, 1, 2],
        |_, _| {},
    )
    .unwrap();
    assert_eq!((proof.elements(), proof.components()), (2, 1));
    assert!(proof.scalar_operations() > 0 && proof.resident_bytes() > 0);
}

// witness: FEA-SHARED-OPERATORS-001/compatible_solver_and_preconditioner_selection @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: A hinge (two tetrahedra sharing only an edge) is a mechanism and is refused with a typed reason naming the unanchored component.
#[test]
fn hinge_glued_by_two_nodes_is_refused_as_a_mechanism() {
    // Elements share only nodes 0 and 1; the anchor fixes element 10 alone.
    let error = prove(
        &POINTS,
        &[[0, 1, 2, 3], [0, 1, 4, 5]],
        &[0, 2, 3],
        |_, _| {},
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            ElasticRigidityError::UnanchoredComponent {
                representative_element: 11,
                elements: 1,
                ..
            }
        ),
        "{error:?}"
    );
}

// witness: FEA-SHARED-OPERATORS-001/compatible_solver_and_preconditioner_selection @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: An anchor that leaves a rotation free (two fixed nodes) or no anchor at all is refused; the refusal counts the constrained DOFs.
#[test]
fn collinear_or_missing_anchor_is_refused() {
    let free_rotation = prove(&POINTS[..4], &[[0, 1, 2, 3]], &[0, 1], |_, _| {}).unwrap_err();
    assert!(matches!(
        free_rotation,
        ElasticRigidityError::UnanchoredComponent {
            constrained_dofs: 6,
            ..
        }
    ));
    let floating = prove(&POINTS[..4], &[[0, 1, 2, 3]], &[], |_, _| {}).unwrap_err();
    assert!(matches!(
        floating,
        ElasticRigidityError::UnanchoredComponent {
            constrained_dofs: 0,
            ..
        }
    ));
}

// witness: FEA-SHARED-OPERATORS-001/compatible_solver_and_preconditioner_selection @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: An element block that is indefinite on the rigid-motion complement, or that does not annihilate rigid motions, is refused naming the element.
#[test]
fn element_without_energy_margin_or_rigid_kernel_is_refused() {
    let indefinite = prove(
        &POINTS[..5],
        &[[0, 1, 2, 3], [1, 2, 3, 4]],
        &[0, 1, 2],
        |e, k| {
            if e == 1 {
                k.iter_mut().for_each(|v| *v = -*v);
            }
        },
    )
    .unwrap_err();
    assert!(
        matches!(
            indefinite,
            ElasticRigidityError::ElementEnergyMargin { element: 11, smallest_ratio_bits, .. }
                if f64::from_bits(smallest_ratio_bits) < 0.0
        ),
        "{indefinite:?}"
    );
    let grounded = prove(
        &POINTS[..5],
        &[[0, 1, 2, 3], [1, 2, 3, 4]],
        &[0, 1, 2],
        |e, k| {
            if e == 0 {
                k[0] *= 2.0;
            }
        },
    )
    .unwrap_err();
    assert!(matches!(
        grounded,
        ElasticRigidityError::RigidMotionEnergy { element: 10, .. }
    ));
}

#[test]
fn free_node_outside_every_element_is_refused() {
    let error = prove(&POINTS[..5], &[[0, 1, 2, 3]], &[0, 1, 2], |_, _| {}).unwrap_err();
    assert_eq!(error, ElasticRigidityError::UncoveredNode { node: 104 });
}

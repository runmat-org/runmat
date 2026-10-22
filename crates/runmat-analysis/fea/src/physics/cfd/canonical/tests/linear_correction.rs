use super::*;
use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};

#[test]
fn correction_resolves_small_defect_against_large_background() {
    let mut f = case(1, ReferenceCase::Convective, false);
    f.policy.linear_relative_tolerance = 1e-8;
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let n = p.topology.cells.len();
    let mut assembler = RealSparseAssembler::new(
        n,
        n,
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
    );
    for i in 0..n {
        assembler.add(i, i, if i < 2 { 3.0 } else { 1.0 });
    }
    assembler.add(0, 1, 1.0);
    assembler.add(1, 0, 1.0);
    let a = assembler.finish().unwrap();
    let current = vec![1.0; n];
    let delta = 1e-9;
    let mut b = vec![1.0; n];
    b[0] = 4.0 + 2.0 * delta;
    b[1] = 4.0 - 2.0 * delta;
    // Independent exact solution: the antisymmetric eigenvector has eigenvalue 2.
    let (x, iterations) =
        linear::solve(&p, &a, &b, &current, linear::LinearSystem::Momentum).unwrap();
    assert!(iterations > 0);
    assert!((x[0] - (1.0 + delta)).abs() < 1e-15);
    assert!((x[1] - (1.0 - delta)).abs() < 1e-15);
    assert!(x[2..].iter().all(|v| *v == 1.0));
    assert!((3.0 * x[0] + x[1] - b[0]).abs() < 1e-14);
    assert!((x[0] + 3.0 * x[1] - b[1]).abs() < 1e-14);

    b[0] = 4.0;
    b[1] = 4.0;
    let (unchanged, iterations) =
        linear::solve(&p, &a, &b, &current, linear::LinearSystem::Momentum).unwrap();
    assert_eq!(unchanged, current);
    assert_eq!(iterations, 0);
    assert!(linear::solve(
        &p,
        &a,
        &b[..n - 1],
        &current,
        linear::LinearSystem::Momentum
    )
    .is_err());
    b[0] = f64::NAN;
    assert!(linear::solve(&p, &a, &b, &current, linear::LinearSystem::Momentum).is_err());
}

use super::*;
use clarabel::algebra::CscMatrix;
use clarabel::solver::{NonnegativeConeT, SupportedConeT, ZeroConeT};
use futures::executor::block_on;
struct Fixture {
    a: CscMatrix<f64>,
    b: Vec<f64>,
    f: Vec<f64>,
    cones: Vec<SupportedConeT<f64>>,
}
impl Fixture {
    fn view(&self) -> Model<'_> {
        Model {
            p: None,
            a: &self.a,
            b: &self.b,
            f: &self.f,
            cones: &self.cones,
            identity: runmat_builtins::LINPROG_CATALOG_ENTRY.identity,
            objective_scale: 1.0,
        }
    }
}
fn cooperation() -> ScalarCooperation {
    ScalarCooperation::for_builtin(runmat_builtins::LINPROG_CATALOG_ENTRY.identity, None)
}
#[test]
fn correction_uses_original_signed_singleton_and_equality_bounds() {
    let fixtures = [
        (
            Fixture {
                a: CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![-2.0]),
                b: vec![-3.0],
                f: vec![1.0],
                cones: vec![NonnegativeConeT(1)],
            },
            BigRational::new(3.into(), 2.into()),
        ),
        (
            Fixture {
                a: CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![-2.0]),
                b: vec![-3.0],
                f: vec![1.0],
                cones: vec![ZeroConeT(1)],
            },
            BigRational::new(3.into(), 2.into()),
        ),
    ];
    for (fixture, expected) in fixtures {
        let view = fixture.view();
        let mut work = cooperation();
        let certificates = block_on(LinearCertificates::borrow(&view, &mut work))
            .unwrap()
            .unwrap();
        assert!(!certificates.is_infeasible());
        let bound = block_on(certificates.lower_bound(&[0.0], &mut work))
            .unwrap()
            .unwrap();
        assert_eq!(bound.value(), &expected);
    }
}
#[test]
fn absent_or_conflicting_bounds_do_not_manufacture_finite_corrections() {
    let free = Fixture {
        a: CscMatrix::zeros((0, 1)),
        b: vec![],
        f: vec![1.0],
        cones: vec![],
    };
    let view = free.view();
    let mut work = cooperation();
    let certificates = block_on(LinearCertificates::borrow(&view, &mut work))
        .unwrap()
        .unwrap();
    assert!(!certificates.is_infeasible());
    assert!(block_on(certificates.lower_bound(&[], &mut work))
        .unwrap()
        .is_none());
    let conflict = Fixture {
        a: CscMatrix::new(2, 1, vec![0, 2], vec![0, 1], vec![-1.0, 1.0]),
        b: vec![-2.0, 1.0],
        f: vec![1.0],
        cones: vec![NonnegativeConeT(2)],
    };
    let view = conflict.view();
    let certificates = block_on(LinearCertificates::borrow(&view, &mut work))
        .unwrap()
        .unwrap();
    assert!(certificates.is_infeasible());
    assert!(block_on(certificates.lower_bound(&[0.0, 0.0], &mut work))
        .unwrap()
        .is_none());
}
#[test]
fn later_recovery_stronger_bound_is_not_hidden_by_first_finite_dyadic() {
    let fixture = Fixture {
        a: CscMatrix::new(2, 1, vec![0, 2], vec![0, 1], vec![-3.0, 1.0]),
        b: vec![-1.0, 100.0],
        f: vec![1.0],
        cones: vec![NonnegativeConeT(2)],
    };
    let view = fixture.view();
    let mut work = cooperation();
    let certificates = block_on(LinearCertificates::borrow(&view, &mut work))
        .unwrap()
        .unwrap();
    let bound = block_on(certificates.lower_bound(&[(1.0_f64 / 3.0).next_up(), 0.0], &mut work))
        .unwrap()
        .unwrap();
    assert_eq!(bound.value(), &BigRational::new(1.into(), 3.into()));
}
#[test]
fn equality_dual_signs_and_all_added_bound_rows_are_in_original_stationarity() {
    let fixture = Fixture {
        a: CscMatrix::new(3, 1, vec![0, 3], vec![0, 1, 2], vec![-2.0, -1.0, 1.0]),
        b: vec![-3.0, -1.0, 2.0],
        f: vec![1.0],
        cones: vec![ZeroConeT(1), NonnegativeConeT(2)],
    };
    let view = fixture.view();
    let mut work = cooperation();
    let certificates = block_on(LinearCertificates::borrow(&view, &mut work))
        .unwrap()
        .unwrap();
    // Negative equality multipliers are free; negative inequality proposals
    // project to zero only with their exact remaining residual retained.
    let bound = block_on(certificates.lower_bound(&[-2.0, -5.0, -3.0], &mut work))
        .unwrap()
        .unwrap();
    assert_eq!(bound.value(), &BigRational::new(3.into(), 2.into()));
}
#[test]
fn exact_primal_check_and_objective_share_original_terminal_owner() {
    let fixture = Fixture {
        a: CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![3.0]),
        b: vec![1.0],
        f: vec![2.0],
        cones: vec![ZeroConeT(1)],
    };
    let view = fixture.view();
    let mut work = cooperation();
    let point = vec![BigRational::new(1.into(), 3.into())];
    assert!(block_on(terminal::feasible_rational_point(&view, &point, &mut work)).unwrap());
    assert_eq!(
        block_on(objective(&view, &point, &mut work)).unwrap(),
        Some(BigRational::new(2.into(), 3.into()))
    );
    assert!(!block_on(terminal::feasible_rational_point(
        &view,
        &[BigRational::zero()],
        &mut work
    ))
    .unwrap());
}

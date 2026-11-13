//! Both rounded-model and original-model acceptance are necessary.
use super::*;
use conic::LinearRowRole;
struct Original {
    accept: bool,
    restricted: bool,
    solutions: usize,
    certificates: usize,
}
impl Validation for Original {
    async fn restricted_convexity(
        &mut self,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<RestrictedCurvature> {
        Ok(if self.restricted {
            RestrictedCurvature::Convex
        } else {
            RestrictedCurvature::Nonconvex
        })
    }
    async fn recession(
        &mut self,
        _: &[num_rational::BigRational],
        _: &[num_rational::BigRational],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(self.accept)
    }
    async fn stopped_candidate(
        &mut self,
        _: &[f64],
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(self.accept)
    }
    async fn objective_limit(
        &mut self,
        _: &[f64],
        cost: f64,
        limit: f64,
        _: f64,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<bool>> {
        Ok(Some(cost < limit && self.accept))
    }
    fn original_convexity(&self) -> bool {
        true
    }
    async fn solution(
        &mut self,
        point: &[f64],
        _: &[LinearRowRole],
        _: &[f64],
        _: f64,
        _: f64,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.solutions += 1;
        assert!((point[0] - 2.).abs() < 1e-5);
        Ok(self.accept)
    }
    async fn incumbent(
        &mut self,
        _: &[f64],
        _: f64,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        Ok(self.accept)
    }
    async fn certificate(
        &mut self,
        _: CertificateCandidate<'_>,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.certificates += 1;
        Ok(self.accept)
    }
}
fn problem(curvature: f64) -> Problem {
    Problem {
        p: Matrix::new(1, 1, vec![0, 1], vec![0], vec![curvature]),
        f: vec![-4.],
        a_ineq: Matrix::new(0, 1, vec![0, 0], vec![], vec![]),
        b_ineq: vec![],
        a_eq: Matrix::new(0, 1, vec![0, 0], vec![], vec![]),
        b_eq: vec![],
        lb: vec![f64::NEG_INFINITY],
        ub: vec![f64::INFINITY],
    }
}
#[test]
fn compiled_qp_preserves_original_acceptance_psd_and_terminal_frontiers() {
    futures::executor::block_on(async {
        for accept in [false, true] {
            let mut authority = Original {
                accept,
                restricted: true,
                solutions: 0,
                certificates: 0,
            };
            let mut cooperation =
                ScalarCooperation::new(runmat_builtins::OptimSolver::Quadprog, None);
            let result = solve_fixture(
                problem(2.),
                None,
                Products::Primal,
                &mut cooperation,
                &mut authority,
            )
            .await
            .unwrap();
            assert_eq!(authority.solutions, 1);
            assert_eq!(result.flag, if accept { 1 } else { -8 });
            if accept {
                assert!((result.fval.unwrap() + 4.).abs() < 1e-5);
            } else {
                assert!(result.x.is_none());
            }
        }
        let mut authority = Original {
            accept: true,
            restricted: true,
            solutions: 0,
            certificates: 0,
        };
        let mut cooperation = ScalarCooperation::new(runmat_builtins::OptimSolver::Quadprog, None);
        let result = solve_fixture(
            problem(-2.),
            None,
            Products::Primal,
            &mut cooperation,
            &mut authority,
        )
        .await
        .unwrap();
        assert_eq!(result.flag, -6);
        assert_eq!(authority.solutions, 0);
        let mut inconsistent = problem(2.);
        inconsistent.lb[0] = 3.;
        inconsistent.ub[0] = 1.;
        authority.accept = false;
        let result = solve_fixture(
            inconsistent,
            None,
            Products::Primal,
            &mut cooperation,
            &mut authority,
        )
        .await
        .unwrap();
        assert_eq!(result.flag, -8);
        assert_eq!(authority.certificates, 1);
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("Algorithm", Value::from("active-set"));
        fields.insert("Display", Value::from("off"));
        let options = Value::Struct(fields.finish());
        for (restricted, accept, flag) in [(false, true, -6), (true, false, -8), (true, true, 1)] {
            let mut original = Original {
                restricted,
                accept,
                solutions: 0,
                certificates: 0,
            };
            let result = solve_fixture(
                problem(2.),
                Some(&options),
                Products::Primal,
                &mut cooperation,
                &mut original,
            )
            .await
            .unwrap();
            assert_eq!(result.flag, flag);
            assert_eq!(original.solutions, usize::from(restricted));
        }
    });
}
async fn solve_fixture<V: Validation>(
    problem: Problem,
    options: Option<&Value>,
    products: Products,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Outcome> {
    let prepared = prepare(problem, options, None, cooperation).await?;
    solve(prepared, Vec::new(), products, cooperation, validation).await
}

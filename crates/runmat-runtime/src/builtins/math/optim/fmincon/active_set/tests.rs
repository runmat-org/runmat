use super::super::model::Role;
use super::*;
use clarabel::solver::SupportedConeT;
use futures::executor::block_on;

#[test]
fn blocking_release_phase_one_and_original_dual_rows_are_real() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let mut cooperation = ScalarCooperation::for_builtin(super::super::IDENTITY, Some(&runtime));
    block_on(async {
        let h = hessian::initial(2, super::super::IDENTITY, &mut cooperation)
            .await
            .unwrap();
        // Starts at both lower bounds. Negative bound duals must release a
        // row; the unconstrained x=2 step then hits x+y=1 and adds its row.
        let model = Linearization {
            semi_infinite: None,
            a: CscMatrix::new(
                3,
                2,
                vec![0, 2, 4],
                vec![0, 1, 0, 2],
                vec![1.0, -1.0, 1.0, -1.0],
            ),
            b: vec![1.0, 0.0, 0.0],
            roles: vec![Role::InequalityLinear(0), Role::Lower(0), Role::Lower(1)],
            equalities: 0,
            cones: vec![SupportedConeT::NonnegativeConeT(3)],
        };
        let quadratic::Result::Step(step) = solve(
            &h,
            &[-2.0, -1.0],
            &model,
            1e-7,
            super::super::IDENTITY,
            &mut cooperation,
        )
        .await
        .unwrap() else {
            panic!("working-set solution")
        };
        assert!((step.direction[0] - 1.0).abs() < 1e-5 && step.direction[1].abs() < 1e-5);
        assert!((step.multipliers[0] - 1.0).abs() < 1e-5);
        assert!(step.multipliers[1].abs() < 1e-5 && step.multipliers[2].abs() < 1e-5);
        let dependent = Linearization {
            semi_infinite: None,
            a: CscMatrix::new(
                4,
                2,
                vec![0, 3, 6],
                vec![0, 1, 2, 0, 1, 3],
                vec![1.0, 2.0, -1.0, 1.0, 2.0, -1.0],
            ),
            b: vec![1.0, 2.0, 0.0, 0.0],
            roles: vec![
                Role::InequalityLinear(0),
                Role::InequalityLinear(1),
                Role::Lower(0),
                Role::Lower(1),
            ],
            equalities: 0,
            cones: vec![SupportedConeT::NonnegativeConeT(4)],
        };
        let quadratic::Result::Step(step) = solve(
            &h,
            &[-2.0, -1.0],
            &dependent,
            1e-7,
            super::super::IDENTITY,
            &mut cooperation,
        )
        .await
        .unwrap() else {
            panic!("dependent original inequalities")
        };
        assert!((step.direction[0] - 1.0).abs() < 1e-5 && step.direction[1].abs() < 1e-5);
        assert!((step.multipliers[0] + 2.0 * step.multipliers[1] - 1.0).abs() < 1e-5);
        // Nonzero equality forces actual Phase I; original equality identity
        // remains first and its unrestricted dual is recovered unchanged.
        let equality = Linearization {
            semi_infinite: None,
            a: model.a.clone(),
            b: model.b.clone(),
            roles: vec![Role::EqualityLinear(0), Role::Lower(0), Role::Lower(1)],
            equalities: 1,
            cones: vec![
                SupportedConeT::ZeroConeT(1),
                SupportedConeT::NonnegativeConeT(2),
            ],
        };
        let quadratic::Result::Step(step) = solve(
            &h,
            &[-2.0, -1.0],
            &equality,
            1e-7,
            super::super::IDENTITY,
            &mut cooperation,
        )
        .await
        .unwrap() else {
            panic!("feasible phase-one equality")
        };
        assert!((step.direction[0] + step.direction[1] - 1.0).abs() < 1e-7);
        assert!((step.direction[0] - 1.0).abs() < 1e-5);
        assert!((step.multipliers[0] - 1.0).abs() < 1e-5);
        // Contradictory original rows do not produce a fabricated solution.
        let inconsistent = Linearization {
            semi_infinite: None,
            a: CscMatrix::new(2, 1, vec![0, 2], vec![0, 1], vec![1.0, -1.0]),
            b: vec![0.0, -1.0],
            roles: vec![Role::Upper(0), Role::Lower(0)],
            equalities: 0,
            cones: vec![SupportedConeT::NonnegativeConeT(2)],
        };
        let h1 = hessian::initial(1, super::super::IDENTITY, &mut cooperation)
            .await
            .unwrap();
        // Both sides are within feasibility tolerance at zero, but they are
        // distinct boundaries, not simultaneous equality constraints.
        let narrow = Linearization {
            semi_infinite: None,
            a: CscMatrix::new(2, 1, vec![0, 2], vec![0, 1], vec![-1.0, 1.0]),
            b: vec![5e-7, 5e-7],
            roles: vec![Role::Lower(0), Role::Upper(0)],
            equalities: 0,
            cones: vec![SupportedConeT::NonnegativeConeT(2)],
        };
        for sign in [-1.0, 1.0] {
            let quadratic::Result::Step(step) = solve(
                &h1,
                &[sign],
                &narrow,
                1e-6,
                super::super::IDENTITY,
                &mut cooperation,
            )
            .await
            .unwrap() else {
                panic!("compatible narrow interval")
            };
            assert!((step.direction[0] + sign * 5e-7).abs() < 1e-8);
            assert!(step.multipliers[if sign < 0.0 { 1 } else { 0 }] > 0.9);
            assert!(step.multipliers[if sign < 0.0 { 0 } else { 1 }].abs() < 1e-8);
        }
        assert!(!matches!(
            solve(
                &h1,
                &[0.0],
                &inconsistent,
                1e-7,
                super::super::IDENTITY,
                &mut cooperation
            )
            .await
            .unwrap(),
            quadratic::Result::Step(_)
        ));
    });
}

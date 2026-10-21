use super::*;
#[test]
fn coupled_slip_tangent_retains_independent_nonsymmetric_reduced_action() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let f = CoupledFixture::new(order, true);
        let operator = f.operator(true);
        let basis =
            CanonicalNonlinearSolidBasis::prepare(&f.model, &f.surface.mesh, &f.summary).unwrap();
        let problem = SolidNewtonProblem::prepare_with_contact(
            &basis,
            &f.summary,
            policy(),
            Some(CoupledFixture::coupling(&operator)),
        )
        .unwrap();
        let initial = problem.initial_state().unwrap();
        let u = problem.reduction.expand(&[0.00035, 0.00015]).unwrap();
        let trial = problem
            .evaluate_trial(&u, &initial.constitutive, initial.contact.as_ref(), 1.0)
            .unwrap();
        let reduced = problem
            .reduction
            .reduce_sparse_operator_planned(
                &trial.evaluation.tangent,
                &problem.contact.as_ref().unwrap().reduced_pattern,
            )
            .unwrap();
        assert_eq!(
            reduced.properties(),
            crate::algebra::MatrixProperties::GENERAL
        );
        for (column, expected) in [([1., 0.], [20., 0.]), ([0., 1.], [60., 240.])] {
            let action = crate::algebra::spmv(&reduced, &column).unwrap();
            for (value, exact) in action.iter().zip(expected) {
                near(*value, exact, 1e-7);
            }
        }
        let direction = [0.3, -0.2];
        let full = problem.reduction.expand_linear(&direction).unwrap();
        let action = crate::algebra::spmv(&reduced, &direction).unwrap();
        for step in [1e-7, 1e-8] {
            let evaluate = |sign: f64| {
                let u = u
                    .iter()
                    .zip(&full)
                    .map(|(u, v)| u + sign * step * v)
                    .collect::<Vec<_>>();
                let trial = problem
                    .evaluate_trial(&u, &initial.constitutive, initial.contact.as_ref(), 1.0)
                    .unwrap();
                problem.residual(&trial.evaluation, 1.).unwrap()
            };
            let plus = evaluate(1.);
            let minus = evaluate(-1.);
            for ((a, p), m) in action.iter().zip(plus).zip(minus) {
                near(*a, (p - m) / (2. * step), 1e-5);
            }
        }
        assert_eq!(initial.contact_configuration().unwrap().ordinal(), 0);
    }
}

use super::super::{inputs::Problem, linear::Data, options::Prepared, IDENTITY};
use super::*;
use clarabel::algebra::CscMatrix;
use runmat_value::CellArray;

fn object(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut result = runmat_value::UnboundedStructValueBuilder::new();
    for (name, value) in fields {
        result.insert(name, value);
    }
    Value::Struct(result.finish())
}
#[test]
fn real_island_reproduction_keeps_elites_bits_exact_provenance_and_predecessor() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
        let controls = object([
            ("PopulationType", Value::from("bitstring")),
            (
                "PopulationSize",
                Value::Tensor(Tensor::new(vec![2., 2.], vec![1, 2]).unwrap()),
            ),
            ("EliteCount", Value::Num(1.)),
            ("CrossoverFraction", Value::Num(0.)),
            (
                "MutationFcn",
                Value::Cell(
                    CellArray::new_with_shape(
                        vec![Value::from("mutationuniform"), Value::Num(0.)],
                        vec![1, 2],
                    )
                    .unwrap(),
                ),
            ),
            ("Display", Value::from("off")),
        ]);
        let problem = Problem {
            objective: Value::FunctionHandle("must_not_evaluate_during_reproduction".into()),
            nonlinear: None,
            variables: 1,
            linear: Some(Data {
                inequality: (CscMatrix::new(0, 1, vec![0, 0], vec![], vec![]), vec![]),
                equality: (CscMatrix::new(0, 1, vec![0, 0], vec![], vec![]), vec![]),
                lower: vec![f64::NEG_INFINITY],
                upper: vec![f64::INFINITY],
                integers: vec![],
            }),
            rng: None,
            public: object([]),
            has_linear: false,
            has_bounds: false,
            mixed_integer: false,
        };
        let prepared = Prepared::new(
            controls,
            crate::builtins::math::optim::ga::options::Activity::Linear,
            &mut cooperation,
        )
        .await
        .unwrap();
        let mut options = options::Options::resolve(prepared, &problem, &mut cooperation)
            .await
            .unwrap();
        let original = Tensor::new(vec![0., 1., 1., 0.], vec![4, 1]).unwrap();
        let pointer = original.as_f64_slice().unwrap().as_ptr();
        let generation = Generation::Numeric(
            common::population::Population::new(POLICY, original.clone(), 4, 1).unwrap(),
        );
        let scores = Value::Tensor(Tensor::new(vec![9., 1., 0., 4.], vec![4, 1]).unwrap());
        let comparison = [
            Score::Defined(9.),
            Score::Defined(1.),
            Score::Defined(0.),
            Score::Defined(4.),
        ];
        let timing = SolverTiming::start(Some(&runtime)).unwrap();
        let mut mutation = Mutation::new(&options, &mut cooperation).await.unwrap();
        let result = next_with::<linear::Domain, _>(
            Current {
                generation: &generation,
                comparison: &comparison,
                public_scores: &scores,
                controls: None,
                state: None,
                problem: &problem.public,
                options: &options,
                timing: &timing,
                record: true,
            },
            None,
            &mut mutation,
            &mut super::super::callbacks::Direct,
            &mut cooperation,
        )
        .await
        .unwrap();
        let result = super::super::callbacks::take_direct(result);
        let Some(result) = result else {
            panic!("unbounded bit reproduction");
        };
        let next = result.generation.numeric().unwrap();
        assert_eq!(next.get(0, 0), 1., "first island copies its own best elite");
        assert_eq!(
            next.get(2, 0),
            1.,
            "second island copies its own best elite"
        );
        for (destination, source) in result.copied_from.iter().enumerate() {
            let source = source.expect("zero-rate bit mutation preserves an exact parent");
            assert_eq!(
                destination / 2,
                source / 2,
                "selection must not cross islands"
            );
            assert_eq!(
                next.get(destination, 0),
                original.as_f64_slice().unwrap()[source]
            );
        }
        assert_eq!(
            result.selection.len(),
            4,
            "selection includes elites and mutation parents"
        );
        assert_eq!(
            result
                .lineage
                .iter()
                .map(|kind| kind.name())
                .collect::<Vec<_>>(),
            ["elite", "retained", "elite", "retained"]
        );
        let expectation = result.expectation.unwrap();
        let weights = expectation.as_f64_slice().unwrap();
        assert!((weights[0] + weights[1] - 1.).abs() < 1e-12);
        assert!((weights[2] + weights[3] - 1.).abs() < 1e-12);
        assert!(!result.unverified_proposals);
        assert_eq!(original.as_f64_slice().unwrap(), [0., 1., 1., 0.]);
        assert_eq!(original.as_f64_slice().unwrap().as_ptr(), pointer);
        assert_ne!(next.positions.as_f64_slice().unwrap().as_ptr(), pointer);
        options.maximum = common::global_time::TimeLimit::Days(0.);
        let before = {
            let mut turn = crate::builtins::common::random::acquire_turn(Some(&runtime), "ga")
                .await
                .unwrap();
            crate::builtins::common::random::snapshot_in_turn(&mut turn).unwrap()
        };
        let stopped = super::next_with::<linear::Domain, _>(
            Current {
                generation: &generation,
                comparison: &comparison,
                public_scores: &scores,
                controls: None,
                state: None,
                problem: &problem.public,
                options: &options,
                timing: &timing,
                record: true,
            },
            None,
            &mut mutation,
            &mut super::super::callbacks::Direct,
            &mut cooperation,
        )
        .await
        .unwrap();
        let stopped = super::super::callbacks::take_direct(stopped);
        assert!(stopped.is_none());
        let after = {
            let mut turn = crate::builtins::common::random::acquire_turn(Some(&runtime), "ga")
                .await
                .unwrap();
            crate::builtins::common::random::snapshot_in_turn(&mut turn).unwrap()
        };
        assert_eq!(
            before, after,
            "an expired generation must not draw creation or variation RNG"
        );
    });
}

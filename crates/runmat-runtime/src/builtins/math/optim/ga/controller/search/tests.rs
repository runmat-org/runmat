use super::super::super::{linear::Data, options::Prepared, IDENTITY};
use super::*;
use clarabel::algebra::CscMatrix;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn object(fields: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    let mut value = runmat_value::UnboundedStructValueBuilder::new();
    for (name, field) in fields {
        value.insert(name, field);
    }
    Value::Struct(value.finish())
}
#[test]
fn repaired_seed_invalidates_only_its_old_supplied_score_and_zero_generations_is_initial_only() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("ga_repaired_score", 29201)],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let _entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, outputs| {
                assert_eq!(id, 29201);
                assert_eq!(outputs, 1);
                let Value::Tensor(point) = &args[0] else {
                    panic!("double genome");
                };
                let x = point.as_f64_slice().unwrap()[0];
                assert_eq!(
                    x, 0.,
                    "the changed seed must be evaluated at its repaired coordinate"
                );
                observed.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(Value::Num(
                        x * x,
                    ))
                })
            },
        )))
        .unwrap();
        let controls = object([
            ("PopulationSize", Value::Num(2.)),
            ("MaxGenerations", Value::Num(0.)),
            ("Display", Value::from("off")),
            (
                "InitialPopulationMatrix",
                Value::Tensor(Tensor::new(vec![-2., 1.], vec![2, 1]).unwrap()),
            ),
            (
                "InitialScoresMatrix",
                Value::Tensor(Tensor::new(vec![999., 1.], vec![2, 1]).unwrap()),
            ),
        ]);
        let problem = Problem {
            objective: Value::FunctionHandle("ga_repaired_score".into()),
            nonlinear: None,
            variables: 1,
            linear: Some(Data {
                inequality: (CscMatrix::new(0, 1, vec![0, 0], vec![], vec![]), vec![]),
                equality: (CscMatrix::new(0, 1, vec![0, 0], vec![], vec![]), vec![]),
                lower: vec![0.],
                upper: vec![1.],
                integers: vec![],
            }),
            rng: None,
            public: object([]),
            has_linear: false,
            has_bounds: true,
            mixed_integer: false,
        };
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
        let prepared = Prepared::new(
            controls,
            crate::builtins::math::optim::ga::options::Activity::Linear,
            &mut cooperation,
        )
        .await
        .unwrap();
        let options = Options::resolve(prepared, &problem, &mut cooperation)
            .await
            .unwrap();
        let timing = SolverTiming::start(Some(&runtime)).unwrap();
        let sources = super::super::super::callbacks::fixture::sources(
            &runtime,
            &_bindings,
            "ga_repaired_score",
            None,
        );
        let mut started = Search::start(problem, options, timing, sources, &mut cooperation)
            .await
            .unwrap();
        assert!(started.stopped.is_none());
        let search = started.ready.take().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            (
                search.objective_counts.points,
                search.objective_counts.callbacks
            ),
            (1, 1)
        );
        assert_eq!(
            search.evaluated.fitness.origins,
            [
                super::super::super::evaluation::fitness::Origin::Measured,
                super::super::super::evaluation::fitness::Origin::SuppliedInitial,
            ]
        );
        assert_eq!(search.best.score(), Some(0.));
        assert_eq!(search.progress.generation, 0);
        assert_eq!(search.progress.total_inner, 0);
        assert_eq!(
            search
                .progress
                .ordinary_stop(&search.options, search.best.score()),
            Some(Termination::Generations)
        );
    });
}

use super::super::super::{
    inputs::Problem,
    linear::Data,
    options::{Options, Prepared},
    IDENTITY,
};
use super::*;
use clarabel::algebra::CscMatrix;
use runmat_value::{Tensor, Value, ValueSequence};
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
fn problem(nonlinear: Option<Value>) -> Problem {
    Problem {
        objective: Value::FunctionHandle("ga_driver_objective".into()),
        nonlinear,
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
    }
}
#[test]
fn completed_al_epoch_has_distinct_inner_outer_counts_and_no_initial_score_or_reporting_replay() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[
                ("ga_driver_objective", 29301),
                ("ga_driver_constraint", 29302),
            ],
        )
        .unwrap();
        let objective_calls = Arc::new(AtomicUsize::new(0));
        let nonlinear_calls = Arc::new(AtomicUsize::new(0));
        let objective = objective_calls.clone();
        let nonlinear = nonlinear_calls.clone();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, requested| {
                let Value::Tensor(x) = &args[0] else {
                    panic!("double point");
                };
                assert_eq!(x.as_f64_slice().unwrap(), [0.]);
                let values = if id == 29301 {
                    assert_eq!(requested, 1);
                    objective.fetch_add(1, Ordering::SeqCst);
                    vec![Value::Num(0.)]
                } else {
                    assert_eq!(id, 29302);
                    assert_eq!(requested, 2);
                    nonlinear.fetch_add(1, Ordering::SeqCst);
                    vec![super::super::super::inputs::raw::empty(), Value::Num(0.)]
                };
                Box::pin(async move {
                    crate::execution::RootedValueSequence::new(
                        ValueSequence::comma_separated(values)
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                    )
                    .map_err(|source| source.into_runtime_error())
                })
            },
        )))
        .unwrap();
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
        let problem = problem(Some(Value::FunctionHandle("ga_driver_constraint".into())));
        let prepared = Prepared::new(
            object([
                ("PopulationSize", Value::Num(2.)),
                ("EliteCount", Value::Num(2.)),
                ("MaxGenerations", Value::Num(3.)),
                ("MaxStallGenerations", Value::Num(1.)),
                ("Display", Value::from("off")),
                (
                    "InitialPopulationMatrix",
                    Value::Tensor(Tensor::new(vec![0., 0.], vec![2, 1]).unwrap()),
                ),
                (
                    "InitialScoresMatrix",
                    Value::from("inactive for nonlinear fitness"),
                ),
            ]),
            crate::builtins::math::optim::ga::options::Activity::Nonlinear,
            &mut cooperation,
        )
        .await
        .unwrap();
        let options = Options::resolve(prepared, &problem, &mut cooperation)
            .await
            .unwrap();
        let ready = super::super::super::initialization::Ready::new(problem, options, &cooperation)
            .unwrap();
        let finished = run(
            ready,
            SolverTiming::start(Some(&runtime)).unwrap(),
            super::super::super::callbacks::fixture::sources(
                &runtime,
                &_bindings,
                "ga_driver_objective",
                Some("ga_driver_constraint"),
            ),
            &mut cooperation,
        )
        .await
        .unwrap();
        assert_eq!(finished.reason, Termination::Augmented);
        let search = finished.search.as_ref().unwrap();
        assert_eq!(
            (
                search.progress.generation,
                search.progress.inner,
                search.progress.total_inner
            ),
            (1, 1, 1)
        );
        assert_eq!(
            (
                objective_calls.load(Ordering::SeqCst),
                nonlinear_calls.load(Ordering::SeqCst)
            ),
            (2, 2)
        );
        assert_eq!(
            (
                search.objective_counts.points,
                search.nonlinear_counts.points
            ),
            (2, 2)
        );
        assert_eq!(search.best.score(), Some(0.));
        assert!(search.best.equalities.iter().all(|value| *value == 0.));
        // Final publication shares existing results and makes no callback.
        super::super::publication::finalize(finished, &cooperation).unwrap();
        assert_eq!(
            (
                objective_calls.load(Ordering::SeqCst),
                nonlinear_calls.load(Ordering::SeqCst)
            ),
            (2, 2)
        );
    });
}
#[test]
fn returned_population_and_declared_score_keep_provenance_and_do_not_manufacture_objective_calls() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _entry = runtime.enter();
        let bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("ga_driver_objective", 29303)],
        )
        .unwrap();
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
        let problem = problem(None);
        let prepared = Prepared::new(
            object([
                ("PopulationSize", Value::Num(2.)),
                ("MaxGenerations", Value::Num(0.)),
                ("Display", Value::from("off")),
                (
                    "InitialPopulationMatrix",
                    Value::Tensor(Tensor::new(vec![0., 1.], vec![2, 1]).unwrap()),
                ),
                (
                    "InitialScoresMatrix",
                    Value::Tensor(Tensor::new(vec![0., 1.], vec![2, 1]).unwrap()),
                ),
            ]),
            crate::builtins::math::optim::ga::options::Activity::Linear,
            &mut cooperation,
        )
        .await
        .unwrap();
        let options = Options::resolve(prepared, &problem, &mut cooperation)
            .await
            .unwrap();
        let mut started = Search::start(
            problem,
            options,
            SolverTiming::start(Some(&runtime)).unwrap(),
            super::super::super::callbacks::fixture::sources(
                &runtime,
                &bindings,
                "ga_driver_objective",
                None,
            ),
            &mut cooperation,
        )
        .await
        .unwrap();
        let mut search = started.ready.take().unwrap();
        let state = RootedValue::in_context(
            object([
                (
                    "Population",
                    Value::Tensor(Tensor::new(vec![0.25, 0.5], vec![2, 1]).unwrap()),
                ),
                (
                    "Score",
                    Value::Tensor(Tensor::new(vec![0.0625, 0.25], vec![2, 1]).unwrap()),
                ),
                ("EvalElites", Value::Bool(false)),
                ("HaveDuplicates", Value::Bool(false)),
            ]),
            &runtime,
        )
        .unwrap();
        assert!(
            search
                .adopt(&state, None, &mut cooperation)
                .await
                .unwrap()
                .changed
        );
        assert_eq!(
            (
                search.objective_counts.points,
                search.objective_counts.callbacks
            ),
            (0, 0)
        );
        assert_eq!(
            search.evaluated.fitness.origins,
            [super::super::super::evaluation::fitness::Origin::ObserverSupplied; 2]
        );
        assert_eq!(search.generation.numeric().unwrap().get(1, 0), 0.5);
        assert!(!search.flags.eval_elites);
        assert_eq!(
            search.progress.generation, 0,
            "a callback edit is not a completed generation"
        );
        assert_eq!(
            search.best.score(),
            Some(0.),
            "the earlier feasible observation remains the historical best"
        );
    });
}

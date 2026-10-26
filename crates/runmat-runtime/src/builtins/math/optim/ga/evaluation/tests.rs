use super::super::population::Generation;
use super::super::{common, ScalarCooperation, POLICY};
use super::{
    fitness::{Objective, Origin, Supplied},
    generation::{Evaluator, Mode},
};
use common::population::{Counts, Measurement, Population};
use runmat_value::{Tensor, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[test]
fn a_borrowed_wave_stop_keeps_actual_counts_without_publishing_partial_fitness() {
    use super::source::{Direct, Request, Source, Wave};
    use common::population::projection::Arguments;
    use std::ops::ControlFlow;
    struct Limited;
    impl super::super::callbacks::Invocation for Limited {
        type Stop = ();
        async fn protocol<
            'a,
            T: 'a,
            F: std::future::Future<Output = crate::BuiltinResult<T>> + 'a,
        >(
            &'a mut self,
            make: impl FnOnce(Option<crate::context::RuntimeContext>) -> F + 'a,
            cooperation: &'a mut ScalarCooperation,
        ) -> crate::BuiltinResult<ControlFlow<(), T>> {
            Ok(ControlFlow::Continue(super::super::callbacks::take_direct(
                super::super::callbacks::Invocation::protocol(&mut Direct, make, cooperation)
                    .await?,
            )))
        }
        async fn call<'a>(
            &'a mut self,
            target: &'a crate::call::descriptor::RepeatableCallable,
            arguments: crate::call::BuiltinArguments,
            outputs: usize,
            event: super::super::callbacks::Event,
            cooperation: &'a mut ScalarCooperation,
        ) -> crate::BuiltinResult<ControlFlow<(), crate::execution::RootedValueSequence>> {
            Ok(ControlFlow::Continue(super::super::callbacks::take_direct(
                super::super::callbacks::Invocation::call(
                    &mut Direct,
                    target,
                    arguments,
                    outputs,
                    event,
                    cooperation,
                )
                .await?,
            )))
        }
    }
    impl Source for Limited {
        async fn next<'a, P: Arguments + 'a>(
            &'a mut self,
            request: Request<'a, P>,
            cooperation: &'a mut ScalarCooperation,
        ) -> crate::BuiltinResult<ControlFlow<(), Wave>> {
            if request.counts.points == 1 {
                return Ok(ControlFlow::Break(()));
            }
            Ok(ControlFlow::Continue(super::super::callbacks::take_direct(
                Direct.next(request, cooperation).await?,
            )))
        }
    }
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("ga_bounded_wave_fitness", 29103)],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let _entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, arguments, outputs| {
                assert_eq!((id, outputs), (29103, 1));
                let Value::Tensor(point) = &arguments[0] else {
                    panic!("actual row");
                };
                let x = point.as_f64_slice().unwrap()[0];
                assert_eq!(x, 2.0);
                observed.fetch_add(1, Ordering::SeqCst);
                Box::pin(async move {
                    crate::execution::RootedValueSequence::new(
                        runmat_value::ValueSequence::single(Value::Num(x * x))
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                    )
                    .map_err(|source| source.into_runtime_error())
                })
            },
        )))
        .unwrap();
        let population = Generation::Numeric(
            Population::new(POLICY, Tensor::new(vec![2., 3.], vec![2, 1]).unwrap(), 2, 1).unwrap(),
        );
        let target = Value::FunctionHandle("ga_bounded_wave_fitness".into());
        let mut cooperation =
            ScalarCooperation::for_builtin(super::super::IDENTITY, Some(&runtime));
        let mut counts = Counts::default();
        let result = Objective {
            function: common::callback_source::Callable::Dynamic(&target),
            vectorized: false,
            check_values: true,
            parallel: None,
            counts: &mut counts,
            aliases: None,
            cooperation: &mut cooperation,
        }
        .evaluate_with(&population, vec![None; 2], &mut Limited)
        .await
        .unwrap();
        assert!(matches!(result, ControlFlow::Break(())));
        assert_eq!((counts.points, counts.callbacks), (1, 1));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn penalty_constraints_precede_fitness_and_exact_duplicates_retain_real_provenance() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let _bindings = crate::user_functions::test_support::install_exact_named_forwarders(
            &runtime,
            &[("ga_test_fitness", 29101), ("ga_test_constraints", 29102)],
        )
        .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let _entry = runtime.enter();
        let _invoker = crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |id, args, outputs| {
                let Value::Tensor(point) = &args[0] else {
                    panic!("row genome");
                };
                let x = point.as_f64_slice().unwrap()[0];
                let values = if id == 29102 {
                    assert_eq!(outputs, 2);
                    vec![
                        Value::Num(x),
                        Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
                    ]
                } else {
                    assert_eq!(id, 29101);
                    assert_eq!(outputs, 1);
                    assert!(x <= 0., "infeasible row must not reach fitness");
                    observed.fetch_add(1, Ordering::SeqCst);
                    vec![Value::Num(x * x + 2.0)]
                };
                Box::pin(async move {
                    crate::execution::RootedValueSequence::new(
                        runmat_value::ValueSequence::comma_separated(values)
                            .map_err(crate::sequence::sequence_error_to_runtime)?,
                    )
                    .map_err(|source| source.into_runtime_error())
                })
            },
        )))
        .unwrap();
        let population = Generation::Numeric(
            Population::new(
                POLICY,
                Tensor::new(vec![-1., -1., 1., 1., -0., 0.], vec![6, 1]).unwrap(),
                6,
                1,
            )
            .unwrap(),
        );
        let mut cooperation =
            ScalarCooperation::for_builtin(super::super::IDENTITY, Some(&runtime));
        let mut objective_counts = Counts::default();
        let mut nonlinear_counts = Counts::default();
        let mut extents = None;
        let aliases = super::duplicates::Aliases::new(&population, true, &mut cooperation)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            aliases.selected(),
            [0, 2, 4, 5],
            "opposite signed zeros are distinct exact points"
        );
        let result = Evaluator {
            objective: common::callback_source::Callable::Dynamic(&Value::FunctionHandle(
                "ga_test_fitness".into(),
            )),
            nonlinear: Some(common::callback_source::Callable::Dynamic(
                &Value::FunctionHandle("ga_test_constraints".into()),
            )),
            objective_parallel: None,
            nonlinear_parallel: None,
            vectorized: false,
            check_values: true,
            mixed_integer: false,
            mode: Mode::Penalty,
            tolerance: 1e-6,
            objective_counts: &mut objective_counts,
            nonlinear_counts: &mut nonlinear_counts,
            extents: &mut extents,
            aliases: Some(&aliases),
            cooperation: &mut cooperation,
        }
        .evaluate(&population, vec![None; 6])
        .await
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(
            (objective_counts.points, objective_counts.callbacks),
            (3, 3)
        );
        assert_eq!(
            (nonlinear_counts.points, nonlinear_counts.callbacks),
            (4, 4)
        );
        assert_eq!(extents, Some((1, 0)));
        assert_eq!(
            result.fitness.origins,
            [
                Origin::Measured,
                Origin::Measured,
                Origin::ConstraintRejected,
                Origin::ConstraintRejected,
                Origin::Measured,
                Origin::Measured
            ]
        );
        assert!(
            matches!(result.fitness.values.measurement(2),Measurement::Real(v) if v==f64::INFINITY)
        );
        assert_eq!(
            result.penalty_scores(1e-6, &mut cooperation).await.unwrap(),
            [Some(3.), Some(3.), Some(4.), Some(4.), Some(2.), Some(2.)]
        );
    });
}

#[test]
fn supplied_invalid_fitness_obeys_funvalcheck_without_replay() {
    futures::executor::block_on(async {
        let runtime = crate::user_functions::test_support::exact_fixture_runtime();
        let mut cooperation =
            ScalarCooperation::for_builtin(super::super::IDENTITY, Some(&runtime));
        let population = Generation::Numeric(
            Population::new(POLICY, Tensor::new(vec![1.], vec![1, 1]).unwrap(), 1, 1).unwrap(),
        );
        let mut counts = Counts::default();
        let error = Objective {
            function: common::callback_source::Callable::Dynamic(&Value::FunctionHandle(
                "not_needed_for_supplied_score".into(),
            )),
            vectorized: false,
            check_values: true,
            parallel: None,
            counts: &mut counts,
            aliases: None,
            cooperation: &mut cooperation,
        }
        .evaluate(
            &population,
            vec![Some(Supplied {
                value: Measurement::Real(f64::NAN),
                origin: Origin::SuppliedInitial,
            })],
        )
        .await
        .err()
        .expect("invalid supplied score");
        assert!(error.to_string().contains("FunValCheck"));
        assert_eq!((counts.points, counts.callbacks), (0, 0));
    });
}

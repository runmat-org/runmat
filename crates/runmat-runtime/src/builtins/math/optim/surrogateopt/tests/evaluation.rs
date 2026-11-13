//! Actual selected semantic callback, point preservation and hard budget.
use super::super::{
    campaign::{Campaign, Provenance},
    domain::BoxGeometry,
    evaluation::{BatchEvaluation, Evaluation, Evaluator},
    queue::Origin,
    rbf::{self, Fit, Prediction, Training},
    IDENTITY,
};
use crate::builtins::math::optim::{
    common::{global_time::TimeLimit, solver_timing::SolverTiming, ScalarCooperation},
    intlinprog::feasibility::{Domain, Input},
};
use crate::{
    call::descriptor::{FunctionNameResolver, RepeatableCallable},
    execution::RootedValue,
};
use clarabel::algebra::CscMatrix;
use runmat_value::Value;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[test]
fn serial_actual_point_is_retained_and_zero_remaining_never_calls_source() {
    let runtime = crate::user_functions::test_support::exact_fixture_runtime();
    let bindings = crate::user_functions::test_support::install_exact_named_forwarders(
        &runtime,
        &[
            ("surrogate_counted", 19031),
            ("surrogate_initial_lanes", 19032),
        ],
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let _invoker = {
        let _installation = runtime.enter();
        crate::user_functions::install_semantic_function_invoker(Some(Arc::new(
            move |function, args, outputs| {
                assert!(matches!(function, 19031 | 19032));
                assert_eq!(outputs, 1);
                counted.fetch_add(1, Ordering::SeqCst);
                let Value::Tensor(point) = &args[0] else {
                    panic!("actual host point");
                };
                let x = point.as_f64_slice().unwrap();
                let rows = point.shape()[0];
                assert_eq!(point.shape()[1], 2);
                let value = if function == 19032 {
                    assert_eq!(rows, 1);
                    let mut fields = runmat_value::UnboundedStructValueBuilder::new();
                    fields.insert("Fval", Value::Num(x[0] * x[0] + x[1] * x[1]));
                    fields.insert("Ineq", Value::Num(x[0] - 2.0));
                    Value::Struct(fields.finish())
                } else if rows == 1 {
                    Value::Num(x[0] * x[0] + x[1] * x[1])
                } else {
                    Value::Tensor(
                        runmat_value::Tensor::new(
                            (0..rows)
                                .map(|i| x[i] * x[i] + x[rows + i] * x[rows + i])
                                .collect(),
                            vec![rows, 1],
                        )
                        .unwrap(),
                    )
                };
                Box::pin(async move {
                    crate::user_functions::test_support::rooted_single_value_sequence(value)
                })
            },
        )))
        .unwrap()
    };
    let source = RootedValue::in_optional_context(
        crate::user_functions::test_support::exact_local_value(
            "surrogate_counted",
            bindings.binding("surrogate_counted"),
        ),
        Some(&runtime),
    )
    .unwrap();
    struct Resolver;
    impl FunctionNameResolver for Resolver {
        fn resolve_function(&self, _: &str) -> Option<runmat_types::FunctionId> {
            None
        }
    }
    let selected = RepeatableCallable::select_rooted(Some(&runtime), source, &Resolver).unwrap();
    futures::executor::block_on(async {
        let mut cooperation = ScalarCooperation::for_builtin(IDENTITY, Some(&runtime));
        let mut serial = Evaluator::new(&selected, 2, 1).unwrap();
        let Evaluation::Completed(completed) = serial
            .evaluate(vec![3.0, 4.0], 1e-3, &mut cooperation)
            .await
            .unwrap()
        else {
            panic!("one accepted expensive evaluation");
        };
        assert_eq!(completed.point.coordinate(0), 3.0);
        assert_eq!(completed.point.coordinate(1), 4.0);
        assert!(matches!(
            completed.observation.objective(),
            super::super::observation::Objective::Measured(25.0)
        ));
        assert_eq!(serial.evaluated(), 1);
        assert_eq!(serial.remaining(), 0);
        assert!(matches!(
            serial
                .evaluate(vec![9.0, 9.0], 1e-3, &mut cooperation)
                .await
                .unwrap(),
            Evaluation::Limit
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let scalar = completed;
        let mut vectorized = Evaluator::new(&selected, 2, 2).unwrap();
        let rows = std::num::NonZeroUsize::new(2).unwrap();
        let BatchEvaluation::Completed(completed) = vectorized
            .evaluate_batch(vec![5.0, 1.0, 12.0, 0.0], rows, 1e-3, &mut cooperation)
            .await
            .unwrap()
        else {
            panic!("two actual vectorized rows")
        };
        assert_eq!(completed[0].point.coordinate(1), 12.0);
        assert_eq!(completed[1].point.coordinate(0), 1.0);
        assert!(matches!(
            completed[1].observation.objective(),
            super::super::observation::Objective::Measured(1.0)
        ));
        assert_eq!(vectorized.evaluated(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let mut campaign = Campaign::new(2, 2, rows).unwrap();
        let original_phase = campaign.phase();
        let first_index = super::super::history::TrialIndex(campaign.history().len());
        let first = campaign
            .record_callback(
                scalar,
                Provenance {
                    phase: original_phase,
                    origin: Origin::Initial,
                },
                &mut cooperation,
            )
            .await
            .unwrap();
        assert!(first.update.is_none());
        assert!(campaign.tracker().state().current.is_none());
        assert!(campaign.reset().is_err());
        assert!(campaign.model_snapshot(&mut cooperation).await.is_err());
        let mut late = completed.into_iter();
        let group_index = super::super::history::TrialIndex(campaign.history().len());
        let group = campaign
            .record_callback(
                late.next().unwrap(),
                Provenance {
                    phase: original_phase,
                    origin: Origin::Initial,
                },
                &mut cooperation,
            )
            .await
            .unwrap();
        assert!(group.update.unwrap().best_changed);
        assert_eq!(campaign.tracker().best(), Some(first_index));
        // Saved indices/counts are derived from the committed measurements,
        // not trusted merely because they are finite and in range.
        use super::super::incumbent::{Tracker, Validation};
        let prefix = campaign.tracker().state();
        let mut wrong_best = prefix;
        wrong_best.best = Some(group_index);
        assert!(Tracker::from_state(
            wrong_best,
            campaign.history(),
            2,
            original_phase.0,
            &mut cooperation
        )
        .await
        .is_err());
        let mut wrong_count = prefix;
        wrong_count.feasible = 1;
        assert!(Tracker::from_state(
            wrong_count,
            campaign.history(),
            2,
            original_phase.0,
            &mut cooperation
        )
        .await
        .is_err());
        let mut restored_summary = Validation::default();
        restored_summary
            .advance(
                prefix,
                campaign.history(),
                2,
                original_phase.0,
                &mut cooperation,
            )
            .await
            .unwrap();
        let snapshot = campaign.model_snapshot(&mut cooperation).await.unwrap();
        let mut domain = Domain::new(
            Input {
                matrix: CscMatrix::new(0, 2, vec![0, 0, 0], vec![], vec![]),
                rhs: vec![],
                equalities: 0,
                lower: vec![0.0; 2],
                upper: vec![20.0; 2],
                integers: vec![],
            },
            SolverTiming::start(Some(&runtime)).unwrap(),
            TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        let shape = BoxGeometry::new(&domain, &mut cooperation).await.unwrap();
        let domain_view = shape.view(&domain).unwrap();
        let (model, generation, original_prediction) = {
            let view = snapshot
                .view(campaign.history(), &[0], &domain_view, &mut cooperation)
                .await
                .unwrap();
            let generation = view.generation();
            let Fit::Poised(model) = rbf::fit(&view, &mut cooperation).await.unwrap() else {
                panic!("two physical sites with one independent affine column")
            };
            let mut original_prediction = [0.0];
            assert_eq!(
                model
                    .predict_into(
                        &view,
                        &[0.2, 0.4],
                        &mut original_prediction,
                        &mut cooperation
                    )
                    .await
                    .unwrap(),
                Prediction::Finite
            );
            (model, generation, original_prediction)
        };
        campaign.reset().unwrap();
        let recorded_index = super::super::history::TrialIndex(campaign.history().len());
        let recorded = campaign
            .record_callback(
                late.next().unwrap(),
                Provenance {
                    phase: original_phase,
                    origin: Origin::Adaptive,
                },
                &mut cooperation,
            )
            .await
            .unwrap();
        assert!(recorded.update.is_none());
        assert_eq!(campaign.tracker().best(), Some(first_index));
        assert!(campaign.flush(&mut cooperation).await.unwrap().best_changed);
        assert_eq!(campaign.tracker().best(), Some(recorded_index));
        let after_reset = campaign.tracker().state();
        assert_eq!(after_reset.incumbent_start, 2);
        assert_eq!(after_reset.incumbent, Some(recorded_index));
        let mut wrong_reset = after_reset;
        wrong_reset.incumbent_start = 0;
        assert!(restored_summary
            .advance(
                wrong_reset,
                campaign.history(),
                3,
                campaign.phase().0,
                &mut cooperation
            )
            .await
            .is_err());
        restored_summary
            .advance(
                after_reset,
                campaign.history(),
                3,
                campaign.phase().0,
                &mut cooperation,
            )
            .await
            .unwrap();
        assert_eq!(restored_summary.into_tracker().state(), after_reset);
        assert_eq!(
            campaign.history().trial(recorded_index).phase,
            original_phase
        );
        assert!(campaign
            .model_rows(&mut cooperation)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(campaign.search().scale(), 0.2);
        // The late adaptive result remains a real best-improving completion,
        // but cannot trigger a new phase's parallel host Tree refinement.
        assert!(campaign
            .refinement(
                super::super::schedule::Applicability {
                    problem: super::super::sampling::Problem::Integer,
                    nonlinear: false,
                    linear: true,
                    parallel: true,
                },
                false
            )
            .is_none());
        let frozen = snapshot
            .view(campaign.history(), &[0], &domain_view, &mut cooperation)
            .await
            .unwrap();
        assert_eq!(frozen.generation(), generation);
        let mut after_late = [0.0];
        assert_eq!(
            model
                .predict_into(&frozen, &[0.2, 0.4], &mut after_late, &mut cooperation)
                .await
                .unwrap(),
            Prediction::Finite
        );
        assert_eq!(original_prediction[0].to_bits(), after_late[0].to_bits());
        assert!(matches!(
            vectorized
                .evaluate_batch(vec![0.0; 4], rows, 1e-3, &mut cooperation)
                .await
                .unwrap(),
            BatchEvaluation::Limit
        ));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        // Known row-oriented Fval is not replayed, and a partial known lane
        // can wait for an actual schema rather than guessing no constraints.
        use super::super::{
            initial::{Discovery, Initial, Next},
            observation::Objective,
        };
        let array =
            |values, shape| Value::Tensor(runmat_value::Tensor::new(values, shape).unwrap());
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("X", array(vec![3.0, 5.0, 4.0, 12.0], vec![2, 2]));
        fields.insert("Fval", array(vec![9.0, 8.0], vec![1, 2]));
        fields.insert("Ineq", array(vec![], vec![2, 0]));
        let mut initial =
            Initial::prepare(Value::Struct(fields.finish()), 2, 1e-3, &mut cooperation)
                .await
                .unwrap();
        let Next::Known(known) = initial
            .next(
                None,
                &mut domain,
                1e-8,
                Discovery::WaitForNewSite,
                &mut cooperation,
            )
            .await
            .unwrap()
        else {
            panic!("complete supplied row")
        };
        assert!(matches!(
            known.observation.objective(),
            Objective::Measured(9.0)
        ));
        assert_eq!(known.point.coordinate(1), 4.0);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("X", array(vec![3.0, 4.0], vec![1, 2]));
        fields.insert("Fval", Value::Num(99.0));
        let mut partial =
            Initial::prepare(Value::Struct(fields.finish()), 2, 1e-3, &mut cooperation)
                .await
                .unwrap();
        assert!(matches!(
            partial
                .next(
                    None,
                    &mut domain,
                    1e-8,
                    Discovery::WaitForNewSite,
                    &mut cooperation
                )
                .await
                .unwrap(),
            Next::NeedSchema
        ));
        let Next::Evaluate { point, seed } = partial
            .next(
                None,
                &mut domain,
                1e-8,
                Discovery::AtUniqueSite,
                &mut cooperation,
            )
            .await
            .unwrap()
        else {
            panic!("real missing-lane discovery")
        };
        let selected = RepeatableCallable::select_rooted(
            Some(&runtime),
            RootedValue::in_context(
                crate::user_functions::test_support::exact_local_value(
                    "surrogate_initial_lanes",
                    bindings.binding("surrogate_initial_lanes"),
                ),
                &runtime,
            )
            .unwrap(),
            &Resolver,
        )
        .unwrap();
        let mut missing = Evaluator::new(&selected, 2, 1).unwrap();
        let Evaluation::Completed(actual) = missing
            .evaluate(point, 1e-3, &mut cooperation)
            .await
            .unwrap()
        else {
            panic!("one actual combined call")
        };
        let actual = actual
            .with_seed(seed, 1e-3, &mut cooperation)
            .await
            .unwrap();
        assert!(matches!(
            actual.observation.objective(),
            Objective::Measured(99.0)
        ));
        assert_eq!(actual.observation.inequality(0), 1.0);
        assert_eq!(missing.evaluated(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("X", array(vec![-1.0, 4.0], vec![1, 2]));
        fields.insert("Fval", Value::Num(99.0));
        fields.insert("Ineq", Value::Num(-1.0));
        let mut repaired =
            Initial::prepare(Value::Struct(fields.finish()), 2, 1e-3, &mut cooperation)
                .await
                .unwrap();
        let Next::Evaluate { point, seed } = repaired
            .next(
                None,
                &mut domain,
                1e-8,
                Discovery::WaitForNewSite,
                &mut cooperation,
            )
            .await
            .unwrap()
        else {
            panic!("changed coordinates require a new complete observation")
        };
        assert_eq!(point, [0.0, 4.0]);
        assert!(seed.is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    });
}

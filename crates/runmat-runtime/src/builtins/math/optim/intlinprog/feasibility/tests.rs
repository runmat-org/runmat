use super::*;
use clarabel::algebra::CscMatrix;
use runmat_builtins::BuiltinCatalogIdentity;

fn context() -> crate::context::RuntimeContext {
    crate::user_functions::test_support::exact_fixture_runtime()
}
fn cooperation(runtime: &crate::context::RuntimeContext) -> ScalarCooperation {
    ScalarCooperation::for_builtin(BuiltinCatalogIdentity { name: "ga" }, Some(runtime))
}
fn binary_triangle() -> Input {
    Input {
        matrix: CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 1.0]),
        rhs: vec![1.0],
        equalities: 0,
        lower: vec![0.0, 0.0],
        upper: vec![1.0, 1.0],
        integers: vec![0, 1],
    }
}

#[test]
fn already_valid_population_point_skips_mip_and_real_repair_does_not_round_relaxation() {
    futures::executor::block_on(async {
        let runtime = context();
        let timing = SolverTiming::start(Some(&runtime)).unwrap();
        let mut cooperation = cooperation(&runtime);
        let mut domain = Domain::new(
            binary_triangle(),
            timing,
            common::global_time::TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        assert!(matches!(
            domain.repair(&[1.0, 0.0], &mut cooperation).await.unwrap(),
            Repair::Unchanged
        ));
        {
            let original = domain.original_linear();
            assert_eq!(original.variables(), 2);
            assert_eq!(original.rows(), 1);
            assert_eq!(original.equalities(), 0);
            assert_eq!(original.rhs(), &[1.0]);
            assert_eq!(original.column(0).unwrap().collect::<Vec<_>>(), [(0, 1.0)]);
            assert_eq!(original.column(1).unwrap().collect::<Vec<_>>(), [(0, 1.0)]);
            assert!(original.column(2).is_none());
            assert_eq!(original.entry(0, 0), Some((0, 1.0)));
            assert_eq!(original.entry(1, 0), Some((0, 1.0)));
            assert!(
                original.entry(0, 1).is_none(),
                "distance row is not original geometry"
            );
            assert!(original.entry(0, usize::MAX).is_none());
            assert!(
                original.entry(2, 0).is_none(),
                "auxiliary column is not original geometry"
            );
        }
        assert_eq!(domain.repair_nodes(), 0);
        let admitted = domain
            .classify(&[1.0, 0.0], 1e-8, &mut cooperation)
            .await
            .unwrap();
        assert!(admitted.feasible);
        assert_eq!(admitted.violation, 0.0);
        assert!(
            !domain
                .classify(&[0.5, 0.0], 1.0, &mut cooperation)
                .await
                .unwrap()
                .feasible
        );
        assert!(
            !domain
                .classify(&[2.0, 0.0], 10.0, &mut cooperation)
                .await
                .unwrap()
                .feasible
        );
        let Repair::Point(point) = domain
            .repair(&[0.75, 0.75], &mut cooperation)
            .await
            .unwrap()
        else {
            panic!("actual original-feasible integer witness required");
        };
        assert_eq!(point.len(), 2);
        assert!(point.iter().all(|value| *value == 0.0 || *value == 1.0));
        assert!(point.iter().sum::<f64>() <= 1.0);
        assert!(domain.repair_nodes() > 0);
        let nodes = domain.repair_nodes();
        assert!(matches!(
            domain.repair(&[0.0, 1.0], &mut cooperation).await.unwrap(),
            Repair::Unchanged
        ));
        assert_eq!(domain.repair_nodes(), nodes);
        assert_eq!(domain.bounds(), (&[0.0, 0.0][..], &[1.0, 1.0][..]));
    });
}

#[test]
fn independent_domains_preserve_valid_signed_zero_and_wide_integer_coordinates() {
    assert_eq!(
        independent::coordinate(-0.0, 0.0, 1.0, true)
            .unwrap_or_else(|_| panic!("valid zero"))
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        independent::coordinate(1e308, 0.0, f64::INFINITY, true)
            .unwrap_or_else(|_| panic!("valid wide integer")),
        1e308
    );
    assert!(matches!(
        independent::coordinate(0.5, 0.2, 0.8, true),
        Err(independent::Refusal::Infeasible)
    ));
}

#[test]
fn bounds_only_domain_retains_original_vectors_without_a_distance_workspace() {
    futures::executor::block_on(async {
        let runtime = context();
        let mut cooperation = cooperation(&runtime);
        let lower = vec![0.0, -2.0];
        let upper = vec![1.0, 2.0];
        let original_lower = lower.as_ptr();
        let original_upper = upper.as_ptr();
        let mut domain = Domain::new(
            Input {
                matrix: CscMatrix::new(0, 2, vec![0, 0, 0], vec![], vec![]),
                rhs: vec![],
                equalities: 0,
                lower,
                upper,
                integers: vec![0],
            },
            SolverTiming::start(Some(&runtime)).unwrap(),
            common::global_time::TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        assert!(matches!(&domain.storage, Storage::Independent(_)));
        {
            let original = domain.original_linear();
            assert_eq!(original.variables(), 2);
            assert_eq!(original.rows(), 0);
            assert_eq!(original.equalities(), 0);
            assert!(original.rhs().is_empty());
            assert_eq!(original.column(0).unwrap().count(), 0);
            assert_eq!(original.column(1).unwrap().count(), 0);
            assert!(original.column(2).is_none());
        }
        assert_eq!(domain.bounds().0.as_ptr(), original_lower);
        assert_eq!(domain.bounds().1.as_ptr(), original_upper);
        assert!(matches!(
            domain.repair(&[0.0, -0.0], &mut cooperation).await.unwrap(),
            Repair::Unchanged
        ));
        let Repair::Point(point) = domain.repair(&[0.75, 3.0], &mut cooperation).await.unwrap()
        else {
            panic!("independent coordinate repair required");
        };
        assert_eq!(point, [1.0, 2.0]);
        assert!(
            domain
                .classify(&point, 0.0, &mut cooperation)
                .await
                .unwrap()
                .feasible
        );
        assert_eq!(domain.repair_nodes(), 0);
    });
}

#[test]
fn infeasible_integer_equality_and_original_time_or_cancellation_are_not_fake_points() {
    futures::executor::block_on(async {
        let runtime = context();
        let mut cooperation = cooperation(&runtime);
        let mut domain = Domain::new(
            Input {
                matrix: CscMatrix::new(1, 1, vec![0, 1], vec![0], vec![1.0]),
                rhs: vec![0.5],
                equalities: 1,
                lower: vec![0.0],
                upper: vec![1.0],
                integers: vec![0],
            },
            SolverTiming::start(Some(&runtime)).unwrap(),
            common::global_time::TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        assert!(matches!(
            domain.repair(&[0.5], &mut cooperation).await.unwrap(),
            Repair::Infeasible
        ));
        let mut timed = Domain::new(
            binary_triangle(),
            SolverTiming::start(Some(&runtime)).unwrap(),
            common::global_time::TimeLimit::Seconds(0.0),
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        assert!(matches!(
            timed.repair(&[0.75, 0.75], &mut cooperation).await.unwrap(),
            Repair::TimeLimit
        ));
        assert_eq!(timed.repair_nodes(), 0);
        let mut day_timed = Domain::new(
            binary_triangle(),
            SolverTiming::start(Some(&runtime)).unwrap(),
            common::global_time::TimeLimit::Days(0.0),
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        assert!(matches!(
            day_timed
                .repair(&[0.75, 0.75], &mut cooperation)
                .await
                .unwrap(),
            Repair::TimeLimit
        ));
        assert_eq!(day_timed.repair_nodes(), 0);

        let source = runmat_execution::cancellation::CancellationSource::new();
        let original = runtime.with_cancellation_token(source.token());
        original.set_error_namespace("GaOriginal");
        let mut cooperation = self::cooperation(&original);
        let mut cancelled = Domain::new(
            binary_triangle(),
            SolverTiming::start(Some(&original)).unwrap(),
            common::global_time::TimeLimit::Unbounded,
            1e-8,
            &mut cooperation,
        )
        .await
        .unwrap();
        source.cancel(runmat_execution::CancellationReason::Quota);
        let error = match cancelled.repair(&[0.75, 0.75], &mut cooperation).await {
            Ok(_) => panic!("original cancellation must escape"),
            Err(error) => error,
        };
        assert_eq!(error.identifier(), Some("GaOriginal:Cancelled"));
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
    });
}

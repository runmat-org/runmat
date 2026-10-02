//! Interior 7/15 Gauss–Kronrod evaluation for open endpoints.
//!
//! Abscissae and weights are the published QUADPACK QK15I rule. Mapping and
//! adaptive acceptance belong to the same quadrature service, not a builtin.

use super::adaptive::QuadratureEvaluator;
use super::algebra::AdaptiveEstimate;
use super::geometry::{midpoint, ScaledEstimate};
use super::mesh::{integrate_partition, MeshRule};
use super::open_algebra::{OpenRuleInputs, OpenSample};
use crate::BuiltinResult;
use num_complex::Complex;

// Positive half of the Kronrod abscissae on [-1, 1]. Even positions are the
// embedded seven-point Gauss rule; zero is evaluated once.
const NODES: [f64; 7] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
];
const KRONROD: [f64; 8] = [
    0.022_935_322_010_529_22,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_2,
    0.140_653_259_715_525_9,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_8,
];
const GAUSS: [f64; 8] = [
    0.0,
    0.129_484_966_168_869_7,
    0.0,
    0.279_705_391_489_276_7,
    0.0,
    0.381_830_050_505_118_9,
    0.0,
    0.417_959_183_673_469_4,
];

/// Transforms an interior coordinate in (0,1) into the callback domain.
#[derive(Clone, Copy, Debug)]
pub(crate) enum OpenDomain {
    Finite,
    RightInfinite {
        bound: f64,
    },
    LeftInfinite {
        bound: f64,
    },
    Contour {
        start: Complex<f64>,
        end: Complex<f64>,
    },
}

impl OpenDomain {
    fn map<E: QuadratureEvaluator>(
        self,
        coordinate: f64,
        evaluator: &E,
    ) -> BuiltinResult<(f64, ScaledEstimate)> {
        match self {
            Self::Finite | Self::Contour { .. } => Ok((coordinate, ScaledEstimate::from_f64(1.0))),
            Self::RightInfinite { bound } | Self::LeftInfinite { bound } => {
                if !(0.0 < coordinate && coordinate < 1.0) {
                    return Err(evaluator.convergence_error());
                }
                let inverse = 1.0 / coordinate;
                let distance = (1.0 - coordinate) / coordinate;
                let argument = match self {
                    Self::RightInfinite { .. } => bound + distance,
                    Self::LeftInfinite { .. } => bound - distance,
                    Self::Finite | Self::Contour { .. } => unreachable!("handled above"),
                };
                if !argument.is_finite() || !inverse.is_finite() {
                    return Err(evaluator.nonrepresentable_error());
                }
                Ok((
                    argument,
                    ScaledEstimate::from_f64(inverse).multiply_f64(inverse),
                ))
            }
        }
    }
}

pub(crate) struct OpenIntervalEstimate<R, E> {
    pub(crate) result: R,
    pub(crate) error: E,
    pub(crate) magnitude: ScaledEstimate,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OpenSegment {
    pub(crate) a: f64,
    pub(crate) b: f64,
    pub(crate) domain: OpenDomain,
}

struct OpenInterval<S: OpenSample> {
    segment: OpenSegment,
    result: S::Result,
    error: S::Error,
    magnitude: ScaledEstimate,
    depth: usize,
}

struct KronrodRule;

impl<E: QuadratureEvaluator> MeshRule<E> for KronrodRule {
    type Segment = OpenSegment;
    type Interval = OpenInterval<E::Sample>;

    const INITIAL_EVALUATIONS: usize = 15;
    const REFINEMENT_EVALUATIONS: usize = 30;

    async fn initial(
        evaluator: &mut E,
        segment: Self::Segment,
        depth: usize,
    ) -> BuiltinResult<Self::Interval> {
        let estimate = evaluate_interval(evaluator, segment.domain, segment.a, segment.b).await?;
        Ok(OpenInterval {
            segment,
            result: estimate.result,
            error: estimate.error,
            magnitude: estimate.magnitude,
            depth,
        })
    }

    fn result(interval: &Self::Interval) -> &<E::Sample as OpenSample>::Result {
        &interval.result
    }

    fn error(interval: &Self::Interval) -> &<E::Sample as OpenSample>::Error {
        &interval.error
    }

    fn magnitude(interval: &Self::Interval) -> ScaledEstimate {
        interval.magnitude
    }

    fn can_refine(_evaluator: &E, interval: &Self::Interval) -> bool {
        let middle = midpoint(interval.segment.a, interval.segment.b);
        interval.depth > 0
            && interval.segment.a < middle
            && middle < interval.segment.b
            && interior_nodes(interval.segment.a, middle).is_some()
            && interior_nodes(middle, interval.segment.b).is_some()
    }

    async fn refine(
        evaluator: &mut E,
        interval: Self::Interval,
    ) -> BuiltinResult<[Self::Interval; 2]> {
        let middle = midpoint(interval.segment.a, interval.segment.b);
        let left = <Self as MeshRule<E>>::initial(
            evaluator,
            OpenSegment {
                b: middle,
                ..interval.segment
            },
            interval.depth - 1,
        )
        .await?;
        let right = <Self as MeshRule<E>>::initial(
            evaluator,
            OpenSegment {
                a: middle,
                ..interval.segment
            },
            interval.depth - 1,
        )
        .await?;
        Ok([left, right])
    }
}

pub(crate) async fn integrate_open<E: QuadratureEvaluator>(
    evaluator: &mut E,
    segments: &[OpenSegment],
) -> BuiltinResult<<E::Sample as OpenSample>::Result> {
    integrate_partition::<E, KronrodRule>(evaluator, segments).await
}

/// Evaluates one interval without sampling its endpoints. All callback
/// evaluations, including transformation and its Jacobian, use the owning
/// evaluator and therefore its original cancellation/effect boundary.
pub(crate) async fn evaluate_interval<E: QuadratureEvaluator>(
    evaluator: &mut E,
    domain: OpenDomain,
    a: f64,
    b: f64,
) -> BuiltinResult<
    OpenIntervalEstimate<<E::Sample as OpenSample>::Result, <E::Sample as OpenSample>::Error>,
> {
    evaluator.checkpoint().await?;
    let (coordinates, half) = interior_nodes(a, b).ok_or_else(|| evaluator.convergence_error())?;
    let mut points = coordinates;
    let mut jacobians = [ScaledEstimate::from_f64(1.0); 15];
    for index in 0..15 {
        let (point, jacobian) = domain.map(coordinates[index], evaluator)?;
        points[index] = point;
        jacobians[index] = jacobian;
    }
    let samples = evaluator.evaluate_open_many(domain, points).await?;
    let half_width = ScaledEstimate::from_f64(half);
    let width = half_width.multiply_f64(2.0);
    let mut base_weights = [ScaledEstimate::from_f64(0.0); 15];
    let mut kronrod_weights = [ScaledEstimate::from_f64(0.0); 15];
    let mut gauss_weights = [ScaledEstimate::from_f64(0.0); 15];
    base_weights[0] = half_width.multiply_f64(KRONROD[7]);
    kronrod_weights[0] = base_weights[0].multiply(jacobians[0]);
    gauss_weights[0] = half_width.multiply_f64(GAUSS[7]).multiply(jacobians[0]);
    for index in 0..7 {
        for side in 0..2 {
            let slot = 1 + index * 2 + side;
            base_weights[slot] = half_width.multiply_f64(KRONROD[index]);
            kronrod_weights[slot] = base_weights[slot].multiply(jacobians[slot]);
            gauss_weights[slot] = half_width
                .multiply_f64(GAUSS[index])
                .multiply(jacobians[slot]);
        }
    }
    let result = <E::Sample as OpenSample>::weighted_sum(&samples, &kronrod_weights, evaluator)
        .await
        .map_err(|cause| evaluator.arithmetic_error(cause))?;
    let gauss = <E::Sample as OpenSample>::weighted_sum(&samples, &gauss_weights, evaluator)
        .await
        .map_err(|cause| evaluator.arithmetic_error(cause))?;
    let error = <E::Sample as OpenSample>::embedded_error(
        OpenRuleInputs {
            samples: &samples,
            base_weights: &base_weights,
            jacobians: &jacobians,
            width,
            kronrod: &result,
            gauss: &gauss,
        },
        evaluator,
    )
    .await
    .map_err(|cause| evaluator.arithmetic_error(cause))?;
    let magnitude = error
        .magnitude(evaluator)
        .await
        .map_err(|cause| evaluator.arithmetic_error(cause))?;
    Ok(OpenIntervalEstimate {
        result,
        error,
        magnitude,
    })
}

fn interior_nodes(a: f64, b: f64) -> Option<([f64; 15], f64)> {
    let center = midpoint(a, b);
    let half = if (b - a).is_finite() {
        (b - a) * 0.5
    } else {
        b * 0.5 - a * 0.5
    };
    if !(a < center && center < b && half.is_finite() && half > 0.0) {
        return None;
    }
    let mut coordinates = [center; 15];
    for (index, &node) in NODES.iter().enumerate() {
        let offset = half * node;
        let left = center - offset;
        let right = center + offset;
        if !(a < left && left < right && right < b) {
            return None;
        }
        coordinates[1 + index * 2] = left;
        coordinates[2 + index * 2] = right;
    }
    Some((coordinates, half))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_runtime_error;
    use crate::builtins::math::optim::quadrature::QuadratureArithmeticError;
    use crate::RuntimeError;
    use futures::task::noop_waker;
    use std::cell::Cell;
    use std::future::Future;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::task::{Context, Poll};

    struct CancellingSine {
        max_evaluations: usize,
        calls: usize,
        charged: u64,
    }

    impl QuadratureEvaluator for CancellingSine {
        type Sample = f64;

        async fn checkpoint(&mut self) -> BuiltinResult<()> {
            Ok(())
        }

        async fn charge(&mut self, units: u64) -> BuiltinResult<()> {
            self.charged += units;
            Ok(())
        }

        async fn evaluate(&mut self, x: f64) -> BuiltinResult<f64> {
            self.calls += 1;
            // Three periods have an analytic zero integral on [0,1], but
            // each half-span has a nonzero integral and the initial embedded
            // rule does not satisfy the requested global tolerance.
            Ok(1.0e5 * (6.0 * std::f64::consts::PI * x).sin())
        }

        fn midpoint(&self, a: f64, b: f64) -> f64 {
            super::super::midpoint(a, b)
        }

        fn tolerances(&self) -> (f64, f64) {
            (1.0e-3, 1.0)
        }

        fn max_evaluations(&self) -> usize {
            self.max_evaluations
        }

        fn max_depth(&self) -> usize {
            30
        }

        fn evaluation_limit_error(&self) -> RuntimeError {
            build_runtime_error("open mesh evaluation limit").build()
        }

        fn convergence_error(&self) -> RuntimeError {
            build_runtime_error("open mesh did not converge").build()
        }

        fn nonrepresentable_error(&self) -> RuntimeError {
            build_runtime_error("open mesh estimate is nonrepresentable").build()
        }

        fn arithmetic_error(&self, error: QuadratureArithmeticError) -> RuntimeError {
            build_runtime_error(format!("open mesh arithmetic: {error:?}")).build()
        }
    }

    #[test]
    fn cancelling_waypoint_segments_use_one_global_error_budget() {
        let segments = [
            OpenSegment {
                a: 0.0,
                b: 0.5,
                domain: OpenDomain::Finite,
            },
            OpenSegment {
                a: 0.5,
                b: 1.0,
                domain: OpenDomain::Finite,
            },
        ];
        let mut insufficient = CancellingSine {
            max_evaluations: 30,
            calls: 0,
            charged: 0,
        };
        let error = futures::executor::block_on(integrate_open(&mut insufficient, &segments))
            .expect_err("canceling segments need global refinement");
        assert!(error.message().contains("evaluation limit"));
        assert_eq!(insufficient.calls, 30);
        assert!(insufficient.charged >= insufficient.calls as u64 * 2);

        let mut enough = CancellingSine {
            max_evaluations: 2000,
            calls: 0,
            charged: 0,
        };
        let value = futures::executor::block_on(integrate_open(&mut enough, &segments))
            .expect("global mesh refines canceling intervals")
            .to_f64()
            .expect("finite result");
        assert!(value.abs() < 1.0e-4);
        assert!(enough.calls > 30);
    }

    #[test]
    fn exhausted_large_interval_does_not_prevent_another_refinement() {
        let interval = |a, b, depth, error: f64| OpenInterval {
            segment: OpenSegment {
                a,
                b,
                domain: OpenDomain::Finite,
            },
            result: ScaledEstimate::from_f64(0.0),
            error: ScaledEstimate::from_f64(error),
            magnitude: ScaledEstimate::from_f64(error),
            depth,
        };
        let exhausted = interval(0.0, 0.5, 0, 10.0);
        let available = interval(0.5, 1.0, 1, 1.0);
        let evaluator = CancellingSine {
            max_evaluations: 100,
            calls: 0,
            charged: 0,
        };
        assert!(!<KronrodRule as MeshRule<CancellingSine>>::can_refine(
            &evaluator, &exhausted
        ));
        assert!(<KronrodRule as MeshRule<CancellingSine>>::can_refine(
            &evaluator, &available
        ));
    }

    struct ArrayWork {
        charged: Rc<Cell<u64>>,
        large_first: bool,
        components: usize,
        pending_at: Option<u64>,
        pending_dropped: Rc<Cell<bool>>,
    }

    impl ArrayWork {
        fn new(components: usize, large_first: bool, pending_at: Option<u64>) -> Self {
            Self {
                charged: Rc::new(Cell::new(0)),
                large_first,
                components,
                pending_at,
                pending_dropped: Rc::new(Cell::new(false)),
            }
        }
    }

    struct PendingCharge(Rc<Cell<bool>>);

    impl Future for PendingCharge {
        type Output = BuiltinResult<()>;

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for PendingCharge {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    impl QuadratureEvaluator for ArrayWork {
        type Sample = Arc<Vec<f64>>;

        async fn checkpoint(&mut self) -> BuiltinResult<()> {
            Ok(())
        }

        async fn charge(&mut self, units: u64) -> BuiltinResult<()> {
            let charged = self.charged.get() + units;
            self.charged.set(charged);
            if self.pending_at == Some(charged) {
                PendingCharge(self.pending_dropped.clone()).await?;
            }
            Ok(())
        }

        async fn evaluate(&mut self, x: f64) -> BuiltinResult<Self::Sample> {
            let mut values = vec![x; self.components];
            if self.large_first {
                values[0] = 1.0e300 * x;
            }
            if self.components > 1 {
                values[1] = x * x;
            }
            Ok(Arc::new(values))
        }

        fn midpoint(&self, a: f64, b: f64) -> f64 {
            super::super::midpoint(a, b)
        }

        fn tolerances(&self) -> (f64, f64) {
            (1.0e-8, 0.0)
        }

        fn max_evaluations(&self) -> usize {
            15
        }

        fn max_depth(&self) -> usize {
            1
        }

        fn evaluation_limit_error(&self) -> RuntimeError {
            build_runtime_error("array evaluation limit").build()
        }

        fn convergence_error(&self) -> RuntimeError {
            build_runtime_error("array convergence failure").build()
        }

        fn nonrepresentable_error(&self) -> RuntimeError {
            build_runtime_error("array nonrepresentable result").build()
        }

        fn arithmetic_error(&self, error: QuadratureArithmeticError) -> RuntimeError {
            build_runtime_error(format!("array arithmetic: {error:?}")).build()
        }
    }

    #[test]
    fn array_weighted_rule_charges_every_coefficient_product() {
        let mut evaluator = ArrayWork::new(2, false, None);
        let estimate = futures::executor::block_on(evaluate_interval(
            &mut evaluator,
            OpenDomain::Finite,
            0.0,
            1.0,
        ))
        .expect("array open rule");
        assert_eq!(estimate.result.0.len(), 2);
        // Two weighted passes (Kronrod and Gauss), each with 15 terms for
        // both components. Additional estimate algebra can charge more.
        assert!(evaluator.charged.get() >= 2 * 15 * 2);
    }

    #[test]
    fn array_error_bounds_are_componentwise_at_extreme_scale() {
        let mut evaluator = ArrayWork::new(2, true, None);
        let estimate = futures::executor::block_on(evaluate_interval(
            &mut evaluator,
            OpenDomain::Finite,
            0.0,
            1.0,
        ))
        .expect("scaled componentwise open rule");
        assert!(estimate.error.0[0].to_f64().expect("large bound") > 1.0e285);
        assert!(estimate.error.0[1].to_f64().expect("ordinary bound") < 1.0e-10);
    }

    #[test]
    fn array_rule_can_suspend_and_drop_after_bounded_actual_coefficient_work() {
        for pending_at in [4096, 2 * 15 * 5000 + 4096] {
            let mut evaluator = ArrayWork::new(5000, false, Some(pending_at));
            let charged = evaluator.charged.clone();
            let dropped = evaluator.pending_dropped.clone();
            let mut invocation = Box::pin(evaluate_interval(
                &mut evaluator,
                OpenDomain::Finite,
                0.0,
                1.0,
            ));
            let waker = noop_waker();
            assert!(invocation
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending());
            assert_eq!(charged.get(), pending_at);
            assert!(!dropped.get());
            drop(invocation);
            assert!(dropped.get());
        }
    }
}

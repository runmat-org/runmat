use super::*;
use futures::executor::block_on;
struct Probe {
    calls: usize,
    limit: usize,
    cooperation: ScalarCooperation,
}
impl Oracle for Probe {
    fn calls(&self) -> usize {
        self.calls
    }
    fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.calls)
    }
    fn cooperation(&mut self) -> &mut ScalarCooperation {
        &mut self.cooperation
    }
    async fn measure(&mut self, x: &[f64]) -> BuiltinResult<Measurement> {
        if self.remaining() == 0 {
            return Ok(Measurement::Stopped(OracleStop::Evaluations));
        }
        self.calls += 1;
        Ok(Measurement::Defined(x.iter().map(|x| x * x).sum()))
    }
}
#[derive(Default)]
struct Phases(Vec<(OutputState, usize)>);
impl Events for Phases {
    async fn notify(
        &mut self,
        state: OutputState,
        _: &Point,
        _: usize,
        calls: usize,
        _: Procedure,
        _: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        self.0.push((state, calls));
        Ok(false)
    }
}
fn policy() -> Policy {
    Policy {
        identity: runmat_builtins::PATTERNSEARCH_CATALOG_ENTRY.identity,
        invalid: &runmat_builtins::PATTERNSEARCH_ERROR_INVALID_ARGUMENT,
        allocation_identifier: "RunMat:patternsearch:AllocationFailed",
        retry: crate::GpuGatherRetry::Never,
    }
}
fn probe(limit: usize) -> Probe {
    Probe {
        calls: 0,
        limit,
        cooperation: ScalarCooperation::for_builtin(policy().identity, None),
    }
}
#[test]
fn known_seed_has_same_simplex_math_without_initial_replay_and_partial_budget_keeps_measured_best()
{
    let controls = Controls {
        tol_x: 1e-8,
        tol_fun: 1e-8,
        max_iter: 1000,
    };
    let mut a = probe(2000);
    let mut b = probe(2000);
    let actual = block_on(minimize(
        policy(),
        Initial::Evaluate(vec![1., 2.]),
        &mut a,
        &controls,
        &mut Phases::default(),
    ))
    .unwrap();
    let mut phases = Phases::default();
    let known = block_on(minimize(
        policy(),
        Initial::Known(Point {
            x: vec![1., 2.],
            f: 5.,
        }),
        &mut b,
        &controls,
        &mut phases,
    ))
    .unwrap();
    assert_eq!(actual.point.x, known.point.x);
    assert_eq!(actual.point.f, known.point.f);
    assert_eq!(actual.iterations, known.iterations);
    assert_eq!(actual.func_count, known.func_count + 1);
    assert_eq!(phases.0.first(), Some(&(OutputState::Init, 0)));
    assert_eq!(phases.0.last(), Some(&(OutputState::Done, b.calls)));
    let mut partial = probe(1);
    let stopped = block_on(minimize(
        policy(),
        Initial::Known(Point {
            x: vec![1., 2.],
            f: 5.,
        }),
        &mut partial,
        &controls,
        &mut Phases::default(),
    ))
    .unwrap();
    assert!(matches!(stopped.reason, Reason::Evaluations));
    assert_eq!(stopped.func_count, 1);
    assert_eq!(
        stopped.point.f,
        stopped.point.x.iter().map(|x| x * x).sum::<f64>()
    );
    assert!(stopped.point.f <= 5.);
}

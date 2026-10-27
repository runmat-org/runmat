//! One actual borrowed-model SQP refinement. The expensive selected source
//! remains rooted but is never invoked by a numerical merit probe.
use super::{
    domain::BoxView,
    invalid,
    merit::{Criterion, Rank},
    multistart::{self, Start},
    numeric,
    oracle::{Controls, MeritOracle, ModelView},
    rbf::Training,
    region::{Region, Scope},
    schedule, tree,
};
use crate::builtins::math::optim::{
    common::ScalarCooperation,
    fmincon::internal::{self, Limits, Problem, Refinement},
    intlinprog::feasibility::Domain,
};
use crate::{execution::RootedValue, BuiltinResult};

/// Tree/MultiStart turns reuse one borrowed model oracle and scratch. Only
/// derived bound slices and the actual starting point change between solves;
/// original linear rows/source/Clock/history are never mirrored or recaptured.
pub(super) struct Session<'a, T, E> {
    domain: &'a Domain,
    source: &'a RootedValue,
    oracle: MeritOracle<'a, T, E>,
    limits: Limits,
    feasibility: bool,
}

/// The request is consumed before borrowing a frozen campaign snapshot for
/// this session. No model refresh or expensive callback occurs in this turn.
pub(super) async fn run<T: Training, E: Training>(
    kind: schedule::Refinement,
    domain: BoxView<'_>,
    anchor: &[f64],
    scale: f64,
    session: &mut Session<'_, T, E>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Refinement>> {
    if !session.matches_domain(&domain) {
        return Err(invalid("scheduled refinement changed its original domain"));
    }
    match kind {
        schedule::Refinement::TreeLocal | schedule::Refinement::TreeReset => {
            let case = if kind == schedule::Refinement::TreeReset {
                tree::Case::Reset
            } else {
                tree::Case::Local
            };
            tree::search(case, domain, anchor, scale, session, cooperation).await
        }
        schedule::Refinement::NonlinearLocal | schedule::Refinement::MultiStart => {
            if session.expired() {
                return Ok(None);
            }
            let mut region = Region::new(domain, anchor, Scope::Local(scale), cooperation).await?;
            let mut point = runmat_value::exact_capacity_vec(anchor.len()).map_err(|source| {
                crate::build_runtime_error("surrogateopt: local starting point allocation failed")
                    .with_builtin(super::NAME)
                    .with_source(source)
                    .build()
            })?;
            for &value in anchor {
                point.push(value);
                cooperation.completed(1).await?;
            }
            if kind == schedule::Refinement::MultiStart {
                multistart::search(Start::Point(point), &mut region, session, cooperation).await
            } else {
                let (lower, upper) = region.bounds();
                let result = session.solve(point, lower, upper, cooperation).await?;
                Ok(session.preferred(&result, None).then_some(result))
            }
        }
    }
}
impl<'a, T: Training, E: Training> Session<'a, T, E> {
    pub(super) async fn new(
        domain: &'a Domain,
        source: &'a RootedValue,
        model: ModelView<'a, T, E>,
        controls: Controls<'a>,
        limits: Limits,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if !model.domain.same_domain(domain) {
            return Err(invalid(
                "local merit and linear geometry have different original domains",
            ));
        }
        let feasibility = matches!(model.criterion, Criterion::Feasibility);
        let oracle = MeritOracle::new(model, controls, cooperation).await?;
        Ok(Self {
            domain,
            source,
            oracle,
            limits,
            feasibility,
        })
    }
    pub(super) async fn solve(
        &mut self,
        initial: Vec<f64>,
        lower: &[f64],
        upper: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Refinement> {
        internal::minimize(
            initial,
            Problem {
                domain: self.domain,
                lower,
                upper,
                source: self.source,
            },
            &mut self.oracle,
            numeric::policy(),
            self.limits,
            cooperation,
        )
        .await
    }
    pub(super) fn expired(&self) -> bool {
        self.oracle.expired()
    }
    pub(super) fn matches_domain(&self, domain: &BoxView<'_>) -> bool {
        domain.same_domain(self.domain)
    }
    pub(super) fn matches_region(&self, region: &Region<'_>) -> bool {
        region.same_domain(self.domain)
    }
    /// Compare modeled local outcomes, never expensive objective values.
    /// Original linear/bound feasibility is separated by typed row roles.
    pub(super) fn preferred(&self, new: &Refinement, old: Option<&Refinement>) -> bool {
        let eligible = |result: &Refinement| {
            !matches!(result.stop, internal::Stop::Time)
                && !self.expired()
                && result.violation.is_some_and(f64::is_finite)
                && result.merit.is_some_and(f64::is_finite)
                && result.constraints.as_ref().is_some_and(|constraints| {
                    constraints.linear_maximum.is_finite()
                        && constraints.linear_maximum <= self.limits.feasibility
                        && (self.feasibility || constraints.nonlinear_violated == 0)
                })
        };
        if !eligible(new) {
            return false;
        }
        let Some(old) = old.filter(|old| eligible(old)) else {
            return true;
        };
        let new_constraints = new.constraints.as_ref().expect("eligible constraints");
        let old_constraints = old.constraints.as_ref().expect("eligible constraints");
        if self.feasibility {
            new_constraints.nonlinear_violated < old_constraints.nonlinear_violated
                || new_constraints.nonlinear_violated == old_constraints.nonlinear_violated
                    && new_constraints.nonlinear_maximum < old_constraints.nonlinear_maximum
        } else {
            new.merit < old.merit
        }
    }
    /// Compare against the actual modeled score of the preceding sample,
    /// never against its expensive objective or a replayed local solve.
    pub(super) fn beats(&self, result: &Refinement, sample: Rank, tolerance: f64) -> bool {
        if !self.preferred(result, None) {
            return false;
        }
        match sample {
            Rank::Objective(merit) => result.merit.is_some_and(|value| value < merit),
            Rank::Feasibility { violated, maximum } => {
                let summary = result
                    .constraints
                    .as_ref()
                    .expect("eligible modeled constraints");
                if summary.nonlinear_violated != violated {
                    return summary.nonlinear_violated < violated;
                }
                let Some(refined) = summary.nonlinear_maximum.filter(|value| value.is_finite())
                else {
                    return false;
                };
                let Some(maximum) = num_rational::BigRational::from_float(maximum) else {
                    return false;
                };
                let Some(tolerance) = num_rational::BigRational::from_float(tolerance) else {
                    return false;
                };
                num_rational::BigRational::from_float(refined).expect("finite modeled residual")
                    < maximum - tolerance
            }
        }
    }
}

//! Adaptive merit ranking and scale transitions. Expensive evaluations and
//! model generations belong to the controller; this owner never invokes them.
use super::{
    invalid,
    rbf::{Model, Prediction, Training},
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CandidateIndex(pub usize);
pub(super) trait Proposals {
    fn len(&self) -> usize;
    fn dimensions(&self) -> usize;
    fn coordinate(&self, row: usize, axis: usize) -> f64;
}
#[derive(Clone, Copy)]
pub(super) enum Phase {
    Feasibility {
        inequality_start: usize,
    },
    Objective {
        objective_lane: usize,
        inequality_start: usize,
    },
}
pub(super) struct SelectionControls {
    pub phase: Phase,
    pub weight: f64,
    pub constraint_tolerance: f64,
    pub min_distance: f64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NoCandidate {
    TooClose,
    Undefined,
    NoFeasiblePrediction,
}
pub(super) enum Selection {
    Point(Selected),
    Reset(NoCandidate),
}
pub(super) struct Selected {
    pub index: CandidateIndex,
    pub criterion: Criterion,
    pub rank: Rank,
}
#[derive(Clone, Copy)]
pub(super) enum Rank {
    Feasibility { violated: usize, maximum: f64 },
    Objective(f64),
}
#[derive(Clone, Copy)]
pub(super) enum Criterion {
    Feasibility,
    Objective(Normalization),
}
#[derive(Clone, Copy)]
pub(super) struct Normalization {
    minimum: f64,
    maximum: f64,
    distance_minimum: f64,
    distance_maximum: f64,
    weight: f64,
}
impl Normalization {
    pub(super) fn value(self, objective: f64, nearest: f64) -> Option<f64> {
        let value = self.weight * normalized(objective, self.minimum, self.maximum)
            + (1.0 - self.weight)
                * (1.0 - normalized(nearest, self.distance_minimum, self.distance_maximum));
        value.is_finite().then_some(value)
    }
}
/// RunMat's reset-Tree scaling when no eligible candidate pool exists:
/// observed current-model objective range and the full unit-box distance
/// range. This remains a real surrogate/distance merit, not a constant
/// fallback or an expensive callback. Primary docs leave this scale unspecified.
pub(super) async fn reset_criterion(
    phase: Phase,
    data: &impl Training,
    weight: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Criterion> {
    match phase {
        Phase::Feasibility { .. } => Ok(Criterion::Feasibility),
        Phase::Objective { objective_lane, .. } => {
            if !data.valid()
                || data.rows().is_empty()
                || objective_lane >= data.lanes()
                || !weight.is_finite()
                || !(0.0..=1.0).contains(&weight)
            {
                return Err(invalid("invalid reset Tree merit scaling"));
            }
            let (mut minimum, mut maximum) = (f64::INFINITY, f64::NEG_INFINITY);
            for row in 0..data.rows().len() {
                let value = data.measurement(row, objective_lane);
                if !value.is_finite() {
                    return Err(invalid("nonfinite reset Tree measurement"));
                }
                minimum = minimum.min(value);
                maximum = maximum.max(value);
                cooperation.completed(1).await?;
            }
            Ok(Criterion::Objective(Normalization {
                minimum,
                maximum,
                distance_minimum: 0.0,
                distance_maximum: (data.dimensions() as f64).sqrt(),
                weight,
            }))
        }
    }
}
struct Score {
    index: usize,
    surrogate: f64,
    nearest: f64,
    violations: usize,
}
fn normalized(value: f64, lower: f64, upper: f64) -> f64 {
    if lower == upper {
        return 0.0;
    }
    if value == lower {
        return 0.0;
    }
    if value == upper {
        return 1.0;
    }
    let span = upper - lower;
    if span.is_finite() {
        (value - lower) / span
    } else {
        (value * 0.5 - lower * 0.5) / (upper * 0.5 - lower * 0.5)
    }
}
async fn buffer<T>(capacity: usize, cooperation: &mut ScalarCooperation) -> BuiltinResult<Vec<T>> {
    cooperation.completed(0).await?;
    runmat_value::exact_capacity_vec(capacity).map_err(|source| {
        crate::build_runtime_error("surrogateopt: adaptive score allocation failed")
            .with_builtin(super::NAME)
            .with_source(source)
            .build()
    })
}

/// Active model sites and all evaluated sites are separate borrowed views of
/// the same history. Old phases still exclude already evaluated coordinates,
/// but do not silently become training data for a new construction phase.
pub(super) async fn select(
    model: &Model,
    training: &impl Training,
    evaluated: &impl Training,
    proposals: &impl Proposals,
    controls: SelectionControls,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Selection> {
    let SelectionControls {
        phase,
        weight,
        constraint_tolerance,
        min_distance,
    } = controls;
    let dimensions = training.dimensions();
    if dimensions != evaluated.dimensions()
        || dimensions != proposals.dimensions()
        || !weight.is_finite()
        || !(0.0..=1.0).contains(&weight)
        || !constraint_tolerance.is_finite()
        || constraint_tolerance < 0.0
        || !min_distance.is_finite()
        || min_distance < 0.0
        || !evaluated.valid()
    {
        return Err(invalid("invalid adaptive merit geometry or controls"));
    }
    let start = match phase {
        Phase::Feasibility { inequality_start } => inequality_start,
        Phase::Objective {
            objective_lane,
            inequality_start,
        } => {
            if objective_lane >= training.lanes() {
                return Err(invalid("missing objective lane"));
            }
            inequality_start
        }
    };
    if start > training.lanes()
        || matches!(phase, Phase::Feasibility { .. }) && start == training.lanes()
        || matches!(phase, Phase::Objective { objective_lane, .. } if objective_lane >= start)
    {
        return Err(invalid("invalid constraint lane extent"));
    }
    let mut point = buffer(dimensions, cooperation).await?;
    point.resize(dimensions, 0.0);
    let mut prediction = buffer(training.lanes(), cooperation).await?;
    prediction.resize(training.lanes(), 0.0);
    let mut scores = buffer(proposals.len(), cooperation).await?;
    let (mut pair_min, mut pair_max) = (f64::INFINITY, 0.0_f64);
    let mut reason = NoCandidate::TooClose;
    for row in 0..proposals.len() {
        for (axis, value) in point.iter_mut().enumerate() {
            *value = proposals.coordinate(row, axis);
            if !value.is_finite() {
                return Err(invalid("nonfinite adaptive proposal coordinate"));
            }
            cooperation.completed(1).await?;
        }
        let mut nearest = f64::INFINITY;
        let (mut local_min, mut local_max) = (f64::INFINITY, 0.0_f64);
        for old in 0..evaluated.rows().len() {
            let mut distance = 0.0_f64;
            for (axis, &value) in point.iter().enumerate() {
                distance = distance.hypot(value - evaluated.coordinate(old, axis));
                cooperation.completed(1).await?;
            }
            nearest = nearest.min(distance);
            local_min = local_min.min(distance);
            local_max = local_max.max(distance);
        }
        if !nearest.is_finite() || !local_max.is_finite() {
            reason = NoCandidate::Undefined;
            continue;
        }
        if nearest < min_distance {
            continue;
        }
        if model
            .predict_into(training, &point, &mut prediction, cooperation)
            .await?
            == Prediction::Undefined
        {
            reason = NoCandidate::Undefined;
            continue;
        }
        let mut violations = 0usize;
        let mut maximum = f64::NEG_INFINITY;
        for &value in &prediction[start..] {
            if value > constraint_tolerance {
                violations += 1;
            }
            maximum = maximum.max(value);
            cooperation.completed(1).await?;
        }
        let surrogate = match phase {
            Phase::Feasibility { .. } => maximum,
            Phase::Objective { objective_lane, .. } => {
                if violations != 0 {
                    reason = NoCandidate::NoFeasiblePrediction;
                    continue;
                }
                prediction[objective_lane]
            }
        };
        pair_min = pair_min.min(local_min);
        pair_max = pair_max.max(local_max);
        scores.push(Score {
            index: row,
            surrogate,
            nearest,
            violations,
        });
    }
    if scores.is_empty() {
        return Ok(Selection::Reset(reason));
    }
    // Phase 1 first restricts candidates to the fewest predicted violations.
    if matches!(phase, Phase::Feasibility { .. }) {
        let mut fewest = usize::MAX;
        for score in &scores {
            fewest = fewest.min(score.violations);
            cooperation.completed(1).await?;
        }
        let mut best = None;
        let mut least = f64::INFINITY;
        let mut farthest = f64::NEG_INFINITY;
        for score in scores {
            if score.violations == fewest
                && (score.surrogate < least || score.surrogate == least && score.nearest > farthest)
            {
                best = Some(CandidateIndex(score.index));
                least = score.surrogate;
                farthest = score.nearest;
            }
            cooperation.completed(1).await?;
        }
        return Ok(Selection::Point(Selected {
            index: best.expect("finite least-violation candidate"),
            criterion: Criterion::Feasibility,
            rank: Rank::Feasibility {
                violated: fewest,
                maximum: least,
            },
        }));
    }
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for score in &scores {
        minimum = minimum.min(score.surrogate);
        maximum = maximum.max(score.surrogate);
        cooperation.completed(1).await?;
    }
    let mut best = None;
    let mut best_merit = f64::INFINITY;
    for score in scores {
        let predicted = normalized(score.surrogate, minimum, maximum);
        let separation = 1.0 - normalized(score.nearest, pair_min, pair_max);
        let merit = weight * predicted + (1.0 - weight) * separation;
        if merit < best_merit {
            best = Some(CandidateIndex(score.index));
            best_merit = merit;
        }
        cooperation.completed(1).await?;
    }
    Ok(Selection::Point(Selected {
        index: best.expect("finite normalized candidate merit"),
        rank: Rank::Objective(best_merit),
        criterion: Criterion::Objective(Normalization {
            minimum,
            maximum,
            distance_minimum: pair_min,
            distance_maximum: pair_max,
            weight,
        }),
    }))
}

#[derive(Clone, Copy)]
pub(super) enum Improvement {
    Success,
    Failure,
}
pub(super) struct Search {
    cycle: usize,
    scale: f64,
    successes: usize,
    failures: usize,
    failure_limit: usize,
}
impl Search {
    pub(super) fn new(dimensions: usize) -> Self {
        Self {
            cycle: 0,
            scale: 0.2,
            successes: 0,
            failures: 0,
            failure_limit: dimensions.max(5),
        }
    }
    pub(super) fn weight(&self) -> f64 {
        [0.3, 0.5, 0.8, 0.95][self.cycle]
    }
    pub(super) fn scale(&self) -> f64 {
        self.scale
    }
    pub(super) fn sampler(
        &self,
        problem: super::sampling::Problem,
    ) -> BuiltinResult<super::sampling::Kind> {
        problem.sampler(self.cycle)
    }
    pub(super) fn selected(&mut self) {
        self.cycle = (self.cycle + 1) % 4;
    }
    pub(super) fn measured(&mut self, improvement: Improvement) {
        match improvement {
            Improvement::Success => self.successes += 1,
            Improvement::Failure => self.failures += 1,
        }
        if self.successes == 3 {
            self.scale = (self.scale * 2.0).min(0.8);
            self.successes = 0;
            self.failures = 0;
        } else if self.failures == self.failure_limit {
            self.scale = (self.scale * 0.5).max(1e-5);
            self.successes = 0;
            self.failures = 0;
        }
    }
}

//! Single canonical measured-trial history with borrowed model projections.
//! Point and measurement generations are committed together; no replay or
//! copied normalized population is used to refresh a model.
use super::{
    domain::BoxView,
    invalid,
    numeric::DoubleArray,
    observation::{Objective, Observation, Quality, Schema},
    queue::{Origin, PhaseId},
    rbf::{Generation, Training, TrainingOwner},
};
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
mod snapshot;
pub(super) use snapshot::{Snapshot, SnapshotView};

pub(super) struct Point {
    values: DoubleArray,
    row: usize,
    rows: usize,
    variables: usize,
}
impl Point {
    pub(super) fn batch_row(
        values: DoubleArray,
        row: usize,
        rows: usize,
        variables: usize,
    ) -> BuiltinResult<Self> {
        if row >= rows || !values.matrix_matches(rows, Some(variables)) {
            return Err(invalid(
                "actual batch point has invalid row/variable extent",
            ));
        }
        Ok(Self {
            values,
            row,
            rows,
            variables,
        })
    }
    pub(super) fn coordinate(&self, column: usize) -> f64 {
        assert!(
            column < self.variables,
            "validated original point coordinate"
        );
        if self.rows == 1 {
            self.values.get(column)
        } else {
            self.values.get(column * self.rows + self.row)
        }
    }
    pub(super) fn variables(&self) -> usize {
        self.variables
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) struct TrialIndex(pub usize);
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(super) enum MeasurementSource {
    Callback,
    InitialValues,
}
pub(super) struct Trial {
    pub point: Point,
    pub observation: Observation,
    pub phase: PhaseId,
    pub origin: Origin,
    pub source: MeasurementSource,
}
pub(super) struct History {
    identity: TrainingOwner,
    trials: Vec<Trial>,
    schema: Option<Schema>,
    generation: Generation,
    variables: usize,
}
impl History {
    pub(super) fn new(variables: usize) -> Self {
        Self {
            identity: TrainingOwner::new(),
            trials: Vec::new(),
            schema: None,
            generation: Generation(0),
            variables,
        }
    }
    pub(super) async fn append(
        &mut self,
        trial: Trial,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<TrialIndex> {
        if trial.point.variables() != self.variables {
            return Err(invalid(
                "measured point changed its original variable extent",
            ));
        }
        for column in 0..self.variables {
            if !trial.point.coordinate(column).is_finite() {
                return Err(invalid("measured point contains a nonfinite coordinate"));
            }
            cooperation.completed(1).await?;
        }
        let schema = trial.observation.schema();
        schema
            .lanes()
            .ok_or_else(|| invalid("measured lane extent overflow"))?;
        if self.schema.is_some_and(|expected| expected != schema) {
            return Err(invalid(
                "objective/constraint lane schema changed across actual evaluations",
            ));
        }
        let generation = self
            .generation
            .0
            .checked_add(1)
            .ok_or_else(|| invalid("trial generation counter overflow"))?;
        self.trials.try_reserve(1).map_err(|source| {
            crate::build_runtime_error("surrogateopt: trial metadata allocation failed")
                .with_builtin(super::NAME)
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_source(source)
                .build()
        })?;
        let index = TrialIndex(self.trials.len());
        self.trials.push(trial);
        self.schema = Some(schema);
        self.generation = Generation(generation);
        Ok(index)
    }
    pub(super) fn trial(&self, index: TrialIndex) -> &Trial {
        &self.trials[index.0]
    }
    pub(super) fn len(&self) -> usize {
        self.trials.len()
    }
    pub(super) fn iter(&self) -> impl Iterator<Item = &Trial> {
        self.trials.iter()
    }
    pub(super) fn variables(&self) -> usize {
        self.variables
    }
    pub(super) async fn evaluated_rows(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<usize>> {
        let mut rows = runmat_value::exact_capacity_vec(self.trials.len()).map_err(|source| {
            crate::build_runtime_error("surrogateopt: evaluated-site metadata allocation failed")
                .with_builtin(super::NAME)
                .with_source(source)
                .build()
        })?;
        for (index, trial) in self.trials.iter().enumerate() {
            if trial.observation.quality == Quality::Finite {
                rows.push(index);
            }
            cooperation.completed(1).await?;
        }
        Ok(rows)
    }
    pub(super) fn schema(&self) -> Option<Schema> {
        self.schema
    }
    pub(super) async fn model_rows(
        &self,
        phase: PhaseId,
        include_adaptive: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<usize>> {
        let mut selected: Vec<usize> = runmat_value::exact_capacity_vec(self.trials.len())
            .map_err(|source| {
                crate::build_runtime_error("surrogateopt: model row-index allocation failed")
                    .with_builtin(super::NAME)
                    .with_source(source)
                    .build()
            })?;
        for (index, trial) in self.trials.iter().enumerate().rev() {
            if trial.phase == phase
                && trial.observation.quality == Quality::Finite
                && (include_adaptive || !matches!(trial.origin, Origin::Adaptive))
            {
                // A zero-distance adaptive revisit is a real new callback,
                // not grounds to transfer or average an old Fval. Retain its
                // newest actual complete observation in the interpolant and
                // every original observation in the public trial history.
                let mut duplicate = false;
                for &later in &selected {
                    let other = &self.trials[later].point;
                    let mut equal = true;
                    for column in 0..self.variables {
                        equal &= trial.point.coordinate(column) == other.coordinate(column);
                        cooperation.completed(1).await?;
                        if !equal {
                            break;
                        }
                    }
                    if equal {
                        duplicate = true;
                        break;
                    }
                }
                if !duplicate {
                    selected.push(index);
                }
            }
            cooperation.completed(1).await?;
        }
        for index in 0..selected.len() / 2 {
            let other = selected.len() - 1 - index;
            selected.swap(index, other);
            cooperation.completed(1).await?;
        }
        Ok(selected)
    }
}

/// Axes select independent affine-tail columns only. Radial coordinates use
/// all original free coordinates; no physical distance is projected away.
/// This view borrows only rows/domain/tail metadata, never trial buffers.
pub(super) struct View<'a, 'domain> {
    history: &'a History,
    rows: &'a [usize],
    axes: &'a [usize],
    domain: &'a BoxView<'domain>,
}
impl Training for View<'_, '_> {
    fn owner(&self) -> &TrainingOwner {
        &self.history.identity
    }
    fn valid(&self) -> bool {
        self.history.variables == self.domain.variables() && !self.axes.is_empty()
    }
    fn generation(&self) -> Generation {
        self.history.generation
    }
    fn rows(&self) -> &[usize] {
        self.rows
    }
    fn dimensions(&self) -> usize {
        self.domain.dimensions()
    }
    fn affine_axes(&self) -> Option<&[usize]> {
        Some(self.axes)
    }
    fn lanes(&self) -> usize {
        self.history.schema.map_or(0, |schema| {
            schema.lanes().expect("validated measured schema")
        })
    }
    fn coordinate(&self, selected_row: usize, axis: usize) -> f64 {
        let free = axis;
        let column = self.domain.free_columns()[free];
        let value = self.history.trials[self.rows[selected_row]]
            .point
            .coordinate(column);
        self.domain.normalize_coordinate(value, free)
    }
    fn measurement(&self, selected_row: usize, lane: usize) -> f64 {
        let observation = &self.history.trials[self.rows[selected_row]].observation;
        match observation.objective() {
            Objective::Measured(value) if lane == 0 => value,
            Objective::Measured(_) => observation.inequality(lane - 1),
            Objective::Absent => observation.inequality(lane),
        }
    }
}

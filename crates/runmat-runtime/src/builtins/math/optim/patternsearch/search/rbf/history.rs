//! Actual callback measurements, bounded by the documented 10*n reset epoch.
//! Coordinate carriers and constraint storage retain immutable backing; no
//! coordinate or constraint payload is copied merely to remember a trial.
use super::super::super::{buffer, evaluation::PointAxis, invalid, push};
use crate::builtins::math::optim::common::{cubic_rbf, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::{ExactArrayStorage, Tensor, Value};
#[cfg(test)]
mod tests;

pub(in crate::builtins::math::optim::patternsearch) enum Coordinates {
    Scalar(f64),
    Vector(Tensor),
    Batch {
        backing: Tensor,
        axis: PointAxis,
        point: usize,
        points: usize,
        variables: usize,
    },
}
impl Coordinates {
    pub fn point(argument: &Value, variables: usize) -> BuiltinResult<Self> {
        match argument {
            Value::Num(value) if variables == 1 && value.is_finite() => Ok(Self::Scalar(*value)),
            Value::Tensor(value) if value.len() == variables && value.as_f64_slice().is_some() => {
                Ok(Self::Vector(value.clone()))
            }
            _ => Err(invalid("invalid retained RBF callback coordinate carrier")),
        }
    }
    pub fn batch(
        argument: &Tensor,
        axis: PointAxis,
        point: usize,
        points: usize,
        variables: usize,
    ) -> BuiltinResult<Self> {
        if point >= points
            || points.checked_mul(variables) != Some(argument.len())
            || axis.point_count(argument.shape(), variables) != Some(points)
            || argument.as_f64_slice().is_none()
        {
            return Err(invalid("invalid retained RBF batch coordinate carrier"));
        }
        Ok(Self::Batch {
            backing: argument.clone(),
            axis,
            point,
            points,
            variables,
        })
    }
    fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Vector(value) => value.len(),
            Self::Batch { variables, .. } => *variables,
        }
    }
    fn at(&self, coordinate: usize) -> f64 {
        match self {
            Self::Scalar(value) => *value,
            Self::Vector(value) => {
                value.as_f64_slice().expect("validated double coordinate")[coordinate]
            }
            Self::Batch {
                backing,
                axis,
                point,
                points,
                variables,
            } => backing
                .as_f64_slice()
                .expect("validated double batch coordinate")
                [axis.offset(*point, coordinate, *points, *variables)],
        }
    }
}
pub(in crate::builtins::math::optim::patternsearch) struct Record {
    coordinates: Coordinates,
    pub f: f64,
    pub inequality: ExactArrayStorage<f64>,
    pub equality: ExactArrayStorage<f64>,
}
pub(in crate::builtins::math::optim::patternsearch) struct History {
    owner: cubic_rbf::TrainingOwner,
    generation: u64,
    dimensions: usize,
    period: usize,
    origin: usize,
    epoch: usize,
    records: Vec<Record>,
}
impl History {
    pub fn new(dimensions: usize, origin: usize) -> BuiltinResult<Self> {
        let period = dimensions
            .checked_mul(10)
            .filter(|value| *value > 0)
            .ok_or_else(|| invalid("RBF history dimension overflow"))?;
        Ok(Self {
            owner: cubic_rbf::TrainingOwner::new(),
            generation: 0,
            dimensions,
            period,
            origin,
            epoch: 0,
            records: Vec::new(),
        })
    }
    fn changed(&mut self) -> BuiltinResult<()> {
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("RBF history generation exhausted"))?;
        Ok(())
    }
    /// Called for each actual attempted objective, not just finite returns.
    pub fn attempted(&mut self, ordinal: usize) -> BuiltinResult<()> {
        let elapsed = ordinal
            .checked_sub(self.origin)
            .ok_or_else(|| invalid("stale RBF trial ordinal"))?;
        let epoch = elapsed / self.period;
        if epoch > self.epoch {
            self.records.clear();
            self.epoch = epoch;
            self.changed()?;
        }
        Ok(())
    }
    pub async fn record(
        &mut self,
        ordinal: usize,
        coordinates: Coordinates,
        f: f64,
        inequality: ExactArrayStorage<f64>,
        equality: ExactArrayStorage<f64>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let elapsed = ordinal
            .checked_sub(self.origin)
            .ok_or_else(|| invalid("stale RBF trial ordinal"))?;
        if elapsed / self.period != self.epoch || elapsed % self.period == 0 {
            return Ok(());
        }
        if coordinates.len() != self.dimensions || !f.is_finite() {
            return Err(invalid("invalid RBF history measurement"));
        }
        for value in inequality.iter().chain(equality.iter()) {
            if !value.is_finite() {
                return Err(invalid("nonfinite RBF history constraint"));
            }
            cooperation.completed(1).await?;
        }
        let mut duplicate = None;
        for (index, record) in self.records.iter().enumerate() {
            let mut same = true;
            for axis in 0..self.dimensions {
                let value = coordinates.at(axis);
                if !value.is_finite() {
                    return Err(invalid("nonfinite RBF history coordinate"));
                }
                if value != record.coordinates.at(axis) {
                    same = false;
                }
                cooperation.completed(1).await?;
            }
            if same {
                duplicate = Some(index);
                break;
            }
        }
        if self.records.is_empty() {
            for axis in 0..self.dimensions {
                if !coordinates.at(axis).is_finite() {
                    return Err(invalid("nonfinite RBF history coordinate"));
                }
                cooperation.completed(1).await?;
            }
        }
        let record = Record {
            coordinates,
            f,
            inequality,
            equality,
        };
        if let Some(index) = duplicate {
            self.records[index] = record;
        } else {
            push(&mut self.records, record)?;
        }
        self.changed()
    }
    pub async fn contains(
        &self,
        point: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        if point.len() != self.dimensions {
            return Ok(false);
        }
        for record in &self.records {
            let mut same = true;
            for (axis, &value) in point.iter().enumerate() {
                same &= value == record.coordinates.at(axis);
                cooperation.completed(1).await?;
            }
            if same {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn records(&self) -> &[Record] {
        &self.records
    }
    pub fn coordinate(&self, row: usize, axis: usize) -> f64 {
        self.records[row].coordinates.at(axis)
    }
    /// The scores belong to one current AL state, while the retained records
    /// remain raw measurements usable after multiplier/penalty transitions.
    pub async fn training<'a>(
        &'a self,
        scores: &'a [f64],
        axes: Option<&'a [usize]>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Training<'a>>> {
        if self.records.len() < self.dimensions.saturating_add(1) {
            return Ok(None);
        }
        if scores.len() != self.records.len() {
            return Err(invalid("RBF score/history extent mismatch"));
        }
        if let Some(axes) = axes {
            let mut previous = None;
            for &axis in axes {
                if axis >= self.dimensions || previous.is_some_and(|value| value >= axis) {
                    return Err(invalid("invalid RBF affine axes"));
                }
                previous = Some(axis);
                cooperation.completed(1).await?;
            }
        }
        let mut rows = buffer(self.records.len())?;
        for (row, score) in scores.iter().enumerate() {
            if !score.is_finite() {
                return Ok(None);
            }
            rows.push(row);
            cooperation.completed(1).await?;
        }
        Ok(Some(Training {
            history: self,
            scores,
            rows,
            axes,
        }))
    }
}
pub(in crate::builtins::math::optim::patternsearch) struct Training<'a> {
    history: &'a History,
    scores: &'a [f64],
    rows: Vec<usize>,
    axes: Option<&'a [usize]>,
}
impl cubic_rbf::Training for Training<'_> {
    fn valid(&self) -> bool {
        self.rows.len() == self.scores.len() && self.rows.len() == self.history.len()
    }
    fn owner(&self) -> &cubic_rbf::TrainingOwner {
        &self.history.owner
    }
    fn generation(&self) -> cubic_rbf::Generation {
        cubic_rbf::Generation(self.history.generation)
    }
    fn rows(&self) -> &[usize] {
        &self.rows
    }
    fn dimensions(&self) -> usize {
        self.history.dimensions
    }
    fn lanes(&self) -> usize {
        1
    }
    fn affine_axes(&self) -> Option<&[usize]> {
        self.axes
    }
    fn coordinate(&self, selected_row: usize, axis: usize) -> f64 {
        self.history.coordinate(self.rows[selected_row], axis)
    }
    fn measurement(&self, selected_row: usize, _lane: usize) -> f64 {
        self.scores[self.rows[selected_row]]
    }
}

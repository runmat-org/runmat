//! Bounded visited-point history. Numeric argument headers retain the existing
//! immutable callback backing; history never makes another full coordinate copy.
use super::invalid;
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
use runmat_value::{Tensor, Value};
use std::collections::VecDeque;

#[derive(Clone, Copy, PartialEq)]
pub(super) struct Parameters {
    pub capacity: usize,
    pub tolerance: f64,
}
enum Coordinates {
    Scalar(f64),
    Array(Tensor),
    MatrixPoint {
        backing: Tensor,
        start: usize,
        stride: usize,
        len: usize,
    },
}
impl Coordinates {
    fn len(&self) -> usize {
        match self {
            Self::MatrixPoint { len, .. } => *len,
            Self::Scalar(_) => 1,
            Self::Array(value) => value.len(),
        }
    }
    fn at(&self, index: usize) -> f64 {
        match self {
            Self::MatrixPoint {
                backing,
                start,
                stride,
                ..
            } => backing.as_f64_slice().expect("double argument")[start + index * stride],
            Self::Scalar(value) => *value,
            Self::Array(value) => value.as_f64_slice().expect("double argument")[index],
        }
    }
    #[cfg(test)]
    fn values(&self) -> Option<&[f64]> {
        match self {
            Self::Scalar(value) => Some(std::slice::from_ref(value)),
            Self::MatrixPoint { .. } => None,
            Self::Array(value) => value.as_f64_slice(),
        }
    }
}
pub(super) struct Cache {
    parameters: Parameters,
    points: VecDeque<Coordinates>,
}
impl Cache {
    fn validate(parameters: Parameters) -> BuiltinResult<()> {
        if parameters.capacity == 0
            || !parameters.tolerance.is_finite()
            || parameters.tolerance <= 0.0
        {
            return Err(invalid(
                "CacheSize must be positive and CacheTol finite positive",
            ));
        }
        Ok(())
    }
    pub fn new(parameters: Parameters) -> BuiltinResult<Self> {
        Self::validate(parameters)?;
        Ok(Self {
            parameters,
            points: VecDeque::new(),
        })
    }
    pub async fn configure(
        current: &mut Option<Self>,
        selected: Option<Parameters>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        match selected {
            None => *current = None,
            Some(parameters) => {
                Self::validate(parameters)?;
                if let Some(current) = current {
                    current.parameters = parameters;
                    while current.points.len() > parameters.capacity {
                        current.points.pop_front();
                        cooperation.completed(1).await?;
                    }
                } else {
                    *current = Some(Self::new(parameters)?);
                }
            }
        }
        Ok(())
    }
    /// A new AL multiplier/penalty generation has a different merit function;
    /// a point visited by its predecessor must not suppress its exploration.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    pub async fn contains(
        &self,
        point: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        for original in &self.points {
            if original.len() != point.len() {
                return Err(invalid("cache coordinate extent changed"));
            }
            let mut distance = 0.0f64;
            for (index, &right) in point.iter().enumerate() {
                distance = distance.hypot(original.at(index) - right);
                cooperation.completed(1).await?;
                if distance > self.parameters.tolerance {
                    break;
                }
            }
            if distance <= self.parameters.tolerance {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The caller has constructed this exact argument for an actual callback.
    /// Scalar copies are inline; Tensor::clone shares HostNumericBuffer and
    /// SharedShape. No struct/cell/object/capture clone is admitted here.
    pub fn record(&mut self, argument: &Value) -> BuiltinResult<()> {
        let point = match argument {
            Value::Num(value) => Coordinates::Scalar(*value),
            Value::Tensor(value) if value.as_f64_slice().is_some() => {
                Coordinates::Array(value.clone())
            }
            _ => {
                return Err(invalid(
                    "cache requires the original real double callback argument",
                ))
            }
        };
        self.insert(point)
    }

    pub fn record_matrix(
        &mut self,
        backing: &Tensor,
        point: usize,
        points: usize,
        variables: usize,
        columns: bool,
    ) -> BuiltinResult<()> {
        let shape = if columns {
            [variables, points]
        } else {
            [points, variables]
        };
        if points == 0
            || variables == 0
            || point >= points
            || backing.shape() != shape
            || backing.as_f64_slice().is_none()
        {
            return Err(invalid(
                "cache matrix point must reference its actual double callback layout",
            ));
        }
        self.insert(Coordinates::MatrixPoint {
            backing: backing.clone(),
            start: if columns { point * variables } else { point },
            stride: if columns { 1 } else { points },
            len: variables,
        })
    }

    fn insert(&mut self, point: Coordinates) -> BuiltinResult<()> {
        if self.points.len() == self.parameters.capacity {
            self.points.pop_front();
        }
        self.points.try_reserve(1).map_err(|source| {
            crate::build_runtime_error("patternsearch: cache metadata allocation failed")
                .with_builtin(super::NAME)
                .with_identifier("RunMat:patternsearch:AllocationFailed")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_source(source)
                .build()
        })?;
        self.points.push_back(point);
        Ok(())
    }

    #[cfg(test)]
    pub fn first_values(&self) -> Option<&[f64]> {
        self.points.front().and_then(Coordinates::values)
    }
}

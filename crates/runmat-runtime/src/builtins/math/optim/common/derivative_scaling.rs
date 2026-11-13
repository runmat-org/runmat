//! One admitted scalar/vector scaling owner for numerical derivative probes.
//! Options are decoded once; coordinate access never clones their vectors.
use super::ScalarOptionFields;
fn optim_error(name: &str, message: &str) -> crate::RuntimeError {
    let mut error = super::optim_error(name, message);
    error.gpu_gather_retry = crate::GpuGatherRetry::Never;
    error
}
use crate::BuiltinResult;
use runmat_builtins::OptimOptionField as Field;
use runmat_value::Value;
use std::sync::Arc;

#[derive(Clone, Debug)]
enum AxisScale {
    Scalar(f64),
    Vector(Arc<[f64]>),
}
impl AxisScale {
    fn get(&self, index: usize) -> f64 {
        match self {
            Self::Scalar(value) => *value,
            Self::Vector(values) => values[index],
        }
    }
    fn validate_len(&self, name: &str, n: usize) -> BuiltinResult<()> {
        if matches!(self, Self::Vector(values) if values.len() != n) {
            return Err(optim_error(
                name,
                "derivative scaling vector must match the original variable count",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct AbsoluteSteps {
    minimum: f64,
    maximum: f64,
}
impl Default for AbsoluteSteps {
    fn default() -> Self {
        Self {
            minimum: 0.0,
            maximum: f64::INFINITY,
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct DerivativeScaling {
    relative: Option<AxisScale>,
    typical: AxisScale,
    typical_explicit: bool,
    absolute: AbsoluteSteps,
}
impl Default for DerivativeScaling {
    fn default() -> Self {
        Self {
            relative: None,
            typical: AxisScale::Scalar(1.0),
            typical_explicit: false,
            absolute: Default::default(),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProbePoints {
    pub first: f64,
    pub second: Option<f64>,
}

/// Validate only class and geometry, before an active operand's provider access.
pub(crate) fn metadata(name: &str, value: &Value) -> BuiltinResult<()> {
    // An active numeric role owns no descendants of an aggregate, closure,
    // object, or foreign carrier. Reject it before recursive fact adaptation.
    if !matches!(
        value,
        Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_)
    ) {
        return Err(optim_error(
            name,
            "derivative scaling must be a nonempty real floating scalar or vector",
        ));
    }
    use runmat_types::*;
    let fact = crate::value_fact::value_fact(value);
    let vector = fact.shape.known_dims().is_none_or(|dims| {
        dims.iter()
            .filter(|dim| dim.is_some_and(|n| n != 1))
            .count()
            <= 1
    });
    if !matches!(
        fact.kind,
        ValueKindFact::Numeric(NumericFact {
            class: NumericClass::Single | NumericClass::Double,
            domain: NumericDomain::Real
        })
    ) || fact.storage == StorageFact::Sparse
        || fact.shape.element_count() == Some(0)
        || !vector
    {
        return Err(optim_error(
            name,
            "derivative scaling must be a nonempty real floating scalar or vector",
        ));
    }
    Ok(())
}

/// Host-only value validation, also used by the canonical option constructor.
pub(crate) fn validate(name: &str, value: &Value) -> BuiltinResult<()> {
    metadata(name, value)?;
    let valid = match value {
        Value::Num(value) => value.is_finite() && *value > 0.0,
        Value::Tensor(value) => (0..value.len()).all(|index| {
            let value = crate::builtins::common::tensor::tensor_value_f64(value, index);
            value.is_finite() && value > 0.0
        }),
        _ => {
            return Err(optim_error(
                name,
                "derivative scaling requires prepared host values",
            ))
        }
    };
    if !valid {
        return Err(optim_error(
            name,
            "derivative scaling entries must be finite and strictly positive",
        ));
    }
    Ok(())
}
fn decode(name: &str, value: &Value) -> BuiltinResult<AxisScale> {
    validate(name, value)?;
    match value {
        Value::Num(value) => Ok(AxisScale::Scalar(*value)),
        Value::Tensor(value) if value.len() == 1 => Ok(AxisScale::Scalar(
            crate::builtins::common::tensor::tensor_value_f64(value, 0),
        )),
        Value::Tensor(value) => {
            let mut values = runmat_value::exact_capacity_vec(value.len()).map_err(|source| {
                crate::build_runtime_error("derivative scaling allocation failed")
                    .with_builtin(name)
                    .with_source(source)
                    .build()
            })?;
            for index in 0..value.len() {
                values.push(crate::builtins::common::tensor::tensor_value_f64(
                    value, index,
                ));
            }
            Ok(AxisScale::Vector(values.into()))
        }
        _ => unreachable!("validated host scaling"),
    }
}
impl DerivativeScaling {
    pub(crate) fn with_absolute_steps(mut self, minimum: f64, maximum: f64) -> BuiltinResult<Self> {
        if !minimum.is_finite()
            || minimum < 0.0
            || maximum.is_nan()
            || maximum <= 0.0
            || maximum < minimum
        {
            return Err(optim_error("optim","DiffMinChange/ DiffMaxChange must define nonnegative minimum and positive maximum step limits"));
        }
        self.absolute = AbsoluteSteps { minimum, maximum };
        Ok(self)
    }
    pub(crate) fn typical_value(&self, index: usize) -> f64 {
        self.typical.get(index)
    }
    /// Supplied derivatives make these operands dormant: no validation, gather,
    /// allocation, or secondary compatibility requirement is acquired.
    pub(crate) fn from_fields(
        name: &str,
        fields: &ScalarOptionFields<'_>,
        active: bool,
    ) -> BuiltinResult<Self> {
        if !active {
            return Ok(Self::default());
        }
        if let Some(mode) = fields.exact(Field::FiniteDifferenceType) {
            if runmat_builtins::OptimSolver::parse(name)
                .is_some_and(|solver| !solver.accepts(Field::FiniteDifferenceType))
            {
                let mode =
                    super::option_string_value(Some(mode), "FiniteDifferenceType", "forward")?;
                if runmat_builtins::OptimDifferenceType::parse(&mode)
                    != Some(runmat_builtins::OptimDifferenceType::Forward)
                {
                    return Err(optim_error(
                        name,
                        "this numerical derivative owner implements forward differences only",
                    ));
                }
            }
        }
        Ok(Self {
            typical_explicit: fields.exact(Field::TypicalX).is_some(),
            absolute: Default::default(),
            relative: fields
                .exact(Field::FiniteDifferenceStepSize)
                .map(|v| decode(name, v))
                .transpose()?,
            typical: fields
                .exact(Field::TypicalX)
                .map(|v| decode(name, v))
                .transpose()?
                .unwrap_or(AxisScale::Scalar(1.0)),
        })
    }
    pub(crate) fn validate_variables(&self, name: &str, n: usize) -> BuiltinResult<()> {
        self.typical.validate_len(name, n)?;
        if self.typical_explicit && matches!(self.typical, AxisScale::Scalar(_)) && n != 1 {
            return Err(optim_error(
                name,
                "TypicalX must have one entry per original variable",
            ));
        }
        if let Some(relative) = &self.relative {
            relative.validate_len(name, n)?;
        }
        Ok(())
    }
    pub(crate) fn magnitude(
        &self,
        name: &str,
        x: f64,
        index: usize,
        central: bool,
    ) -> BuiltinResult<f64> {
        let factor = self
            .relative
            .as_ref()
            .map(|v| v.get(index))
            .unwrap_or_else(|| {
                if central {
                    f64::EPSILON.cbrt()
                } else {
                    f64::EPSILON.sqrt()
                }
            });
        if !x.is_finite() {
            return Err(optim_error(name, "finite-difference point is undefined"));
        }
        let h = (factor * x.abs().max(self.typical.get(index)))
            .max(self.absolute.minimum)
            .min(self.absolute.maximum);
        if !h.is_finite() || h < 0.0 {
            return Err(optim_error(
                name,
                "derivative step is not finite and nonnegative at this point",
            ));
        }
        // Positive admitted factors can underflow to zero. points() then
        // selects adjacent representable feasible values rather than inventing
        // a zero denominator or rejecting an available actual offset.
        Ok(h)
    }
    /// Forward uses sign'(x), with sign'(0)=1. Bounds may force the
    /// opposite direction. Central uses both feasible sides when available;
    /// actual offsets, not nominal h, own the interpolation denominators.
    pub(crate) fn points(
        &self,
        name: &str,
        x: f64,
        index: usize,
        lower: f64,
        upper: f64,
        central: bool,
    ) -> BuiltinResult<Option<ProbePoints>> {
        if lower == upper {
            return Ok(None);
        }
        let h = self.magnitude(name, x, index, central)?;
        let up = (x + h).min(upper);
        let up = if up > x && up.is_finite() {
            Some(up)
        } else {
            let next = x.next_up();
            (next <= upper && next.is_finite()).then_some(next)
        };
        let down = (x - h).max(lower);
        let down = if down < x && down.is_finite() {
            Some(down)
        } else {
            let next = x.next_down();
            (next >= lower && next.is_finite()).then_some(next)
        };
        let selected = match (central, up, down) {
            (true, Some(first), Some(second)) => Some(ProbePoints {
                first,
                second: Some(second),
            }),
            (false, _, Some(first)) if x < 0.0 => Some(ProbePoints {
                first,
                second: None,
            }),
            (_, Some(first), _) | (_, None, Some(first)) => Some(ProbePoints {
                first,
                second: None,
            }),
            (_, None, None) => None,
        };
        if let Some(points) = selected {
            let first_offset = points.first - x;
            if !first_offset.is_finite()
                || first_offset == 0.0
                || first_offset.abs() > self.absolute.maximum
            {
                return Err(optim_error(
                    name,
                    "actual finite-difference offset is unusable",
                ));
            }
            if let Some(second) = points.second {
                let second_offset = second - x;
                let span = points.first - second;
                if !second_offset.is_finite()
                    || second_offset.abs() > self.absolute.maximum
                    || second_offset == 0.0
                    || !span.is_finite()
                    || span <= 0.0
                {
                    return Err(optim_error(
                        name,
                        "actual central finite-difference offsets or span are unusable",
                    ));
                }
            }
        }
        Ok(selected)
    }
    pub(crate) fn unbounded(
        &self,
        name: &str,
        x: f64,
        index: usize,
        central: bool,
    ) -> BuiltinResult<ProbePoints> {
        let points = self
            .points(name, x, index, f64::NEG_INFINITY, f64::INFINITY, central)?
            .ok_or_else(|| {
                optim_error(name, "finite-difference coordinate is not representable")
            })?;
        if central && points.second.is_none() {
            return Err(optim_error(
                name,
                "central finite-difference coordinates are not representable",
            ));
        }
        Ok(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_resident_aggregate_needs_no_provider_and_unusable_span_never_probes() {
        use runmat_accelerate_api::{
            test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
        };
        let allocation = TestAllocationOwner::new()
            .allocate_zeroed(vec![1, 1], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        assert!(
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
                .is_none()
        );
        let child = Value::GpuTensor(allocation.handle.clone());
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("resident_child", child.clone());
        let structure = Value::Struct(fields.finish());
        let cell =
            Value::Cell(runmat_value::CellArray::new_with_shape(vec![child], vec![1, 1]).unwrap());
        for malformed in [structure, cell] {
            assert!(metadata("fminunc", &malformed).is_err());
            assert!(validate("fminunc", &malformed).is_err());
        }
        assert!(
            crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
                .is_none()
        );

        let tiny = DerivativeScaling {
            relative: Some(AxisScale::Scalar(1e-300)),
            typical: AxisScale::Scalar(1e-300),
            typical_explicit: true,
            absolute: Default::default(),
        };
        assert_eq!(tiny.magnitude("fminunc", 0.0, 0, true).unwrap(), 0.0);
        let adjacent = tiny.unbounded("fminunc", 0.0, 0, true).unwrap();
        assert_eq!(adjacent.first, 0.0f64.next_up());
        assert_eq!(adjacent.second, Some(0.0f64.next_down()));
        let identity = |x: f64| x;
        assert_eq!(
            (identity(adjacent.first) - identity(adjacent.second.unwrap()))
                / (adjacent.first - adjacent.second.unwrap()),
            1.0
        );

        let huge = DerivativeScaling {
            relative: Some(AxisScale::Scalar(f64::MAX)),
            typical: AxisScale::Scalar(1.0),
            typical_explicit: true,
            absolute: Default::default(),
        };
        assert!((f64::MAX - (-f64::MAX)).is_infinite());
        let mut calls = 0;
        // The linear objective would be misreported as zero gradient if the
        // finite endpoints were divided by their overflowing central span.
        let measured = huge
            .points("fmincon", 0.0, 0, -f64::MAX, f64::MAX, true)
            .map(|points| {
                let points = points.unwrap();
                let mut objective = |x: f64| {
                    calls += 1;
                    x
                };
                (objective(points.first) - objective(points.second.unwrap()))
                    / (points.first - points.second.unwrap())
            });
        assert!(measured.is_err());
        assert_eq!(calls, 0);
        assert!(huge.unbounded("fminunc", 0.0, 0, true).is_err());
    }

    #[test]
    fn scalar_vector_defaults_signed_zero_and_feasible_central_points() {
        let scaling = DerivativeScaling {
            relative: Some(AxisScale::Vector(Arc::from([0.1, 0.2]))),
            typical: AxisScale::Vector(Arc::from([2.0, 3.0])),
            typical_explicit: true,
            absolute: Default::default(),
        };
        scaling.validate_variables("fsolve", 2).unwrap();
        assert!(scaling.validate_variables("fsolve", 3).is_err());
        let mut fields = runmat_value::UnboundedStructValueBuilder::new();
        fields.insert("FinDiffRelStep", Value::Num(0.25));
        fields.insert("TypicalX", Value::Num(2.0));
        let fields = fields.finish();
        let selected = ScalarOptionFields::from_struct(Some(&fields));
        let explicit = DerivativeScaling::from_fields("fsolve", &selected, true).unwrap();
        assert!(explicit.validate_variables("fsolve", 2).is_err());
        assert_eq!(
            explicit.unbounded("fsolve", 0.0, 0, false).unwrap().first,
            0.5
        );
        let mut malformed = runmat_value::UnboundedStructValueBuilder::new();
        malformed.insert("FiniteDifferenceStepSize", Value::from("not numeric"));
        malformed.insert("TypicalX", Value::Num(-1.0));
        let malformed = malformed.finish();
        let fields = ScalarOptionFields::from_struct(Some(&malformed));
        assert!(DerivativeScaling::from_fields("fsolve", &fields, true).is_err());
        assert!(DerivativeScaling::from_fields("fsolve", &fields, false).is_ok());
        let negative = scaling.unbounded("fsolve", -1.0, 0, false).unwrap();
        assert!((negative.first + 1.2).abs() < 1e-14);
        assert!((scaling.unbounded("fsolve", 0.0, 1, false).unwrap().first - 0.6).abs() < 1e-14);
        let bounded = scaling
            .points("fmincon", 1.0, 0, 0.9, 1.05, true)
            .unwrap()
            .unwrap();
        assert_eq!(bounded.first, 1.05);
        assert_eq!(bounded.second, Some(0.9));
        assert!(scaling
            .points("fmincon", 1.0, 0, 1.0, 1.0, true)
            .unwrap()
            .is_none());
        let defaults = DerivativeScaling::default();
        assert_eq!(
            defaults.magnitude("fsolve", 0.0, 0, false).unwrap(),
            f64::EPSILON.sqrt()
        );
        assert_eq!(
            defaults.magnitude("fsolve", 0.0, 0, true).unwrap(),
            f64::EPSILON.cbrt()
        );
        let limited = defaults.clone().with_absolute_steps(1e-4, 1e-3).unwrap();
        assert_eq!(limited.magnitude("fseminf", 0.0, 0, false).unwrap(), 1e-4);
        let capped = scaling.clone().with_absolute_steps(0.0, 0.01).unwrap();
        assert_eq!(capped.magnitude("fseminf", 0.0, 0, false).unwrap(), 0.01);
        assert!(capped.unbounded("fseminf", f64::MAX, 0, false).is_err());
        assert!(defaults.clone().with_absolute_steps(0.1, 0.01).is_err());
        let point = defaults.unbounded("fsolve", f64::MAX, 0, false).unwrap();
        assert!(point.first < f64::MAX);
    }
}

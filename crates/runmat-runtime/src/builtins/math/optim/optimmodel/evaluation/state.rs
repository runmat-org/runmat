//! Numerical values versus genuinely unassigned expression slots. Missing
//! slots never publish fabricated zeros. Only real branch copies allocate.
use super::super::*;
use runmat_value::{NumericScalar, NumericStorage, Tensor};
pub(super) struct Partial {
    pub shape: SharedShape,
    pub slots: Vec<Option<f64>>,
}
/// A materialized numeric value or the still-unassigned slots that prevent
/// its publication. Neither representation boxes the actual value payload.
pub(super) enum State {
    Ready(RootedValue),
    Partial(Partial),
    Uniform {
        shape: SharedShape,
        value: Option<f64>,
    },
}
impl State {
    pub fn ready(value: RootedValue) -> Self {
        Self::Ready(value)
    }
    pub fn partial(shape: SharedShape, slots: Vec<Option<f64>>) -> Self {
        Self::Partial(Partial { shape, slots })
    }
    pub fn uniform(shape: SharedShape, value: Option<f64>) -> Self {
        Self::Uniform { shape, value }
    }
    pub fn ready_ref(&self) -> Option<&RootedValue> {
        match self {
            Self::Ready(value) => Some(value),
            _ => None,
        }
    }
    pub fn partial_mut(&mut self) -> Option<&mut Partial> {
        match self {
            Self::Partial(value) => Some(value),
            _ => None,
        }
    }
    pub fn is_partial(&self) -> bool {
        matches!(self, Self::Partial(_))
    }

    pub fn shape(&self) -> BuiltinResult<crate::builtins::common::shape::ValueDimensions<'_>> {
        match self {
            Self::Ready(value) => {
                crate::builtins::common::shape::value_dimensions_borrowed(value.value())
            }
            Self::Uniform { shape, .. } | Self::Partial(Partial { shape, .. }) => Ok(
                crate::builtins::common::shape::ValueDimensions::Stored(shape),
            ),
        }
    }
    pub fn count(&self, policy: Policy) -> BuiltinResult<usize> {
        match self {
            Self::Ready(value) => {
                let shape =
                    crate::builtins::common::shape::value_dimensions_borrowed(value.value())?;
                count(&shape, policy)
            }
            Self::Partial(Partial { slots, .. }) => Ok(slots.len()),
            Self::Uniform { shape, .. } => count(shape, policy),
        }
    }
    pub fn at(&self, index: usize, policy: Policy) -> BuiltinResult<Option<f64>> {
        match self {
            Self::Uniform { shape, value } => {
                if index >= count(shape, policy)? {
                    return Err(policy.invalid("expression coordinate exceeds its shape"));
                }
                Ok(*value)
            }
            Self::Partial(Partial { slots, .. }) => slots
                .get(index)
                .copied()
                .ok_or_else(|| policy.invalid("expression coordinate exceeds its shape")),
            Self::Ready(value) => Ok(Some(match value.value() {
                Value::Num(value) if index == 0 => *value,
                Value::Tensor(value) => {
                    let Some(NumericScalar::F64(value)) = value.numeric_value_at(index) else {
                        return Err(policy.invalid(
                            "expression numeric storage differs from its declared class/shape",
                        ));
                    };
                    value
                }
                Value::SparseTensor(value) => {
                    if index
                        >= value
                            .rows
                            .checked_mul(value.cols)
                            .ok_or_else(|| policy.invalid("sparse expression shape overflow"))?
                    {
                        return Err(policy.invalid("expression coordinate exceeds sparse shape"));
                    }
                    value
                        .get(index % value.rows, index / value.rows)
                        .ok_or_else(|| {
                            policy.invalid(
                                "sparse expression storage differs from its declared class",
                            )
                        })?
                }
                _ => {
                    return Err(policy.invalid("expression evaluated to a non-real-double carrier"))
                }
            })),
        }
    }
    pub async fn duplicate(
        &self,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Ok(match self {
            Self::Uniform { shape, value } => Self::uniform(shape.clone(), *value),
            Self::Ready(value) => Self::ready(value.clone()),
            Self::Partial(Partial { shape, slots }) => {
                let mut copy = runmat_value::exact_capacity_vec(slots.len())
                    .map_err(|source| policy.storage(source))?;
                for &slot in slots {
                    copy.push(slot);
                    cooperation.completed(1).await?;
                }
                Self::partial(shape.clone(), copy)
            }
        })
    }
    pub async fn into_ready(
        self,
        policy: Policy,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<RootedValue> {
        match self {
            Self::Ready(value) => Ok(value),
            Self::Uniform { shape, value } => {
                let count = count(&shape, policy)?;
                if count != 0 && value.is_none() {
                    return Err(policy.invalid("expression contains an unassigned coordinate"));
                }
                let mut values = runmat_value::exact_capacity_vec(count)
                    .map_err(|source| policy.storage(source))?;
                for _ in 0..count {
                    values.push(value.expect("nonempty uniform value checked"));
                    cooperation.completed(1).await?;
                }
                output(values, shape, policy, cooperation.runtime())
            }
            Self::Partial(Partial { shape, slots }) => {
                let mut values = runmat_value::exact_capacity_vec(slots.len())
                    .map_err(|source| policy.storage(source))?;
                for slot in slots {
                    values.push(slot.ok_or_else(|| {
                        policy.invalid("expression contains an unassigned coordinate")
                    })?);
                    cooperation.completed(1).await?;
                }
                output(values, shape, policy, cooperation.runtime())
            }
        }
    }
}
pub(in crate::builtins::math::optim::optimmodel) fn count(
    shape: &[usize],
    policy: Policy,
) -> BuiltinResult<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, d| count.checked_mul(*d))
        .ok_or_else(|| policy.invalid("expression extent overflows usize"))
}
pub(super) fn output(
    values: Vec<f64>,
    shape: SharedShape,
    policy: Policy,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<RootedValue> {
    let tensor = Tensor::from_numeric_storage(NumericStorage::F64(values), shape)
        .map_err(|source| policy.invalid(source))?;
    RootedValue::in_optional_context(Value::Tensor(tensor), runtime)
}

//! A public Best history uses the existing admitted numeric assignment/COW
//! owner. No parallel Vec mirror or unconditional history payload clone.
use super::{invalid, ScalarCooperation, POLICY};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_value::{IntValue, NumericDType, Tensor, Value};

pub(super) struct History {
    value: Option<RootedValue>,
    len: usize,
}
impl History {
    pub fn new(cooperation: &ScalarCooperation) -> BuiltinResult<Self> {
        let value = Value::Tensor(Tensor::new(Vec::new(), vec![1, 0]).map_err(invalid)?);
        Ok(Self {
            value: Some(RootedValue::in_optional_context(
                value,
                cooperation.runtime(),
            )?),
            len: 0,
        })
    }
    pub fn value(&self) -> Value {
        self.value
            .as_ref()
            .expect("history retained outside append")
            .value()
            .clone()
    }
    pub async fn adopt(
        &mut self,
        value: RootedValue,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        use super::common::constrained_numeric as numeric;
        let value = value
            .with_owned_async(|value| async {
                let value = numeric::prepare(
                    value,
                    "state.Best",
                    numeric::Geometry::Vector,
                    &super::numeric::INPUT_POLICY,
                    cooperation,
                )
                .await?
                .ok_or_else(|| invalid("state.Best requires a real numeric vector"))?;
                let value = match value {
                    Value::Tensor(value) if value.numeric_dtype() == NumericDType::F64 => {
                        let len = value.len();
                        Value::Tensor(value.reshape(vec![1, len]).map_err(invalid)?)
                    }
                    value => {
                        let numeric::Numeric::Real { values, .. } = numeric::materialize_raw(
                            value,
                            "state.Best",
                            &super::numeric::INPUT_POLICY,
                            cooperation,
                        )
                        .await?
                        else {
                            return Err(invalid("state.Best requires a real numeric vector"));
                        };
                        let len = values.len();
                        Value::Tensor(Tensor::new(values, vec![1, len]).map_err(invalid)?)
                    }
                };
                RootedValue::in_optional_context(value, cooperation.runtime())
            })
            .await?;
        let Value::Tensor(tensor) = value.value() else {
            unreachable!("normalized history");
        };
        self.len = tensor.len();
        self.value = Some(value);
        Ok(())
    }
    /// The controller retires its previous state's own Best edge before
    /// appending. A history retained by user code remains an ordinary COW
    /// predecessor; it is never modified or presumed uniquely owned.
    pub async fn append(
        &mut self,
        best: f64,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        use crate::object::indexing::{
            ObjectIndexComponent, ObjectIndexSelector, ObjectSubscript, ObjectSubscriptPath,
        };
        cooperation.completed(0).await?;
        let runtime = cooperation.runtime().cloned().ok_or_else(|| {
            let mut error = crate::context::RuntimeCapabilityError::new(
                crate::context::RuntimeCapability::Builtin,
                "genetic history requires its selected execution context",
            )
            .into_runtime_error();
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })?;
        let next = self
            .len
            .checked_add(1)
            .ok_or_else(|| invalid("genetic history extent overflow"))?;
        let index = u64::try_from(next).map_err(|_| invalid("genetic history index overflow"))?;
        let components =
            crate::context::host_allocation::initialize_host_array(Some(&runtime), 1, |_| {
                Ok(ObjectIndexComponent::Value(Value::Int(IntValue::U64(
                    index,
                ))))
            })?;
        let path = ObjectSubscriptPath::single_in_context(
            Some(&runtime),
            ObjectSubscript::parentheses(ObjectIndexSelector::IndexValues { components }),
        )?;
        let mut values = POLICY.capacity(1, Some(&runtime))?;
        values.push(Value::Num(best));
        let previous = self.value.take().expect("history owned outside append");
        let returned = previous
            .with_owned_async(|value| async {
                crate::object::protocol::assign_subscript_path_in_context(
                    &runtime,
                    value,
                    path,
                    values,
                    Some("ga"),
                )
                .await
            })
            .await?;
        let valid = matches!(returned.value(),Value::Tensor(value) if value.numeric_dtype()==NumericDType::F64 && value.shape()==[1,next]);
        if !valid {
            return Err(invalid(
                "numeric history assignment returned invalid shape or storage",
            ));
        }
        self.value = Some(returned);
        self.len = next;
        cooperation.completed(1).await?;
        Ok(())
    }
}

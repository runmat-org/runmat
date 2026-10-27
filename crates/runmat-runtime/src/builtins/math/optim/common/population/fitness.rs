//! Actual measurements are distinct from finite comparison scores.
use super::{Policy, ScalarCooperation};
use crate::BuiltinResult;
use runmat_value::{ComplexTensor, NumericDType, Tensor, Value};
#[derive(Clone, Copy)]
pub(crate) enum Measurement {
    Real(f64),
    Complex(f64, f64),
}
impl Measurement {
    pub fn score(self) -> Option<f64> {
        match self {
            Self::Real(v) if v.is_finite() => Some(v),
            _ => None,
        }
    }
}
pub(crate) struct Fitness {
    policy: Policy,
    value: Value,
    len: usize,
    real_rows: Option<Vec<bool>>,
}
impl Fitness {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn value(&self) -> Value {
        self.value.clone()
    }
    pub async fn mean(&self, cooperation: &mut ScalarCooperation) -> BuiltinResult<Value> {
        let mut real = 0.0;
        let mut imaginary = 0.0;
        let mut complex = false;
        for index in 0..self.len {
            match self.measurement(index) {
                Measurement::Real(v) => real += v / self.len as f64,
                Measurement::Complex(a, b) => {
                    real += a / self.len as f64;
                    imaginary += b / self.len as f64;
                    complex = true;
                }
            }
            cooperation.completed(1).await?;
        }
        Ok(if complex {
            Value::Complex(real, imaginary)
        } else {
            Value::Num(real)
        })
    }
    pub fn measurement(&self, index: usize) -> Measurement {
        match &self.value {
            Value::Tensor(v) => Measurement::Real(v.as_f64_slice().expect("double fitness")[index]),
            Value::ComplexTensor(v) => {
                let (re, im) = v.numeric_value_at(index).expect("fitness extent");
                if self.real_rows.as_ref().is_some_and(|rows| rows[index]) {
                    Measurement::Real(re.materialize_f64())
                } else {
                    Measurement::Complex(re.materialize_f64(), im.materialize_f64())
                }
            }
            _ => unreachable!("admitted fitness vector"),
        }
    }
    pub async fn scores(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<Option<f64>>> {
        let mut values = self.policy.capacity(self.len, cooperation.runtime())?;
        for i in 0..self.len {
            values.push(self.measurement(i).score());
            cooperation.completed(1).await?;
        }
        Ok(values)
    }
    pub async fn from_measurements(
        policy: Policy,
        values: Vec<Measurement>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let len = values.len();
        let mut complex = false;
        for value in &values {
            complex |= matches!(value, Measurement::Complex(..));
            cooperation.completed(1).await?;
        }
        let mut real_rows = if complex {
            Some(policy.capacity(len, cooperation.runtime())?)
        } else {
            None
        };
        let value = if complex {
            let mut data = policy.capacity(len, cooperation.runtime())?;
            for value in values {
                real_rows
                    .as_mut()
                    .expect("complex mask")
                    .push(matches!(value, Measurement::Real(_)));
                data.push(match value {
                    Measurement::Real(v) => (v, 0.0),
                    Measurement::Complex(a, b) => (a, b),
                });
                cooperation.completed(1).await?;
            }
            Value::ComplexTensor(
                ComplexTensor::new(data, vec![len, 1]).map_err(|error| policy.invalid(error))?,
            )
        } else {
            let mut data = policy.capacity(len, cooperation.runtime())?;
            for value in values {
                let Measurement::Real(value) = value else {
                    unreachable!("real vector")
                };
                data.push(value);
                cooperation.completed(1).await?;
            }
            Value::Tensor(Tensor::new(data, vec![len, 1]).map_err(|error| policy.invalid(error))?)
        };
        Ok(Self {
            policy,
            value,
            len,
            real_rows,
        })
    }
    pub async fn decode(
        policy: Policy,
        mut value: Value,
        len: usize,
        check: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if matches!(value, Value::GpuTensor(_)) {
            crate::builtins::math::optim::common::preflight_host_residency(
                policy.identity.name,
                &value,
                None,
                cooperation.runtime(),
            )?;
            let fact = crate::value_fact::value_fact(&value);
            if !matches!(fact.kind,runmat_types::ValueKindFact::Numeric(n) if n.class==runmat_types::NumericClass::Double)
                || fact.shape.element_count().is_some_and(|n| n != len)
            {
                return Err(policy.invalid(
                    "objective result must be a double vector matching the evaluated rows",
                ));
            }
            value =
                crate::dispatcher::gather_if_needed_in_context_async(&value, cooperation.runtime())
                    .await?;
        }
        let value=match value {
            Value::Num(v) if len==1=>Value::Tensor(Tensor::new(vec![v],vec![1,1]).map_err(|error|policy.invalid(error))?),
            Value::Complex(a,b) if len==1=>Value::ComplexTensor(ComplexTensor::new(vec![(a,b)],vec![1,1]).map_err(|error|policy.invalid(error))?),
            Value::Tensor(v) if v.numeric_dtype()==NumericDType::F64&&v.len()==len&&vector(v.shape())=>
                Value::Tensor(v.reshape(vec![len,1]).map_err(|error|policy.invalid(error))?),
            Value::ComplexTensor(mut v) if v.numeric_dtype()==NumericDType::F64&&v.len()==len&&vector(v.shape())=>{
                v.reshape_in_place(vec![len,1]).map_err(|error|policy.invalid(error))?;Value::ComplexTensor(v)
            }
            _=>return Err(policy.invalid("objective must return a full double scalar/vector with one value per evaluated row")),
        };
        let real_rows = None;
        let result = Self {
            policy,
            value,
            len,
            real_rows,
        };
        for i in 0..len {
            if check && result.measurement(i).score().is_none() {
                return Err(
                    policy.invalid("FunValCheck rejected a complex, Inf or NaN objective value")
                );
            }
            cooperation.completed(1).await?;
        }
        Ok(result)
    }
}
fn vector(shape: &[usize]) -> bool {
    shape.len() <= 2 && shape.iter().filter(|&&n| n != 1).count() <= 1
}

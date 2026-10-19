mod classify;
mod empty;

#[cfg(not(test))]
use classify::ClassifiedValue;
#[cfg(test)]
pub(in crate::builtins::math::elementwise::bsxfun) use classify::{
    value as classify_value, ClassifiedValue, ComplexClassedValue,
};
use runmat_value::{
    ComplexTensor, IntegerComplexStorage, LogicalArray, NumericDType, NumericScalar,
    NumericStorage, Tensor, Value,
};

#[derive(Default)]
pub(super) enum UniformCollector {
    #[default]
    Pending,
    Numeric {
        class: runmat_value::NumericDType,
        values: Vec<runmat_value::NumericScalar>,
    },
    Logical(Vec<u8>),
    Complex {
        class: NumericDType,
        values: Vec<(NumericScalar, NumericScalar)>,
    },
    Character(Vec<char>),
}

impl UniformCollector {
    pub(super) fn push(&mut self, value: Value) -> crate::BuiltinResult<()> {
        let classified = classify::value(&value)?;
        match self {
            Self::Pending => {
                *self = match classified {
                    ClassifiedValue::Logical(value) => Self::Logical(vec![u8::from(value)]),
                    ClassifiedValue::Numeric(value) => Self::Numeric {
                        class: value.class,
                        values: vec![value.value],
                    },
                    ClassifiedValue::Complex(value) => Self::Complex {
                        class: value.class,
                        values: vec![(value.real, value.imaginary)],
                    },
                    ClassifiedValue::Character(value) => Self::Character(vec![value]),
                };
                Ok(())
            }
            Self::Numeric { class, values } => match classified {
                ClassifiedValue::Numeric(value) if value.class == *class => {
                    values.push(value.value);
                    Ok(())
                }
                other => Err(classify::inconsistent(class.class_name(), &other)),
            },
            Self::Logical(values) => match classified {
                ClassifiedValue::Logical(value) => {
                    values.push(u8::from(value));
                    Ok(())
                }
                other => Err(classify::inconsistent("logical", &other)),
            },
            Self::Complex { class, values } => match classified {
                ClassifiedValue::Complex(value) if value.class == *class => {
                    values.push((value.real, value.imaginary));
                    Ok(())
                }
                other => Err(classify::inconsistent(class.class_name(), &other)),
            },
            Self::Character(values) => match classified {
                ClassifiedValue::Character(value) => {
                    values.push(value);
                    Ok(())
                }
                other => Err(classify::inconsistent("char", &other)),
            },
        }
    }

    pub(super) fn finish(
        self,
        shape: &[usize],
        contract: super::callback::OutputContract,
    ) -> crate::BuiltinResult<Value> {
        match self {
            Self::Pending => empty::value(shape, contract),
            Self::Numeric { class, values } => numeric(class, values, shape),
            Self::Logical(values) => LogicalArray::new(values, shape.to_vec())
                .map(Value::LogicalArray)
                .map_err(super::error::internal),
            Self::Complex { class, values } => complex(class, values, shape),
            Self::Character(values) => empty::characters(values, shape),
        }
    }
}

fn complex(
    class: NumericDType,
    values: Vec<(NumericScalar, NumericScalar)>,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    let tensor = match class {
        NumericDType::F64 => ComplexTensor::new(
            values
                .into_iter()
                .map(|(real, imaginary)| match (real, imaginary) {
                    (NumericScalar::F64(real), NumericScalar::F64(imaginary)) => (real, imaginary),
                    _ => unreachable!("complex collector class is validated while collecting"),
                })
                .collect(),
            shape.to_vec(),
        ),
        NumericDType::F32 => ComplexTensor::from_f32(
            values
                .into_iter()
                .map(|(real, imaginary)| match (real, imaginary) {
                    (NumericScalar::F32(real), NumericScalar::F32(imaginary)) => (real, imaginary),
                    _ => unreachable!("complex collector class is validated while collecting"),
                })
                .collect(),
            shape.to_vec(),
        ),
        _ => {
            let mut real = NumericStorage::zeros(class, values.len());
            let mut imaginary = NumericStorage::zeros(class, values.len());
            for (index, (real_value, imaginary_value)) in values.into_iter().enumerate() {
                real.set_value(index, real_value)
                    .map_err(super::error::internal)?;
                imaginary
                    .set_value(index, imaginary_value)
                    .map_err(super::error::internal)?;
            }
            let real = real
                .into_integer_storage()
                .map_err(|_| super::error::internal("complex output class must be integer"))?;
            let imaginary = imaginary
                .into_integer_storage()
                .map_err(|_| super::error::internal("complex output class must be integer"))?;
            IntegerComplexStorage::new(real, imaginary)
                .and_then(|storage| ComplexTensor::new_integer(storage, shape.to_vec()))
        }
    }
    .map_err(super::error::internal)?;
    Ok(Value::ComplexTensor(tensor))
}

fn numeric(
    class: runmat_value::NumericDType,
    values: Vec<runmat_value::NumericScalar>,
    shape: &[usize],
) -> crate::BuiltinResult<Value> {
    let mut storage = NumericStorage::zeros(class, values.len());
    for (index, value) in values.into_iter().enumerate() {
        storage
            .set_value(index, value)
            .map_err(super::error::internal)?;
    }
    Tensor::from_numeric_storage(storage, shape.to_vec())
        .map(Value::Tensor)
        .map_err(super::error::internal)
}

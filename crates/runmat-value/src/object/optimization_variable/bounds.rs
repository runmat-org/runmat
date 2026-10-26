use crate::{NumericDType, Tensor, Value};

/// One original canonical Tensor carrier, never a separate payload mirror.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationBoundArray(Value);
impl OptimizationBoundArray {
    pub fn try_from_tensor(tensor: Tensor) -> Result<Self, Tensor> {
        if tensor.numeric_dtype() != NumericDType::F64 || tensor.as_f64_slice().is_none() {
            return Err(tensor);
        }
        Ok(Self(Value::Tensor(tensor)))
    }
    pub fn value(&self) -> &Value {
        &self.0
    }
    fn tensor(&self) -> &Tensor {
        let Value::Tensor(tensor) = &self.0 else {
            unreachable!("validated bound carrier")
        };
        tensor
    }
    pub fn numeric_dtype(&self) -> NumericDType {
        self.tensor().numeric_dtype()
    }
    pub fn len(&self) -> usize {
        self.tensor().len()
    }
    pub fn is_empty(&self) -> bool {
        self.tensor().is_empty()
    }
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        self.tensor().as_f64_slice()
    }
}

/// Bounds use double storage, independently of a solver's working precision.
/// A uniform bound needs no element-sized allocation until a caller requests
/// its array presentation or changes individual elements.
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizationBoundStorage {
    Uniform(f64),
    PerElement(OptimizationBoundArray),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizationBoundRole {
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizationBoundView<'a> {
    Uniform(f64),
    PerElement(&'a [f64]),
}

impl OptimizationBoundStorage {
    pub(super) fn numeric_work_len(&self) -> usize {
        match self {
            Self::Uniform(_) => 0,
            Self::PerElement(values) => values.len(),
        }
    }

    pub(super) fn value(&self, index: usize) -> Option<f64> {
        match self {
            Self::Uniform(value) => Some(*value),
            Self::PerElement(values) => values.as_f64_slice()?.get(index).copied(),
        }
    }

    pub(super) fn view(&self) -> OptimizationBoundView<'_> {
        match self {
            Self::Uniform(value) => OptimizationBoundView::Uniform(*value),
            Self::PerElement(values) => OptimizationBoundView::PerElement(
                values.as_f64_slice().expect("validated double bounds"),
            ),
        }
    }
}

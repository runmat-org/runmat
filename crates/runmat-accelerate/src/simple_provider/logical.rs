//! Native floating views for logical predicates and comparisons.
use super::*;
use runmat_builtins::RelationalOperator;

#[derive(Clone, Copy)]
pub(super) enum BinaryPredicate {
    And,
    Or,
    Xor,
    Compare(RelationalOperator),
}

#[derive(Clone, Copy)]
pub(super) enum UnaryPredicate {
    Not,
    Finite,
    Nan,
    Infinite,
}

#[derive(Clone, Copy)]
enum FloatingView<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

impl<'a> FloatingView<'a> {
    fn admitted(data: &'a InProcessNumericData, handle: &GpuTensorHandle) -> Result<Self> {
        if handle.physical().storage() != GpuTensorStorage::Real {
            return Err(runmat_accelerate_api::unsupported_provider_operation(
                "logical predicate requires real storage",
            ));
        }
        match (data, handle.physical().element_type()) {
            (InProcessNumericData::F32(values), NumericElementType::F32) => Ok(Self::F32(values)),
            (InProcessNumericData::F64(values), NumericElementType::F64) => Ok(Self::F64(values)),
            _ => Err(runmat_accelerate_api::unsupported_provider_operation(
                "logical predicate requires native floating storage",
            )),
        }
    }

    fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    fn nonzero(self, index: usize) -> bool {
        match self {
            Self::F32(values) => values[index] != 0.0,
            Self::F64(values) => values[index] != 0.0,
        }
    }

    fn unary(self, index: usize, predicate: UnaryPredicate) -> bool {
        match self {
            Self::F32(values) => evaluate_unary(values[index], predicate),
            Self::F64(values) => evaluate_unary(values[index], predicate),
        }
    }

    fn compare(
        self,
        index: usize,
        other: Self,
        other_index: usize,
        predicate: RelationalOperator,
    ) -> bool {
        let test = |a: f64, b: f64| match predicate {
            RelationalOperator::Equal => a == b,
            RelationalOperator::NotEqual => a != b,
            RelationalOperator::LessThan => a < b,
            RelationalOperator::LessThanOrEqual => a <= b,
            RelationalOperator::GreaterThan => a > b,
            RelationalOperator::GreaterThanOrEqual => a >= b,
        };
        match (self, other) {
            (Self::F32(a), Self::F32(b)) => test(f64::from(a[index]), f64::from(b[other_index])),
            (Self::F64(a), Self::F64(b)) => test(a[index], b[other_index]),
            (Self::F32(a), Self::F64(b)) => test(f64::from(a[index]), b[other_index]),
            (Self::F64(a), Self::F32(b)) => test(a[index], f64::from(b[other_index])),
        }
    }
}

trait FloatPredicate: Copy {
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_zero(self) -> bool;
}

impl FloatPredicate for f32 {
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
}

impl FloatPredicate for f64 {
    fn is_finite(self) -> bool {
        self.is_finite()
    }
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.is_infinite()
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
}

fn evaluate_unary<T: FloatPredicate>(value: T, predicate: UnaryPredicate) -> bool {
    match predicate {
        UnaryPredicate::Not => value.is_zero(),
        UnaryPredicate::Finite => value.is_finite(),
        UnaryPredicate::Nan => value.is_nan(),
        UnaryPredicate::Infinite => value.is_infinite(),
    }
}

fn checked_len(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |n, &extent| n.checked_mul(extent))
        .ok_or_else(|| anyhow!("logical predicate element count overflow"))
}

impl InProcessProvider {
    pub(super) fn binary_predicate_impl(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        predicate: BinaryPredicate,
    ) -> Result<GpuTensorHandle> {
        let (output, shape) = self.prepare_binary_predicate(a, b, predicate)?;
        self.allocate_logical_tensor(output, shape)
    }

    pub(super) fn binary_predicate_execution(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        predicate: BinaryPredicate,
    ) -> runmat_accelerate_api::ProviderExecutionResult<GpuTensorHandle> {
        // Class/shape/owner admission cannot publish provider output. Preserve
        // its typed unsupported disposition so exact host fallback is legal.
        let (output, shape) = self
            .prepare_binary_predicate(a, b, predicate)
            .map_err(runmat_accelerate_api::ProviderExecutionError::from)?;
        self.allocate_logical_tensor(output, shape)
            .map_err(|error| {
                runmat_accelerate_api::ProviderExecutionError::from(error).after_effect()
            })
    }

    fn prepare_binary_predicate(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        predicate: BinaryPredicate,
    ) -> Result<(Vec<f64>, Vec<usize>)> {
        let guard = numeric_registry()
            .lock_for(self.device_id)
            .map_err(|_| anyhow!("logical predicate owner is poisoned"))?;
        // Authenticate both allocations before class admission or output allocation.
        let a_data = guard
            .get(a)
            .ok_or_else(|| anyhow!("logical left operand is not live and owned"))?;
        let b_data = guard
            .get(b)
            .ok_or_else(|| anyhow!("logical right operand is not live and owned"))?;
        let a_data = FloatingView::admitted(a_data, a)?;
        let b_data = FloatingView::admitted(b_data, b)?;
        ensure!(
            a_data.len() == checked_len(a.shape())? && b_data.len() == checked_len(b.shape())?,
            "logical predicate payload length mismatch"
        );
        let shape = runtime_broadcast_shapes("logical predicate", a.shape(), b.shape())
            .map_err(|error| anyhow!(error))?;
        let count = checked_len(&shape)?;
        let a_strides = runtime_compute_strides(a.shape());
        let b_strides = runtime_compute_strides(b.shape());
        let mut output = Vec::new();
        output.try_reserve_exact(count)?;
        for index in 0..count {
            let left = runtime_broadcast_index(index, &shape, a.shape(), &a_strides);
            let right = runtime_broadcast_index(index, &shape, b.shape(), &b_strides);
            let truth = match predicate {
                BinaryPredicate::And => a_data.nonzero(left) && b_data.nonzero(right),
                BinaryPredicate::Or => a_data.nonzero(left) || b_data.nonzero(right),
                BinaryPredicate::Xor => a_data.nonzero(left) ^ b_data.nonzero(right),
                BinaryPredicate::Compare(operator) => a_data.compare(left, b_data, right, operator),
            };
            output.push(if truth { 1.0 } else { 0.0 });
        }
        Ok((output, shape))
    }

    pub(super) fn unary_predicate_impl(
        &self,
        input: &GpuTensorHandle,
        predicate: UnaryPredicate,
    ) -> Result<GpuTensorHandle> {
        let output = {
            let guard = numeric_registry()
                .lock_for(self.device_id)
                .map_err(|_| anyhow!("logical predicate owner is poisoned"))?;
            let data = guard
                .get(input)
                .ok_or_else(|| anyhow!("logical operand is not live and owned"))?;
            let data = FloatingView::admitted(data, input)?;
            ensure!(
                data.len() == checked_len(input.shape())?,
                "logical predicate payload length mismatch"
            );
            let mut output = Vec::new();
            output.try_reserve_exact(data.len())?;
            for index in 0..data.len() {
                output.push(if data.unary(index, predicate) {
                    1.0
                } else {
                    0.0
                });
            }
            output
        };
        self.allocate_logical_tensor(output, input.shape().to_vec())
    }
}

#[cfg(test)]
mod tests;

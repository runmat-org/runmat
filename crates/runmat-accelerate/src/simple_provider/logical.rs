//! Native floating views for logical predicates and comparisons.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum BinaryPredicate {
    And,
    Or,
    Xor,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
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
        predicate: BinaryPredicate,
    ) -> bool {
        let test = |a: f64, b: f64| match predicate {
            BinaryPredicate::Equal => a == b,
            BinaryPredicate::NotEqual => a != b,
            BinaryPredicate::LessThan => a < b,
            BinaryPredicate::LessThanOrEqual => a <= b,
            BinaryPredicate::GreaterThan => a > b,
            BinaryPredicate::GreaterThanOrEqual => a >= b,
            _ => unreachable!("a numerical comparison predicate is required"),
        };
        match (self, other) {
            (Self::F32(a), Self::F32(b)) => test(f64::from(a[index]), f64::from(b[other_index])),
            (Self::F64(a), Self::F64(b)) => test(a[index], b[other_index]),
            (Self::F32(a), Self::F64(b)) => test(f64::from(a[index]), b[other_index]),
            (Self::F64(a), Self::F32(b)) => test(a[index], f64::from(b[other_index])),
        }
    }
}

trait FloatPredicate {
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
        let (output, shape) = {
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
                    BinaryPredicate::Equal
                    | BinaryPredicate::NotEqual
                    | BinaryPredicate::LessThan
                    | BinaryPredicate::LessThanOrEqual
                    | BinaryPredicate::GreaterThan
                    | BinaryPredicate::GreaterThanOrEqual => {
                        a_data.compare(left, b_data, right, predicate)
                    }
                };
                output.push(if truth { 1.0 } else { 0.0 });
            }
            (output, shape)
        };
        self.allocate_logical_tensor(output, shape)
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
mod tests {
    use super::*;
    use futures::executor::block_on;

    #[test]
    fn native_single_and_double_predicates_preserve_logical_outputs() {
        let provider = InProcessProvider::new();
        for input in [
            HostNumericDataView::F32(&[f32::NAN, 0.0, f32::INFINITY, -1.0]),
            HostNumericDataView::F64(&[f64::NAN, 0.0, f64::INFINITY, -1.0]),
        ] {
            let data = provider
                .upload_numeric(&HostNumericTensorView {
                    data: input,
                    shape: &[2, 2],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            let zero = provider.zeros_like(&data, &[2, 2]).unwrap();
            let finite = provider.logical_isfinite(&data).unwrap();
            let nonzero = block_on(provider.elem_ne(&data, &zero)).unwrap();
            let selected = provider.logical_and(&finite, &nonzero).unwrap();
            assert_eq!(
                block_on(provider.download(&selected)).unwrap().data,
                [0.0, 0.0, 0.0, 1.0]
            );
            assert_eq!(
                block_on(provider.download(&provider.logical_or(&finite, &nonzero).unwrap()))
                    .unwrap()
                    .data,
                [1.0, 1.0, 1.0, 1.0]
            );
            assert_eq!(
                block_on(provider.download(&provider.logical_xor(&finite, &nonzero).unwrap()))
                    .unwrap()
                    .data,
                [1.0, 1.0, 1.0, 0.0]
            );
            assert_eq!(
                block_on(provider.download(&provider.logical_not(&data).unwrap()))
                    .unwrap()
                    .data,
                [0.0, 1.0, 0.0, 0.0]
            );
            assert_eq!(
                block_on(provider.download(&provider.logical_isnan(&data).unwrap()))
                    .unwrap()
                    .data,
                [1.0, 0.0, 0.0, 0.0]
            );
            assert_eq!(
                block_on(provider.download(&provider.logical_isinf(&data).unwrap()))
                    .unwrap()
                    .data,
                [0.0, 0.0, 1.0, 0.0]
            );
            assert_eq!(
                block_on(provider.download(&block_on(provider.elem_eq(&data, &zero)).unwrap()))
                    .unwrap()
                    .data,
                [0.0, 1.0, 0.0, 0.0]
            );
            assert!(selected.view().is_logical());
            assert!(!selected.shares_allocation_with(&data));
        }
    }

    #[test]
    fn native_relational_predicates_respect_nan_infinity_signed_zero_and_broadcast() {
        let provider = InProcessProvider::new();
        for (left, right) in [
            (
                HostNumericDataView::F32(&[f32::NAN, f32::INFINITY, -0.0, -2.0]),
                HostNumericDataView::F32(&[0.0, 0.0, 0.0, -1.0]),
            ),
            (
                HostNumericDataView::F64(&[f64::NAN, f64::INFINITY, -0.0, -2.0]),
                HostNumericDataView::F64(&[0.0, 0.0, 0.0, -1.0]),
            ),
        ] {
            let left = provider
                .upload_numeric(&HostNumericTensorView {
                    data: left,
                    shape: &[2, 2],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            let right = provider
                .upload_numeric(&HostNumericTensorView {
                    data: right,
                    shape: &[2, 2],
                    storage: GpuTensorStorage::Real,
                })
                .unwrap();
            for (predicate, expected) in [
                (BinaryPredicate::LessThan, [0.0, 0.0, 0.0, 1.0]),
                (BinaryPredicate::LessThanOrEqual, [0.0, 0.0, 1.0, 1.0]),
                (BinaryPredicate::GreaterThan, [0.0, 1.0, 0.0, 0.0]),
                (BinaryPredicate::GreaterThanOrEqual, [0.0, 1.0, 1.0, 0.0]),
            ] {
                let output = provider
                    .binary_predicate_impl(&left, &right, predicate)
                    .unwrap();
                assert_eq!(block_on(provider.download(&output)).unwrap().data, expected);
                assert!(output.view().is_logical());
                assert!(!output.shares_allocation_with(&left));
            }
        }
        let left = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[-1.0, 2.0]),
                shape: &[2, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let right = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.0, 3.0, -2.0]),
                shape: &[1, 3],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let output = block_on(provider.elem_lt(&left, &right)).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(
            block_on(provider.download(&output)).unwrap().data,
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn integer_complex_and_foreign_inputs_do_not_enter_floating_predicates() {
        let provider = InProcessProvider::new();
        let integer = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::I64(&[i64::MAX]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &provider.logical_isfinite(&integer).unwrap_err()
        ));
        let complex = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&[1.0, 0.0]),
                shape: &[1, 1],
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .unwrap();
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &provider.logical_isfinite(&complex).unwrap_err()
        ));
        let foreign = InProcessProvider::new()
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .unwrap();
        assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
            &provider.logical_and(&integer, &foreign).unwrap_err()
        ));
    }
}

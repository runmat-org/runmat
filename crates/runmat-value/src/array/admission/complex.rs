use super::{synchronous_quantum, ShapeAdmission};
use crate::{ComplexStorage, ComplexTensor};
use std::num::NonZeroUsize;

/// Incremental owner for one dense complex tensor candidate.
pub struct ComplexTensorAdmission {
    storage: ComplexStorage,
    shape: ShapeAdmission,
    validated: bool,
}

/// Sealed, once-consumed complex tensor parts.
pub struct ValidatedComplexTensorParts {
    storage: ComplexStorage,
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
}

impl ComplexTensorAdmission {
    pub fn new(storage: ComplexStorage, shape: Vec<usize>) -> Result<Self, String> {
        if let ComplexStorage::Integer(integer) = &storage {
            if integer.real().numeric_dtype() != integer.imag().numeric_dtype()
                || integer.real().len() != integer.imag().len()
            {
                return Err(
                    "complex integer components must have matching class and length".into(),
                );
            }
        }
        Ok(Self {
            storage,
            shape: ShapeAdmission::new(shape),
            validated: false,
        })
    }

    /// Check at most `maximum_work` shape dimensions or the terminal length
    /// comparison. Returns the exact number of completed work items.
    pub fn advance(&mut self, maximum_work: NonZeroUsize) -> Result<usize, String> {
        if self.validated {
            return Ok(0);
        }
        let completed = self.shape.advance(maximum_work, |shape| {
            format!("complex tensor shape {shape:?} overflows usize")
        })?;
        if self.shape.is_complete() {
            let expected = self.shape.elements();
            if self.storage.len() != expected {
                return Err(format!(
                    "complex {} storage length {} doesn't match shape {:?} ({} elements)",
                    self.storage.numeric_dtype().class_name(),
                    self.storage.len(),
                    self.shape.shape,
                    expected
                ));
            }
            self.validated = true;
        }
        Ok(completed)
    }

    pub fn is_complete(&self) -> bool {
        self.validated
    }

    pub fn into_validated(self) -> Result<ValidatedComplexTensorParts, Self> {
        if !self.validated {
            return Err(self);
        }
        let (rows, cols) = self.shape.dimensions();
        Ok(ValidatedComplexTensorParts {
            storage: self.storage,
            shape: self.shape.shape,
            rows,
            cols,
        })
    }

    pub fn finish_synchronously(mut self) -> Result<ValidatedComplexTensorParts, String> {
        while !self.is_complete() {
            self.advance(synchronous_quantum())?;
        }
        self.into_validated()
            .map_err(|_| "complex tensor admission unexpectedly incomplete".to_string())
    }
}

impl ValidatedComplexTensorParts {
    /// Construct the admitted tensor without rescanning shape or storage.
    pub fn into_tensor(self) -> ComplexTensor {
        ComplexTensor::from_validated_parts(self)
    }

    pub(in crate::array) fn into_components(self) -> (ComplexStorage, Vec<usize>, usize, usize) {
        (self.storage, self.shape, self.rows, self.cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComplexElement, HostComplexBuffer, IntegerComplexStorage, IntegerStorage};

    fn finish(
        storage: ComplexStorage,
        shape: Vec<usize>,
        quantum: usize,
    ) -> Result<ComplexTensor, String> {
        let mut admission = ComplexTensorAdmission::new(storage, shape)?;
        let quantum = NonZeroUsize::new(quantum).expect("test quantum is nonzero");
        while !admission.is_complete() {
            admission.advance(quantum)?;
        }
        Ok(admission
            .into_validated()
            .map_err(|_| "incomplete admission".to_string())?
            .into_tensor())
    }

    #[test]
    fn stepped_and_synchronous_admission_preserve_all_complex_classes() -> Result<(), String> {
        let storages = vec![
            ComplexStorage::F64(vec![ComplexElement(1.0, -1.0)].into()),
            ComplexStorage::F32(vec![ComplexElement(1.0_f32, -1.0)].into()),
            ComplexStorage::Integer(IntegerComplexStorage::new(
                IntegerStorage::I8(vec![1]),
                IntegerStorage::I8(vec![-1]),
            )?),
            ComplexStorage::Integer(IntegerComplexStorage::new(
                IntegerStorage::I16(vec![1]),
                IntegerStorage::I16(vec![-1]),
            )?),
            ComplexStorage::Integer(IntegerComplexStorage::new(
                IntegerStorage::I32(vec![1]),
                IntegerStorage::I32(vec![-1]),
            )?),
            ComplexStorage::Integer(IntegerComplexStorage::new(
                IntegerStorage::I64(vec![1]),
                IntegerStorage::I64(vec![-1]),
            )?),
            ComplexStorage::Integer(IntegerComplexStorage::new(
                IntegerStorage::U8(vec![1]),
                IntegerStorage::U8(vec![2]),
            )?),
            ComplexStorage::Integer(IntegerComplexStorage::new(
                IntegerStorage::U16(vec![1]),
                IntegerStorage::U16(vec![2]),
            )?),
            ComplexStorage::Integer(IntegerComplexStorage::new(
                IntegerStorage::U32(vec![1]),
                IntegerStorage::U32(vec![2]),
            )?),
            ComplexStorage::Integer(IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1]),
                IntegerStorage::U64(vec![2]),
            )?),
        ];
        for storage in storages {
            let stepped = finish(storage.clone(), vec![1, 1], 1)?;
            let synchronous = ComplexTensorAdmission::new(storage, vec![1, 1])?
                .finish_synchronously()?
                .into_tensor();
            assert_eq!(stepped, synchronous);
        }
        Ok(())
    }

    #[test]
    fn integer_pair_constructor_precedes_shape_admission() {
        let error =
            IntegerComplexStorage::new(IntegerStorage::I8(vec![1]), IntegerStorage::U8(Vec::new()))
                .expect_err("invalid integer pair");
        assert_eq!(
            error,
            "complex integer components must have matching class and length"
        );
    }

    #[test]
    fn empty_scalar_overflow_and_length_diagnostics_match_sync_constructors() -> Result<(), String>
    {
        assert_eq!(
            finish(
                ComplexStorage::F64(Vec::<ComplexElement<f64>>::new().into()),
                vec![0, 7],
                1,
            )?
            .shape(),
            vec![0, 7]
        );
        assert_eq!(
            finish(
                ComplexStorage::F32(vec![ComplexElement(2.0, 3.0)].into()),
                Vec::new(),
                1,
            )?
            .shape(),
            Vec::<usize>::new()
        );
        let overflow = finish(
            ComplexStorage::F64(Vec::<ComplexElement<f64>>::new().into()),
            vec![usize::MAX, 2],
            1,
        )
        .expect_err("shape overflow");
        assert_eq!(
            overflow,
            format!(
                "complex tensor shape {:?} overflows usize",
                vec![usize::MAX, 2]
            )
        );
        let mismatch = finish(
            ComplexStorage::F64(vec![ComplexElement(1.0, 2.0)].into()),
            vec![2],
            1,
        )
        .expect_err("length mismatch");
        assert_eq!(
            mismatch,
            "complex double storage length 1 doesn't match shape [2] (2 elements)"
        );
        Ok(())
    }

    #[test]
    fn completed_parts_preserve_shared_floating_storage_owner() -> Result<(), String> {
        let storage = HostComplexBuffer::from(vec![ComplexElement(1.0_f32, 2.0)]);
        let alias = storage.clone();
        let tensor = ComplexTensorAdmission::new(ComplexStorage::F32(storage), vec![1])?
            .finish_synchronously()?
            .into_tensor();
        let ComplexStorage::F32(actual) = tensor.complex_storage() else {
            panic!("single complex storage");
        };
        assert!(actual.shares_allocation_with(&alias));
        Ok(())
    }
}

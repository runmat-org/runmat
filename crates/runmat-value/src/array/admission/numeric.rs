use super::{synchronous_quantum, ShapeAdmission};
use crate::{HostNumericBuffer, NumericStorage, Tensor};
use std::num::NonZeroUsize;

enum NumericCandidate {
    Native(NumericStorage),
    Host(HostNumericBuffer),
}

impl NumericCandidate {
    fn len(&self) -> usize {
        match self {
            Self::Native(storage) => storage.len(),
            Self::Host(storage) => storage.len(),
        }
    }

    fn class_name(&self) -> &'static str {
        match self {
            Self::Native(storage) => storage.class_name(),
            Self::Host(storage) => storage.numeric_dtype().class_name(),
        }
    }

    fn into_host(self) -> HostNumericBuffer {
        match self {
            Self::Native(storage) => HostNumericBuffer::from_numeric_storage(storage),
            Self::Host(storage) => storage,
        }
    }

    fn overflow_error(&self, shape: &[usize]) -> String {
        match self {
            Self::Native(_) => format!("numeric tensor shape {shape:?} overflows usize"),
            Self::Host(_) => format!(
                "{} tensor data length {} doesn't match shape {:?}",
                self.class_name(),
                self.len(),
                shape
            ),
        }
    }

    fn length_error(&self, shape: &[usize], expected: usize) -> String {
        match self {
            Self::Native(_) => format!(
                "{} storage length {} doesn't match shape {:?} ({} elements)",
                self.class_name(),
                self.len(),
                shape,
                expected
            ),
            Self::Host(_) => format!(
                "{} tensor data length {} doesn't match shape {:?}",
                self.class_name(),
                self.len(),
                shape
            ),
        }
    }
}

/// Incremental owner for one dense numeric tensor candidate.
pub struct NumericTensorAdmission {
    storage: NumericCandidate,
    shape: ShapeAdmission,
    validated: bool,
}

/// Sealed, once-consumed numeric tensor parts.
pub struct ValidatedNumericTensorParts {
    storage: HostNumericBuffer,
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
}

impl NumericTensorAdmission {
    pub fn from_numeric_storage(storage: NumericStorage, shape: Vec<usize>) -> Self {
        Self {
            storage: NumericCandidate::Native(storage),
            shape: ShapeAdmission::new(shape),
            validated: false,
        }
    }

    pub fn from_host_buffer(storage: HostNumericBuffer, shape: Vec<usize>) -> Self {
        Self {
            storage: NumericCandidate::Host(storage),
            shape: ShapeAdmission::new(shape),
            validated: false,
        }
    }

    /// Check at most `maximum_work` shape dimensions or the terminal length
    /// comparison. Returns the exact number of completed work items.
    pub fn advance(&mut self, maximum_work: NonZeroUsize) -> Result<usize, String> {
        if self.validated {
            return Ok(0);
        }
        let storage = &self.storage;
        let completed = self
            .shape
            .advance(maximum_work, |shape| storage.overflow_error(shape))?;
        if self.shape.is_complete() {
            let expected = self.shape.elements();
            if self.storage.len() != expected {
                return Err(self.storage.length_error(&self.shape.shape, expected));
            }
            self.validated = true;
        }
        Ok(completed)
    }

    pub fn is_complete(&self) -> bool {
        self.validated
    }

    pub fn into_validated(self) -> Result<ValidatedNumericTensorParts, Self> {
        if !self.validated {
            return Err(self);
        }
        let (rows, cols) = self.shape.dimensions();
        Ok(ValidatedNumericTensorParts {
            storage: self.storage.into_host(),
            shape: self.shape.shape,
            rows,
            cols,
        })
    }

    pub fn finish_synchronously(mut self) -> Result<ValidatedNumericTensorParts, String> {
        while !self.is_complete() {
            self.advance(synchronous_quantum())?;
        }
        self.into_validated()
            .map_err(|_| "numeric tensor admission unexpectedly incomplete".to_string())
    }
}

impl ValidatedNumericTensorParts {
    /// Construct the admitted tensor without rescanning shape or storage.
    pub fn into_tensor(self) -> Tensor {
        Tensor::from_validated_parts(self)
    }

    pub(in crate::array) fn into_components(self) -> (HostNumericBuffer, Vec<usize>, usize, usize) {
        (self.storage, self.shape, self.rows, self.cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finish(
        storage: NumericStorage,
        shape: Vec<usize>,
        quantum: usize,
    ) -> Result<Tensor, String> {
        let mut admission = NumericTensorAdmission::from_numeric_storage(storage, shape);
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
    fn stepped_and_synchronous_admission_preserve_all_numeric_classes() -> Result<(), String> {
        let storages = vec![
            NumericStorage::F64(vec![1.0]),
            NumericStorage::F32(vec![1.0]),
            NumericStorage::I8(vec![1]),
            NumericStorage::I16(vec![1]),
            NumericStorage::I32(vec![1]),
            NumericStorage::I64(vec![1]),
            NumericStorage::U8(vec![1]),
            NumericStorage::U16(vec![1]),
            NumericStorage::U32(vec![1]),
            NumericStorage::U64(vec![1]),
        ];
        for storage in storages {
            let stepped = finish(storage.clone(), vec![1, 1], 1)?;
            let synchronous = NumericTensorAdmission::from_numeric_storage(storage, vec![1, 1])
                .finish_synchronously()?
                .into_tensor();
            assert_eq!(stepped, synchronous);
        }
        Ok(())
    }

    #[test]
    fn empty_scalar_overflow_and_length_diagnostics_preserve_constructor_contract(
    ) -> Result<(), String> {
        assert_eq!(
            finish(NumericStorage::I16(Vec::new()), vec![0, 3], 1)?.shape(),
            vec![0, 3]
        );
        assert_eq!(
            finish(NumericStorage::U64(vec![9]), Vec::new(), 1)?.shape(),
            Vec::<usize>::new()
        );
        let overflow = finish(NumericStorage::F64(Vec::new()), vec![usize::MAX, 2], 1)
            .expect_err("shape overflow");
        assert_eq!(
            overflow,
            format!(
                "numeric tensor shape {:?} overflows usize",
                vec![usize::MAX, 2]
            )
        );
        assert_eq!(
            finish(NumericStorage::U8(vec![1]), vec![2], 1).expect_err("length mismatch"),
            "uint8 storage length 1 doesn't match shape [2] (2 elements)"
        );
        Ok(())
    }

    #[test]
    fn host_buffer_keeps_legacy_error_and_shared_owner() -> Result<(), String> {
        let storage = HostNumericBuffer::from_numeric_storage(NumericStorage::F32(vec![1.0]));
        let alias = storage.clone();
        let tensor = NumericTensorAdmission::from_host_buffer(storage, vec![1])
            .finish_synchronously()?
            .into_tensor();
        assert!(tensor.host_buffer().shares_allocation_with(&alias));

        let error = NumericTensorAdmission::from_host_buffer(alias, vec![usize::MAX, 2])
            .finish_synchronously()
            .err()
            .expect("legacy host overflow diagnostic");
        assert_eq!(
            error,
            format!(
                "single tensor data length 1 doesn't match shape {:?}",
                vec![usize::MAX, 2]
            )
        );
        Ok(())
    }

    #[test]
    fn native_storage_allocation_moves_into_tensor_without_repack() -> Result<(), String> {
        let values = vec![1.0_f64, 2.0];
        let original_pointer = values.as_ptr();
        let tensor =
            NumericTensorAdmission::from_numeric_storage(NumericStorage::F64(values), vec![1, 2])
                .finish_synchronously()?
                .into_tensor();
        assert_eq!(
            tensor.as_f64_slice().expect("double tensor").as_ptr(),
            original_pointer
        );
        Ok(())
    }
}

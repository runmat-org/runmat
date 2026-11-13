//! Closed role owner, without a variable-capacity publication ledger.
use super::*;
#[derive(Clone, Copy)]
pub(super) enum FactorRole {
    Eigenvalues,
    Diagonal,
    Right,
    Left,
}
pub(super) struct PublishedFactors {
    handles: [Option<GpuTensorHandle>; 4],
}
impl PublishedFactors {
    pub(super) fn new() -> Self {
        Self {
            handles: [None, None, None, None],
        }
    }
    pub(super) fn handles(&self) -> impl Iterator<Item = &GpuTensorHandle> {
        self.handles.iter().flatten()
    }
    pub(super) fn insert(&mut self, role: FactorRole, handle: GpuTensorHandle) {
        assert!(self.handles[role as usize].replace(handle).is_none());
    }
    pub(super) fn reject(self, error: crate::RuntimeError) -> crate::RuntimeError {
        gpu_helpers::reject_gpu_outputs(self.handles.into_iter().flatten(), error)
    }
    pub(super) fn retire(self) -> BuiltinResult<()> {
        let mut result = Ok(());
        for handle in self.handles.into_iter().flatten() {
            let retired = handle.release().map(|_| ()).map_err(|cause| {
                build_runtime_error("eig: partial factor retirement failed")
                    .with_builtin(BUILTIN_NAME)
                    .with_source(cause)
                    .build()
            });
            result = crate::runtime_error::complete_with_cleanup(result, retired);
        }
        result
    }
    pub(super) fn finish(self, vector_output: bool) -> EigEval {
        let [eigenvalues, diagonal, right, left] = self.handles;
        let eigenvalues = Value::GpuTensor(eigenvalues.expect("eigenvalue role published"));
        let diagonal_matrix = Value::GpuTensor(diagonal.expect("diagonal role published"));
        let diagonal_output = if vector_output {
            eigenvalues.clone()
        } else {
            diagonal_matrix.clone()
        };
        EigEval {
            eigenvalues,
            diagonal_matrix,
            diagonal_output,
            right: Value::GpuTensor(right.expect("right role published")),
            left: left.map(Value::GpuTensor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuTensorStorage, NumericElementType,
    };
    #[test]
    fn partial_bundle_retires_only_its_owned_factors() {
        let owner = TestAllocationOwner::new();
        let source = owner
            .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
            .unwrap();
        let first = owner
            .allocate_zeroed(
                vec![2, 1],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let second = owner
            .allocate_zeroed(
                vec![2, 2],
                NumericElementType::F64,
                GpuTensorStorage::ComplexInterleaved,
            )
            .unwrap();
        let mut outputs = PublishedFactors::new();
        outputs.insert(FactorRole::Eigenvalues, first.handle);
        outputs.insert(FactorRole::Diagonal, second.handle);
        assert_eq!(outputs.handles().count(), 2);
        outputs.retire().unwrap();
        assert_eq!(first.probe.retirements(), 1);
        assert_eq!(second.probe.retirements(), 1);
        assert_eq!(source.probe.retirements(), 0);
        drop(source.handle);
        assert_eq!(source.probe.retirements(), 1);
    }
}

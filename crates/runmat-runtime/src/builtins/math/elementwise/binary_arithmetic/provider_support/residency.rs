use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn resident_output_from_sources<
    'a,
>(
    output: GpuTensorHandle,
    sources: impl IntoIterator<Item = &'a GpuTensorHandle>,
) -> crate::BuiltinResult<Value> {
    let output = gpu_helpers::propagate_owned_output_provenance(output, sources)?;
    gpu_helpers::resident_gpu_value(output)
}

#[cfg(test)]
mod tests {
    use super::resident_output_from_sources;
    use runmat_accelerate_api::{
        test_support::TestAllocationOwner, GpuHandleProvenance, GpuTensorStorage,
        NumericElementType,
    };
    use runmat_value::Value;

    #[test]
    fn preserves_explicit_source_intent() {
        let owner = TestAllocationOwner::new();
        let allocate = || {
            owner
                .allocate_zeroed(vec![2, 2], NumericElementType::F64, GpuTensorStorage::Real)
                .unwrap()
        };
        let automatic = allocate()
            .handle
            .with_provenance(GpuHandleProvenance::Automatic)
            .unwrap();
        let explicit = allocate()
            .handle
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let output = allocate().handle;

        let Value::GpuTensor(output) =
            resident_output_from_sources(output, [&automatic, &explicit]).unwrap()
        else {
            panic!("resident output must remain a GPU tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&output),
            Some(GpuHandleProvenance::Explicit)
        );
    }
}

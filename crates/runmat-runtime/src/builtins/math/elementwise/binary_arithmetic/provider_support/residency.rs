use runmat_accelerate_api::GpuTensorHandle;
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;

pub(in crate::builtins::math::elementwise::binary_arithmetic) fn resident_output_from_sources<
    'a,
>(
    mut output: GpuTensorHandle,
    sources: impl IntoIterator<Item = &'a GpuTensorHandle>,
) -> Value {
    let provenance = if sources
        .into_iter()
        .any(runmat_accelerate_api::handle_is_explicit)
    {
        runmat_accelerate_api::GpuHandleProvenance::Explicit
    } else {
        runmat_accelerate_api::GpuHandleProvenance::Automatic
    };
    runmat_accelerate_api::set_handle_provenance(&mut output, provenance);
    gpu_helpers::resident_gpu_value(output)
}

#[cfg(test)]
mod tests {
    use super::resident_output_from_sources;
    use runmat_accelerate_api::{GpuHandleProvenance, GpuTensorHandle};
    use runmat_value::Value;

    #[test]
    fn preserves_explicit_source_intent() {
        let automatic =
            GpuTensorHandle::new(vec![2, 2], 1, 1).with_provenance(GpuHandleProvenance::Automatic);
        let explicit =
            GpuTensorHandle::new(vec![2, 2], 1, 2).with_provenance(GpuHandleProvenance::Explicit);
        let output = GpuTensorHandle::new(vec![2, 2], 1, 3);

        let Value::GpuTensor(output) =
            resident_output_from_sources(output, [&automatic, &explicit])
        else {
            panic!("resident output must remain a GPU tensor");
        };
        assert_eq!(
            runmat_accelerate_api::handle_provenance(&output),
            Some(GpuHandleProvenance::Explicit)
        );
    }
}

//! Extrema result descriptors are resolved by the numerical owner, not the
//! selected provider's preferred precision or an output's self-reported class.
use super::{gpu_helpers, ExtremaKind};
use crate::BuiltinResult;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, GpuTensorStorage, NumericElementType};

pub(super) fn pair_contract(
    left: &GpuTensorHandle,
    right: &GpuTensorHandle,
    shape: Vec<usize>,
) -> Option<gpu_helpers::BinaryGpuOutputContract> {
    let class = |input: &GpuTensorHandle| {
        if input.view().is_logical() {
            NumericElementType::F64
        } else {
            input.physical().element_type()
        }
    };
    let a = class(left);
    let b = class(right);
    let complex = left.physical().storage() != GpuTensorStorage::Real
        || right.physical().storage() != GpuTensorStorage::Real;
    let element = if a.integer_type().is_some() || b.integer_type().is_some() {
        if complex {
            return None;
        }
        if a == b {
            a
        } else if a.integer_type().is_some()
            && b == NumericElementType::F64
            && !right.view().is_logical()
            && right.shape().iter().all(|extent| *extent == 1)
        {
            a
        } else if b.integer_type().is_some()
            && a == NumericElementType::F64
            && !left.view().is_logical()
            && left.shape().iter().all(|extent| *extent == 1)
        {
            b
        } else {
            return None;
        }
    } else if a == NumericElementType::F32 && b == NumericElementType::F32 {
        NumericElementType::F32
    } else {
        NumericElementType::F64
    };
    Some(gpu_helpers::BinaryGpuOutputContract {
        shape,
        storage: if complex {
            GpuTensorStorage::ComplexInterleaved
        } else {
            GpuTensorStorage::Real
        },
        precision: element.precision(),
        integer: element.integer_type(),
        logical: false,
        alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
    })
}

pub(super) fn like_contract(
    input: &GpuTensorHandle,
    shape: Vec<usize>,
) -> gpu_helpers::BinaryGpuOutputContract {
    gpu_helpers::BinaryGpuOutputContract {
        shape,
        storage: input.physical().storage(),
        precision: runmat_accelerate_api::handle_precision(input),
        integer: runmat_accelerate_api::handle_integer_type(input),
        logical: input.view().is_logical(),
        alias: gpu_helpers::GpuOutputAliasPolicy::AllowInput,
    }
}

impl ExtremaKind {
    pub(super) fn admit_output(
        self,
        provider: &'static dyn AccelProvider,
        left: &GpuTensorHandle,
        right: &GpuTensorHandle,
        contract: &gpu_helpers::BinaryGpuOutputContract,
        mut output: GpuTensorHandle,
    ) -> BuiltinResult<GpuTensorHandle> {
        if !gpu_helpers::binary_gpu_output_matches(&output, left, right, provider, contract) {
            return Err(gpu_helpers::reject_gpu_output(
                output,
                self.contract_error(
                    "elementwise extrema stage changed owner, shape, storage, or resolved class",
                ),
            ));
        }
        if let Err(source) = output.allocation_lease().validate_live() {
            let mut error = self.contract_error(format!(
                "elementwise extrema stage returned unavailable allocation: {source}"
            ));
            error.source = Some(Box::new(source));
            return Err(gpu_helpers::reject_gpu_output(output, error));
        }
        if let Err(error) = gpu_helpers::propagate_output_provenance(&mut output, [left, right]) {
            return Err(gpu_helpers::reject_gpu_output(output, error));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests;

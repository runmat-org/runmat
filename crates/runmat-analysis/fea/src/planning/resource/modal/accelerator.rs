//! Modal accelerator inventory; CPU placement never claims a device.
use super::*;

pub(super) fn modal_accelerator_requirements(
    backend: ComputeBackend,
    requests: &[AcceleratorRequest],
    reduced_dimension: u64,
    stiffness_nonzeros: u64,
) -> Result<Vec<AcceleratorRequirement>, ResourcePlanningError> {
    if backend != ComputeBackend::Gpu {
        if !requests.is_empty() {
            return Err(ResourcePlanningError::Invalid(
                "CPU modal execution cannot claim accelerator resources".into(),
            ));
        }
        return Ok(Vec::new());
    }
    let [request] = requests else {
        return Err(ResourcePlanningError::Invalid(
            "GPU modal execution requires exactly one accelerator request".into(),
        ));
    };
    let required =
        structural_modal_accelerator_minimum_bytes(reduced_dimension, stiffness_nonzeros)?;
    if request.count != 1 || request.minimum_allocation_bytes < required {
        return Err(ResourcePlanningError::Invalid(format!(
            "the modal accelerator requires one device with at least {required} bytes"
        )));
    }
    Ok(vec![AcceleratorRequirement {
        class: request.class.clone(),
        count: request.count,
        memory_bytes_each: ByteCount(request.minimum_allocation_bytes),
    }])
}

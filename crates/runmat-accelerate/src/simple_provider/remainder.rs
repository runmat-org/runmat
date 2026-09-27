//! Floating remainder hooks using the owning runtime numerical policy.
use super::*;
use runmat_runtime::builtins::common::broadcast::BroadcastPlan;
use runmat_types::FloatingRemainderKind;

impl InProcessProvider {
    pub(super) fn floating_remainder_impl(
        &self,
        a: &GpuTensorHandle,
        b: &GpuTensorHandle,
        kind: FloatingRemainderKind,
    ) -> Result<GpuTensorHandle> {
        for input in [a, b] {
            ensure!(
                input.device_id() == self.device_id(),
                "remainder operand belongs to another provider"
            );
            if input.descriptor().element_type != Some(NumericElementType::F64) {
                return Err(runmat_accelerate_api::unsupported_provider_operation(
                    "in-process floating remainder requires native double",
                ));
            }
            ensure!(
                input.descriptor().storage == Some(GpuTensorStorage::Real)
                    && !runmat_accelerate_api::handle_is_logical(input)
                    && runmat_accelerate_api::handle_class_identity(input)
                        .is_none_or(|class| class == NumericElementType::F64.class_identity()),
                "floating remainder operand metadata mismatch"
            );
        }
        let shape = runtime_broadcast_shapes("remainder", a.shape(), b.shape())
            .map_err(|error| anyhow!(error))?;
        let empty = shape.contains(&0);
        if !empty {
            shape
                .iter()
                .try_fold(1_usize, |n, d| n.checked_mul(*d))
                .ok_or_else(|| anyhow!("remainder output element count overflow"))?;
        }
        let output = {
            let registry = registry(self.device_id).lock().unwrap();
            let left = registry
                .get(a)
                .ok_or_else(|| anyhow!("remainder left buffer missing"))?;
            let right = registry
                .get(b)
                .ok_or_else(|| anyhow!("remainder right buffer missing"))?;
            ensure_storage_len("remainder", left, a.shape(), &GpuTensorStorage::Real)?;
            ensure_storage_len("remainder", right, b.shape(), &GpuTensorStorage::Real)?;
            if empty {
                Vec::new()
            } else {
                let plan =
                    BroadcastPlan::new(a.shape(), b.shape()).map_err(|error| anyhow!(error))?;
                plan.iter()
                    .map(|(_, l, r)| {
                        runmat_runtime::builtins::math::rounding::floating_remainder::apply_f64(
                            left[l], right[r], kind,
                        )
                    })
                    .collect()
            }
        };
        self.allocate_tensor_with_storage(output, shape, GpuTensorStorage::Real)
    }
}

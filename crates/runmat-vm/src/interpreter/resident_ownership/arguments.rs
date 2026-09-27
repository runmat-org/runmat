//! Compact roots for arguments moved out of a VM stack during suspension.
use runmat_gc::{GcHandle, GcRoot, Trace, Tracer, ValueRoot};
use runmat_value::Value;
use std::collections::HashSet;

pub(super) struct ArgumentRoot {
    resident: Vec<Value>,
    managed: Vec<GcHandle>,
}

impl ArgumentRoot {
    pub fn is_empty(&self) -> bool {
        self.resident.is_empty() && self.managed.is_empty()
    }

    pub fn new(
        values: &[Value],
        handles: impl Iterator<Item = runmat_accelerate_api::GpuTensorHandle>,
    ) -> Self {
        struct Managed(HashSet<GcHandle>);
        impl Tracer for Managed {
            fn mark(&mut self, handle: GcHandle) {
                self.0.insert(handle);
            }
        }
        let mut managed = Managed(HashSet::new());
        for value in values {
            value.trace(&mut managed);
        }
        Self {
            resident: handles.map(Value::GpuTensor).collect(),
            managed: managed.0.into_iter().collect(),
        }
    }
}

impl GcRoot for ArgumentRoot {
    fn scan(&self) -> Vec<GcHandle> {
        self.managed.clone()
    }
    fn description(&self) -> String {
        "vm_consumed_arguments".into()
    }
}

impl ValueRoot for ArgumentRoot {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        for value in &self.resident {
            visitor(value)?;
        }
        // Managed argument targets can acquire new resident children while a
        // callback is suspended. Revisit the actual guarded target, not a stale
        // payload snapshot or a synthesized handle-object value.
        for handle in &self.managed {
            runmat_gc::gc_with_value(handle, &mut *visitor).ok()??;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_arguments_register_only_resource_identities_not_host_payloads() {
        let handle = runmat_accelerate_api::GpuTensorHandle::new(vec![1, 1], 9912, 1);
        let values = vec![
            Value::Tensor(runmat_value::Tensor::new(vec![2.0; 65_536], vec![256, 256]).unwrap()),
            Value::GpuTensor(handle.clone()),
        ];
        let root = ArgumentRoot::new(&values, std::iter::once(handle));
        assert_eq!(root.resident.len(), 1);
        assert!(root.managed.is_empty());
        root.visit_retained_values(&mut |value| {
            assert!(
                matches!(value, Value::GpuTensor(_)),
                "host tensor must never enter the owned argument projection"
            );
            Some(())
        })
        .unwrap();
    }
}

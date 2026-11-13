//! GC liveness for caller Values held by a pending preference service future.
//! The durable document owns encoded payloads; these roots only cover the
//! preparation interval, including an IndexedDB observer after caller drop.

use super::PreferenceServiceError;
use crate::execution::RootedValueHandles;
use runmat_value::Value;

pub(super) struct PreferenceInputRoots {
    _roots: RootedValueHandles,
}

impl PreferenceInputRoots {
    pub(super) fn for_values(values: &[Value]) -> Result<Self, PreferenceServiceError> {
        Self::new(values.iter())
    }

    pub(super) fn for_entries(entries: &[(String, Value)]) -> Result<Self, PreferenceServiceError> {
        Self::new(entries.iter().map(|(_, value)| value))
    }

    fn new<'a>(values: impl Iterator<Item = &'a Value>) -> Result<Self, PreferenceServiceError> {
        let roots = RootedValueHandles::for_values(values).map_err(|source| {
            PreferenceServiceError::Storage(
                crate::build_runtime_error("could not retain preference input")
                    .with_source(source)
                    .build(),
            )
        })?;
        Ok(Self { _roots: roots })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::HandleRef;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn nested_inputs_keep_managed_handles_alive_until_preparation_ends() {
        runmat_gc::gc_test_context(|| {
            let initial = runmat_gc::gc_allocate_rooted(Value::String("managed".into())).unwrap();
            let handle = initial.handle();
            let input = Value::HandleObject(HandleRef {
                class_name: "PreferenceInput".into(),
                target: handle,
                valid: true,
            });
            let roots = PreferenceInputRoots::for_values(&[input]).unwrap();
            initial.unroot().unwrap();
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            drop(roots);
            runmat_gc::gc_collect_major().unwrap();
            assert!(runmat_gc::gc_clone_value(&handle).is_err());
        });
    }
}

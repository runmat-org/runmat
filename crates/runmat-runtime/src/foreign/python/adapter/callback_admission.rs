use std::cell::RefCell;
use std::collections::BTreeMap;

use runmat_python::PythonValue;

use super::{PythonAdapter, PythonCallbackRegistration, RuntimeContext, RuntimeError};

/// Callback tokens created while converting one call's inputs have not reached
/// Python yet. Retire only those tokens if admission fails before dispatch.
pub(super) struct CallbackAdmission<'a> {
    callbacks: &'a RefCell<BTreeMap<u64, PythonCallbackRegistration>>,
    unpublished: Vec<u64>,
}

impl<'a> CallbackAdmission<'a> {
    pub(super) fn new(callbacks: &'a RefCell<BTreeMap<u64, PythonCallbackRegistration>>) -> Self {
        Self {
            callbacks,
            unpublished: Vec::new(),
        }
    }

    pub(super) fn convert<V: Into<crate::call::RootedArgumentValue>>(
        &mut self,
        adapter: &PythonAdapter,
        context: &RuntimeContext,
        value: V,
    ) -> Result<PythonValue, RuntimeError> {
        let converted = adapter.argument_to_python(context, value.into())?;
        if let PythonValue::Callback(id) = &converted {
            self.unpublished.push(*id);
        }
        Ok(converted)
    }

    pub(super) fn convert_arguments<V: Into<crate::call::RootedArgumentValue>>(
        &mut self,
        adapter: &PythonAdapter,
        context: &RuntimeContext,
        arguments: Vec<V>,
    ) -> Result<Vec<PythonValue>, RuntimeError> {
        arguments
            .into_iter()
            .map(|value| self.convert(adapter, context, value))
            .collect()
    }

    pub(super) fn convert_named_inputs<V: Into<crate::call::RootedArgumentValue>>(
        &mut self,
        adapter: &PythonAdapter,
        context: &RuntimeContext,
        inputs: Vec<(String, V)>,
    ) -> Result<Vec<(String, PythonValue)>, RuntimeError> {
        inputs
            .into_iter()
            .map(|(name, value)| Ok((name, self.convert(adapter, context, value)?)))
            .collect()
    }

    /// After this point Python may retain any published token, including when
    /// execution fails or is canceled, so the adapter owns each registration.
    pub(super) fn publish(mut self) {
        self.unpublished.clear();
    }
}

impl Drop for CallbackAdmission<'_> {
    fn drop(&mut self) {
        if self.unpublished.is_empty() {
            return;
        }
        let retired = {
            let mut callbacks = self.callbacks.borrow_mut();
            self.unpublished
                .drain(..)
                .filter_map(|id| callbacks.remove(&id))
                .collect::<Vec<_>>()
        };
        // Dropping a rooted registration can reenter GC observers. The table
        // borrow must end before the last root of any callback is retired.
        drop(retired);
    }
}

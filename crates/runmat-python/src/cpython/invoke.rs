use super::interpreter::Interpreter;
use super::support::{attr, c_string, capture_error, checked};
use crate::{PythonCall, PythonError, PythonValue};

impl Interpreter {
    pub(crate) fn invoke(&self, call: PythonCall) -> Result<Vec<PythonValue>, PythonError> {
        self.with_gil(|this| this.invoke_with_gil(call))
    }

    pub(super) fn invoke_with_gil(
        &self,
        call: PythonCall,
    ) -> Result<Vec<PythonValue>, PythonError> {
        match call {
            PythonCall::InvokeQualified { name, arguments } => {
                let callable = self.resolve_qualified(&name)?;
                let result = self.call_object(callable, arguments);
                self.decref(callable);
                result.map(|value| vec![value])
            }
            PythonCall::GetMember { receiver, name } => {
                let receiver = self.object(receiver)?;
                let value = unsafe { attr(&self.api, receiver, &name)? };
                self.convert_owned(value).map(|value| vec![value])
            }
            PythonCall::SetMember {
                receiver,
                name,
                value,
            } => {
                let receiver = self.object(receiver)?;
                let value = self.to_owned(value)?;
                let name = c_string(&name, "attribute name")?;
                // SAFETY: receiver and value are live lane-owned references.
                let status =
                    unsafe { (self.api.py_object_set_attr_string)(receiver, name.as_ptr(), value) };
                self.decref(value);
                self.status(status, "set Python attribute")?;
                Ok(Vec::new())
            }
            PythonCall::InvokeMember {
                receiver,
                name,
                arguments,
            } => {
                let receiver = self.object(receiver)?;
                let callable = unsafe { attr(&self.api, receiver, &name)? };
                let result = self.call_object(callable, arguments);
                self.decref(callable);
                result.map(|value| vec![value])
            }
            PythonCall::GetItem { receiver, index } => {
                let receiver = self.object(receiver)?;
                let index = self.to_owned(index)?;
                // SAFETY: both references are live for the call.
                let value = unsafe { (self.api.py_object_get_item)(receiver, index) };
                self.decref(index);
                let value = unsafe { checked(&self.api, value, "index Python object")? };
                self.convert_owned(value).map(|value| vec![value])
            }
            PythonCall::SetItem {
                receiver,
                index,
                value,
            } => {
                let receiver = self.object(receiver)?;
                let index = self.to_owned(index)?;
                let value = self.to_owned(value)?;
                // SAFETY: all references are live for the call.
                let status = unsafe { (self.api.py_object_set_item)(receiver, index, value) };
                self.decref(index);
                self.decref(value);
                self.status(status, "assign Python object item")?;
                Ok(Vec::new())
            }
            PythonCall::Iterate { receiver } => {
                let receiver = self.object(receiver)?;
                // SAFETY: receiver is a live lane-owned reference.
                let iterator = unsafe { (self.api.py_object_get_iter)(receiver) };
                let iterator = unsafe { checked(&self.api, iterator, "iterate Python object")? };
                let mut values = Vec::new();
                loop {
                    // SAFETY: iterator remains live until the loop completes.
                    let item = unsafe { (self.api.py_iter_next)(iterator) };
                    if item.is_null() {
                        break;
                    }
                    values.push(self.convert_owned(item)?);
                }
                self.decref(iterator);
                // A null iterator result means exhaustion only if no error is set.
                if unsafe { (self.api.py_err_occurred)() }.is_null() {
                    Ok(values)
                } else {
                    Err(unsafe { capture_error(&self.api, "iterate Python object") })
                }
            }
            PythonCall::ExecutePersistent {
                code,
                inputs,
                outputs,
            } => self.execute(&code, inputs, outputs, self.workspace),
            PythonCall::ExecuteFile {
                path,
                arguments,
                inputs,
                outputs,
            } => self.execute_file(&path, arguments, inputs, outputs),
            PythonCall::Release { handle } => {
                if let Some(object) = self.objects.borrow_mut().remove(&handle) {
                    self.decref(object);
                }
                Ok(Vec::new())
            }
        }
    }
}

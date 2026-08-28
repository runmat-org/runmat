use std::ffi::c_void;
use std::sync::mpsc;

use super::api::{capsule_pointer, PyMethodDef, PyObject};
use super::interpreter::Interpreter;
use super::support::{c_string, checked};
use crate::object::{PythonCallbackCommand, PythonLaneEvent};
use crate::{PythonCallbackInvocation, PythonError};

const METH_VARARGS: i32 = 0x0001;
const CALLBACK_CAPSULE_NAME: &std::ffi::CStr = c"runmat.python.callback";

struct CallbackCapsule {
    interpreter: std::rc::Weak<Interpreter>,
    callback: u64,
    _definition: Box<PyMethodDef>,
}

impl Interpreter {
    pub(super) fn callback_to_python(&self, callback: u64) -> Result<*mut PyObject, PythonError> {
        let definition = Box::new(PyMethodDef {
            name: c"runmat_callback".as_ptr(),
            method: invoke_callback,
            flags: METH_VARARGS,
            documentation: c"RunMat callback".as_ptr(),
        });
        let definition_pointer = std::ptr::from_ref(definition.as_ref()).cast_mut();
        let capsule_data = Box::new(CallbackCapsule {
            interpreter: self.weak_reference(),
            callback,
            _definition: definition,
        });
        let capsule_data = Box::into_raw(capsule_data);
        // SAFETY: the capsule owns capsule_data until release_callback_capsule
        // recovers it with the same private name.
        let capsule = unsafe {
            (self.api.py_capsule_new)(
                capsule_data.cast::<c_void>(),
                CALLBACK_CAPSULE_NAME.as_ptr(),
                Some(release_callback_capsule),
            )
        };
        if capsule.is_null() {
            // SAFETY: CPython did not take the allocation on failure.
            unsafe { drop(Box::from_raw(capsule_data)) };
            return Err(unsafe {
                super::support::capture_error(&self.api, "create Python callback owner")
            });
        }
        // SAFETY: definition_pointer points into capsule_data and therefore
        // outlives the function object, which owns a reference to capsule.
        let function = unsafe {
            (self.api.py_cfunction_new_ex)(definition_pointer, capsule, std::ptr::null_mut())
        };
        self.decref(capsule);
        unsafe { checked(&self.api, function, "create Python callback") }
    }
}

unsafe extern "C" fn invoke_callback(
    capsule: *mut PyObject,
    arguments: *mut PyObject,
) -> *mut PyObject {
    let pointer = unsafe { capsule_pointer(capsule, CALLBACK_CAPSULE_NAME.as_ptr()) };
    let Some(capsule) = (unsafe { pointer.cast::<CallbackCapsule>().as_ref() }) else {
        return std::ptr::null_mut();
    };
    let Some(interpreter) = capsule.interpreter.upgrade() else {
        return std::ptr::null_mut();
    };
    let arguments = match interpreter.tuple_values(arguments, "read Python callback arguments") {
        Ok(arguments) => arguments,
        Err(error) => return callback_error(&interpreter, error),
    };
    let Some(events) = interpreter.active_callbacks.borrow().clone() else {
        return callback_error(
            &interpreter,
            PythonError::host(
                "PythonCallbackError",
                "Python callback has no active RunMat invocation",
            ),
        );
    };
    let (commands, command_receiver) = mpsc::sync_channel(1);
    if events
        .send(PythonLaneEvent::Callback {
            invocation: PythonCallbackInvocation {
                callback: capsule.callback,
                arguments,
            },
            commands,
        })
        .is_err()
    {
        return callback_error(
            &interpreter,
            PythonError::host(
                "PythonCallbackError",
                "Python callback's originating RunMat invocation has ended",
            ),
        );
    }
    loop {
        match command_receiver.recv() {
            Ok(PythonCallbackCommand::Return(Ok(value))) => {
                return match Interpreter::to_owned(interpreter.as_ref(), value) {
                    Ok(value) => value,
                    Err(error) => callback_error(&interpreter, error),
                };
            }
            Ok(PythonCallbackCommand::Return(Err(error))) => {
                return callback_error(&interpreter, error);
            }
            Ok(PythonCallbackCommand::Reenter { call, reply }) => {
                let _ = reply.send(interpreter.invoke_with_gil(call));
            }
            Err(_) => {
                return callback_error(
                    &interpreter,
                    PythonError::host(
                        "PythonCallbackError",
                        "Python callback response channel closed",
                    ),
                );
            }
        }
    }
}

fn callback_error(interpreter: &Interpreter, error: PythonError) -> *mut PyObject {
    let message = c_string(&error.to_string(), "Python callback error")
        .unwrap_or_else(|_| c"RunMat callback failed".to_owned());
    // SAFETY: RuntimeError and the message are live Python/C objects for this
    // call. CPython retains the exception state rather than the string pointer.
    unsafe {
        (interpreter.api.py_err_set_string)(interpreter.builtins.runtime_error, message.as_ptr())
    };
    std::ptr::null_mut()
}

unsafe extern "C" fn release_callback_capsule(capsule: *mut PyObject) {
    let pointer = unsafe { capsule_pointer(capsule, CALLBACK_CAPSULE_NAME.as_ptr()) };
    if !pointer.is_null() {
        // SAFETY: callback_to_python created exactly one box at this address.
        unsafe { drop(Box::from_raw(pointer.cast::<CallbackCapsule>())) };
    }
}

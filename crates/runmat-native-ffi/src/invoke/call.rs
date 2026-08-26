use libffi::middle::{Arg, Cif, Ret};
use runmat_value::Value;

use crate::loader::LoadedLibrary;
use crate::{NativeLibraryMetadata, NativeType, SymbolPrototype};

use super::abi::{calling_convention, ffi_type};
use super::arguments::prepare_arguments;
use super::callback::{closure, CallbackBinding, CallbackState};
use super::outputs::{decode_output_parameters, decode_return, return_slot, ReturnSlot};
use super::pointer::PointerBinding;
use super::InvocationError;

#[derive(Debug, Clone, PartialEq)]
pub enum InvocationValue {
    Value(Value),
    Pointer(crate::NativePointer),
}

#[derive(Debug, Clone, PartialEq)]
pub struct InvocationResult {
    pub return_value: Option<InvocationValue>,
    pub output_parameters: Vec<(usize, Value)>,
}

pub fn invoke_symbol(
    library: &LoadedLibrary,
    prototype: &SymbolPrototype,
    arguments: &[Value],
    metadata: &NativeLibraryMetadata,
) -> Result<InvocationResult, InvocationError> {
    invoke_symbol_with_bindings(library, prototype, arguments, metadata, &[], &[])
}

pub fn invoke_symbol_with_callbacks(
    library: &LoadedLibrary,
    prototype: &SymbolPrototype,
    arguments: &[Value],
    metadata: &NativeLibraryMetadata,
    callbacks: &[CallbackBinding<'_>],
) -> Result<InvocationResult, InvocationError> {
    invoke_symbol_with_bindings(library, prototype, arguments, metadata, callbacks, &[])
}

pub fn invoke_symbol_with_bindings(
    library: &LoadedLibrary,
    prototype: &SymbolPrototype,
    arguments: &[Value],
    metadata: &NativeLibraryMetadata,
    callbacks: &[CallbackBinding<'_>],
    pointers: &[PointerBinding<'_>],
) -> Result<InvocationResult, InvocationError> {
    if prototype.variadic {
        return Err(InvocationError::Abi {
            symbol: prototype.name.clone(),
            message: "variadic calls require explicit promoted argument metadata".into(),
        });
    }
    if arguments.len() != prototype.parameters.len() {
        return Err(InvocationError::Arity {
            symbol: prototype.name.clone(),
            expected: prototype.parameters.len(),
            actual: arguments.len(),
        });
    }
    let symbol = library.symbol(&prototype.exported_name)?;
    let abi = calling_convention(&prototype.name, prototype.calling_convention)?;
    let argument_types = prototype
        .parameters
        .iter()
        .map(|parameter| ffi_type(&prototype.name, &parameter.ty, metadata))
        .collect::<Result<Vec<_>, _>>()?;
    let result_type = ffi_type(&prototype.name, &prototype.return_type, metadata)?;
    let cif = Cif::try_new_with_abi(argument_types, result_type, abi).map_err(|error| {
        InvocationError::Abi {
            symbol: prototype.name.clone(),
            message: format!("could not prepare call interface: {error:?}"),
        }
    })?;
    let mut seen_callbacks = std::collections::BTreeSet::new();
    let callback_states = callbacks
        .iter()
        .map(|binding| {
            let parameter = prototype
                .parameters
                .get(binding.argument_index)
                .ok_or_else(|| InvocationError::Argument {
                    symbol: prototype.name.clone(),
                    argument: binding.argument_index + 1,
                    name: "callback".into(),
                    message: "callback binding index is out of range".into(),
                })?;
            if !seen_callbacks.insert(binding.argument_index) {
                return Err(InvocationError::Argument {
                    symbol: prototype.name.clone(),
                    argument: binding.argument_index + 1,
                    name: parameter.name.clone(),
                    message: "callback argument has more than one binding".into(),
                });
            }
            CallbackState::new(&parameter.ty, binding.dispatch).map_err(|message| {
                InvocationError::Argument {
                    symbol: prototype.name.clone(),
                    argument: binding.argument_index + 1,
                    name: parameter.name.clone(),
                    message,
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let callback_closures = callbacks
        .iter()
        .zip(&callback_states)
        .map(|(binding, state)| {
            let NativeType::Callback {
                calling_convention: convention,
                ..
            } = &prototype.parameters[binding.argument_index].ty
            else {
                unreachable!("CallbackState validated the parameter type")
            };
            state
                .cif(&prototype.name, *convention, metadata)
                .map(|cif| closure(cif, state))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let callback_addresses = callbacks
        .iter()
        .zip(&callback_closures)
        .map(|(binding, closure)| {
            let code = libffi::middle::CodePtr::from_fun(*closure.code_ptr());
            (binding.argument_index, code.as_mut_ptr())
        })
        .collect::<BTreeMap<_, _>>();
    let mut pointer_addresses = BTreeMap::new();
    for binding in pointers {
        let parameter = prototype
            .parameters
            .get(binding.argument_index)
            .ok_or_else(|| InvocationError::Argument {
                symbol: prototype.name.clone(),
                argument: binding.argument_index + 1,
                name: "pointer".into(),
                message: "pointer binding index is out of range".into(),
            })?;
        let NativeType::Pointer { pointee, .. } = &parameter.ty else {
            return Err(InvocationError::Argument {
                symbol: prototype.name.clone(),
                argument: binding.argument_index + 1,
                name: parameter.name.clone(),
                message: "pointer binding points to a non-pointer argument".into(),
            });
        };
        if pointee.as_ref() != binding.pointee() {
            return Err(InvocationError::Argument {
                symbol: prototype.name.clone(),
                argument: binding.argument_index + 1,
                name: parameter.name.clone(),
                message: "pointer resource type does not match the declared pointee".into(),
            });
        }
        if pointer_addresses
            .insert(binding.argument_index, binding.address())
            .is_some()
        {
            return Err(InvocationError::Argument {
                symbol: prototype.name.clone(),
                argument: binding.argument_index + 1,
                name: parameter.name.clone(),
                message: "pointer argument has more than one binding".into(),
            });
        }
    }
    let argument_slots = prepare_arguments(
        &prototype.name,
        &prototype.parameters,
        arguments,
        metadata,
        &callback_addresses,
        &pointer_addresses,
    )?;
    let ffi_arguments = argument_slots
        .iter()
        .map(|slot| slot.ffi_arg())
        .collect::<Vec<Arg<'_>>>();
    let mut return_slot = return_slot(&prototype.name, &prototype.return_type, metadata)?;

    // SAFETY: The CIF is derived from the validated normalized prototype; each
    // argument slot has matching storage and remains alive through the call.
    // The loaded symbol borrows its library, and return storage matches the CIF.
    unsafe {
        match &mut return_slot {
            ReturnSlot::Void => {
                cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::void())
            }
            ReturnSlot::Scalar(slot) => match slot {
                super::arguments::ScalarSlot::I8(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::U8(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::I16(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::U16(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::I32(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::U32(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::I64(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::U64(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::Isize(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::Usize(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::F32(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
                super::arguments::ScalarSlot::F64(value) => {
                    cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(value))
                }
            },
            ReturnSlot::Pointer(pointer) => {
                cif.call_return_into(symbol.code_ptr(), &ffi_arguments, Ret::new(pointer))
            }
            ReturnSlot::Structure(storage) => cif.call_return_into(
                symbol.code_ptr(),
                &ffi_arguments,
                Ret::new(storage.bytes_mut()),
            ),
        }
    }
    if let Some(message) = callback_states.iter().find_map(CallbackState::take_failure) {
        return Err(InvocationError::Callback {
            symbol: prototype.name.clone(),
            message,
        });
    }
    Ok(InvocationResult {
        return_value: decode_return(&prototype.name, prototype, &return_slot, metadata)?,
        output_parameters: decode_output_parameters(
            &prototype.name,
            prototype,
            &argument_slots,
            metadata,
        )?
        .into_iter()
        .filter(|(index, _)| !pointer_addresses.contains_key(index))
        .collect(),
    })
}
use std::collections::BTreeMap;

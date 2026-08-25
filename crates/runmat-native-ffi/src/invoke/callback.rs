use std::cell::RefCell;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use libffi::low::ffi_cif;
use libffi::middle::{Cif, Closure, Type};
use runmat_value::{IntValue, Value};

use crate::{NativeLibraryMetadata, NativeScalar, NativeType};

use super::abi::{calling_convention, ffi_type};
use super::arguments::{scalar_from_value, write_scalar_to_pointer};
use super::InvocationError;

pub trait CallbackDispatch {
    fn invoke(&self, arguments: &[Value]) -> Result<Value, String>;
}

impl<F> CallbackDispatch for F
where
    F: Fn(&[Value]) -> Result<Value, String>,
{
    fn invoke(&self, arguments: &[Value]) -> Result<Value, String> {
        self(arguments)
    }
}

pub struct CallbackBinding<'callback> {
    pub argument_index: usize,
    pub dispatch: &'callback dyn CallbackDispatch,
}

impl std::fmt::Debug for CallbackBinding<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CallbackBinding")
            .field("argument_index", &self.argument_index)
            .finish_non_exhaustive()
    }
}

pub(super) struct CallbackState<'callback> {
    return_type: NativeType,
    parameters: Vec<NativeType>,
    dispatch: &'callback dyn CallbackDispatch,
    failure: RefCell<Option<String>>,
}

impl std::fmt::Debug for CallbackState<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CallbackState")
            .field("return_type", &self.return_type)
            .field("parameters", &self.parameters)
            .field("failed", &self.failure.borrow().is_some())
            .finish_non_exhaustive()
    }
}

impl<'callback> CallbackState<'callback> {
    pub(super) fn new(
        ty: &NativeType,
        dispatch: &'callback dyn CallbackDispatch,
    ) -> Result<Self, String> {
        let NativeType::Callback {
            return_type,
            parameters,
            ..
        } = ty
        else {
            return Err("callback binding points to a non-callback argument".into());
        };
        if !matches!(
            return_type.as_ref(),
            NativeType::Scalar { .. } | NativeType::Enumeration { .. }
        ) || parameters.iter().any(|parameter| {
            !matches!(
                parameter.ty,
                NativeType::Scalar { .. } | NativeType::Enumeration { .. }
            )
        }) {
            return Err("callbacks currently require scalar parameters and a scalar return".into());
        }
        Ok(Self {
            return_type: return_type.as_ref().clone(),
            parameters: parameters
                .iter()
                .map(|parameter| parameter.ty.clone())
                .collect(),
            dispatch,
            failure: RefCell::new(None),
        })
    }

    pub(super) fn cif(
        &self,
        symbol: &str,
        convention: crate::CallingConvention,
        metadata: &NativeLibraryMetadata,
    ) -> Result<Cif, InvocationError> {
        let arguments = self
            .parameters
            .iter()
            .map(|ty| ffi_type(symbol, ty, metadata))
            .collect::<Result<Vec<Type>, _>>()?;
        let result = ffi_type(symbol, &self.return_type, metadata)?;
        let abi = calling_convention(symbol, convention)?;
        Cif::try_new_with_abi(arguments, result, abi).map_err(|error| InvocationError::Abi {
            symbol: symbol.into(),
            message: format!("could not prepare callback interface: {error:?}"),
        })
    }

    pub(super) fn take_failure(&self) -> Option<String> {
        self.failure.borrow_mut().take()
    }
}

pub(super) fn closure<'state>(cif: Cif, state: &'state CallbackState<'_>) -> Closure<'state> {
    Closure::new(cif, callback_trampoline, state)
}

unsafe extern "C" fn callback_trampoline(
    _cif: &ffi_cif,
    result: &mut u8,
    arguments: *const *const c_void,
    state: &CallbackState<'_>,
) {
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        let values = state
            .parameters
            .iter()
            .enumerate()
            .map(|(index, ty)| {
                // SAFETY: libffi supplies one pointer for every CIF argument;
                // the type is the validated callback parameter type.
                let pointer = unsafe { *arguments.add(index) };
                unsafe { read_scalar_value(ty, pointer) }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let value = state.dispatch.invoke(&values)?;
        let scalar = match &state.return_type {
            NativeType::Scalar { scalar }
            | NativeType::Enumeration {
                storage: scalar, ..
            } => *scalar,
            _ => return Err("callback return type is not scalar".into()),
        };
        let slot = scalar_from_value(scalar, &value)?;
        // SAFETY: `result` is the libffi return buffer described by `scalar`.
        unsafe { write_scalar_to_pointer(&slot, (result as *mut u8).cast()) };
        Ok::<(), String>(())
    }));
    let failure = match outcome {
        Ok(Ok(())) => None,
        Ok(Err(message)) => Some(message),
        Err(_) => Some("callback panicked".into()),
    };
    if let Some(message) = failure {
        *state.failure.borrow_mut() = Some(message);
        let scalar = match &state.return_type {
            NativeType::Scalar { scalar }
            | NativeType::Enumeration {
                storage: scalar, ..
            } => *scalar,
            _ => return,
        };
        let zero = super::arguments::zero_scalar(scalar);
        // SAFETY: The return buffer matches the normalized scalar return type.
        unsafe { write_scalar_to_pointer(&zero, (result as *mut u8).cast()) };
    }
}

unsafe fn read_scalar_value(ty: &NativeType, pointer: *const c_void) -> Result<Value, String> {
    let scalar = match ty {
        NativeType::Scalar { scalar }
        | NativeType::Enumeration {
            storage: scalar, ..
        } => *scalar,
        _ => return Err("callback argument type is not scalar".into()),
    };
    macro_rules! read {
        ($type:ty, $value:expr) => {{
            // SAFETY: The callback CIF guarantees a readable value of this type.
            let value = unsafe { pointer.cast::<$type>().read() };
            $value(value)
        }};
    }
    Ok(match scalar {
        NativeScalar::Bool => read!(u8, |value| Value::Bool(value != 0)),
        NativeScalar::Char | NativeScalar::SignedChar | NativeScalar::I8 => {
            read!(i8, |value| Value::Int(IntValue::I8(value)))
        }
        NativeScalar::UnsignedChar | NativeScalar::U8 => {
            read!(u8, |value| Value::Int(IntValue::U8(value)))
        }
        NativeScalar::Short | NativeScalar::I16 => {
            read!(i16, |value| Value::Int(IntValue::I16(value)))
        }
        NativeScalar::UnsignedShort | NativeScalar::U16 => {
            read!(u16, |value| Value::Int(IntValue::U16(value)))
        }
        NativeScalar::Int | NativeScalar::I32 => {
            read!(i32, |value| Value::Int(IntValue::I32(value)))
        }
        NativeScalar::UnsignedInt | NativeScalar::U32 => {
            read!(u32, |value| Value::Int(IntValue::U32(value)))
        }
        NativeScalar::Long if std::mem::size_of::<std::ffi::c_long>() == 4 => {
            read!(i32, |value| Value::Int(IntValue::I32(value)))
        }
        NativeScalar::Long => read!(i64, |value| Value::Int(IntValue::I64(value))),
        NativeScalar::UnsignedLong if std::mem::size_of::<std::ffi::c_ulong>() == 4 => {
            read!(u32, |value| Value::Int(IntValue::U32(value)))
        }
        NativeScalar::UnsignedLong => read!(u64, |value| Value::Int(IntValue::U64(value))),
        NativeScalar::LongLong | NativeScalar::I64 => {
            read!(i64, |value| Value::Int(IntValue::I64(value)))
        }
        NativeScalar::UnsignedLongLong | NativeScalar::U64 => {
            read!(u64, |value| Value::Int(IntValue::U64(value)))
        }
        NativeScalar::Isize => read!(isize, |value| Value::Int(IntValue::I64(value as i64))),
        NativeScalar::Usize => read!(usize, |value| Value::Int(IntValue::U64(value as u64))),
        NativeScalar::F32 => read!(f32, |value| Value::Num(f64::from(value))),
        NativeScalar::F64 => read!(f64, Value::Num),
    })
}

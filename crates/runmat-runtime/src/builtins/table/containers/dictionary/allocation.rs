use super::*;
use crate::context::host_allocation::{
    host_allocation_runtime_error, HostAllocationError, HostAllocationPlan, HostAllocationPolicy,
    HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use runmat_value::{
    ExactArrayInitializationError, ExactHostAllocationPermit, ExactHostTensorAllocation,
    Utf16DecodeError, Utf16TextCopyError, Utf8Text,
};

pub(super) fn allocation(
    runtime: Option<&RuntimeContext>,
    cause: HostAllocationError,
) -> crate::RuntimeError {
    host_allocation_runtime_error(
        runtime,
        "RunMat:dictionary:HostAllocation",
        "dictionary: host allocation authority failed",
        cause,
    )
}
pub(super) fn exact_error<E>(
    runtime: Option<&RuntimeContext>,
    cause: ExactArrayInitializationError<E>,
) -> crate::RuntimeError
where
    E: Into<crate::RuntimeError>,
{
    match cause {
        ExactArrayInitializationError::Allocation(cause) => allocation(runtime, cause.into()),
        ExactArrayInitializationError::Element(cause) => cause.into(),
    }
}
fn utf16_decode_error(source: Utf16DecodeError) -> crate::RuntimeError {
    let mut error = invalid_variable("dictionary: character input is not valid UTF-16");
    error.source = Some(Box::new(source));
    error
}

fn utf16_error(
    runtime: Option<&RuntimeContext>,
    cause: ExactArrayInitializationError<Utf16TextCopyError<crate::RuntimeError>>,
) -> crate::RuntimeError {
    match cause {
        ExactArrayInitializationError::Allocation(source) => allocation(runtime, source.into()),
        ExactArrayInitializationError::Element(Utf16TextCopyError::Decode(source)) => {
            utf16_decode_error(source)
        }
        ExactArrayInitializationError::Element(Utf16TextCopyError::Checkpoint(error)) => error,
    }
}

pub(super) fn add(a: u64, b: u64, runtime: Option<&RuntimeContext>) -> BuiltinResult<u64> {
    a.checked_add(b)
        .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))
}
pub(super) fn exact_bytes(
    raw: usize,
    claims: usize,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<u64> {
    let raw = u64::try_from(raw).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?;
    HostAllocationPlan::retained_exact_output_bytes(raw, claims)
        .map_err(|cause| allocation(runtime, cause))
}

pub(super) struct Work<'a> {
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}
impl<'a> Work<'a> {
    pub(super) fn new(
        runtime: Option<&'a RuntimeContext>,
        scratch: u64,
        retained: u64,
    ) -> BuiltinResult<Self> {
        if runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(allocation(runtime, HostAllocationError::Cancelled));
        }
        let plan = HostAllocationPlan::new(scratch, retained)
            .map_err(|cause| allocation(runtime, cause))?;
        let work = Self {
            receipt: HostAllocationReceipt::admit(runtime, plan)
                .map_err(|cause| allocation(runtime, cause))?,
            runtime,
        };
        work.check()?;
        Ok(work)
    }
    pub(super) fn check(&self) -> BuiltinResult<()> {
        if self.runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(allocation(self.runtime, HostAllocationError::Cancelled));
        }
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed context"))
                .map_err(|cause| allocation(self.runtime, cause))?;
        } else if HostAllocationReceipt::policy(self.runtime)
            .map_err(|cause| allocation(self.runtime, cause))?
            != HostAllocationPolicy::Unbounded
        {
            return Err(allocation(
                self.runtime,
                HostAllocationError::StaleAuthority,
            ));
        }
        Ok(())
    }
    pub(super) fn construct<T>(
        &self,
        construct: impl FnOnce(Option<&dyn ExactHostAllocationPermit>) -> BuiltinResult<T>,
    ) -> BuiltinResult<T> {
        self.check()?;
        let output = match &self.receipt {
            Some(receipt) => receipt
                .with_output_construction(self.runtime.expect("managed context"), |permit| {
                    construct(Some(permit))
                })
                .map_err(|cause| allocation(self.runtime, cause))??,
            None => construct(None)?,
        };
        self.check()?;
        Ok(output)
    }
}

enum IndexStorage {
    Managed(HostTypedWorkspace<usize>),
    Unbounded(Vec<usize>),
}
// Storage retires before its scratch receipt.
pub(super) struct Indices<'a> {
    storage: IndexStorage,
    work: Work<'a>,
}
impl<'a> Indices<'a> {
    pub(super) fn new(runtime: Option<&'a RuntimeContext>, len: usize) -> BuiltinResult<Self> {
        let bytes = HostTypedWorkspace::<usize>::quoted_bytes(len)
            .map_err(|cause| allocation(runtime, cause))?;
        let work = Work::new(runtime, bytes, 0)?;
        let storage = match &work.receipt {
            Some(receipt) => IndexStorage::Managed(
                receipt
                    .temporary_typed(work.runtime.expect("managed context"), len)
                    .map_err(|cause| allocation(runtime, cause))?,
            ),
            None => IndexStorage::Unbounded(vec![0; len]),
        };
        work.check()?;
        Ok(Self { storage, work })
    }
    pub(super) fn check(&self) -> BuiltinResult<()> {
        self.work.check()
    }
    pub(super) fn as_slice(&self) -> &[usize] {
        match &self.storage {
            IndexStorage::Managed(value) => value.as_slice(),
            IndexStorage::Unbounded(value) => value,
        }
    }
    pub(super) fn as_mut_slice(&mut self) -> &mut [usize] {
        match &mut self.storage {
            IndexStorage::Managed(value) => value.as_mut_slice(),
            IndexStorage::Unbounded(value) => value,
        }
    }
}

impl stream::Element<'_> {
    pub(super) fn retained_bytes(self, runtime: Option<&RuntimeContext>) -> BuiltinResult<u64> {
        match self {
            Self::Numeric(NumericScalar::F32(_)) => {
                HostAllocationPlan::retained_tensor_bytes(NumericDType::F32, &[1, 1])
                    .map_err(|cause| allocation(runtime, cause))
            }
            Self::Chars(chars) => {
                let (_, raw) = Utf8Text::requested_bytes_for_utf16_with(chars, |_| {
                    if runtime.is_some_and(RuntimeContext::is_cancelled) {
                        Err(allocation(runtime, HostAllocationError::Cancelled))
                    } else {
                        Ok(())
                    }
                })
                .map_err(|cause| utf16_error(runtime, cause))?;
                exact_bytes(raw, usize::from(!chars.is_empty()), runtime)
            }
            Self::Literal(text) => {
                let raw = Utf8Text::requested_bytes_for_capacity(text.len())
                    .map_err(|cause| allocation(runtime, cause.into()))?;
                exact_bytes(raw, usize::from(!text.is_empty()), runtime)
            }
            _ => Ok(0),
        }
    }
    pub(super) fn materialize(
        self,
        runtime: Option<&RuntimeContext>,
        permit: Option<&dyn ExactHostAllocationPermit>,
    ) -> BuiltinResult<Value> {
        Ok(match self {
            Self::Borrowed(value) => value.clone(),
            Self::Text(value) => Value::String(value.clone()),
            Self::Logical(value) => Value::Bool(value),
            Self::Numeric(NumericScalar::F64(value)) => Value::Num(value),
            Self::Numeric(NumericScalar::F32(value)) => {
                if let Some(permit) = permit {
                    let mut exact =
                        ExactHostTensorAllocation::new(NumericDType::F32, &[1, 1], permit)
                            .map_err(|cause| allocation(runtime, cause.into()))?;
                    exact.as_f32_slice_mut().expect("native Single")[0] = value;
                    Value::Tensor(
                        exact
                            .into_tensor()
                            .map_err(|cause| allocation(runtime, cause.into()))?,
                    )
                } else {
                    // Only the explicitly unrestricted branch reaches this.
                    Value::Tensor(
                        Tensor::from_f32(vec![value], vec![1, 1]).map_err(invalid_variable)?,
                    )
                }
            }
            Self::Numeric(value) => Value::Int(value.into_int_value().expect("integer scalar")),
            Self::Chars(chars) => Value::String(
                (match permit {
                    Some(permit) => Utf8Text::try_copy_utf16_exact_with(chars, permit, |_| {
                        if runtime.is_some_and(RuntimeContext::is_cancelled) {
                            Err(allocation(runtime, HostAllocationError::Cancelled))
                        } else {
                            Ok(())
                        }
                    })
                    .map_err(|cause| utf16_error(runtime, cause))?,
                    None => runmat_value::decode_utf16_code_units(chars)
                        .map_err(utf16_decode_error)?
                        .into(),
                })
                .into(),
            ),
            Self::Literal(text) => Value::String(
                (match permit {
                    Some(permit) => Utf8Text::try_copy_str_exact_with(text, permit, |_| {
                        Ok::<(), crate::RuntimeError>(())
                    })
                    .map_err(|cause| exact_error(runtime, cause))?,
                    None => text.into(),
                })
                .into(),
            ),
        })
    }
}

/// The shared standard-substruct parser still owns unqualified path/component
/// Vecs. Refuse this dictionary-only ingress before that parser allocates.
/// Already typed dictionary reads/edits below remain finite-admissible.
pub(in crate::builtins::table) fn dictionary_protocol_ingress(
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    if runtime.is_some_and(RuntimeContext::is_cancelled) {
        return Err(allocation(runtime, HostAllocationError::Cancelled));
    }
    match HostAllocationReceipt::policy(runtime).map_err(|cause| allocation(runtime, cause))? {
        HostAllocationPolicy::Unbounded => Ok(()),
        HostAllocationPolicy::Managed => Err(host_allocation_runtime_error(
            runtime,
            "RunMat:dictionary:SubstructWorkspace",
            "dictionary: shared standard-substruct path/component workspace has no Managed owner",
            HostAllocationError::MissingManagedOwner,
        )),
    }
}

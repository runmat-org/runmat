//! Exact, invocation-owned character-vector component control text.
use super::super::{allocation_error, never_retry, Work};
use crate::builtins::datetime::datetime_error;
use crate::builtins::datetime::{
    datetime_utf16_error,
    text_parsing::{encode_utf16_chunk, utf16_chunk_end},
};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
};
use crate::context::RuntimeContext;
use crate::BuiltinResult;
use runmat_value::Value;

pub(in crate::builtins::datetime) struct PreparedControl<'a> {
    borrowed: Option<&'a str>,
    _receipt: Option<HostAllocationReceipt>,
    scratch: Option<HostTypedWorkspace<u8>>,
    encoded_len: usize,
}

impl<'a> PreparedControl<'a> {
    pub(in crate::builtins::datetime) async fn new(
        runtime: &RuntimeContext,
        work: &mut Work<'_>,
        value: Option<&'a Value>,
    ) -> BuiltinResult<Self> {
        match value {
            None => Ok(Self { borrowed: None, _receipt: None, scratch: None, encoded_len: 0 }),
            Some(Value::String(text)) => Ok(Self {
                borrowed: Some(crate::builtins::datetime::nonmissing_text(text, work.label).map_err(never_retry)?), _receipt: None, scratch: None, encoded_len: 0,
            }),
            Some(Value::StringArray(array)) if array.data().len() == 1 => Ok(Self {
                borrowed: Some(crate::builtins::datetime::nonmissing_text(&array.data()[0], work.label).map_err(never_retry)?),
                _receipt: None, scratch: None, encoded_len: 0,
            }),
            Some(Value::CharArray(array)) if array.rows == 1 => {
                array.validate_layout().map_err(|source| {
                    never_retry(datetime_error(format!("{}: {source}", work.label)))
                })?;
                let len = array.cols.checked_mul(4).map(|len| len.max(1)).ok_or_else(|| {
                    allocation_error(Some(runtime), work.label, HostAllocationError::Overflow)
                })?;
                let bytes = HostTypedWorkspace::<u8>::quoted_bytes(len)
                    .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
                let plan = HostAllocationPlan::new(bytes, 0)
                    .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
                let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
                    .map_err(|source| allocation_error(Some(runtime), work.label, source))?
                    .ok_or_else(|| allocation_error(Some(runtime), work.label, HostAllocationError::StaleAuthority))?;
                work.check()?;
                let mut scratch = receipt.temporary_typed::<u8>(runtime, len)
                    .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
                let mut end = 0;
                let mut source = 0;
                while source < array.data().len() {
                    let next = utf16_chunk_end(array.data(), source);
                    end += encode_utf16_chunk(&array.data()[source..next], source, &mut scratch.as_mut_slice()[end..])
                        .map_err(|cause| never_retry(datetime_utf16_error(cause)))?;
                    if next < array.data().len() {
                        work.cooperate(u64::try_from(next - source).unwrap_or(u64::MAX)).await?;
                        receipt.checkpoint(runtime)
                            .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
                    }
                    source = next;
                }
                work.check()?;
                receipt.checkpoint(runtime)
                    .map_err(|source| allocation_error(Some(runtime), work.label, source))?;
                Ok(Self { borrowed: None, _receipt: Some(receipt), scratch: Some(scratch), encoded_len: end })
            }
            Some(_) => Err(never_retry(datetime_error(format!(
                "datetime: {} component type or date format must be a string scalar or character vector",
                work.label,
            )))),
        }
    }

    pub(in crate::builtins::datetime) fn text(&self) -> Option<&str> {
        self.borrowed.or_else(|| {
            self.scratch.as_ref().map(|scratch| {
                std::str::from_utf8(&scratch.as_slice()[..self.encoded_len])
                    .expect("character control encoded into admitted UTF-8")
            })
        })
    }
}

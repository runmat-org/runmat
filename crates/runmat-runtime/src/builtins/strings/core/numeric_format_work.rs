//! Bounded, reusable numeric text staging for `int2str` and `num2str`.
//!
//! The backing belongs to the caller's admitted temporary workspace. Numeric
//! formatting must not create a fresh heap `String` for every matrix element.

use std::fmt::{self, Write};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use runmat_value::{CharArray, ExactArrayStorage, SharedShape, Utf16CodeUnit};

/// Empty numeric text has no cell-width workspace, even when the input has a
/// huge virtual zero axis. Its two-element shape remains an admitted owner.
pub(super) fn admitted_empty(
    runtime: &RuntimeContext,
    rows: usize,
) -> Result<CharArray, HostAllocationError> {
    let shape_bytes =
        SharedShape::requested_bytes_for_rank(2).map_err(HostAllocationError::Numeric)?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(shape_bytes).map_err(|_| HostAllocationError::Overflow)?,
        usize::from(shape_bytes != 0),
    )?;
    let receipt =
        HostAllocationReceipt::admit(Some(runtime), HostAllocationPlan::new(0, retained)?)?
            .ok_or(HostAllocationError::MissingManagedOwner)?;
    let permit = receipt.output_permit(runtime)?;
    let shape = SharedShape::new_admitted_from_slice(&[rows, 0], &permit)
        .map_err(HostAllocationError::Numeric)?;
    let payload = ExactArrayStorage::<Utf16CodeUnit>::try_begin(0, &permit)
        .map_err(HostAllocationError::Numeric)?
        .try_finish()
        .expect("zero-length exact builder is complete");
    let output = CharArray::from_admitted(payload, shape)
        .expect("admitted zero-length payload matches its zero-axis shape");
    receipt.checkpoint(runtime)?;
    Ok(output)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) struct FormatCapacityExceeded;

pub(super) struct FormatBuffer<'a> {
    bytes: &'a mut [u8],
    len: usize,
    exceeded: bool,
}

impl<'a> FormatBuffer<'a> {
    pub(super) fn new(bytes: &'a mut [u8]) -> Self {
        Self {
            bytes,
            len: 0,
            exceeded: false,
        }
    }

    pub(super) fn clear(&mut self) {
        self.len = 0;
        self.exceeded = false;
    }

    pub(super) fn as_str(&self) -> &str {
        // Every writer entrypoint appends an entire UTF-8 str or an ASCII byte.
        std::str::from_utf8(&self.bytes[..self.len]).expect("format buffer writes UTF-8")
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn truncate(&mut self, len: usize) {
        assert!(len <= self.len && self.as_str().is_char_boundary(len));
        self.len = len;
    }

    pub(super) fn remove_range(&mut self, start: usize, end: usize) {
        assert!(start <= end && end <= self.len);
        assert!(self.as_str().is_char_boundary(start) && self.as_str().is_char_boundary(end));
        self.bytes.copy_within(end..self.len, start);
        self.len -= end - start;
    }

    pub(super) fn upper_ascii(&mut self) {
        self.bytes[..self.len].make_ascii_uppercase();
    }

    pub(super) fn swap_ascii(&mut self, left: usize, right: usize) {
        assert!(left < self.len && right < self.len);
        self.bytes.swap(left, right);
    }

    pub(super) fn append(&mut self, value: &str) -> Result<(), FormatCapacityExceeded> {
        let end = self
            .len
            .checked_add(value.len())
            .filter(|end| *end <= self.bytes.len())
            .ok_or(FormatCapacityExceeded)?;
        self.bytes[self.len..end].copy_from_slice(value.as_bytes());
        self.len = end;
        Ok(())
    }

    pub(super) fn ascii(&mut self, byte: u8) -> Result<(), FormatCapacityExceeded> {
        debug_assert!(byte.is_ascii());
        if self.len == self.bytes.len() {
            return Err(FormatCapacityExceeded);
        }
        self.bytes[self.len] = byte;
        self.len += 1;
        Ok(())
    }

    pub(super) fn repeat_ascii(
        &mut self,
        byte: u8,
        count: usize,
    ) -> Result<(), FormatCapacityExceeded> {
        debug_assert!(byte.is_ascii());
        let end = self
            .len
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(FormatCapacityExceeded)?;
        self.bytes[self.len..end].fill(byte);
        self.len = end;
        Ok(())
    }

    pub(super) fn insert_ascii(
        &mut self,
        index: usize,
        byte: u8,
    ) -> Result<(), FormatCapacityExceeded> {
        debug_assert!(byte.is_ascii());
        if index > self.len || self.len == self.bytes.len() {
            return Err(FormatCapacityExceeded);
        }
        self.bytes.copy_within(index..self.len, index + 1);
        self.bytes[index] = byte;
        self.len += 1;
        Ok(())
    }

    pub(super) fn pad_left(
        &mut self,
        count: usize,
        byte: u8,
    ) -> Result<(), FormatCapacityExceeded> {
        let end = self
            .len
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(FormatCapacityExceeded)?;
        self.bytes.copy_within(0..self.len, count);
        self.bytes[..count].fill(byte);
        self.len = end;
        Ok(())
    }

    pub(super) fn checked_format(
        &mut self,
        args: fmt::Arguments<'_>,
    ) -> Result<(), FormatCapacityExceeded> {
        fmt::write(self, args).map_err(|_| FormatCapacityExceeded)?;
        if self.exceeded {
            return Err(FormatCapacityExceeded);
        }
        Ok(())
    }
}

impl Write for FormatBuffer<'_> {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        self.append(value).map_err(|_| {
            self.exceeded = true;
            fmt::Error
        })
    }
}

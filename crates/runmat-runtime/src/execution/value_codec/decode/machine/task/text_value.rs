use super::super::*;
use super::text::{ShapeCopy, TextCopy};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use runmat_value::{Utf16CodeUnit, Utf8Text, Utf8TextExactCopy};
use std::num::NonZeroUsize;

#[path = "text_value/character.rs"]
mod character;
use character::ManagedCharacter;
#[path = "text_value/strings.rs"]
mod strings;
use strings::Strings;

pub(in crate::execution::value_codec::decode::machine) enum TextValueTask<'a> {
    String {
        text: TextCopy<'a>,
        path: ControlPath<'a>,
    },
    ManagedString {
        text: Option<Utf8TextExactCopy<'a>>,
        receipt: HostAllocationReceipt,
        runtime: &'a RuntimeContext,
        path: ControlPath<'a>,
    },
    Character {
        shape: ShapeCopy<'a>,
        source: &'a [u16],
        code_units: Vec<Utf16CodeUnit>,
        path: ControlPath<'a>,
    },
    ManagedCharacter {
        owner: ControlBox<ManagedCharacter<'a>>,
        path: ControlPath<'a>,
    },
    StringArray {
        shape: ShapeCopy<'a>,
        source: &'a [runmat_execution::value::StringElement],
        values: Vec<runmat_value::StringElement>,
        current: Option<TextCopy<'a>>,
        path: ControlPath<'a>,
    },
    ManagedStrings {
        owner: ControlBox<Strings<'a>>,
        path: ControlPath<'a>,
    },
}

impl<'a> TextValueTask<'a> {
    pub(super) fn string(
        source: &'a str,
        path: ControlPath<'a>,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        let policy = HostAllocationReceipt::policy(runtime)
            .map_err(|error| ValueCodecError::allocation(&path, error))?;
        if policy == HostAllocationPolicy::Unbounded {
            return Ok(Self::String {
                text: TextCopy::new(source),
                path,
            });
        }
        let runtime = runtime.expect("managed policy has a runtime");
        let raw = Utf8Text::requested_bytes_for_capacity(source.len()).map_err(|error| {
            ValueCodecError::allocation(&path, HostAllocationError::Numeric(error))
        })?;
        let raw = u64::try_from(raw)
            .map_err(|_| ValueCodecError::allocation(&path, HostAllocationError::Overflow))?;
        let retained =
            HostAllocationPlan::retained_exact_output_bytes(raw, usize::from(!source.is_empty()))
                .map_err(|error| ValueCodecError::allocation(&path, error))?;
        let plan = HostAllocationPlan::new(0, retained)
            .map_err(|error| ValueCodecError::allocation(&path, error))?;
        let receipt = HostAllocationReceipt::admit(Some(runtime), plan)
            .map_err(|error| ValueCodecError::allocation(&path, error))?
            .ok_or_else(|| {
                ValueCodecError::allocation(&path, HostAllocationError::StaleAuthority)
            })?;
        let permit = receipt
            .output_permit(runtime)
            .map_err(|error| ValueCodecError::allocation(&path, error))?;
        let text = Utf8Text::try_begin_copy_exact(source, &permit).map_err(|error| {
            ValueCodecError::allocation(
                &path,
                receipt
                    .checkpoint(runtime)
                    .err()
                    .unwrap_or(HostAllocationError::Numeric(error)),
            )
        })?;
        receipt
            .checkpoint(runtime)
            .map_err(|error| ValueCodecError::allocation(&path, error))?;
        Ok(Self::ManagedString {
            text: Some(text),
            receipt,
            runtime,
            path,
        })
    }

    pub(super) fn character(
        shape: &'a [u64],
        source: &'a [u16],
        path: ControlPath<'a>,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        if let Some(owner) = ManagedCharacter::new(shape, source, runtime, &path)? {
            let owner = path.owner().boxed(owner, &path)?;
            return Ok(Self::ManagedCharacter { owner, path });
        }
        Ok(Self::Character {
            shape: ShapeCopy::new(shape, path.try_clone()?),
            source,
            code_units: Vec::with_capacity(source.len()),
            path,
        })
    }

    pub(super) fn string_array(
        shape: &'a [u64],
        source: &'a [runmat_execution::value::StringElement],
        path: ControlPath<'a>,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        if let Some(owner) = Strings::new(shape, source, runtime, &path)? {
            let owner = path.owner().boxed(owner, &path)?;
            return Ok(Self::ManagedStrings { owner, path });
        }
        Ok(Self::StringArray {
            shape: ShapeCopy::new(shape, path.try_clone()?),
            source,
            values: Vec::with_capacity(source.len()),
            current: None,
            path,
        })
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        match self {
            Self::String { text, path: _ } => {
                let completed = text.advance(maximum_work);
                if !text.is_complete() {
                    return Ok((completed, None));
                }
                let text = std::mem::replace(text, TextCopy::new("")).finish();
                Ok((completed, Some(Value::string(text))))
            }
            Self::ManagedString {
                text,
                receipt,
                runtime,
                path,
            } => {
                let copy = text.as_mut().expect("active managed UTF-8 copy");
                receipt
                    .checkpoint(runtime)
                    .map_err(|error| ValueCodecError::allocation(path, error))?;
                let completed =
                    copy.advance(NonZeroUsize::new(maximum_work).expect("decode work is nonzero"));
                receipt
                    .checkpoint(runtime)
                    .map_err(|error| ValueCodecError::allocation(path, error))?;
                if !copy.is_complete() {
                    return Ok((completed.max(1), None));
                }
                let text = text
                    .take()
                    .expect("completed managed UTF-8 copy")
                    .try_finish()
                    .unwrap_or_else(|_| unreachable!("completed UTF-8 copy"));
                Ok((completed.max(1), Some(Value::string(text))))
            }
            Self::ManagedCharacter { owner, path } => owner.advance(maximum_work, path),
            Self::ManagedStrings { owner, path } => owner.advance(maximum_work, path),
            Self::Character {
                shape,
                source,
                code_units,
                path,
            } => {
                if !shape.is_complete() {
                    let completed = shape.advance(maximum_work)?;
                    return Ok((completed, None));
                }
                let count = (source.len() - code_units.len()).min(maximum_work);
                for unit in &source[code_units.len()..code_units.len() + count] {
                    code_units.push(Utf16CodeUnit::from_raw(*unit));
                }
                if code_units.len() != source.len() {
                    return Ok((count.max(1), None));
                }
                let shape = shape.take_finished();
                let code_units = std::mem::take(code_units);
                let value = CharArray::from_code_units_with_shape(code_units, shape)
                    .map(Value::CharArray)
                    .map_err(|error| ValueCodecError::invalid(path, error))?;
                Ok((count.max(1), Some(value)))
            }
            Self::StringArray {
                shape,
                source,
                values,
                current,
                path,
            } => {
                if !shape.is_complete() {
                    let completed = shape.advance(maximum_work)?;
                    return Ok((completed, None));
                }
                if values.len() == source.len() {
                    let shape = shape.take_finished();
                    let values = std::mem::take(values);
                    let value = Value::StringArray(
                        StringArray::from_elements_unbounded(values, shape)
                            .map_err(|error| ValueCodecError::invalid(path, error))?,
                    );
                    return Ok((1, Some(value)));
                }
                let Some(text) = source[values.len()].as_str() else {
                    values.push(runmat_value::StringElement::Missing);
                    return Ok((1, None));
                };
                let copy = current.get_or_insert_with(|| TextCopy::new(text));
                let completed = copy.advance(maximum_work);
                if copy.is_complete() {
                    values.push(current.take().expect("active text copy").finish().into());
                }
                Ok((completed, None))
            }
        }
    }

    pub(super) fn path(&self) -> &ControlPath<'a> {
        match self {
            Self::String { path, .. }
            | Self::ManagedString { path, .. }
            | Self::Character { path, .. }
            | Self::ManagedCharacter { path, .. }
            | Self::ManagedStrings { path, .. }
            | Self::StringArray { path, .. } => path,
        }
    }
}

//! Inline metadata slots bounded by the existing numeric program validator.
use crate::{NativeExecutorError, NativeExecutorResult};
use runmat_runtime::native::NativeValueRef;
use runmat_runtime::numeric_region::{MAX_NODES, MAX_OUTPUTS};
use runmat_value::Value;

#[derive(Clone, Debug)]
pub(crate) struct NumericRegionReferences {
    slots: [NativeValueRef; MAX_NODES],
    len: usize,
}

impl NumericRegionReferences {
    pub(crate) fn try_from_iter(
        values: impl IntoIterator<Item = NativeExecutorResult<NativeValueRef>>,
    ) -> NativeExecutorResult<Self> {
        let mut result = Self {
            slots: [NativeValueRef::NULL; MAX_NODES],
            len: 0,
        };
        for value in values {
            let slot = result.slots.get_mut(result.len).ok_or_else(bounds_error)?;
            *slot = value?;
            result.len += 1;
        }
        Ok(result)
    }
}

impl std::ops::Deref for NumericRegionReferences {
    type Target = [NativeValueRef];
    fn deref(&self) -> &Self::Target {
        &self.slots[..self.len]
    }
}

#[derive(Debug)]
pub(crate) struct NumericRegionInputViews<'a> {
    slots: Option<[&'a Value; MAX_NODES]>,
    len: usize,
}

impl<'a> NumericRegionInputViews<'a> {
    pub(crate) fn try_from_iter(
        values: impl IntoIterator<Item = NativeExecutorResult<&'a Value>>,
    ) -> NativeExecutorResult<Self> {
        let mut values = values.into_iter();
        let Some(first) = values.next().transpose()? else {
            return Ok(Self {
                slots: None,
                len: 0,
            });
        };
        let mut result = Self {
            slots: Some([first; MAX_NODES]),
            len: 1,
        };
        for value in values {
            let slot = result
                .slots
                .as_mut()
                .expect("nonempty views")
                .get_mut(result.len)
                .ok_or_else(bounds_error)?;
            *slot = value?;
            result.len += 1;
        }
        Ok(result)
    }
}

impl<'a> std::ops::Deref for NumericRegionInputViews<'a> {
    type Target = [&'a Value];
    fn deref(&self) -> &Self::Target {
        self.slots.as_ref().map_or(&[], |slots| &slots[..self.len])
    }
}

fn bounds_error() -> NativeExecutorError {
    NativeExecutorError::Host("numeric region metadata exceeds validated program limits".into())
}

/// Source-level inline metadata only; not a claim about compiler frame spills.
pub(crate) const NUMERIC_REGION_INLINE_METADATA_BYTES: usize =
    std::mem::size_of::<NumericRegionReferences>()
        + std::mem::size_of::<NumericRegionInputViews<'static>>()
        + std::mem::size_of::<[NativeValueRef; MAX_OUTPUTS]>();
const _: () = assert!(NUMERIC_REGION_INLINE_METADATA_BYTES <= 8192);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_metadata_uses_existing_validator_limits_without_heap_backing() {
        let refs = NumericRegionReferences::try_from_iter(
            (0..MAX_NODES).map(|_| Ok(NativeValueRef::NULL)),
        )
        .unwrap();
        assert_eq!(refs.len(), MAX_NODES);
        assert!(NumericRegionReferences::try_from_iter(
            (0..=MAX_NODES).map(|_| Ok(NativeValueRef::NULL))
        )
        .is_err());
        let value = Value::Num(42.0);
        let views =
            NumericRegionInputViews::try_from_iter((0..MAX_NODES).map(|_| Ok(&value))).unwrap();
        assert_eq!(views.len(), MAX_NODES);
        assert!(
            NumericRegionInputViews::try_from_iter((0..=MAX_NODES).map(|_| Ok(&value))).is_err()
        );
    }
}

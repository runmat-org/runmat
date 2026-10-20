//! Allocation mode for decoder aggregate destinations. Exact builders own the
//! final backing; ordinary builders exist only after canonical policy admission.
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use crate::execution::value_codec::{object_output::ObjectText, ValueCodecError};
use runmat_value::{ExactArrayStorage, ExactArrayStorageBuilder, Utf8Text};

#[derive(Default)]
pub(super) struct Quote {
    bytes: usize,
    claims: usize,
}
impl Quote {
    pub(super) fn add(
        &mut self,
        bytes: usize,
        claims: usize,
        path: &str,
    ) -> Result<(), ValueCodecError> {
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| ValueCodecError::allocation(path, HostAllocationError::Overflow))?;
        self.claims = self
            .claims
            .checked_add(claims)
            .ok_or_else(|| ValueCodecError::allocation(path, HostAllocationError::Overflow))?;
        Ok(())
    }
    pub(super) fn text(&mut self, source: &str, path: &str) -> Result<(), ValueCodecError> {
        let bytes = Utf8Text::requested_bytes_for_capacity(source.len())
            .map_err(|e| ValueCodecError::allocation(path, e.into()))?;
        self.add(bytes, usize::from(!source.is_empty()), path)
    }
    pub(super) fn array<T>(&mut self, count: usize, path: &str) -> Result<(), ValueCodecError> {
        let bytes = ExactArrayStorage::<T>::requested_bytes(count)
            .map_err(|e| ValueCodecError::allocation(path, e.into()))?;
        self.add(bytes, usize::from(count != 0), path)
    }
}

pub(super) struct Admission<'a> {
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}
impl<'a> Admission<'a> {
    pub(super) fn is_managed(&self) -> bool {
        self.receipt.is_some()
    }
    pub(super) fn new(
        quote: Quote,
        runtime: Option<&'a RuntimeContext>,
        path: &str,
    ) -> Result<Self, ValueCodecError> {
        let error = |e| ValueCodecError::allocation(path, e);
        let bytes = u64::try_from(quote.bytes).map_err(|_| error(HostAllocationError::Overflow))?;
        let retained =
            HostAllocationPlan::retained_exact_output_bytes(bytes, quote.claims).map_err(error)?;
        let plan = HostAllocationPlan::new(0, retained).map_err(error)?;
        let receipt = HostAllocationReceipt::admit(runtime, plan).map_err(error)?;
        Ok(Self { receipt, runtime })
    }
    pub(super) fn checkpoint(&self, path: &str) -> Result<(), ValueCodecError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed aggregate runtime"))
                .map_err(|e| ValueCodecError::allocation(path, e))?;
        }
        if self.runtime.is_some_and(RuntimeContext::is_cancelled) {
            return Err(ValueCodecError::allocation(
                path,
                HostAllocationError::Cancelled,
            ));
        }
        Ok(())
    }
    pub(super) fn construct<T>(
        &self,
        path: &str,
        build: impl FnOnce(
            Option<&dyn runmat_value::ExactHostAllocationPermit>,
        ) -> Result<T, ValueCodecError>,
    ) -> Result<T, ValueCodecError> {
        self.checkpoint(path)?;
        let output = if let Some(receipt) = &self.receipt {
            let permit = receipt
                .output_permit(self.runtime.expect("managed aggregate runtime"))
                .map_err(|e| ValueCodecError::allocation(path, e))?;
            build(Some(&permit))
        } else {
            build(None)
        };
        self.checkpoint(path)?;
        output
    }
    pub(super) fn text<'s>(
        &self,
        source: &'s str,
        path: &str,
    ) -> Result<ObjectText<'s>, ValueCodecError> {
        self.construct(path, |permit| match permit {
            Some(permit) => Utf8Text::try_begin_copy_exact(source, permit)
                .map(ObjectText::Exact)
                .map_err(|e| ValueCodecError::allocation(path, e.into())),
            None => Ok(ObjectText::Ordinary(super::text::TextCopy::new(source))),
        })
    }
    pub(super) fn array<T>(&self, count: usize, path: &str) -> Result<Array<T>, ValueCodecError> {
        self.construct(path, |permit| match permit {
            Some(permit) => ExactArrayStorage::try_begin(count, permit)
                .map(Array::Exact)
                .map_err(|e| ValueCodecError::allocation(path, e.into())),
            None => Ok(Array::Ordinary(Vec::with_capacity(count))),
        })
    }
}

pub(super) enum Array<T> {
    Exact(ExactArrayStorageBuilder<T>),
    Ordinary(Vec<T>),
}
impl<T> Array<T> {
    pub(super) fn push(&mut self, value: T) {
        match self {
            Self::Exact(builder) => assert!(
                builder.try_push(value).is_ok(),
                "decoded child count was reserved"
            ),
            Self::Ordinary(values) => values.push(value),
        }
    }
    pub(super) fn finish(self) -> ExactArrayStorage<T> {
        match self {
            Self::Exact(builder) => match builder.try_finish() {
                Ok(values) => values,
                Err(_) => panic!("decoded child count matches destination"),
            },
            Self::Ordinary(values) => ExactArrayStorage::shared_unbounded(values),
        }
    }
}

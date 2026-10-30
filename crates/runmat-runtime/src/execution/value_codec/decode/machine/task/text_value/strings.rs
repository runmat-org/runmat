//! String-array decoding writes directly into its final typed slot owner.
use super::super::aggregate_output::{Admission, Array, Quote};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPolicy, HostAllocationReceipt},
    RuntimeContext,
};
use crate::execution::value_codec::{object_output::ObjectText, ValueCodecError};
use runmat_value::{SharedShape, SharedShapeBuilder, StringArray, StringElement, Value};

pub(in crate::execution::value_codec::decode::machine) struct Strings<'a> {
    shape: Option<SharedShapeBuilder>,
    final_shape: Option<SharedShape>,
    values: Option<Array<StringElement>>,
    copy: Option<ObjectText<'a>>,
    source_shape: &'a [u64],
    source: &'a [runmat_execution::value::StringElement],
    offset: usize,
    admission: Admission<'a>,
}

impl<'a> Strings<'a> {
    pub(super) fn new(
        source_shape: &'a [u64],
        source: &'a [runmat_execution::value::StringElement],
        runtime: Option<&'a RuntimeContext>,
        path: &str,
    ) -> Result<Option<Self>, ValueCodecError> {
        if HostAllocationReceipt::policy(runtime)
            .map_err(|error| ValueCodecError::allocation(path, error))?
            == HostAllocationPolicy::Unbounded
        {
            return Ok(None);
        }
        let mut quote = Quote::default();
        quote.array::<StringElement>(source.len(), path)?;
        quote.add(
            SharedShape::requested_bytes_for_rank(source_shape.len())
                .map_err(|error| ValueCodecError::allocation(path, error.into()))?,
            1,
            path,
        )?;
        for (index, element) in source.iter().enumerate() {
            if index.is_multiple_of(4096) && runtime.is_some_and(RuntimeContext::is_cancelled) {
                return Err(ValueCodecError::allocation(
                    path,
                    HostAllocationError::Cancelled,
                ));
            }
            if let Some(text) = element.as_str() {
                quote.text(text, path)?;
            }
        }
        let admission = Admission::new(quote, runtime, path)?;
        let values = Some(admission.array(source.len(), path)?);
        let shape = Some(admission.construct(path, |permit| {
            match permit {
                Some(permit) => SharedShape::begin_exact(source_shape.len(), permit)
                    .map_err(|error| ValueCodecError::allocation(path, error.into())),
                None => Err(ValueCodecError::allocation(
                    path,
                    HostAllocationError::StaleAuthority,
                )),
            }
        })?);
        Ok(Some(Self {
            shape,
            final_shape: None,
            values,
            copy: None,
            source_shape,
            source,
            offset: 0,
            admission,
        }))
    }

    pub(super) fn advance(
        &mut self,
        work: usize,
        path: &str,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        self.admission.checkpoint(path)?;
        if let Some(shape) = &mut self.shape {
            let start = shape.len();
            let end = self.source_shape.len().min(start.saturating_add(work));
            for &extent in &self.source_shape[start..end] {
                shape
                    .try_push(usize::try_from(extent).map_err(|_| {
                        ValueCodecError::invalid(path, "value exceeds platform limits")
                    })?)
                    .map_err(|_| {
                        ValueCodecError::invalid(path, "string shape exceeds admitted rank")
                    })?;
            }
            if end == self.source_shape.len() {
                self.final_shape = Some(
                    self.shape
                        .take()
                        .expect("active shape")
                        .try_finish()
                        .unwrap_or_else(|_| unreachable!("complete shape")),
                );
            }
            self.admission.checkpoint(path)?;
            return Ok(((end - start).max(1), None));
        }
        if let Some(element) = self.source.get(self.offset) {
            if let Some(text) = element.as_str() {
                if self.copy.is_none() {
                    self.copy = Some(self.admission.text(text, path)?);
                }
                let copy = self.copy.as_mut().expect("active string copy");
                let completed = copy.advance(work);
                if copy.is_complete() {
                    self.values
                        .as_mut()
                        .expect("active slots")
                        .push(StringElement::Text(
                            self.copy.take().expect("complete string copy").finish(),
                        ));
                    self.offset += 1;
                }
                self.admission.checkpoint(path)?;
                return Ok((completed.max(1), None));
            }
            self.values
                .as_mut()
                .expect("active slots")
                .push(StringElement::Missing);
            self.offset += 1;
            self.admission.checkpoint(path)?;
            return Ok((1, None));
        }
        let values = self.values.take().expect("complete slots").finish();
        let shape = self.final_shape.take().expect("complete shape");
        let array = if self.admission.is_managed() {
            StringArray::from_admitted(values, shape)
        } else {
            StringArray::from_shared_storage(values, shape)
        }
        .map_err(|error| ValueCodecError::invalid(path, error))?;
        self.admission.checkpoint(path)?;
        Ok((1, Some(Value::StringArray(array))))
    }
}

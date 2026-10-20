//! One canonical Struct generation. Managed keys copy incrementally into
//! final retained UTF-8 owners; no temporary String or child Value clone.
//! Fixed-index initialization and lookup hashing are still synchronous work.
use super::super::*;
use super::text::TextCopy;
use crate::context::{host_allocation::HostAllocationReceipt, RuntimeContext};
use runmat_execution::value::StructField;
use runmat_value::{StructValueBuilder, UnboundedStructValueBuilder, Utf8Text, Utf8TextExactCopy};

enum Output {
    Exact(StructValueBuilder),
    Ordinary(UnboundedStructValueBuilder),
}
enum Name<'a> {
    Exact(Utf8TextExactCopy<'a>),
    Ordinary(TextCopy<'a>),
}
pub(in crate::execution::value_codec::decode::machine) struct StructTask<'a> {
    // Partial payloads retire before the original rooted children and receipt.
    output: Option<Output>,
    name: Option<Name<'a>>,
    children: TakenValues<'a>,
    fields: &'a [StructField],
    index: usize,
    path: ControlPath<'a>,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
}
impl<'a> StructTask<'a> {
    pub(super) fn new(
        fields: &'a [StructField],
        children: TakenValues<'a>,
        path: ControlPath<'a>,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        let receipt = crate::execution::structure::admit_struct_generation(
            runtime,
            fields.iter().map(|field| field.name.as_str()),
        )
        .map_err(|source| ValueCodecError::allocation(&path, source))?;
        let output = if let Some(receipt) = &receipt {
            let runtime = runtime.expect("managed receipt has runtime");
            let permit = receipt
                .output_permit(runtime)
                .map_err(|source| ValueCodecError::allocation(&path, source))?;
            let builder = StructValueBuilder::try_new(fields.len(), &permit);
            receipt
                .checkpoint(runtime)
                .map_err(|source| ValueCodecError::allocation(&path, source))?;
            Output::Exact(
                builder.map_err(|source| ValueCodecError::allocation(&path, source.into()))?,
            )
        } else {
            Output::Ordinary(UnboundedStructValueBuilder::with_capacity(fields.len()))
        };
        Ok(Self {
            output: Some(output),
            name: None,
            children,
            fields,
            index: 0,
            path,
            receipt,
            runtime,
        })
    }
    fn checkpoint(&self) -> Result<(), ValueCodecError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed runtime"))
                .map_err(|source| ValueCodecError::allocation(&self.path, source))?;
        }
        Ok(())
    }
    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, bool), ValueCodecError> {
        self.checkpoint()?;
        if self.index == self.fields.len() {
            return Ok((1, true));
        }
        if self.name.is_none() {
            let source = self.fields[self.index].name.as_str();
            self.name = Some(if let Some(receipt) = &self.receipt {
                let permit = receipt
                    .output_permit(self.runtime.expect("managed runtime"))
                    .map_err(|source| ValueCodecError::allocation(&self.path, source))?;
                let copy = Utf8Text::try_begin_copy_exact(source, &permit);
                self.checkpoint()?;
                Name::Exact(
                    copy.map_err(|source| ValueCodecError::allocation(&self.path, source.into()))?,
                )
            } else {
                Name::Ordinary(TextCopy::new(source))
            });
        }
        let (completed, complete) = match self.name.as_mut().expect("active field name") {
            Name::Exact(copy) => (
                copy.advance(NonZeroUsize::new(maximum_work.max(1)).unwrap())
                    .max(1),
                copy.is_complete(),
            ),
            Name::Ordinary(copy) => (copy.advance(maximum_work), copy.is_complete()),
        };
        if complete {
            let key = match self.name.take().expect("complete field name") {
                Name::Exact(copy) => copy.try_finish().expect("completed UTF-8 field name"),
                Name::Ordinary(copy) => Utf8Text::new(copy.finish()),
            };
            let value = self.children.next()?.expect("one decoded value per field");
            match self.output.as_mut().expect("unpublished structure") {
                Output::Exact(builder) => {
                    builder.try_insert_prepared(key, value).map_err(|_| {
                        ValueCodecError::invalid(
                            &self.path,
                            "quoted structure rejected an admitted field",
                        )
                    })?;
                }
                Output::Ordinary(builder) => {
                    builder.insert_text(key, value);
                }
            }
            self.index += 1;
        }
        self.checkpoint()?;
        Ok((completed, self.index == self.fields.len()))
    }
    pub(super) fn finish(&mut self) -> Value {
        Value::Struct(match self.output.take().expect("unpublished structure") {
            Output::Exact(builder) => builder.finish(),
            Output::Ordinary(builder) => builder.finish(),
        })
    }
    pub(super) fn take_children(&mut self) -> TakenValues<'a> {
        std::mem::replace(&mut self.children, DecodeMachine::empty_taken_values())
    }
}

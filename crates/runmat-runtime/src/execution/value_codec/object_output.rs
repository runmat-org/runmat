//! Decoder destinations for one original Object generation. Decoded children
//! remain rooted by TakenValues until final publication by the machine.
use super::ValueCodecError;
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use runmat_execution::value::DurableDynamicProperty;
use runmat_value::{
    DynamicPropertyDef, ExactObjectBuilder, PreparedObjectClassName, SharedClassIdentity,
    UnboundedObjectBuilder, Utf8Text, Utf8TextExactCopy,
};
use runmat_value::{ObjectInstance, Value};
use std::num::NonZeroUsize;

enum Builder {
    Exact(ExactObjectBuilder),
    Ordinary(UnboundedObjectBuilder),
}

pub(super) enum ObjectText<'a> {
    Exact(Utf8TextExactCopy<'a>),
    Ordinary(super::text_copy::TextCopy<'a>),
}
impl ObjectText<'_> {
    pub(super) fn advance(&mut self, work: usize) -> usize {
        match self {
            Self::Exact(copy) => copy.advance(NonZeroUsize::new(work.max(1)).unwrap()).max(1),
            Self::Ordinary(copy) => copy.advance(work),
        }
    }
    pub(super) fn is_complete(&self) -> bool {
        match self {
            Self::Exact(copy) => copy.is_complete(),
            Self::Ordinary(copy) => copy.is_complete(),
        }
    }
    pub(super) fn finish(self) -> Utf8Text {
        match self {
            Self::Exact(copy) => copy.try_finish().expect("completed decoder text"),
            Self::Ordinary(copy) => copy.finish().into(),
        }
    }
}

pub(super) struct ObjectOutput<'a> {
    builder: Builder,
    receipt: Option<HostAllocationReceipt>,
    runtime: Option<&'a RuntimeContext>,
    selected: Option<std::rc::Rc<runmat_value::SelectedObjectClassFacts>>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum DynamicMetadataCopy {
    Incremental,
    Synchronous,
}
impl<'a> ObjectOutput<'a> {
    pub(super) fn new<'n>(
        class: &str,
        names: impl ExactSizeIterator<Item = &'n str> + Clone,
        dynamic: &[DurableDynamicProperty],
        copy: DynamicMetadataCopy,
        runtime: Option<&'a RuntimeContext>,
        path: &str,
    ) -> Result<Self, ValueCodecError> {
        Self::build(class, names, dynamic, copy, runtime, path, false)
    }

    pub(super) fn new_numeric_factor<'n>(
        class: &str,
        names: impl ExactSizeIterator<Item = &'n str> + Clone,
        dynamic: &[DurableDynamicProperty],
        runtime: Option<&'a RuntimeContext>,
        path: &str,
    ) -> Result<Self, ValueCodecError> {
        Self::build(
            class,
            names,
            dynamic,
            DynamicMetadataCopy::Incremental,
            runtime,
            path,
            true,
        )
    }

    fn build<'n>(
        class: &str,
        names: impl ExactSizeIterator<Item = &'n str> + Clone,
        dynamic: &[DurableDynamicProperty],
        copy: DynamicMetadataCopy,
        runtime: Option<&'a RuntimeContext>,
        path: &str,
        numeric_factor: bool,
    ) -> Result<Self, ValueCodecError> {
        let allocation = |e| ValueCodecError::allocation(path, e);
        let quote =
            ExactObjectBuilder::quote_with_dynamic_capacity(class, names.len(), dynamic.len())
                .map_err(|e| allocation(HostAllocationError::Numeric(e)))?;
        let mut bytes = quote.requested_bytes;
        let mut claims = quote.retained_claims;
        if numeric_factor {
            bytes = bytes
                .checked_add(ExactObjectBuilder::numeric_factor_native_requested_bytes())
                .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
            claims = claims
                .checked_add(ExactObjectBuilder::numeric_factor_native_retained_claim_count())
                .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
        }
        let mut text = |value: &str| -> Result<(), ValueCodecError> {
            bytes = bytes
                .checked_add(
                    Utf8Text::requested_bytes_for_capacity(value.len())
                        .map_err(|e| allocation(e.into()))?,
                )
                .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
            claims = claims
                .checked_add(usize::from(!value.is_empty()))
                .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
            Ok(())
        };
        for name in names.clone() {
            text(name)?;
        }
        for value in dynamic {
            text(&value.name)?;
            text(&value.definition_name)?;
            // This text is incremental decoder scratch, independently funded
            // from the final SharedClassIdentity prepared below.
            if copy == DynamicMetadataCopy::Incremental {
                text(value.defining_class.display_name())?;
            }
            text(&value.description)?;
        }
        for value in dynamic {
            bytes = bytes
                .checked_add(
                    SharedClassIdentity::requested_bytes_for_capacity(
                        value.defining_class.display_name().len(),
                    )
                    .map_err(|e| allocation(e.into()))?,
                )
                .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
            claims = claims
                .checked_add(1)
                .ok_or_else(|| allocation(HostAllocationError::Overflow))?;
        }
        let retained = HostAllocationPlan::retained_exact_output_bytes(
            u64::try_from(bytes).map_err(|_| allocation(HostAllocationError::Overflow))?,
            claims,
        )
        .map_err(allocation)?;
        let receipt = HostAllocationReceipt::admit(
            runtime,
            HostAllocationPlan::new(0, retained).map_err(allocation)?,
        )
        .map_err(allocation)?;
        let builder = if let Some(receipt) = &receipt {
            let runtime = runtime.expect("managed object receipt has runtime");
            let permit = receipt.output_permit(runtime).map_err(allocation)?;
            let class = PreparedObjectClassName::prepare(class, &permit);
            receipt.checkpoint(runtime).map_err(allocation)?;
            let class = class.map_err(|e| class_error(path, e))?;
            let builder = ExactObjectBuilder::try_new_with_dynamic_capacity(
                class,
                names.len(),
                dynamic.len(),
                &permit,
            );
            receipt.checkpoint(runtime).map_err(allocation)?;
            Builder::Exact(builder.map_err(|e| allocation(e.into()))?)
        } else {
            Builder::Ordinary(UnboundedObjectBuilder::with_property_capacity(
                class,
                names.len(),
            ))
        };
        Ok(Self {
            builder,
            receipt,
            runtime,
            selected: None,
        })
    }
    pub(super) fn selected_class(
        &mut self,
        facts: std::rc::Rc<runmat_value::SelectedObjectClassFacts>,
        path: &str,
    ) -> Result<(), ValueCodecError> {
        self.checkpoint(path)?;
        match &mut self.builder {
            Builder::Exact(builder) => {
                if !builder.try_set_selected_class(facts) {
                    return Err(ValueCodecError::invalid(
                        path,
                        "selected class differs from decoded object",
                    ));
                }
            }
            Builder::Ordinary(_) => self.selected = Some(facts),
        }
        Ok(())
    }
    pub(super) fn copy_text(&self, source: &str, path: &str) -> Result<Utf8Text, ValueCodecError> {
        let mut text = self.text(source, path)?;
        text.advance(usize::MAX);
        self.checkpoint(path)?;
        Ok(text.finish())
    }
    pub(super) fn metadata_handle(
        &mut self,
        name: &str,
        handle: runmat_gc::GcHandle,
        path: &str,
    ) -> Result<(), ValueCodecError> {
        self.checkpoint(path)?;
        let found = match &mut self.builder {
            Builder::Exact(builder) => builder.try_set_dynamic_metadata_handle(name, handle),
            Builder::Ordinary(builder) => match builder.dynamic_property_mut(name) {
                Some(definition) => {
                    definition.metadata_handle = Some(handle);
                    true
                }
                None => false,
            },
        };
        if !found {
            return Err(ValueCodecError::invalid(
                path,
                "selected metadata property is absent",
            ));
        }
        Ok(())
    }
    pub(super) fn insert_dynamic_source(
        &mut self,
        source: &DurableDynamicProperty,
        metadata: Option<runmat_gc::GcHandle>,
        path: &str,
    ) -> Result<bool, ValueCodecError> {
        let key = self.copy_text(&source.name, path)?;
        let definition = DynamicPropertyDef {
            name: self.copy_text(&source.definition_name, path)?,
            defining_class: self.class(source.defining_class.display_name(), path)?,
            description: self.copy_text(&source.description, path)?,
            metadata_handle: metadata,
            get_access: source.get_access,
            set_access: source.set_access,
            dependent: source.dependent,
            hidden: source.hidden,
            transient: source.transient,
            non_copyable: source.non_copyable,
            abort_set: source.abort_set,
            set_observable: source.set_observable,
            get_observable: source.get_observable,
        };
        self.insert_dynamic(key, definition, path)
    }
    pub(super) fn checkpoint(&self, path: &str) -> Result<(), ValueCodecError> {
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed runtime"))
                .map_err(|e| ValueCodecError::allocation(path, e))?;
        }
        Ok(())
    }
    pub(super) fn text<'s>(
        &self,
        value: &'s str,
        path: &str,
    ) -> Result<ObjectText<'s>, ValueCodecError> {
        self.checkpoint(path)?;
        Ok(if let Some(receipt) = &self.receipt {
            let permit = receipt
                .output_permit(self.runtime.expect("managed runtime"))
                .map_err(|e| ValueCodecError::allocation(path, e))?;
            let copy = Utf8Text::try_begin_copy_exact(value, &permit);
            self.checkpoint(path)?;
            ObjectText::Exact(copy.map_err(|e| ValueCodecError::allocation(path, e.into()))?)
        } else {
            ObjectText::Ordinary(super::text_copy::TextCopy::new(value))
        })
    }
    pub(super) fn insert(
        &mut self,
        name: Utf8Text,
        value: Value,
        path: &str,
    ) -> Result<bool, ValueCodecError> {
        self.checkpoint(path)?;
        let result = match &mut self.builder {
            Builder::Exact(builder) => builder.try_insert_prepared(name, value).map_err(|_| {
                ValueCodecError::invalid(path, "object rejected an admitted property")
            })?,
            Builder::Ordinary(builder) => builder.insert_text(name, value),
        };
        self.checkpoint(path)?;
        Ok(result.is_some())
    }
    pub(super) fn class(
        &self,
        source: &str,
        path: &str,
    ) -> Result<SharedClassIdentity, ValueCodecError> {
        self.checkpoint(path)?;
        let class = if let Some(receipt) = &self.receipt {
            let permit = receipt
                .output_permit(self.runtime.expect("managed runtime"))
                .map_err(|e| ValueCodecError::allocation(path, e))?;
            let class = PreparedObjectClassName::prepare(source, &permit);
            self.checkpoint(path)?;
            class.map_err(|e| class_error(path, e))?.into_identity()
        } else {
            runmat_types::ClassIdentity::new(source.to_owned())
                .map_err(|e| ValueCodecError::invalid(path, e.to_string()))?
                .into()
        };
        self.checkpoint(path)?;
        Ok(class)
    }
    pub(super) fn insert_dynamic(
        &mut self,
        key: Utf8Text,
        value: DynamicPropertyDef,
        path: &str,
    ) -> Result<bool, ValueCodecError> {
        self.checkpoint(path)?;
        let previous = match &mut self.builder {
            Builder::Exact(builder) => builder
                .try_insert_dynamic_key_prepared(key, value)
                .map_err(|_| {
                    ValueCodecError::invalid(path, "object rejected admitted dynamic metadata")
                })?,
            Builder::Ordinary(builder) => {
                builder.insert_dynamic(key.into_string_unbounded(), value)
            }
        };
        self.checkpoint(path)?;
        Ok(previous.is_some())
    }

    pub(super) fn attach_numeric_factor(
        &mut self,
        factor: runmat_value::NumericFactorData,
        path: &str,
    ) -> Result<(), ValueCodecError> {
        self.checkpoint(path)?;
        match &mut self.builder {
            Builder::Exact(builder) => {
                let receipt = self.receipt.as_ref().expect("exact object receipt");
                let permit = receipt
                    .output_permit(self.runtime.expect("managed runtime"))
                    .map_err(|error| ValueCodecError::allocation(path, error))?;
                if !builder
                    .try_set_numeric_factor_data(factor, &permit)
                    .map_err(|error| ValueCodecError::allocation(path, error.into()))?
                {
                    return Err(ValueCodecError::invalid(
                        path,
                        "numerical factor data was not admitted",
                    ));
                }
            }
            Builder::Ordinary(builder) => builder.replace_numeric_factor_data(factor),
        }
        self.checkpoint(path)
    }
    pub(super) fn finish(self, path: &str) -> Result<ObjectInstance, ValueCodecError> {
        self.checkpoint(path)?;
        let object = match self.builder {
            Builder::Exact(builder) => builder
                .finish()
                .map_err(|e| ValueCodecError::allocation(path, e.into()))?,
            Builder::Ordinary(builder) => {
                let object = builder.finish();
                match self.selected {
                    Some(facts) => object.with_selected_class(facts).ok_or_else(|| {
                        ValueCodecError::invalid(path, "selected class differs from decoded object")
                    })?,
                    None => object,
                }
            }
        };
        if let Some(receipt) = &self.receipt {
            receipt
                .checkpoint(self.runtime.expect("managed runtime"))
                .map_err(|e| ValueCodecError::allocation(path, e))?;
        }
        Ok(object)
    }
}

fn class_error(path: &str, error: runmat_value::ObjectClassPreparationError) -> ValueCodecError {
    match error {
        runmat_value::ObjectClassPreparationError::Invalid(source) => {
            ValueCodecError::invalid(path, source.to_string())
        }
        runmat_value::ObjectClassPreparationError::Allocation(source) => {
            ValueCodecError::allocation(path, source.into())
        }
    }
}

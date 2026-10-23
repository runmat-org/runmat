//! Original-generation object literal construction from already-rooted inputs.

mod initialization;
pub(crate) mod replacement;
pub(crate) use initialization::ObjectNativeInitialization;

use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::execution::{RootedValue, RootedValues};
use crate::RuntimeError;
use runmat_value::{
    ExactObjectBuilder, PreparedObjectClassName, SelectedObjectClassFacts, UnboundedObjectBuilder,
    Utf8Text, Value,
};
use std::rc::Rc;

/// Construct from stable caller-owned field values without collecting a second
/// payload/reference buffer. Their original source roots remain held through
/// exact destination construction and final root publication.
pub(crate) fn construct_object_from_borrowed_fields<'a, I>(
    runtime: Option<&RuntimeContext>,
    class_name: &str,
    properties: I,
) -> Result<RootedValue, RuntimeError>
where
    I: Iterator<Item = (&'a str, &'a Value)> + Clone,
{
    construct_object_from_fields_with_native(runtime, class_name, properties, None)
}

pub(crate) fn construct_object_from_fields_with_native<'a, I>(
    runtime: Option<&RuntimeContext>,
    class_name: &str,
    properties: I,
    native: Option<ObjectNativeInitialization<'_>>,
) -> Result<RootedValue, RuntimeError>
where
    I: Iterator<Item = (&'a str, &'a Value)> + Clone,
{
    use super::{RootedValueAdmissionError, RootedValueHandles};
    use crate::context::host_allocation::{HostAllocationPolicy, HostAllocationReceipt};
    if let Some(runtime) = runtime {
        let _roots = RootedValueHandles::for_values_iterator_in_context(
            properties.clone().map(|(_, value)| value),
            runtime,
        )
        .map_err(|source| match source {
            RootedValueAdmissionError::Host(source) => allocation(runtime, source),
            RootedValueAdmissionError::Gc(source) => {
                crate::runtime_error::semantic_error_with_source(
                    "ObjectConstruction",
                    "object source root admission failed",
                    source,
                )
            }
        })?;
        return construct_object_with_metadata_in_context(
            runtime, class_name, properties, None, native,
        );
    }
    let policy = HostAllocationReceipt::policy(None).map_err(|source| {
        host_allocation_runtime_error(
            None,
            "RunMat:ObjectConstruction",
            "object construction policy is unavailable",
            source,
        )
    })?;
    if policy != HostAllocationPolicy::Unbounded {
        return Err(crate::runtime_error::semantic_error(
            "ObjectConstruction",
            "object construction has no selected managed context",
        ));
    }
    let _roots = RootedValueHandles::for_values_iterator_unbounded(
        properties.clone().map(|(_, value)| value),
    )
    .map_err(|source| {
        crate::runtime_error::semantic_error_with_source(
            "ObjectConstruction",
            "object source root admission failed",
            source,
        )
    })?;
    let mut builder = UnboundedObjectBuilder::with_property_capacity(
        class_name.to_owned(),
        properties.clone().count(),
    );
    for (name, value) in properties {
        builder.insert(name.to_owned(), value.clone());
    }
    let object = builder.finish();
    let object = match native {
        Some(native) => native.install_ordinary(object),
        None => object,
    };
    RootedValue::new(Value::Object(object)).map_err(|source| source.into_runtime_error())
}

/// Construct a class-registry object from defaults whose source owners remain
/// rooted by the registry. Values clone only their canonical shared owners;
/// every new class/key/map/payload allocation is prepared before publication.
pub(crate) fn construct_registered_object_in_context<'a, I>(
    runtime: &RuntimeContext,
    class_name: &str,
    properties: I,
    selected: Option<Rc<SelectedObjectClassFacts>>,
) -> Result<RootedValue, RuntimeError>
where
    I: Iterator<Item = (&'a str, &'a Value)> + Clone,
{
    construct_object_with_metadata_in_context(runtime, class_name, properties, selected, None)
}

fn construct_object_with_metadata_in_context<'a, I>(
    runtime: &RuntimeContext,
    class_name: &str,
    properties: I,
    selected: Option<Rc<SelectedObjectClassFacts>>,
    native: Option<ObjectNativeInitialization<'_>>,
) -> Result<RootedValue, RuntimeError>
where
    I: Iterator<Item = (&'a str, &'a Value)> + Clone,
{
    let property_count = properties.clone().count();
    let quote = ExactObjectBuilder::quote(class_name, property_count)
        .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?;
    let mut physical = quote.requested_bytes;
    let mut claims = quote.retained_claims;
    if let Some(native) = native.as_ref() {
        let (bytes, native_claims) = native.additional_quote();
        physical = physical
            .checked_add(bytes)
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        claims = claims
            .checked_add(native_claims)
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    }
    for (name, _) in properties.clone() {
        let bytes = Utf8Text::requested_bytes_for_capacity(name.len())
            .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?;
        physical = physical
            .checked_add(bytes)
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        claims = claims
            .checked_add(usize::from(!name.is_empty()))
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|source| allocation(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(|source| allocation(runtime, source))?,
    )
    .map_err(|source| allocation(runtime, source))?;
    let object = if let Some(receipt) = &receipt {
        let permit = receipt
            .output_permit(runtime)
            .map_err(|source| allocation(runtime, source))?;
        let class_name = PreparedObjectClassName::prepare(class_name, &permit).map_err(
            |source| match source {
                runmat_value::ObjectClassPreparationError::Invalid(source) => {
                    crate::runtime_error::semantic_error_with_source(
                        "ObjectConstruction",
                        "object class identity is invalid",
                        source,
                    )
                }
                runmat_value::ObjectClassPreparationError::Allocation(source) => {
                    allocation(runtime, HostAllocationError::Numeric(source))
                }
            },
        )?;
        let mut builder = ExactObjectBuilder::try_new(class_name, property_count, &permit)
            .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?;
        if let Some(selected) = selected {
            if !builder.try_set_selected_class(selected) {
                return Err(crate::runtime_error::semantic_error(
                    "ObjectConstruction",
                    "selected object class differs from its constructor",
                ));
            }
        }
        if let Some(native) = native {
            if !native
                .install_exact(&mut builder, &permit)
                .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?
            {
                return Err(crate::runtime_error::semantic_error(
                    "ObjectConstruction",
                    "native object owner was already initialized",
                ));
            }
        }
        for (name, value) in properties {
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation(runtime, source))?;
            let key =
                Utf8Text::try_copy_str_exact_with(name, &permit, |_| receipt.checkpoint(runtime))
                    .map_err(|source| match source {
                    runmat_value::ExactArrayInitializationError::Allocation(source) => {
                        allocation(runtime, HostAllocationError::Numeric(source))
                    }
                    runmat_value::ExactArrayInitializationError::Element(source) => {
                        allocation(runtime, source)
                    }
                })?;
            builder
                .try_insert_prepared(key, value.clone())
                .map_err(|source| {
                    crate::runtime_error::semantic_error_with_source(
                        "ObjectConstruction",
                        "object property insertion failed",
                        source.into_parts().0,
                    )
                })?;
        }
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
        builder
            .finish()
            .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?
    } else {
        let mut builder =
            UnboundedObjectBuilder::with_property_capacity(class_name, property_count);
        for (name, value) in properties {
            builder.insert(name.to_owned(), value.clone());
        }
        let object = builder.finish();
        let object = match native {
            Some(native) => native.install_ordinary(object),
            None => object,
        };
        match selected {
            Some(selected) => object.with_selected_class(selected).ok_or_else(|| {
                crate::runtime_error::semantic_error(
                    "ObjectConstruction",
                    "selected object class differs from its constructor",
                )
            })?,
            None => object,
        }
    };
    if let Some(receipt) = &receipt {
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
    }
    RootedValue::in_context(Value::Object(object), runtime)
}

pub fn construct_object_in_context<'a, I>(
    runtime: &RuntimeContext,
    class_name: &str,
    names: I,
    values: RootedValues,
) -> Result<RootedValue, RuntimeError>
where
    I: ExactSizeIterator<Item = &'a str> + Clone,
{
    if names.len() != values.len() {
        return Err(crate::runtime_error::semantic_error(
            "ObjectConstruction",
            "object property names and values have different lengths",
        ));
    }
    let quote = ExactObjectBuilder::quote(class_name, names.len())
        .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?;
    let mut physical = quote.requested_bytes;
    let mut claims = quote.retained_claims;
    for name in names.clone() {
        let bytes = Utf8Text::requested_bytes_for_capacity(name.len())
            .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?;
        physical = physical
            .checked_add(bytes)
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        claims = claims
            .checked_add(usize::from(!name.is_empty()))
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|source| allocation(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(|source| allocation(runtime, source))?,
    )
    .map_err(|source| allocation(runtime, source))?;
    let mut values = values.try_into_values()?;
    let object = if let Some(receipt) = &receipt {
        let permit = receipt
            .output_permit(runtime)
            .map_err(|source| allocation(runtime, source))?;
        let class_name = PreparedObjectClassName::prepare(class_name, &permit).map_err(
            |source| match source {
                runmat_value::ObjectClassPreparationError::Invalid(source) => {
                    crate::runtime_error::semantic_error_with_source(
                        "ObjectConstruction",
                        "object class identity is invalid",
                        source,
                    )
                }
                runmat_value::ObjectClassPreparationError::Allocation(source) => {
                    allocation(runtime, HostAllocationError::Numeric(source))
                }
            },
        )?;
        let mut builder = ExactObjectBuilder::try_new(class_name, names.len(), &permit)
            .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?;
        for name in names {
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation(runtime, source))?;
            let key =
                Utf8Text::try_copy_str_exact_with(name, &permit, |_| receipt.checkpoint(runtime))
                    .map_err(|source| match source {
                    runmat_value::ExactArrayInitializationError::Allocation(source) => {
                        allocation(runtime, HostAllocationError::Numeric(source))
                    }
                    runmat_value::ExactArrayInitializationError::Element(source) => {
                        allocation(runtime, source)
                    }
                })?;
            let value = values.next().ok_or_else(iterator_changed)?;
            builder.try_insert_prepared(key, value).map_err(|source| {
                crate::runtime_error::semantic_error_with_source(
                    "ObjectConstruction",
                    "object property insertion failed",
                    source.into_parts().0,
                )
            })?;
        }
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
        builder
            .finish()
            .map_err(|source| allocation(runtime, HostAllocationError::Numeric(source)))?
    } else {
        let mut builder = UnboundedObjectBuilder::with_property_capacity(class_name, names.len());
        for name in names {
            builder.insert(name.to_owned(), values.next().ok_or_else(iterator_changed)?);
        }
        builder.finish()
    };
    if values.len() != 0 {
        return Err(iterator_changed());
    }
    if let Some(receipt) = &receipt {
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
    }
    RootedValue::in_context(Value::Object(object), runtime)
}

fn allocation(runtime: &RuntimeContext, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(runtime),
        "RunMat:Object:Allocation",
        "object literal backing admission failed",
        source,
    )
}

fn iterator_changed() -> RuntimeError {
    crate::runtime_error::semantic_error(
        "ObjectConstruction",
        "object property iterator changed during construction",
    )
}

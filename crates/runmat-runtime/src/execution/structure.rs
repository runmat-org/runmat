//! Original-generation scalar structure construction from already-rooted inputs.
//! This admits final map/key backing only, not arbitrary caller-owned children
//! or async-yielding cooperation. Index initialization and text copying are
//! synchronous; explicit checkpoints fence cancellation and selected authority.

use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationCheckpoint, HostAllocationError,
        HostAllocationPlan, HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::{
    execution::{RootedValue, RootedValues},
    RuntimeError,
};
use runmat_value::{ExactOrderedMap, StructValueBuilder, UnboundedStructValueBuilder, Value};

mod array;
mod array_values;
mod borrowed;
pub(crate) mod cell;
mod generated;
pub(crate) mod replacement;
pub use array::construct_struct_array_from_columns_in_context;
pub use array_values::construct_struct_array_from_values_in_context;
pub use borrowed::construct_struct_from_fields_in_context;
pub(crate) use borrowed::construct_struct_from_fields_in_optional_context;
pub(crate) use generated::construct_generated_structure;
pub(crate) use generated::construct_generated_structure_owned;

pub fn construct_struct_in_context<'a, I>(
    runtime: &RuntimeContext,
    names: I,
    values: RootedValues,
) -> Result<RootedValue, RuntimeError>
where
    I: ExactSizeIterator<Item = &'a str> + Clone,
{
    let owner = HostAllocationReceipt::capture_owner(Some(runtime))
        .map_err(|source| allocation(runtime, source))?;
    if names.len() != values.len() {
        return Err(crate::runtime_error::semantic_error(
            "StructConstruction",
            "structure names and values have different lengths",
        ));
    }
    let receipt = admit_struct_generation(Some(runtime), names.clone())
        .map_err(|source| allocation(runtime, source))?;
    check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
    // Keep the original roots alive through every moved child and final root
    // publication. Builders declared below retire first on any early error.
    let mut values = values.try_into_values()?;
    let structure = if let Some(receipt) = &receipt {
        let permit = receipt
            .output_permit(runtime)
            .map_err(|source| allocation(runtime, source))?;
        let builder = StructValueBuilder::try_new(names.len(), &permit);
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
        let mut builder = builder.map_err(|source| allocation(runtime, source.into()))?;
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
        for name in names {
            check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
            receipt
                .checkpoint(runtime)
                .map_err(|source| allocation(runtime, source))?;
            let key = copy_key(runtime, receipt, &permit, name)?;
            let value = values.next().ok_or_else(iterator_changed)?;
            builder.try_insert_prepared(key, value).map_err(|source| {
                if let Err(cause) = receipt.checkpoint(runtime) {
                    return allocation(runtime, cause);
                }
                crate::runtime_error::semantic_error_with_source(
                    "StructConstruction",
                    "structure insertion failed",
                    source.into_parts().0,
                )
            })?;
        }
        receipt
            .checkpoint(runtime)
            .map_err(|source| allocation(runtime, source))?;
        builder.finish()
    } else {
        let mut builder = UnboundedStructValueBuilder::with_capacity(names.len());
        for name in names {
            builder.insert(name, values.next().ok_or_else(iterator_changed)?);
        }
        builder.finish()
    };
    if values.len() != 0 {
        return Err(iterator_changed());
    }
    check_owner(Some(runtime), &owner).map_err(|source| allocation(runtime, source))?;
    let result = RootedValue::in_context(Value::Struct(structure), runtime)?;
    drop(values);
    Ok(result)
}

fn allocation(runtime: &RuntimeContext, source: HostAllocationError) -> RuntimeError {
    host_allocation_runtime_error(
        Some(runtime),
        "RunMat:Struct:Allocation",
        "structure backing admission failed",
        source,
    )
}
/// Quote final map/index and each incoming key's original generation. This
/// does not claim that hashing/index initialization is asynchronously bounded.
pub(crate) fn admit_struct_generation<'a, I>(
    runtime: Option<&RuntimeContext>,
    names: I,
) -> Result<Option<HostAllocationReceipt>, HostAllocationError>
where
    I: ExactSizeIterator<Item = &'a str> + Clone,
{
    admit_struct_generation_counted(runtime, names.clone(), names.len())
}

fn admit_struct_generation_counted<'a, I>(
    runtime: Option<&RuntimeContext>,
    names: I,
    count: usize,
) -> Result<Option<HostAllocationReceipt>, HostAllocationError>
where
    I: Iterator<Item = &'a str> + Clone,
{
    admit_struct_generation_for::<Value, _>(runtime, names, count)
}

fn admit_struct_generation_for<'a, V, I>(
    runtime: Option<&RuntimeContext>,
    names: I,
    count: usize,
) -> Result<Option<HostAllocationReceipt>, HostAllocationError>
where
    I: Iterator<Item = &'a str> + Clone,
{
    admit_struct_generation_extra_for::<V, _>(runtime, names, count, 0, 0)
}

fn admit_struct_generation_extra_for<'a, V, I>(
    runtime: Option<&RuntimeContext>,
    names: I,
    count: usize,
    extra_bytes: usize,
    extra_claims: usize,
) -> Result<Option<HostAllocationReceipt>, HostAllocationError>
where
    I: Iterator<Item = &'a str> + Clone,
{
    let owner = HostAllocationReceipt::capture_owner(runtime)?;
    let quote = ExactOrderedMap::<V>::quote(count)?;
    let (raw, claims) = names.clone().try_fold(
        (
            quote
                .requested_bytes
                .checked_add(extra_bytes)
                .ok_or(HostAllocationError::Overflow)?,
            quote
                .retained_claims
                .checked_add(extra_claims)
                .ok_or(HostAllocationError::Overflow)?,
        ),
        |(bytes, claims), name| {
            check_owner(runtime, &owner)?;
            let (key_bytes, key_claims) =
                ExactOrderedMap::<V>::key_quote(name).map_err(HostAllocationError::Numeric)?;
            Ok::<_, HostAllocationError>((
                bytes
                    .checked_add(key_bytes)
                    .ok_or(HostAllocationError::Overflow)?,
                claims
                    .checked_add(key_claims)
                    .ok_or(HostAllocationError::Overflow)?,
            ))
        },
    )?;
    let raw = u64::try_from(raw).map_err(|_| HostAllocationError::Overflow)?;
    let output = HostAllocationPlan::retained_exact_output_bytes(raw, claims)?;
    let plan = HostAllocationPlan::new(0, output)?;
    check_owner(runtime, &owner)?;
    let receipt = HostAllocationReceipt::admit(runtime, plan)?;
    check_owner(runtime, &owner)?;
    Ok(receipt)
}

fn check_owner(
    runtime: Option<&RuntimeContext>,
    owner: &Option<HostAllocationCheckpoint>,
) -> Result<(), HostAllocationError> {
    match owner {
        Some(owner) => owner.check(runtime.ok_or(HostAllocationError::MissingManagedOwner)?),
        None => match HostAllocationReceipt::policy(runtime)? {
            HostAllocationPolicy::Unbounded => Ok(()),
            HostAllocationPolicy::Managed => Err(HostAllocationError::StaleAuthority),
        },
    }
}
fn iterator_changed() -> RuntimeError {
    crate::runtime_error::semantic_error(
        "StructConstruction",
        "structure name/value iterator changed during publication",
    )
}
pub(crate) fn copy_key(
    runtime: &RuntimeContext,
    receipt: &HostAllocationReceipt,
    permit: &dyn runmat_value::ExactHostAllocationPermit,
    name: &str,
) -> Result<runmat_value::Utf8Text, RuntimeError> {
    let key = runmat_value::Utf8Text::try_copy_str_exact_with(name, permit, |_| {
        receipt.checkpoint(runtime)
    });
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation(runtime, source))?;
    key.map_err(|source| {
        allocation(
            runtime,
            match source {
                runmat_value::ExactArrayInitializationError::Allocation(source) => source.into(),
                runmat_value::ExactArrayInitializationError::Element(source) => source,
            },
        )
    })
}

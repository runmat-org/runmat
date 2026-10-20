//! One physical destination quote and publication fence for Object edits.
use super::*;
use crate::context::{host_allocation::HostAllocationCheckpoint, RuntimeContext};

#[derive(Clone, Copy)]
pub(super) enum Omission<'a> {
    None,
    Property(&'a str),
    Dynamic(&'a str),
}
pub(super) struct Shape<'a> {
    pub properties: usize,
    pub dynamic: usize,
    pub omission: Omission<'a>,
}

#[derive(Default)]
pub(super) struct ExtraMetadata {
    pub requested_bytes: usize,
    pub retained_claims: usize,
}

pub(super) fn admit<'a>(
    runtime: &RuntimeContext,
    source: &ObjectInstance,
    shape: &Shape<'_>,
    new_keys: impl Iterator<Item = &'a str>,
    error: &impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
) -> Result<HostAllocationReceipt, crate::RuntimeError> {
    admit_with_metadata(
        runtime,
        source,
        shape,
        new_keys,
        ExtraMetadata::default(),
        error,
    )
}

pub(super) fn admit_with_metadata<'a>(
    runtime: &RuntimeContext,
    source: &ObjectInstance,
    shape: &Shape<'_>,
    new_keys: impl Iterator<Item = &'a str>,
    extra: ExtraMetadata,
    error: &impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
) -> Result<HostAllocationReceipt, crate::RuntimeError> {
    let quote = ExactObjectBuilder::replacement_quote(shape.properties, shape.dynamic)
        .map_err(|source| error("object replacement geometry is invalid", source.into()))?;
    let metadata = source.unclaimed_metadata_quote().map_err(|source| {
        error(
            "object metadata admission geometry is invalid",
            source.into(),
        )
    })?;
    let overflow = || {
        error(
            "object replacement geometry overflow",
            HostAllocationError::Overflow,
        )
    };
    let mut bytes = quote
        .requested_bytes
        .checked_add(metadata.requested_bytes)
        .and_then(|bytes| bytes.checked_add(extra.requested_bytes))
        .ok_or_else(overflow)?;
    let mut claims = quote
        .retained_claims
        .checked_add(metadata.retained_claims)
        .and_then(|claims| claims.checked_add(extra.retained_claims))
        .ok_or_else(overflow)?;
    for name in new_keys {
        bytes =
            bytes
                .checked_add(Utf8Text::requested_bytes_for_capacity(name.len()).map_err(
                    |source| error("object property key geometry is invalid", source.into()),
                )?)
                .ok_or_else(overflow)?;
        claims = claims
            .checked_add(usize::from(!name.is_empty()))
            .ok_or_else(overflow)?;
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes).map_err(|_| overflow())?,
        claims,
    )
    .map_err(|source| error("object replacement retained quote overflow", source))?;
    HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained)
            .map_err(|source| error("object replacement quote failed", source))?,
    )
    .map_err(|source| error("object replacement admission failed", source))?
    .ok_or_else(|| {
        error(
            "object replacement has no managed owner",
            HostAllocationError::MissingManagedOwner,
        )
    })
}

pub(super) fn prepare(
    runtime: &RuntimeContext,
    source: &ObjectInstance,
    shape: &Shape<'_>,
    receipt: &HostAllocationReceipt,
    error: &impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
    presentation: &impl Fn(String) -> crate::RuntimeError,
) -> Result<ExactObjectBuilder, crate::RuntimeError> {
    let permit = receipt
        .output_permit(runtime)
        .map_err(|source| error("object replacement permit is no longer current", source))?;
    source
        .try_admit_metadata_exact(&permit)
        .map_err(|source| error("object metadata admission failed", source.into()))?;
    let result = match shape.omission {
        Omission::None => {
            ExactObjectBuilder::prepare_copy(source, shape.properties, shape.dynamic, &permit)
        }
        Omission::Property(name) => {
            ExactObjectBuilder::prepare_copy_excluding_property(source, name, &permit)
        }
        Omission::Dynamic(name) => {
            ExactObjectBuilder::prepare_copy_excluding_dynamic_property(source, name, &permit)
        }
    };
    result.map_err(|source| match source {
        ObjectEditPreparationError::Allocation(source) => {
            error("object replacement allocation failed", source.into())
        }
        source => presentation(format!(
            "object replacement cannot retain its source: {source}"
        )),
    })
}

pub(super) fn publish(
    runtime: &RuntimeContext,
    destination: &mut ObjectInstance,
    builder: ExactObjectBuilder,
    receipt: &HostAllocationReceipt,
    owner: &Option<HostAllocationCheckpoint>,
    error: &impl Fn(&'static str, HostAllocationError) -> crate::RuntimeError,
) -> Result<(), crate::RuntimeError> {
    receipt.checkpoint(runtime).map_err(|source| {
        error(
            "object replacement was cancelled before publication",
            source,
        )
    })?;
    let value = builder.finish().map_err(|source| {
        error(
            "object replacement publication allocation failed",
            source.into(),
        )
    })?;
    receipt.checkpoint(runtime).map_err(|source| {
        error(
            "object replacement was cancelled before publication",
            source,
        )
    })?;
    if let Some(owner) = owner {
        owner.check(runtime).map_err(|source| {
            error(
                "object property edit source authority changed before publication",
                source,
            )
        })?;
    }
    *destination = value;
    Ok(())
}

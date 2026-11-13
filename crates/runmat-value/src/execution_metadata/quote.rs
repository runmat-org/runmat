//! Exact visible allocation inventory. Opaque tree nodes are an explicit
//! remaining ownership frontier, never replaced by serialized byte estimates.

use super::ExecutionMetadataQuoteError as Error;
use crate::ExactHostNumericAllocationError;
use runmat_types::{
    DimensionFact, DistributionScheme, ExecutionFact, ResidencyFact, ShapeFact, ValueFact,
    ValueKindFact,
};

fn add(left: usize, right: usize) -> Result<usize, Error> {
    left.checked_add(right).ok_or(Error::Allocation(
        ExactHostNumericAllocationError::LayoutOverflow,
    ))
}
fn vector<T>(values: &Vec<T>) -> Result<usize, Error> {
    values
        .capacity()
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(Error::Allocation(
            ExactHostNumericAllocationError::LayoutOverflow,
        ))
}
fn optional_text(text: &Option<String>) -> usize {
    text.as_ref().map_or(0, String::capacity)
}
fn outputs(contract: &runmat_execution::OutputContract) -> Result<usize, Error> {
    let Some(work) = &contract.owned_work else {
        return Ok(0);
    };
    let mut bytes = vector(&work.dense)?;
    for output in &work.dense {
        bytes = add(bytes, vector(&output.shape)?)?;
        bytes = add(bytes, vector(&output.allowed_element_types)?)?;
    }
    Ok(bytes)
}

pub(super) fn unchanged<T>(_: &mut T) {}
pub(super) fn normalize_distributed(value: &mut runmat_execution::DistributedValueHandle) {
    normalize_fact(&mut value.value, 0);
}
pub(super) fn normalize_composite(value: &mut runmat_execution::CompositeHandle) {
    normalize_fact(&mut value.value, 0);
}

/// Empty standard trees may still retain an allocated root after removal.
/// Replace only semantically empty trees once before immutable publication,
/// proving an empty-tree quote has no remaining physical node allocation.
fn normalize_fact(value: &mut ValueFact, depth: usize) {
    if depth >= 128 {
        return;
    }
    if value.invalidation.0.is_empty() {
        value.invalidation.0 = Default::default();
    }
    let next = depth + 1;
    match &mut value.kind {
        ValueKindFact::Cell(cell) => {
            normalize_fact(&mut cell.element, next);
            for value in &mut cell.elements {
                normalize_fact(value, next);
            }
        }
        ValueKindFact::Struct(structure) => {
            if structure.fields.is_empty() {
                structure.fields = Default::default();
            }
            for value in structure.fields.values_mut() {
                normalize_fact(value, next);
            }
            for fields in &mut structure.elements {
                if fields.is_empty() {
                    *fields = Default::default();
                }
                for value in fields.values_mut() {
                    normalize_fact(value, next);
                }
            }
        }
        ValueKindFact::Object(object) => {
            if object.properties.is_empty() {
                object.properties = Default::default();
            }
            for value in object.properties.values_mut() {
                normalize_fact(value, next);
            }
        }
        ValueKindFact::Callable(callable) => {
            if callable.capabilities.0.is_empty() {
                callable.capabilities.0 = Default::default();
            }
            for value in callable
                .parameters
                .iter_mut()
                .chain(&mut callable.outputs)
                .chain(&mut callable.captures)
            {
                normalize_fact(value, next);
            }
        }
        ValueKindFact::OutputList(sequence) => {
            for value in &mut sequence.outputs {
                normalize_fact(value, next);
            }
        }
        ValueKindFact::Execution(execution) => match execution {
            ExecutionFact::Future { output, .. }
            | ExecutionFact::Task { output, .. }
            | ExecutionFact::Job { output } => {
                for value in &mut output.outputs {
                    normalize_fact(value, next);
                }
            }
            ExecutionFact::Pool => {}
        },
        ValueKindFact::Distributed(distributed) => normalize_fact(&mut distributed.value, next),
        ValueKindFact::Composite(composite) => normalize_fact(&mut composite.value, next),
        ValueKindFact::Never
        | ValueKindFact::Unknown
        | ValueKindFact::Void
        | ValueKindFact::Numeric(_)
        | ValueKindFact::Logical
        | ValueKindFact::Character
        | ValueKindFact::String
        | ValueKindFact::Symbolic
        | ValueKindFact::ClassReference(_)
        | ValueKindFact::Exception(_)
        | ValueKindFact::Foreign(_) => {}
    }
}
pub(super) fn future(value: &runmat_execution::FutureHandle) -> Result<usize, Error> {
    outputs(&value.outputs)
}
pub(super) fn task(value: &runmat_execution::TaskHandle) -> Result<usize, Error> {
    outputs(&value.outputs)
}
pub(super) fn job(value: &runmat_execution::JobHandle) -> Result<usize, Error> {
    outputs(&value.outputs)
}
pub(super) fn distributed(
    value: &runmat_execution::DistributedValueHandle,
) -> Result<usize, Error> {
    add(
        add(fact(&value.value, 0)?, vector(&value.global_shape)?)?,
        scheme(&value.scheme)?,
    )
}
pub(super) fn composite(value: &runmat_execution::CompositeHandle) -> Result<usize, Error> {
    fact(&value.value, 0)
}
fn scheme(value: &DistributionScheme) -> Result<usize, Error> {
    match value {
        DistributionScheme::OneDimensional { partition, .. } => vector(partition),
        DistributionScheme::Custom { partitioner } => Ok(partitioner.capacity()),
        DistributionScheme::Replicated
        | DistributionScheme::Block { .. }
        | DistributionScheme::Cyclic { .. }
        | DistributionScheme::TwoDimensionalBlockCyclic { .. } => Ok(0),
    }
}
fn facts(values: &Vec<ValueFact>, depth: usize) -> Result<usize, Error> {
    let mut bytes = vector(values)?;
    for value in values {
        bytes = add(bytes, fact(value, depth + 1)?)?;
    }
    Ok(bytes)
}
fn boxed(value: &ValueFact, depth: usize) -> Result<usize, Error> {
    add(std::mem::size_of::<ValueFact>(), fact(value, depth + 1)?)
}
fn fact(value: &ValueFact, depth: usize) -> Result<usize, Error> {
    if depth >= 128 {
        return Err(Error::FactDepth);
    }
    if !value.invalidation.0.is_empty() {
        return Err(Error::OpaqueFactTree);
    }
    let mut bytes = match &value.shape {
        ShapeFact::Shaped { dims } => {
            let mut bytes = vector(dims)?;
            for dim in dims {
                if let DimensionFact::Symbolic(symbol) = dim {
                    bytes = add(bytes, symbol.0.capacity())?;
                }
            }
            bytes
        }
        ShapeFact::Unknown | ShapeFact::Scalar | ShapeFact::Ranked { .. } => 0,
    };
    bytes = add(
        bytes,
        match &value.residency {
            ResidencyFact::Device { provider } => optional_text(provider),
            ResidencyFact::Remote { pool } => optional_text(pool),
            ResidencyFact::Unknown | ResidencyFact::Host => 0,
        },
    )?;
    let nested = match &value.kind {
        ValueKindFact::Cell(cell) => {
            add(boxed(&cell.element, depth)?, facts(&cell.elements, depth)?)?
        }
        ValueKindFact::Struct(structure) => {
            if !structure.fields.is_empty()
                || structure.elements.iter().any(|fields| !fields.is_empty())
            {
                return Err(Error::OpaqueFactTree);
            }
            vector(&structure.elements)?
        }
        ValueKindFact::Object(object) => {
            if !object.properties.is_empty() {
                return Err(Error::OpaqueFactTree);
            }
            object
                .runtime_class
                .as_ref()
                .map_or(0, |class| class.allocation_capacity())
        }
        ValueKindFact::ClassReference(class) => class
            .runtime_class
            .as_ref()
            .map_or(0, |class| class.allocation_capacity()),
        ValueKindFact::Callable(callable) => {
            if !callable.capabilities.0.is_empty() {
                return Err(Error::OpaqueFactTree);
            }
            if callable.identity.is_some() {
                return Err(Error::CallableIdentity);
            }
            add(
                add(
                    facts(&callable.parameters, depth)?,
                    facts(&callable.outputs, depth)?,
                )?,
                facts(&callable.captures, depth)?,
            )?
        }
        ValueKindFact::OutputList(sequence) => facts(&sequence.outputs, depth)?,
        ValueKindFact::Exception(exception) => optional_text(&exception.identifier),
        ValueKindFact::Execution(execution) => match execution {
            ExecutionFact::Future { output, .. }
            | ExecutionFact::Task { output, .. }
            | ExecutionFact::Job { output } => facts(&output.outputs, depth)?,
            ExecutionFact::Pool => 0,
        },
        ValueKindFact::Distributed(distributed) => add(
            boxed(&distributed.value, depth)?,
            distributed
                .scheme
                .as_ref()
                .map(scheme)
                .transpose()?
                .unwrap_or(0),
        )?,
        ValueKindFact::Composite(composite) => boxed(&composite.value, depth)?,
        ValueKindFact::Foreign(foreign) => {
            add(foreign.family.capacity(), optional_text(&foreign.type_name))?
        }
        ValueKindFact::Never
        | ValueKindFact::Unknown
        | ValueKindFact::Void
        | ValueKindFact::Numeric(_)
        | ValueKindFact::Logical
        | ValueKindFact::Character
        | ValueKindFact::String
        | ValueKindFact::Symbolic => 0,
    };
    add(bytes, nested)
}

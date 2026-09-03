use runmat_accelerate_api::{GpuTensorHandle, GpuTensorStorage};
use runmat_builtins::{BuiltinCatalogEntry, BuiltinErrorDescriptor, MetadataPredicate};
use runmat_types::{NumericDomain, StorageFact, ValueFact, ValueKindFact};
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct MetadataBoundary {
    entry: &'static BuiltinCatalogEntry,
    internal: &'static BuiltinErrorDescriptor,
    too_many_outputs: &'static BuiltinErrorDescriptor,
    predicate: MetadataPredicate,
}

impl MetadataBoundary {
    pub(super) const fn new(
        entry: &'static BuiltinCatalogEntry,
        internal: &'static BuiltinErrorDescriptor,
        too_many_outputs: &'static BuiltinErrorDescriptor,
        predicate: MetadataPredicate,
    ) -> Self {
        Self {
            entry,
            internal,
            too_many_outputs,
            predicate,
        }
    }

    pub(super) fn execute(&self, value: Value) -> BuiltinResult<Value> {
        self.reject_excess_outputs()?;
        let result = match &value {
            Value::GpuTensor(handle) => self.classify_resident(handle)?,
            host => self.classify_host(host),
        };
        Ok(Value::Bool(result))
    }

    fn classify_resident(&self, handle: &GpuTensorHandle) -> BuiltinResult<bool> {
        match self.predicate {
            MetadataPredicate::GpuArray => {
                return Ok(runmat_accelerate_api::handle_is_explicit(handle));
            }
            MetadataPredicate::Cell | MetadataPredicate::CellString => return Ok(false),
            MetadataPredicate::Logical
            | MetadataPredicate::Numeric
            | MetadataPredicate::Real
            | MetadataPredicate::Sparse => {}
        }
        validate_resident_numeric_metadata(handle).map_err(|detail| self.internal(detail))?;
        let storage = runmat_accelerate_api::handle_storage(handle);
        let logical = runmat_accelerate_api::handle_is_logical(handle);
        Ok(match self.predicate {
            MetadataPredicate::GpuArray => unreachable!("handled before metadata validation"),
            MetadataPredicate::Cell | MetadataPredicate::CellString => {
                unreachable!("handled before metadata validation")
            }
            MetadataPredicate::Logical => logical,
            MetadataPredicate::Numeric => !logical,
            MetadataPredicate::Real => storage == GpuTensorStorage::Real,
            MetadataPredicate::Sparse => false,
        })
    }

    fn classify_host(&self, value: &Value) -> bool {
        match self.predicate {
            MetadataPredicate::Cell => match value {
                Value::Cell(_) => true,
                Value::Distributed(handle) => fact_is_cell(&handle.value),
                _ => false,
            },
            MetadataPredicate::CellString => match value {
                Value::Cell(cell) => cell
                    .data
                    .iter()
                    .all(|element| matches!(element, Value::CharArray(_))),
                _ => false,
            },
            MetadataPredicate::GpuArray => false,
            MetadataPredicate::Logical => match value {
                Value::Bool(_) | Value::LogicalArray(_) => true,
                Value::SparseTensor(value) => value.is_logical(),
                Value::Distributed(handle) => fact_is_logical(&handle.value),
                _ => false,
            },
            MetadataPredicate::Numeric => match value {
                Value::Num(_)
                | Value::Int(_)
                | Value::Complex(_, _)
                | Value::Tensor(_)
                | Value::ComplexTensor(_) => true,
                Value::SparseTensor(value) => !value.is_logical(),
                Value::Distributed(handle) => fact_is_numeric(&handle.value),
                _ => false,
            },
            MetadataPredicate::Real => match value {
                Value::Num(_)
                | Value::Int(_)
                | Value::Bool(_)
                | Value::Tensor(_)
                | Value::LogicalArray(_)
                | Value::CharArray(_)
                | Value::Symbolic(_)
                | Value::SymbolicArray(_) => true,
                Value::SparseTensor(value) => !value.is_complex(),
                Value::Object(value)
                    if value.is_class(runmat_types::standard::DURATION)
                        || value.is_class(runmat_types::standard::CALENDAR_DURATION) =>
                {
                    true
                }
                Value::ObjectArray(value)
                    if value.class_name().is(runmat_types::standard::DURATION)
                        || value
                            .class_name()
                            .is(runmat_types::standard::CALENDAR_DURATION) =>
                {
                    true
                }
                Value::Distributed(handle) => fact_is_real(&handle.value),
                _ => false,
            },
            MetadataPredicate::Sparse => match value {
                Value::SparseTensor(_) => true,
                Value::Distributed(handle) => handle.value.storage == StorageFact::Sparse,
                _ => false,
            },
        }
    }

    fn reject_excess_outputs(&self) -> BuiltinResult<()> {
        if matches!(crate::output_count::current_output_count(), Some(count) if count > 1) {
            return Err(self.error(self.too_many_outputs, "only one output is defined"));
        }
        Ok(())
    }

    fn internal(&self, detail: impl std::fmt::Display) -> RuntimeError {
        self.error(self.internal, detail)
    }

    fn error(
        &self,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl std::fmt::Display,
    ) -> RuntimeError {
        let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
            .with_builtin(self.entry.identity.name);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }
}

pub(super) fn validate_resident_numeric_metadata(handle: &GpuTensorHandle) -> Result<(), String> {
    if gpu_helpers::exact_provider_for_handle(handle).is_none() {
        return Err("no acceleration provider owns the input handle".into());
    }
    let storage = runmat_accelerate_api::handle_storage(handle);
    let integer = runmat_accelerate_api::handle_integer_type(handle);
    let logical = runmat_accelerate_api::handle_is_logical(handle);
    let precision = runmat_accelerate_api::handle_precision(handle);
    let coherent = if logical {
        integer.is_none()
            && precision.is_some()
            && storage == GpuTensorStorage::Real
            && gpu_helpers::gpu_class_metadata_matches(handle, precision, None, true)
    } else if let Some(integer) = integer {
        precision.is_none()
            && matches!(
                storage,
                GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
            )
            && gpu_helpers::gpu_class_metadata_matches(handle, None, Some(integer), false)
    } else {
        precision.is_some()
            && matches!(
                storage,
                GpuTensorStorage::Real | GpuTensorStorage::ComplexInterleaved
            )
            && gpu_helpers::gpu_class_metadata_matches(handle, precision, None, false)
    };
    if coherent {
        return Ok(());
    }
    Err(format!(
        "resident class metadata contradicts physical storage (storage={storage:?}, integer={integer:?}, logical={logical}, precision={precision:?}, class={:?})",
        runmat_accelerate_api::handle_class_identity(handle)
    ))
}

fn fact_is_logical(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Logical)
}

fn fact_is_cell(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Cell(_))
}

fn fact_is_numeric(value: &ValueFact) -> bool {
    matches!(value.kind, ValueKindFact::Numeric(_))
}

fn fact_is_real(value: &ValueFact) -> bool {
    matches!(
        value.kind,
        ValueKindFact::Numeric(runmat_types::NumericFact {
            domain: NumericDomain::Real,
            ..
        }) | ValueKindFact::Logical
            | ValueKindFact::Character
    )
}

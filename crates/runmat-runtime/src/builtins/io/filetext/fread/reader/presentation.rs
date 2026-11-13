//! Selected output class and residency. Precision conversion precedes Like
//! conversion, but both happen scalar-wise in the one final output owner.

use super::*;
use crate::builtins::common::gpu_helpers::host_upload::GpuHostUpload;
use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostLogicalTensorView, HostNumericTensorView,
    ProviderPrecision,
};

#[derive(Clone, Copy)]
pub(super) enum Destination {
    Character,
    Numeric(NumericDType),
    Logical,
}

#[derive(Clone, Copy)]
pub(in crate::builtins::io::filetext::fread) struct Projection<'a> {
    pub(super) precision: OutputKind,
    pub(super) destination: Destination,
    resident: Option<&'a GpuTensorHandle>,
}

impl<'a> Projection<'a> {
    pub(in crate::builtins::io::filetext::fread) fn select(
        precision: OutputKind,
        prototype: Option<&'a Value>,
    ) -> BuiltinResult<Self> {
        let mut projection = Self {
            precision,
            destination: match precision {
                OutputKind::Char => Destination::Character,
                OutputKind::Numeric(dtype) => Destination::Numeric(dtype),
            },
            resident: None,
        };
        projection.destination = match prototype {
            None => return Ok(projection),
            Some(Value::GpuTensor(prototype)) => {
                projection.resident = Some(prototype);
                if runmat_accelerate_api::handle_is_logical(prototype) {
                    Destination::Logical
                } else if let Some(integer) = runmat_accelerate_api::handle_integer_type(prototype)
                {
                    Destination::Numeric(NumericDType::from(runmat_types::IntegerClass::from(
                        integer,
                    )))
                } else {
                    Destination::Numeric(match runmat_accelerate_api::handle_precision(prototype) {
                        Some(ProviderPrecision::F32) => NumericDType::F32,
                        Some(ProviderPrecision::F64) => NumericDType::F64,
                        None => {
                            return Err(fread_error_with_detail(
                                &FREAD_ERROR_INVALID_OPTION,
                                "resident 'like' prototype is missing precision metadata",
                            ))
                        }
                    })
                }
            }
            Some(Value::LogicalArray(_) | Value::Bool(_)) => Destination::Logical,
            Some(Value::Tensor(tensor)) => Destination::Numeric(tensor.numeric_dtype()),
            Some(Value::Int(value)) => Destination::Numeric(value.numeric_dtype()),
            Some(Value::Num(_)) => Destination::Numeric(NumericDType::F64),
            Some(Value::CharArray(_) | Value::String(_) | Value::StringArray(_)) => {
                Destination::Character
            }
            Some(_) => {
                return Err(fread_error_with_detail(
                    &FREAD_ERROR_INVALID_OPTION,
                    "unsupported 'like' prototype",
                ))
            }
        };
        Ok(projection)
    }

    fn precision_scalar(self, source: NumericScalar) -> NumericScalar {
        match self.precision {
            OutputKind::Numeric(dtype) => scalar::convert(source, dtype),
            OutputKind::Char => {
                unreachable!("character projection does not use numeric publication")
            }
        }
    }

    pub(super) fn numeric<T: scalar::ReadValue>(
        self,
        source: T,
        dtype: NumericDType,
    ) -> NumericScalar {
        scalar::convert(self.precision_scalar(source.numeric()), dtype)
    }

    pub(super) fn logical<T: scalar::ReadValue>(self, source: T) -> Result<u8, String> {
        match self.precision {
            OutputKind::Numeric(_) => Ok(scalar::nonzero(self.precision_scalar(source.numeric()))),
            OutputKind::Char => Ok(u8::from(source.character()?.raw() != 0)),
        }
    }

    pub(super) fn character<T: scalar::ReadValue>(
        self,
        source: T,
    ) -> Result<runmat_value::Utf16CodeUnit, String> {
        source.character()
    }

    pub(in crate::builtins::io::filetext::fread) async fn publish(
        self,
        runtime: Option<&crate::context::RuntimeContext>,
        value: Value,
        cooperation: &mut super::super::super::cooperation::FileCooperation,
    ) -> BuiltinResult<Value> {
        let Some(prototype) = self.resident else {
            return Ok(value);
        };
        cooperation.completed(0).await?;
        let output = match &value {
            Value::Tensor(tensor) => {
                let view = gpu_helpers::tensor_numeric_view(tensor);
                let mut upload = GpuHostUpload::prepare_borrowed_native_output_in_optional_context(
                    runtime,
                    BUILTIN_NAME,
                    prototype,
                    tensor.shape(),
                    view.element_type(),
                    GpuTensorStorage::Real,
                    0,
                )?;
                upload.publish(&HostNumericTensorView {
                    data: view,
                    shape: tensor.shape(),
                    storage: GpuTensorStorage::Real,
                })?
            }
            Value::LogicalArray(array) => {
                let mut upload =
                    GpuHostUpload::prepare_borrowed_logical_output_in_optional_context(
                        runtime,
                        BUILTIN_NAME,
                        prototype,
                        &array.shape,
                    )?;
                upload.publish_logical(&HostLogicalTensorView {
                    data: &array.data,
                    shape: &array.shape,
                })?
            }
            _ => {
                return Err(fread_error_with_detail(
                    &FREAD_ERROR_INVALID_OPTION,
                    "character output cannot be resident",
                ))
            }
        };
        // The shared upload owner has already checked class, shape, provider,
        // provenance and protected prototype before publishing this handle.
        if let Err(error) = cooperation.completed(0).await {
            return Err(gpu_helpers::reject_gpu_output(output, error));
        }
        Ok(Value::GpuTensor(output))
    }
}

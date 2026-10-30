use super::super::numerical_execution::NumericalCall;
use super::*;
use crate::call::FunctionArgumentView;
use runmat_builtins::ProductControl;
#[derive(Clone, PartialEq, Eq)]
pub(super) enum DimSelection {
    Auto,
    Dim(usize),
    Vec(Vec<usize>),
    All,
}
pub(super) struct ParsedArguments {
    pub selection: DimSelection,
    pub nan_mode: ReductionNaN,
    pub output: OutputTemplate,
}
fn control(value: &Value) -> Option<ProductControl> {
    match value {
        Value::String(runmat_value::StringElement::Text(text)) => {
            ProductControl::from_public_option(text.as_str())
        }
        Value::StringArray(array) if array.data().len() == 1 && !array.data()[0].is_missing() => {
            ProductControl::from_public_option(
                array.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str(),
            )
        }
        Value::CharArray(array)
            if array.rows == 1
                && array.shape().iter().skip(2).all(|n| *n == 1)
                && array.validate_layout().is_ok() =>
        {
            ProductControl::from_public_code_units(array.data().iter().map(|unit| unit.raw()))
        }
        _ => None,
    }
}
pub(super) async fn parse(
    args: FunctionArgumentView<'_>,
    call: &NumericalCall<'_>,
) -> BuiltinResult<ParsedArguments> {
    let mut selection = DimSelection::Auto;
    let mut selection_set = false;
    let mut nan_mode = ReductionNaN::Include;
    let mut output = OutputTemplate::Default;
    let mut output_set = false;
    for value in args {
        if let Some(control) = control(value) {
            match control {
                ProductControl::Missing(policy) => nan_mode = policy,
                ProductControl::Output(class) => {
                    if output_set {
                        return Err(prod_descriptor_error_with_detail(
                            &PROD_ERROR_INVALID_ARGUMENT,
                            "multiple output class specifications provided",
                        ));
                    }
                    output = class;
                    output_set = true;
                }
                ProductControl::All => {
                    if selection_set && selection != DimSelection::Auto {
                        return Err(prod_descriptor_error_with_detail(
                            &PROD_ERROR_INVALID_ARGUMENT,
                            "all cannot be combined with explicit dimensions",
                        ));
                    }
                    selection = DimSelection::All;
                    selection_set = true;
                }
            }
            continue;
        }
        if selection_set && selection != DimSelection::Auto {
            return Err(prod_descriptor_error_with_detail(
                &PROD_ERROR_INVALID_ARGUMENT,
                "unrecognised argument",
            ));
        }
        let shape = match value {
            Value::Tensor(t) => Some(t.shape()),
            Value::LogicalArray(a) => Some(a.shape.as_slice()),
            Value::GpuTensor(h) => Some(h.shape()),
            _ => None,
        };
        if shape.is_some_and(|shape| !vector_shape(shape)) {
            return Err(prod_descriptor_error_with_detail(
                &PROD_ERROR_INVALID_ARGUMENT,
                "dimension vector must be a row or column vector",
            ));
        }
        let gathered;
        let value = if let Value::GpuTensor(handle) = value {
            call.fallback([value])?;
            let readback =
                runmat_accelerate_api::numeric_readback_for_handle(handle).map_err(|source| {
                    crate::runtime_error::with_provider_execution_source(
                        prod_internal_error("dimension has no owning provider"),
                        source.into(),
                    )
                })?;
            gathered = gpu_helpers::download_readback_value_in_context(call.runtime, &readback)
                .await
                .map_err(super::super::numerical_execution::terminal)?;
            call.check()?;
            &gathered
        } else {
            value
        };
        let mut dims = tensor::borrowed_host_shape_dimensions(value).ok_or_else(|| {
            prod_descriptor_error_with_detail(
                &PROD_ERROR_INVALID_ARGUMENT,
                "expected a positive dimension, dimension vector, or supported option",
            )
        })?;
        selection = match dims.len() {
            0 => DimSelection::Auto,
            1 => DimSelection::Dim(positive(dims.next().expect("one selector"))?),
            _ => {
                call.host()?;
                DimSelection::Vec(dims.map(positive).collect::<BuiltinResult<Vec<_>>>()?)
            }
        };
        selection_set = true;
    }
    Ok(ParsedArguments {
        selection,
        nan_mode,
        output,
    })
}
fn vector_shape(shape: &[usize]) -> bool {
    match shape.len() {
        0 | 1 => true,
        2 => shape[0] == 1 || shape[1] == 1,
        _ => shape.iter().filter(|n| **n > 1).count() <= 1,
    }
}
fn positive(value: Result<usize, tensor::DimensionDecodeError>) -> BuiltinResult<usize> {
    let value = value.map_err(|source| {
        let error =
            prod_descriptor_error_with_detail(&PROD_ERROR_INVALID_ARGUMENT, source.to_string());
        source.attach_cause(error)
    })?;
    if value == 0 {
        Err(prod_descriptor_error_with_detail(
            &PROD_ERROR_INVALID_ARGUMENT,
            "dimension indices must be >= 1",
        ))
    } else {
        Ok(value)
    }
}

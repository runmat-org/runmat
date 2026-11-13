//! Sparse incomplete Cholesky factorization.

use super::chol::{sparse::publish_factor, CholTriangle};
use crate::builtins::common::tensor;
use crate::builtins::math::linalg::ops::sparse_factor::{
    IncompleteCholeskyKind, IncompleteCholeskyOptions, ManagedCholeskyScalar,
    ManagedSparseCholesky, ManagedSparseCholeskyError,
};
use crate::context::RuntimeContext;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::{
    BuiltinErrorDescriptor, ICHOL_ERROR_INVALID_ARGUMENT, ICHOL_ERROR_INVALID_INPUT,
    ICHOL_ERROR_NOT_POSITIVE_DEFINITE, ICHOL_ERROR_RUNTIME,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, SparseTensor, StructValue, Value};

const NAME: &str = "ichol";

#[derive(Clone, Copy, Debug)]
struct Options {
    kind: IncompleteCholeskyKind,
    drop_tolerance: f64,
    modified: bool,
    diagonal_compensation: f64,
    triangle: CholTriangle,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            kind: IncompleteCholeskyKind::NoFill,
            drop_tolerance: 0.0,
            modified: false,
            diagonal_compensation: 0.0,
            triangle: CholTriangle::Lower,
        }
    }
}

#[runtime_builtin(
    name = "ichol",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::math::linalg::factor::ichol"
)]
async fn ichol_builtin(
    runtime: Option<RuntimeContext>,
    value: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(error(&ICHOL_ERROR_INVALID_ARGUMENT, None));
    }
    let Value::SparseTensor(sparse) = value else {
        return Err(error(&ICHOL_ERROR_INVALID_INPUT, None));
    };
    let options = parse_options(rest.view()).await?;
    let runtime = runtime
        .or_else(crate::context::legacy::active)
        .ok_or_else(|| {
            error(
                &ICHOL_ERROR_RUNTIME,
                Some("ichol: runtime context unavailable"),
            )
        })?;
    let output = evaluate(sparse, options, &runtime).await?;
    if crate::output_count::current_output_count() == Some(0) {
        Ok(Value::output_list(Vec::new()))
    } else if crate::output_count::current_output_count().is_some() {
        Ok(Value::output_list(vec![output]))
    } else {
        Ok(output)
    }
}

async fn parse_options(args: crate::call::FunctionArgumentView<'_>) -> BuiltinResult<Options> {
    if args.len() > 1 {
        return Err(error(&ICHOL_ERROR_INVALID_ARGUMENT, None));
    }
    let mut options = Options::default();
    let Some(raw) = args.first() else {
        return Ok(options);
    };
    let Value::Struct(structure) = raw else {
        return Err(error(
            &ICHOL_ERROR_INVALID_ARGUMENT,
            Some("ichol: options must be a scalar struct"),
        ));
    };
    parse_struct(structure, &mut options).await?;
    Ok(options)
}

async fn parse_struct(structure: &StructValue, options: &mut Options) -> BuiltinResult<()> {
    for (name, value) in &structure.fields {
        match name.to_ascii_lowercase().as_str() {
            "type" => match text(value)?
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("nofill") => options.kind = IncompleteCholeskyKind::NoFill,
                Some("ict") => options.kind = IncompleteCholeskyKind::Threshold,
                _ => {
                    return Err(error(
                        &ICHOL_ERROR_INVALID_ARGUMENT,
                        Some("ichol: type must be 'nofill' or 'ict'"),
                    ))
                }
            },
            "droptol" => {
                let scalar = numeric_scalar(value).await?;
                if !scalar.is_finite() || scalar < 0.0 {
                    return Err(error(
                        &ICHOL_ERROR_INVALID_ARGUMENT,
                        Some("ichol: droptol must be a finite nonnegative scalar"),
                    ));
                }
                options.drop_tolerance = scalar;
            }
            "michol" => match text(value)?
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("off") => options.modified = false,
                Some("on") => options.modified = true,
                _ => {
                    return Err(error(
                        &ICHOL_ERROR_INVALID_ARGUMENT,
                        Some("ichol: michol must be 'off' or 'on'"),
                    ))
                }
            },
            "diagcomp" => {
                let scalar = numeric_scalar(value).await?;
                if !scalar.is_finite() || scalar < 0.0 {
                    return Err(error(
                        &ICHOL_ERROR_INVALID_ARGUMENT,
                        Some("ichol: diagcomp must be a finite nonnegative scalar"),
                    ));
                }
                options.diagonal_compensation = scalar;
            }
            "shape" => match text(value)?
                .as_deref()
                .map(str::to_ascii_lowercase)
                .as_deref()
            {
                Some("lower") => options.triangle = CholTriangle::Lower,
                Some("upper") => options.triangle = CholTriangle::Upper,
                _ => {
                    return Err(error(
                        &ICHOL_ERROR_INVALID_ARGUMENT,
                        Some("ichol: shape must be 'lower' or 'upper'"),
                    ))
                }
            },
            _ => {
                return Err(error(
                    &ICHOL_ERROR_INVALID_ARGUMENT,
                    Some("ichol: unknown option field"),
                ))
            }
        }
    }
    Ok(())
}

async fn numeric_scalar(value: &Value) -> BuiltinResult<f64> {
    tensor::scalar_f64_from_value_async(value)
        .await
        .map_err(|message| error(&ICHOL_ERROR_INVALID_ARGUMENT, Some(&message)))?
        .ok_or_else(|| {
            error(
                &ICHOL_ERROR_INVALID_ARGUMENT,
                Some("ichol: option must be a real scalar"),
            )
        })
}

fn text(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(runmat_value::StringElement::Text(value)) => Some(value.as_str().to_owned()),
        Value::StringArray(value) if value.data().len() == 1 && !value.data()[0].is_missing() => {
            Some(
                value.data()[0]
                    .text_ref()
                    .expect("checked nonmissing scalar string")
                    .as_str()
                    .to_owned(),
            )
        }
        Value::CharArray(value) if value.rows == 1 => Some(
            runmat_value::decode_utf16_code_units(value.data()).map_err(|source| {
                let mut error = error(&ICHOL_ERROR_INVALID_ARGUMENT, Some(&source.to_string()));
                error.source = Some(Box::new(source));
                error
            })?,
        ),
        _ => None,
    })
}

async fn evaluate(
    sparse: SparseTensor,
    options: Options,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    match sparse.numeric_dtype() {
        Some(NumericDType::F32) if sparse.as_complex_f32_slice().is_some() => {
            evaluate_typed::<faer::c32>(sparse, options, runtime).await
        }
        Some(NumericDType::F64) if sparse.as_complex_f64_slice().is_some() => {
            evaluate_typed::<faer::c64>(sparse, options, runtime).await
        }
        Some(NumericDType::F32) => evaluate_typed::<f32>(sparse, options, runtime).await,
        Some(NumericDType::F64) => evaluate_typed::<f64>(sparse, options, runtime).await,
        _ => Err(error(&ICHOL_ERROR_INVALID_INPUT, None)),
    }
}

async fn evaluate_typed<T: ManagedCholeskyScalar>(
    sparse: SparseTensor,
    options: Options,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    let factor = ManagedSparseCholesky::<T>::factor_incomplete(
        &sparse,
        IncompleteCholeskyOptions {
            kind: options.kind,
            drop_tolerance: options.drop_tolerance,
            modified: options.modified,
            diagonal_compensation: options.diagonal_compensation,
        },
        matches!(options.triangle, CholTriangle::Lower),
        runtime,
    )
    .await
    .map_err(|failure| factor_error(failure, runtime))?;
    publish_factor(&factor, &sparse, options.triangle, runtime)
        .await
        .map_err(|source| {
            let mut builder = build_runtime_error(ICHOL_ERROR_RUNTIME.message)
                .with_builtin(NAME)
                .with_source(source);
            if let Some(identifier) = ICHOL_ERROR_RUNTIME.identifier {
                builder = builder.with_identifier(identifier);
            }
            builder.build()
        })
}

fn factor_error(error_value: ManagedSparseCholeskyError, runtime: &RuntimeContext) -> RuntimeError {
    match error_value {
        ManagedSparseCholeskyError::NumericalBreakdown => {
            error(&ICHOL_ERROR_NOT_POSITIVE_DEFINITE, None)
        }
        ManagedSparseCholeskyError::NotSquare
        | ManagedSparseCholeskyError::UnsupportedClass
        | ManagedSparseCholeskyError::NonFiniteInput
        | ManagedSparseCholeskyError::InvalidTopology(_) => {
            error(&ICHOL_ERROR_INVALID_INPUT, Some(&error_value.to_string()))
        }
        ManagedSparseCholeskyError::Admission(source) => {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(runtime),
                "RunMat:ichol:Runtime",
                "ichol: sparse factor allocation",
                source,
            )
        }
        ManagedSparseCholeskyError::Execution(source) => {
            source.into_runtime_error_in_context(Some(runtime))
        }
        other => error(&ICHOL_ERROR_RUNTIME, Some(&other.to_string())),
    }
}

fn error(descriptor: &'static BuiltinErrorDescriptor, detail: Option<&str>) -> RuntimeError {
    let mut builder = build_runtime_error(detail.unwrap_or(descriptor.message)).with_builtin(NAME);
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::host_effect::ManagedHostEffectLedger;
    use crate::context::{
        ParallelCapability, RuntimeParallelResources, RuntimeParallelService, RuntimeServicePorts,
    };
    use crate::execution::RuntimeExecutionService;
    use futures::executor::block_on;
    use std::rc::Rc;

    struct Parallel(ManagedHostEffectLedger);
    impl RuntimeParallelService for Parallel {
        fn supports(&self, _: ParallelCapability) -> bool {
            false
        }
        fn placement_resources(&self) -> RuntimeParallelResources {
            let state = self.0.snapshot();
            RuntimeParallelResources {
                memory_available_bytes: Some(state.free_bytes()),
                epoch: state.epoch,
                ..Default::default()
            }
        }
        fn managed_host_effect_ledger(&self) -> Option<ManagedHostEffectLedger> {
            Some(self.0.clone())
        }
    }

    fn runtime() -> RuntimeContext {
        let ledger = ManagedHostEffectLedger::new(1_000_000);
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
            RuntimeServicePorts::default().with_parallel(Rc::new(Parallel(ledger))),
        )
    }

    fn ordinary_runtime() -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
    }

    fn matrix() -> SparseTensor {
        SparseTensor::new(
            3,
            3,
            vec![0, 2, 5, 7],
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
        )
        .unwrap()
    }

    #[test]
    fn public_nofill_retains_sparse_lower_pattern() {
        let runtime = runtime();
        let output = block_on(ichol_builtin(
            Some(runtime),
            Value::SparseTensor(matrix()),
            crate::execution::RootedValues::new(Vec::new())
                .unwrap()
                .into(),
        ))
        .unwrap();
        let Value::SparseTensor(output) = output else {
            panic!("sparse output")
        };
        assert_eq!((output.rows, output.cols), (3, 3));
        assert!(output.nnz() <= 5);
    }

    #[test]
    fn public_nofill_publishes_under_ordinary_unbounded_policy() {
        let output = block_on(ichol_builtin(
            Some(ordinary_runtime()),
            Value::SparseTensor(matrix()),
            crate::execution::RootedValues::new(Vec::new())
                .unwrap()
                .into(),
        ))
        .unwrap();
        let Value::SparseTensor(output) = output else {
            panic!("sparse output")
        };
        assert_eq!((output.rows, output.cols), (3, 3));
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F64));
    }

    #[test]
    fn invalid_options_are_rejected_before_factor_allocation() {
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.fields.insert("type".into(), Value::from("other"));
        let args =
            crate::execution::RootedValues::new(vec![Value::Struct(options.finish())]).unwrap();
        let failure = block_on(ichol_builtin(
            Some(runtime()),
            Value::SparseTensor(matrix()),
            args.into(),
        ))
        .unwrap_err();
        assert_eq!(failure.identifier(), Some("RunMat:ichol:InvalidArgument"));
    }

    #[test]
    fn threshold_upper_form_preserves_physical_single_class() {
        let sparse = SparseTensor::new_f32(
            3,
            3,
            vec![0, 1, 3, 5],
            vec![0, 0, 1, 1, 2],
            vec![4.0, 1.0, 3.0, 1.0, 2.0],
        )
        .unwrap();
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.fields.insert("type".into(), Value::from("ict"));
        options.fields.insert("droptol".into(), Value::Num(0.0));
        options.fields.insert("shape".into(), Value::from("upper"));
        let args =
            crate::execution::RootedValues::new(vec![Value::Struct(options.finish())]).unwrap();
        let output = block_on(ichol_builtin(
            Some(runtime()),
            Value::SparseTensor(sparse),
            args.into(),
        ))
        .unwrap();
        let Value::SparseTensor(output) = output else {
            panic!("sparse output")
        };
        assert_eq!(output.numeric_dtype(), Some(NumericDType::F32));
        for column in 0..output.cols {
            assert!(
                output.row_indices[output.col_ptrs[column]..output.col_ptrs[column + 1]]
                    .iter()
                    .all(|&row| row <= column)
            );
        }
    }
}

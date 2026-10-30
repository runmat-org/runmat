use super::*;

const COMMON_STATFUNS: [&str; 48] = [
    "bootci",
    "bootstrp",
    "crossval",
    "factoran",
    "fitglm",
    "fitlm",
    "fitlme",
    "fitnlm",
    "fitrgp",
    "gamfit",
    "gevfit",
    "glmfit",
    "gmdistribution",
    "gpfit",
    "kmeans",
    "kmedoids",
    "lasso",
    "lassoglm",
    "lognfit",
    "mlecustom",
    "mlecov",
    "mvncdf",
    "mvtcdf",
    "nbinfit",
    "nlinfit",
    "nnmf",
    "normfit",
    "parallel",
    "pca",
    "plsregress",
    "ppca",
    "rocmetrics",
    "sequentialfs",
    "tsne",
    "wblfit",
    "copulafit",
    "coxphfit",
    "evfit",
    "fitcox",
    "fitglme",
    "fitlmematrix",
    "mdscale",
    "nlmefitsa",
    "treebagger",
    "candexch",
    "cordexch",
    "daugment",
    "dcovary",
];

pub(super) fn merge_old_into_new(
    runtime: Option<&crate::context::RuntimeContext>,
    oldopts: &StructValue,
    newopts: &StructValue,
) -> BuiltinResult<StructValue> {
    let canonical_new = canonicalize_options(runtime, newopts)?;
    build_options(runtime, |field| {
        let next = lookup_struct_field(&canonical_new, field.name());
        Ok(if next.is_some_and(|value| !is_empty_value(value)) {
            next.expect("checked nonempty option").clone()
        } else {
            match lookup_struct_field(oldopts, field.name()) {
                Some(value) => value.clone(),
                None => empty_option_value(runtime, field)?,
            }
        })
    })
}

pub(super) fn canonicalize_options(
    runtime: Option<&crate::context::RuntimeContext>,
    options: &StructValue,
) -> BuiltinResult<StructValue> {
    for name in options.fields.keys() {
        canonical_option_name(name)?;
    }
    build_options(runtime, |field| {
        options
            .fields
            .iter()
            .find_map(|(name, value)| {
                canonical_option_name(name)
                    .ok()
                    .filter(|canonical| *canonical == field)
                    .map(|_| value)
            })
            .map(|value| validate_option_value(field, value))
            .unwrap_or_else(|| empty_option_value(runtime, field))
    })
}

pub(super) fn defaults_for_statfun(
    runtime: Option<&crate::context::RuntimeContext>,
    statfun: &str,
) -> BuiltinResult<StructValue> {
    let key = statfun.to_ascii_lowercase();
    if !COMMON_STATFUNS.contains(&key.as_str()) {
        return Err(statset_error(
            &STATSET_ERROR_INVALID_STATFUN,
            format!("statset: unsupported statistics function '{statfun}'"),
        ));
    }
    build_options(runtime, |field| match (key.as_str(), field) {
        (
            "fitglm" | "fitlm" | "fitnlm" | "glmfit" | "lasso" | "lassoglm" | "nlinfit" | "normfit"
            | "wblfit",
            StatOptionField::Display,
        ) => text_literal(runtime, "off"),
        (
            "fitglm" | "fitlm" | "fitnlm" | "glmfit" | "lasso" | "lassoglm" | "nlinfit" | "normfit"
            | "wblfit",
            StatOptionField::MaxIter,
        ) => Ok(Value::Num(100.0)),
        (
            "fitglm" | "fitlm" | "fitnlm" | "glmfit" | "lasso" | "lassoglm" | "nlinfit" | "normfit"
            | "wblfit",
            StatOptionField::TolX,
        ) => Ok(Value::Num(1.0e-6)),
        ("factoran", StatOptionField::Display) => text_literal(runtime, "off"),
        ("factoran", StatOptionField::MaxIter) => Ok(Value::Num(100.0)),
        ("factoran", StatOptionField::TolX) => Ok(Value::Num(1.0e-8)),
        ("nbinfit", StatOptionField::Display) => text_literal(runtime, "off"),
        ("nbinfit", StatOptionField::MaxFunEvals) => Ok(Value::Num(400.0)),
        ("nbinfit", StatOptionField::MaxIter) => Ok(Value::Num(200.0)),
        ("nbinfit", StatOptionField::TolBnd | StatOptionField::TolFun | StatOptionField::TolX) => {
            Ok(Value::Num(1.0e-6))
        }
        (
            "kmeans" | "tsne" | "pca" | "ppca" | "gmdistribution" | "kmedoids",
            StatOptionField::Display,
        ) => text_literal(runtime, "off"),
        (
            "kmeans" | "tsne" | "pca" | "ppca" | "gmdistribution" | "kmedoids",
            StatOptionField::MaxIter,
        ) => Ok(Value::Num(100.0)),
        (
            "bootci" | "bootstrp" | "crossval" | "parallel" | "sequentialfs",
            StatOptionField::UseParallel | StatOptionField::UseSubstreams,
        ) => Ok(Value::Bool(false)),
        (
            "bootci" | "bootstrp" | "crossval" | "parallel" | "sequentialfs",
            StatOptionField::Streams,
        ) => Ok(empty_cell()),
        _ => empty_option_value(runtime, field),
    })
}

pub(super) fn empty_options(
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<StructValue> {
    build_options(runtime, |field| empty_option_value(runtime, field))
}

pub(super) fn replace_option(
    runtime: Option<&crate::context::RuntimeContext>,
    options: &StructValue,
    field: StatOptionField,
    value: Value,
) -> BuiltinResult<StructValue> {
    build_options(runtime, |candidate| {
        Ok(if candidate == field {
            value.clone()
        } else {
            match lookup_struct_field(options, candidate.name()) {
                Some(value) => value.clone(),
                None => empty_option_value(runtime, candidate)?,
            }
        })
    })
}

fn empty_option_value(
    runtime: Option<&crate::context::RuntimeContext>,
    field: StatOptionField,
) -> BuiltinResult<Value> {
    if field == StatOptionField::Streams {
        let cell = crate::execution::structure::cell::from_values(runtime, &[0, 0], &[])?;
        Ok(cell.into_value_and_lease().0)
    } else {
        exact_empty_numeric(runtime)
    }
}

fn exact_empty_numeric(runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<Value> {
    let Some(runtime) = runtime else {
        return Ok(empty_numeric());
    };
    use crate::context::host_allocation::{HostAllocationPlan, HostAllocationReceipt};
    use runmat_value::{ExactHostTensorAllocation, NumericDType};
    let allocation = |source| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "RunMat:statset:Allocation",
            "statistics option empty numeric allocation failed",
            source,
        )
    };
    let retained = HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[0, 0])
        .map_err(allocation)?;
    let plan = HostAllocationPlan::new(0, retained).map_err(allocation)?;
    let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan).map_err(allocation)?
    else {
        return Ok(empty_numeric());
    };
    let output = receipt
        .with_output_allocation(runtime, |permit| {
            ExactHostTensorAllocation::new(NumericDType::F64, &[0, 0], permit)
        })
        .map_err(allocation)?;
    let output = output.into_tensor().map_err(|source| {
        crate::runtime_error::semantic_error_with_source(
            "StatsetOutput",
            "empty numeric publication failed",
            source,
        )
    })?;
    receipt.checkpoint(runtime).map_err(allocation)?;
    Ok(Value::Tensor(output))
}

fn text_literal(
    runtime: Option<&crate::context::RuntimeContext>,
    text: &str,
) -> BuiltinResult<Value> {
    let Some(runtime) = runtime else {
        return Ok(Value::string(text.to_owned()));
    };
    use crate::context::host_allocation::{
        HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
    };
    let allocation = |source| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "RunMat:statset:Allocation",
            "statistics option text allocation failed",
            source,
        )
    };
    let physical = runmat_value::Utf8Text::requested_bytes_for_capacity(text.len())
        .map_err(|source| allocation(HostAllocationError::Numeric(source)))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical).map_err(|_| allocation(HostAllocationError::Overflow))?,
        usize::from(!text.is_empty()),
    )
    .map_err(allocation)?;
    let plan = HostAllocationPlan::new(0, retained).map_err(allocation)?;
    let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan).map_err(allocation)?
    else {
        return Ok(Value::string(text.to_owned()));
    };
    let permit = receipt.output_permit(runtime).map_err(allocation)?;
    let value = runmat_value::Utf8Text::try_copy_str_exact_with(text, &permit, |_| {
        receipt.checkpoint(runtime)
    })
    .map_err(|source| match source {
        runmat_value::ExactArrayInitializationError::Allocation(source) => {
            allocation(HostAllocationError::Numeric(source))
        }
        runmat_value::ExactArrayInitializationError::Element(source) => allocation(source),
    })?;
    receipt.checkpoint(runtime).map_err(allocation)?;
    Ok(Value::String(value.into()))
}

fn build_options(
    runtime: Option<&crate::context::RuntimeContext>,
    mut initialize: impl FnMut(StatOptionField) -> BuiltinResult<Value>,
) -> BuiltinResult<StructValue> {
    let count = StatOptionField::ALL.len();
    if let Some(runtime) = runtime {
        let values =
            crate::execution::RootedValues::try_initialize_in_context(count, runtime, |index| {
                initialize(StatOptionField::ALL[index])
            })?;
        return values.with_owned_output_storage(|values| {
            let output = crate::execution::construct_struct_from_fields_in_context(
                runtime,
                StatOptionField::ALL
                    .iter()
                    .zip(values.as_slice())
                    .map(|(field, value)| (field.name(), value)),
            )?;
            let (output, _roots) = output.into_value_and_lease();
            let Value::Struct(output) = output else {
                unreachable!("structure constructor publishes Struct")
            };
            Ok(output)
        });
    }
    let allocation = |source| {
        crate::context::host_allocation::host_allocation_runtime_error(
            None,
            "RunMat:statset:Allocation",
            "statistics option publication requires its managed runtime",
            source,
        )
    };
    match crate::context::host_allocation::HostAllocationReceipt::policy(None)
        .map_err(allocation)?
    {
        crate::context::host_allocation::HostAllocationPolicy::Managed => {
            return Err(allocation(
                crate::context::host_allocation::HostAllocationError::MissingManagedOwner,
            ));
        }
        crate::context::host_allocation::HostAllocationPolicy::Unbounded => {}
    }
    let mut builder = runmat_value::UnboundedStructValueBuilder::with_capacity(count);
    for field in StatOptionField::ALL {
        builder.insert(field.name(), initialize(field)?);
    }
    Ok(builder.finish())
}

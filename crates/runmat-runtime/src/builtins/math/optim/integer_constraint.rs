//! Data-only extended-integer domain descriptor. Properties are its sole authority.
use crate::builtins::math::optim::common::{self, ScalarCooperation};
use crate::{BuiltinResult, RuntimeError};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};
pub(crate) mod schema;
#[cfg(test)]
mod tests;

const NAME: &str = "integerConstraint";

pub(crate) fn ensure_registered() {
    static SCHEMA: std::sync::Once = std::sync::Once::new();
    SCHEMA.call_once(|| {
        crate::execution::value_codec::register_immutable_value_class_with_scratch(
            crate::execution::value_codec::ImmutableValueClassSchema {
                class_name: schema::CLASS.display_name(),
                type_identity: schema::TYPE_IDENTITY,
                schema_version: 1,
                allowed_fields: schema::FIELDS,
                required_fields: schema::FIELDS,
                validate: schema::validate_structure,
            },
            crate::execution::value_codec::ImmutableValueClassScratch {
                len: schema::scratch_len,
                validate: schema::validate_indices,
            },
        )
        .expect("unique IntegerConstraint portable schema");
    });
    static CLASS: crate::class_registry::ClassRegistration =
        crate::class_registry::ClassRegistration::new(schema::CLASS);
    CLASS.ensure(|| {
        let properties = schema::FIELDS
            .iter()
            .map(|&name| {
                (
                    name.into(),
                    crate::class_registry::RuntimeProperty {
                        name: name.into(),
                        is_static: false,
                        is_constant: false,
                        is_dependent: false,
                        get_access: runmat_types::MemberAccess::Public,
                        set_access: runmat_types::MemberAccess::Public,
                        default_value: None,
                    },
                )
            })
            .collect();
        crate::class_registry::register_builtin_class(
            schema::CLASS,
            crate::class_registry::RuntimeClass {
                name: schema::CLASS.owned(),
                parent: None,
                properties,
                methods: Default::default(),
                events: Default::default(),
            },
        );
    });
}

#[runtime_builtin(
    name = "integerConstraint",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::integer_constraint"
)]
async fn integer_constraint_builtin_registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> BuiltinResult<crate::execution::RootedValue> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::INTEGER_CONSTRAINT_DESCRIPTOR,
        &runmat_builtins::INTEGER_CONSTRAINT_ERROR_INVALID_INPUT,
        NAME,
    )?;
    if !args.len().is_multiple_of(2) {
        return Err(invalid("expected name/value pairs and at most one output"));
    }
    let runtime = runtime.as_ref();
    crate::context::host_allocation::require_unbounded_workspace(runtime, NAME,
        "finite host-memory execution is unsupported; constructor text normalization and default-property metadata have no complete finite certificate")?;
    let mut sources: [Option<&Value>; schema::Property::ALL.len()] =
        [None; schema::Property::ALL.len()];
    for pair in args.chunks_exact(2) {
        let name = crate::builtins::common::validation::value_text_scalar_string(&pair[0])
            .ok_or_else(|| invalid("property names must be text scalars"))?;
        let key = schema::Property::parse(&name)
            .ok_or_else(|| invalid(format!("unknown property '{name}'")))?;
        let slot = schema::Property::ALL
            .iter()
            .position(|field| *field == key)
            .expect("canonical property");
        let value = pair.get(1).expect("complete property pair");
        if sources[slot].replace(value).is_some() {
            return Err(invalid(format!(
                "{} was specified more than once",
                key.name()
            )));
        }
        preflight(value, runtime)?;
    }
    let mut cooperation = ScalarCooperation::for_builtin(
        runmat_builtins::BuiltinCatalogIdentity { name: NAME },
        runtime,
    );
    cooperation.completed(0).await?;
    let mut values: [Value; schema::Property::ALL.len()] = std::array::from_fn(|_| empty());
    for (source, output) in sources.into_iter().zip(&mut values) {
        if let Some(source) = source {
            *output = crate::dispatcher::gather_if_needed_in_context_async(source, runtime).await?;
            let indices = schema::Indices::new(output)
                .ok_or_else(|| invalid("indices must be full double vectors"))?;
            for _ in indices.iter() {
                cooperation.completed(1).await?;
            }
        }
    }
    {
        let _entry = runtime.map(crate::context::RuntimeContext::enter);
        ensure_registered();
    }
    let result = crate::execution::object::construct_object_from_borrowed_fields(
        runtime,
        schema::CLASS.display_name(),
        schema::Property::ALL
            .into_iter()
            .zip(&values)
            .map(|(key, value)| (key.name(), value)),
    )?;
    let Value::Object(object) = result.value() else {
        unreachable!("object construction")
    };
    schema::validate(object, runtime).map_err(|error| validation_error(runtime, error))?;
    Ok(result)
}

fn preflight(value: &Value, runtime: Option<&crate::context::RuntimeContext>) -> BuiltinResult<()> {
    if !matches!(
        value,
        Value::Num(_) | Value::Tensor(_) | Value::GpuTensor(_)
    ) {
        return Err(invalid("indices must be full double vectors"));
    }
    if crate::builtins::introspection::underlying_type::underlying_class_identity_for_value_checked(
        value,
    )? != runmat_types::standard::DOUBLE
    {
        return Err(invalid("indices must have double storage"));
    }
    let fact = crate::value_fact::value_fact(value);
    if !matches!(
        fact.kind,
        runmat_types::ValueKindFact::Numeric(runmat_types::NumericFact {
            domain: runmat_types::NumericDomain::Real,
            ..
        })
    ) {
        return Err(invalid("indices must be real double vectors"));
    }
    if fact.shape.element_count() != Some(0)
        && (fact.shape.rank().is_some_and(|rank| rank > 2)
            || fact.shape.known_dims().is_some_and(|dims| {
                dims.len() == 2 && dims[0].is_some_and(|n| n > 1) && dims[1].is_some_and(|n| n > 1)
            }))
    {
        return Err(invalid("indices must be vectors"));
    }
    common::preflight_host_residency(NAME, value, None, runtime)?;
    if !matches!(value, Value::GpuTensor(_)) && schema::Indices::new(value).is_none() {
        return Err(invalid("indices must be vectors"));
    }
    Ok(())
}
fn empty() -> Value {
    Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty vector"))
}
fn invalid(detail: impl std::fmt::Display) -> RuntimeError {
    crate::build_runtime_error(format!("integerConstraint: {detail}"))
        .with_builtin(NAME)
        .with_identifier(
            runmat_builtins::INTEGER_CONSTRAINT_ERROR_INVALID_INPUT
                .identifier
                .expect("canonical error identity"),
        )
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

pub(crate) fn validation_error(
    runtime: Option<&crate::context::RuntimeContext>,
    error: crate::execution::value_codec::ValueCodecError,
) -> RuntimeError {
    match error {
        crate::execution::value_codec::ValueCodecError::Allocation { source, .. } => {
            let mut error = crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:integerConstraint:Allocation",
                "IntegerConstraint validation scratch admission failed",
                source,
            );
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        }
        error => invalid(error),
    }
}

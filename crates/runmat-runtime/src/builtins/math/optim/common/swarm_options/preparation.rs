//! Preflight every active carrier before transfer. The vectorized route owns
//! precedence, so a dormant parallel control is never inspected or materialized.
use super::{invalid, metadata, validate, Field, Owner};
use crate::builtins::math::optim::common::{ScalarCooperation, ScalarOptionFields};
use crate::{execution::RootedValue, BuiltinResult};
use runmat_value::{StructValue, Value};
pub(crate) struct Prepared(RootedValue);
impl Prepared {
    pub(crate) fn structure(&self) -> &StructValue {
        let Value::Struct(v) = self.0.value() else {
            unreachable!("prepared swarm controls")
        };
        v
    }
}
pub(crate) async fn prepare(
    value: Value,
    variables: usize,
    numeric: &'static runmat_builtins::BuiltinExtensionDescriptor,
    resident: &'static runmat_builtins::BuiltinExtensionDescriptor,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Prepared> {
    let owner = Owner::Solver;
    let value = if matches!(&value,Value::Tensor(v)if v.is_empty()) {
        Value::Struct(runmat_value::UnboundedStructValueBuilder::new().finish())
    } else {
        value
    };
    let source = RootedValue::in_optional_context(value, cooperation.runtime())?;
    let Value::Struct(options) = source.value() else {
        return Err(invalid(
            owner,
            "options must be a scalar options structure or []",
        ));
    };
    for (name, value) in &options.fields {
        if name.eq_ignore_ascii_case("Solver") {
            if runmat_builtins::OptimSolver::parse(&super::text(owner, value, Field::Display)?)
                != Some(runmat_builtins::OptimSolver::Particleswarm)
            {
                return Err(invalid(owner, "options Solver must be particleswarm"));
            }
            continue;
        }
        let field = Field::parse_for_solver(name, runmat_builtins::OptimSolver::Particleswarm)
            .filter(|field| runmat_builtins::OptimSolver::Particleswarm.accepts(*field))
            .ok_or_else(|| invalid(owner, format!("unsupported particleswarm option {name}")))?;
        if field != Field::UseParallel {
            metadata(owner, field, value, resident, cooperation.runtime())?;
            dimensions(field, value, variables)?;
            integer_extension(value, numeric, cooperation.runtime())?;
            validate(owner, field, value)?;
        }
    }
    let fields = ScalarOptionFields::from_struct(Some(options));
    let vectorized = if let Some(value) = fields.exact(Field::UseVectorized) {
        let value = materialize(Field::UseVectorized, value, cooperation).await?;
        validate(owner, Field::UseVectorized, &value)?;
        Some(value)
    } else {
        None
    };
    let active_parallel = !vectorized
        .as_ref()
        .map(|v| super::switch(owner, v, Field::UseVectorized))
        .transpose()?
        .unwrap_or(false);
    if active_parallel {
        if let Some(value) = fields.exact(Field::UseParallel) {
            metadata(
                owner,
                Field::UseParallel,
                value,
                resident,
                cooperation.runtime(),
            )?;
            integer_extension(value, numeric, cooperation.runtime())?;
            validate(owner, Field::UseParallel, value)?;
        }
    }
    let mut replacements: [Option<(Field, Value)>; Field::ALL.len()] =
        std::array::from_fn(|_| None);
    for (index, field) in Field::ALL.into_iter().enumerate() {
        if field == Field::UseParallel && !active_parallel {
            continue;
        }
        let Some(value) = fields.exact(field) else {
            continue;
        };
        let replacement = if field == Field::UseVectorized {
            vectorized.clone().expect("selected vectorized value")
        } else {
            materialize(field, value, cooperation).await?
        };
        validate(owner, field, &replacement)?;
        replacements[index] = Some((field, replacement));
        cooperation.completed(1).await?;
    }
    let selected = options.fields.iter().map(|(name, original)| {
        let selected = Field::parse(name)
            .and_then(|field| {
                replacements
                    .iter()
                    .flatten()
                    .find_map(|(candidate, value)| (*candidate == field).then_some(value))
            })
            .unwrap_or(original);
        (name, selected)
    });
    let output = crate::execution::construct_struct_from_fields_in_optional_context(
        cooperation.runtime(),
        selected,
    )?;
    Ok(Prepared(output))
}
fn dimensions(field: Field, value: &Value, variables: usize) -> BuiltinResult<()> {
    let owner = Owner::Solver;
    if matches!(value,Value::Tensor(v)if v.is_empty()) {
        return Ok(());
    }
    if matches!(field, Field::InitialSwarmSpan | Field::InertiaRange) {
        let count = crate::value_fact::value_fact(value).shape.element_count();
        if count.is_some_and(|n| {
            if field == Field::InertiaRange {
                n != 2
            } else {
                n != 1 && n != variables
            }
        }) {
            return Err(invalid(
                owner,
                "active span/inertia extent does not match its documented role",
            ));
        }
    }
    if field == Field::InitialPoints {
        let (x, f) = if let Value::Struct(points) = value {
            (&points.fields["X"], points.fields.get("Fval"))
        } else {
            (value, None)
        };
        let x = crate::value_fact::value_fact(x);
        let dims = x.shape.known_dims();
        if dims
            .as_ref()
            .is_some_and(|d| d.len() != 2 || d[1].is_some_and(|n| n != variables))
        {
            return Err(invalid(owner, "InitialPoints.X must have nvars columns"));
        }
        if let Some(f) = f {
            let f = crate::value_fact::value_fact(f);
            if f.shape.element_count() != Some(0)
                && (f.shape.known_dims().is_some_and(|d| {
                    d.len() > 2 || d.iter().filter(|n| n.is_some_and(|n| n != 1)).count() > 1
                }) || dims
                    .as_ref()
                    .and_then(|d| d.first().copied().flatten())
                    .zip(f.shape.element_count())
                    .is_some_and(|(rows, n)| n > rows))
            {
                return Err(invalid(
                    owner,
                    "InitialPoints.Fval must be a vector no longer than X rows",
                ));
            }
        }
    }
    Ok(())
}
fn integer_extension(
    value: &Value,
    extension: &'static runmat_builtins::BuiltinExtensionDescriptor,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if crate::builtins::math::optim::common::is_typed_integer_value(value) {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            runtime,
            extension,
            "particleswarm",
        )?;
    }
    Ok(())
}
async fn materialize(
    field: Field,
    value: &Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    if matches!(field, Field::InitialSwarmSpan | Field::InertiaRange) {
        return crate::builtins::math::optim::common::gather_exact_floating_value_with_cooperation(
            "particleswarm",
            value.clone(),
            field.name(),
            cooperation,
        )
        .await;
    }
    if field == Field::InitialPoints {
        if let Value::Struct(points) = value {
            let mut fields = runmat_value::UnboundedStructValueBuilder::new();
            for (name, array) in &points.fields {
                let gathered = crate::dispatcher::gather_if_needed_in_context_async(
                    array,
                    cooperation.runtime(),
                )
                .await?;
                fields.insert(name, gathered);
                cooperation.completed(1).await?;
            }
            return Ok(Value::Struct(fields.finish()));
        }
    }
    if field.is_callback()
        || matches!(field, Field::Display | Field::FunValCheck)
        || crate::builtins::duration::is_duration_object(value)
    {
        return Ok(value.clone());
    }
    crate::dispatcher::gather_if_needed_in_context_async(value, cooperation.runtime()).await
}

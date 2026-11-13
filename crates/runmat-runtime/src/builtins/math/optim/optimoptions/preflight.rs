//! Preserve builder grammar and host diagnostics before any option readback.
use super::*;

pub(super) fn arguments<'a>(
    args: crate::call::FunctionArgumentView<'a>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<ArgumentRoles<'a>> {
    let Some(first) = args.first() else {
        return Err(optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
            "optimoptions: expected a solver name or options struct",
        ));
    };
    let mut roles =
        crate::context::host_allocation::prepare_host_array_optional(runtime, args.len())?;
    let explicit = !matches!(first, Value::Struct(_));
    let mut solver = match first {
        Value::Struct(options) => {
            let solver = solver_from_options(options)?;
            fields(options, solver, runtime)?;
            solver
        }
        value => parse_solver(value)?,
    };
    roles
        .try_push(if matches!(first, Value::Struct(_)) {
            ArgumentRole::Structure { solver }
        } else {
            ArgumentRole::Selector
        })
        .expect("first argument slot");
    let mut index = 1;
    while let Some(argument) = args.get(index) {
        if let Value::Struct(options) = argument {
            let next = solver_from_options(options)?;
            if !explicit && next != Solver::Generic {
                solver = next;
            }
            fields(options, solver, runtime)?;
            roles
                .try_push(ArgumentRole::Structure { solver })
                .expect("structure argument slot");
            index += 1;
        } else {
            let name = expect_string_scalar(
                argument,
                "optimoptions: option names must be character vectors or string scalars",
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_NAME,
            )?;
            let Some(value) = args.get(index + 1) else {
                return Err(optimoptions_error_with(
                    &OPTIMOPTIONS_ERROR_MISSING_OPTION_VALUE,
                    format!("optimoptions: missing value for option '{name}'"),
                ));
            };
            let field = admitted_option_field(solver, &name)?;
            value_for_field(solver, field, value, runtime)?;
            roles
                .try_push(ArgumentRole::Selector)
                .expect("option name slot");
            roles
                .try_push(ArgumentRole::Option { solver, field })
                .expect("option value slot");
            index += 2;
        }
    }
    let roles = roles.try_finish().map_err(|_| {
        optimoptions_error_with(
            &OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
            "optimoptions argument-role cardinality changed",
        )
    })?;
    Ok(ArgumentRoles {
        roles,
        source: args,
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) enum ArgumentRole {
    Selector,
    Option {
        solver: Solver,
        field: OptimOptionField,
    },
    Structure {
        solver: Solver,
    },
}
pub(super) struct ArgumentRoles<'a> {
    roles: runmat_value::ExactArrayStorage<ArgumentRole>,
    source: crate::call::FunctionArgumentView<'a>,
}
impl ArgumentRoles<'_> {
    pub fn len(&self) -> usize {
        self.roles.len()
    }
    pub fn get(&self, index: usize) -> BuiltinResult<ArgumentRole> {
        if self.roles.len() != self.source.len() {
            return Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
                "optimoptions argument-role cardinality changed",
            ));
        }
        self.roles.get(index).copied().ok_or_else(|| {
            optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_ARGUMENT,
                "optimoptions argument role missing",
            )
        })
    }
}

fn fields(
    options: &StructValue,
    solver: Solver,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    for (name, value) in &options.fields {
        if name.eq_ignore_ascii_case("Solver") {
            continue;
        }
        value_for_field(solver, admitted_option_field(solver, name)?, value, runtime)?;
    }
    Ok(())
}

fn value_for_field(
    solver: Solver,
    field: OptimOptionField,
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if solver == Solver::Surrogateopt {
        use crate::builtins::math::optim::common::surrogate_options as controls;
        return controls::metadata(controls::Policy::constructor(), field, value, runtime);
    }
    if solver == Solver::Ga {
        return crate::builtins::math::optim::ga::options::admission::metadata(
            crate::builtins::math::optim::ga::options::Owner::Constructor,
            field,
            value,
            runtime,
        );
    }
    if solver == Solver::Patternsearch {
        return pattern::metadata(field, value, runtime);
    }
    if solver == Solver::Simulannealbnd {
        let owner = crate::builtins::math::optim::common::anneal_options::Owner::Constructor;
        crate::builtins::math::optim::common::anneal_options::metadata(
            owner,
            field,
            value,
            Some(&RESIDENT_OPTION_EXTENSION),
            runtime,
        )?;
        if !crate::builtins::math::optim::common::anneal_options::empty(value)
            && crate::builtins::math::optim::common::is_typed_integer_value(value)
        {
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                runtime,
                &INTEGER_OPTION_EXTENSION,
                NAME,
            )?;
        }
        return Ok(());
    }
    if solver == Solver::Particleswarm {
        let owner = crate::builtins::math::optim::common::swarm_options::Owner::Constructor;
        crate::builtins::math::optim::common::swarm_options::metadata(
            owner,
            field,
            value,
            &RESIDENT_OPTION_EXTENSION,
            runtime,
        )?;
        if crate::builtins::math::optim::common::is_typed_integer_value(value) {
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                runtime,
                &INTEGER_OPTION_EXTENSION,
                NAME,
            )?;
        }
        return crate::builtins::math::optim::common::swarm_options::validate(owner, field, value);
    }
    if matches!(
        field,
        OptimOptionField::HessianApproximation | OptimOptionField::HessUpdate
    ) {
        prepared_option_value(solver, field, value)?;
        if crate::builtins::math::optim::common::quasi_newton_controls::history_operand(value)
            .is_some_and(crate::builtins::math::optim::common::is_typed_integer_value)
        {
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                runtime,
                &INTEGER_OPTION_EXTENSION,
                NAME,
            )?;
        }
    }
    if !field.is_callback() {
        crate::builtins::math::optim::common::preflight_host_residency(
            NAME,
            value,
            Some(&RESIDENT_OPTION_EXTENSION),
            runtime,
        )?;
        if crate::builtins::math::optim::common::is_typed_integer_value(value)
            || matches!(value, Value::SparseTensor(t) if t.integer_storage().is_some())
        {
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                runtime,
                &INTEGER_OPTION_EXTENSION,
                NAME,
            )?;
        }
    }
    if matches!(value, Value::GpuTensor(_)) {
        let fact = crate::value_fact::value_fact(value);
        let scalar_real = fact.shape.element_count() == Some(1)
            && (matches!(fact.kind, runmat_types::ValueKindFact::Numeric(numeric)
                if numeric.domain == runmat_types::NumericDomain::Real)
                || matches!(fact.kind, runmat_types::ValueKindFact::Logical));
        let pattern = field == OptimOptionField::HessPattern && pattern_metadata(value);
        let scaling = matches!(
            field,
            OptimOptionField::FiniteDifferenceStepSize | OptimOptionField::TypicalX
        );
        if scaling {
            crate::builtins::math::optim::common::derivative_scaling::metadata(NAME, value)?;
        }
        if (!scalar_real && !pattern && !scaling)
            || matches!(
                field,
                OptimOptionField::Display
                    | OptimOptionField::FiniteDifferenceType
                    | OptimOptionField::MeritFunction
                    | OptimOptionField::Algorithm
                    | OptimOptionField::OutputFcn
                    | OptimOptionField::PlotFcn
                    | OptimOptionField::HessianFcn
                    | OptimOptionField::JacobianMultiplyFcn
                    | OptimOptionField::HessianMultiplyFcn
                    | OptimOptionField::SubproblemAlgorithm
                    | OptimOptionField::HessianApproximation
                    | OptimOptionField::HessUpdate
            )
        {
            return Err(optimoptions_error_with(
                &OPTIMOPTIONS_ERROR_INVALID_OPTION_VALUE,
                format!(
                    "optimoptions: option {} has unsupported scalar class or geometry",
                    field.name()
                ),
            ));
        }
    } else {
        prepared_option_value(solver, field, value)?;
    }
    Ok(())
}

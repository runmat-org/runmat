//! Resolve only the controls that determine effective algorithm/model roles.
//! Dormant fields remain original rooted carriers, not validation work.
use super::*;
pub(super) enum InputContext {
    Unspecified,
    FminuncModel,
    ResidualModel,
    Fmincon { nonlinear_constraints: bool },
    Multiobjective { nonlinear_constraints: bool },
}

pub(super) struct Plan {
    quadratic_time: bool,
    fminunc: bool,
    residual: bool,
    numerical_derivatives: bool,
    fmincon: bool,
    multiobjective: bool,
    fmincon_nonlinear: bool,
    constraint_gradient: bool,
    trust: bool,
    supplied_gradient: bool,
    objective: bool,
    multiply: bool,
    factorization: bool,
    direct: bool,
    gradient: Option<Value>,
    constraint: Option<Value>,
    bandwidth: Option<Value>,
}

impl Plan {
    pub(super) fn applies(&self, field: OptimOptionField) -> bool {
        use OptimOptionField::*;
        if self.residual
            && matches!(
                field,
                JacobianMultiplyFcn
                    | OptimalityTolerance
                    | SubproblemAlgorithm
                    | PrecondBandWidth
                    | MaxPCGIter
                    | TolPCG
            )
        {
            // Residual extent determines the actual controller. These original
            // rooted carriers are read only by the selected reflective phase.
            return false;
        }
        if self.quadratic_time && field == MaxTime {
            return false;
        }
        if matches!(field, FiniteDifferenceStepSize | TypicalX) {
            return self.numerical_derivatives;
        }
        if (self.fmincon || self.multiobjective) && field == SpecifyConstraintGradient {
            return self.fmincon_nonlinear && self.constraint_gradient;
        }
        if !self.fminunc && !self.fmincon {
            return true;
        }
        if field.trust_region_only() && !self.trust {
            return false;
        }
        match field {
            HessianApproximation | HessUpdate => !self.trust,
            UseParallel => !self.supplied_gradient,
            HessianMultiplyFcn => !self.objective,
            HessPattern => !self.objective && !self.multiply,
            PrecondBandWidth => !self.factorization,
            MaxPCGIter | TolPCG => !self.factorization && !self.direct,
            _ => true,
        }
    }
    pub(super) fn take(&mut self, field: OptimOptionField) -> Option<Value> {
        match field {
            OptimOptionField::SpecifyObjectiveGradient => self.gradient.take(),
            OptimOptionField::SpecifyConstraintGradient => self.constraint.take(),
            OptimOptionField::PrecondBandWidth => self.bandwidth.take(),
            _ => None,
        }
    }
}

pub(super) async fn prepare(
    solver: OptimSolver,
    fields: &ScalarOptionFields<'_>,
    inputs: InputContext,
    numeric_extension: Option<&'static BuiltinExtensionDescriptor>,
    resident_extension: Option<&'static BuiltinExtensionDescriptor>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Plan> {
    use OptimOptionField::*;
    let mut plan = Plan {
        quadratic_time: solver == OptimSolver::Quadprog,
        fminunc: solver == OptimSolver::Fminunc,
        residual: solver == OptimSolver::Lsqnonlin,
        numerical_derivatives: true,
        fmincon: solver == OptimSolver::Fmincon,
        multiobjective: matches!(solver, OptimSolver::Fminimax | OptimSolver::Fgoalattain),
        fmincon_nonlinear: match inputs {
            InputContext::Unspecified => true,
            InputContext::FminuncModel | InputContext::ResidualModel => false,
            InputContext::Fmincon {
                nonlinear_constraints,
            }
            | InputContext::Multiobjective {
                nonlinear_constraints,
            } => nonlinear_constraints,
        },
        constraint_gradient: false,
        trust: false,
        supplied_gradient: false,
        objective: false,
        multiply: false,
        factorization: false,
        direct: false,
        gradient: None,
        constraint: None,
        bandwidth: None,
    };
    if plan.fmincon || plan.multiobjective {
        if plan.fmincon {
            let algorithm = fields
                .exact(Algorithm)
                .map(|value| {
                    super::super::super::option_string_value(
                        Some(value),
                        "Algorithm",
                        "interior-point",
                    )
                })
                .transpose()?
                .map(|text| {
                    OptimAlgorithm::parse(&text)
                        .ok_or_else(|| optim_error(solver.name(), "unsupported Algorithm"))
                })
                .transpose()?
                .unwrap_or(OptimAlgorithm::InteriorPoint);
            plan.trust = algorithm == OptimAlgorithm::TrustRegionReflective;
            if plan.trust && plan.fmincon_nonlinear {
                return Err(crate::builtins::math::optim::fmincon::invalid(
                    "trust-region-reflective does not support nonlinear constraints",
                ));
            }
        }
        if let Some(value) = fields.exact(SpecifyObjectiveGradient) {
            preflight(
                solver,
                value,
                numeric_extension,
                resident_extension,
                runtime,
            )?;
            if matches!(value, Value::GpuTensor(_)) {
                validate_switch_metadata(solver.name(), SpecifyObjectiveGradient, value)?;
                let value = crate::dispatcher::gather_if_needed_in_context_async(
                    value,
                    cooperation.runtime(),
                )
                .await?;
                plan.constraint_gradient =
                    switch_value(solver.name(), SpecifyObjectiveGradient.name(), &value)?;
                plan.gradient = Some(value);
                cooperation.completed(0).await?;
            } else {
                plan.constraint_gradient =
                    switch_value(solver.name(), SpecifyObjectiveGradient.name(), value)?;
            }
        }
        // Preserve the requested role separately from the decoded switch.
        // A gathered false constraint flag still replaces its original carrier.
        plan.numerical_derivatives = !plan.constraint_gradient;
        plan.supplied_gradient = plan.constraint_gradient;
        if plan.trust && !plan.constraint_gradient {
            return Err(crate::builtins::math::optim::fmincon::invalid(
                "trust-region-reflective requires SpecifyObjectiveGradient=true",
            ));
        }
        if plan.fmincon_nonlinear && plan.constraint_gradient {
            let supplied = if let Some(value) = fields.exact(SpecifyConstraintGradient) {
                preflight(
                    solver,
                    value,
                    numeric_extension,
                    resident_extension,
                    runtime,
                )?;
                if matches!(value, Value::GpuTensor(_)) {
                    validate_switch_metadata(solver.name(), SpecifyConstraintGradient, value)?;
                    let value = crate::dispatcher::gather_if_needed_in_context_async(
                        value,
                        cooperation.runtime(),
                    )
                    .await?;
                    let enabled =
                        switch_value(solver.name(), SpecifyConstraintGradient.name(), &value)?;
                    plan.constraint = Some(value);
                    cooperation.completed(0).await?;
                    enabled
                } else {
                    switch_value(solver.name(), SpecifyConstraintGradient.name(), value)?
                }
            } else {
                false
            };
            plan.numerical_derivatives = !supplied;
        }
        if plan.trust {
            return prepare_model_roles(
                plan,
                solver,
                fields,
                numeric_extension,
                resident_extension,
                runtime,
                cooperation,
            )
            .await;
        }
        return Ok(plan);
    }
    if plan.residual {
        if let Some(value) = fields.exact(Algorithm) {
            if !matches!(
                value,
                Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
            ) {
                return Err(optim_error(solver.name(), "Algorithm must be scalar text"));
            }
            let text = super::super::super::option_string_value(Some(value), "Algorithm", "")?;
            if !OptimAlgorithm::parse(&text)
                .is_some_and(|algorithm| solver.accepts_algorithm(algorithm))
            {
                return Err(optim_error(
                    solver.name(),
                    "unsupported lsqnonlin Algorithm",
                ));
            }
        }
    }
    if matches!(solver, OptimSolver::Fsolve | OptimSolver::Lsqnonlin) {
        if let Some(value) = fields.exact(SpecifyObjectiveGradient) {
            preflight(
                solver,
                value,
                numeric_extension,
                resident_extension,
                runtime,
            )?;
            let enabled = if matches!(value, Value::GpuTensor(_)) {
                validate_switch_metadata(solver.name(), SpecifyObjectiveGradient, value)?;
                let value = crate::dispatcher::gather_if_needed_in_context_async(
                    value,
                    cooperation.runtime(),
                )
                .await?;
                let enabled = switch_value(solver.name(), SpecifyObjectiveGradient.name(), &value)?;
                plan.gradient = Some(value);
                cooperation.completed(0).await?;
                enabled
            } else {
                switch_value(solver.name(), SpecifyObjectiveGradient.name(), value)?
            };
            if enabled && matches!(inputs, InputContext::ResidualModel) {
                return Err(optim_error(
                    solver.name(),
                    "modeled supplied Jacobian declarations are unavailable for this consumer",
                ));
            }
            plan.numerical_derivatives = !enabled;
        }
    }
    if !plan.fminunc {
        return Ok(plan);
    }
    let algorithm = fields
        .exact(Algorithm)
        .map(|value| {
            let text =
                super::super::super::option_string_value(Some(value), "Algorithm", "quasi-newton")?;
            OptimAlgorithm::parse(&text)
                .filter(|algorithm| solver.accepts_algorithm(*algorithm))
                .ok_or_else(|| optim_error(solver.name(), "unsupported Algorithm"))
        })
        .transpose()?
        .unwrap_or(OptimAlgorithm::QuasiNewton);
    plan.supplied_gradient = if let Some(value) = fields.exact(SpecifyObjectiveGradient) {
        preflight(
            solver,
            value,
            numeric_extension,
            resident_extension,
            runtime,
        )?;
        if matches!(value, Value::GpuTensor(_)) {
            validate_switch_metadata(solver.name(), SpecifyObjectiveGradient, value)?;
            let value =
                crate::dispatcher::gather_if_needed_in_context_async(value, cooperation.runtime())
                    .await?;
            let enabled = switch_value(solver.name(), SpecifyObjectiveGradient.name(), &value)?;
            plan.gradient = Some(value);
            cooperation.completed(0).await?;
            enabled
        } else {
            switch_value(solver.name(), SpecifyObjectiveGradient.name(), value)?
        }
    } else {
        false
    };
    if matches!(inputs, InputContext::FminuncModel) && plan.supplied_gradient {
        return Err(optim_error(
            solver.name(),
            "modeled analytic objective derivatives are unavailable for this consumer",
        ));
    }
    plan.numerical_derivatives = !plan.supplied_gradient;
    plan.trust = algorithm == OptimAlgorithm::TrustRegion && plan.supplied_gradient;
    if !plan.trust {
        return Ok(plan);
    }
    prepare_model_roles(
        plan,
        solver,
        fields,
        numeric_extension,
        resident_extension,
        runtime,
        cooperation,
    )
    .await
}
async fn prepare_model_roles(
    mut plan: Plan,
    solver: OptimSolver,
    fields: &ScalarOptionFields<'_>,
    numeric_extension: Option<&'static BuiltinExtensionDescriptor>,
    resident_extension: Option<&'static BuiltinExtensionDescriptor>,
    runtime: Option<&crate::context::RuntimeContext>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Plan> {
    use OptimOptionField::*;
    plan.objective = fields
        .exact(HessianFcn)
        .map(|value| hessian_source(solver.name(), value))
        .transpose()?
        .is_some_and(|source| source == runmat_builtins::OptimHessianSource::Objective);
    plan.multiply = !plan.objective && fields.exact(HessianMultiplyFcn).is_some();
    plan.factorization = fields
        .exact(SubproblemAlgorithm)
        .map(|value| subproblem(solver.name(), value))
        .transpose()?
        .is_some_and(|algorithm| {
            algorithm == runmat_builtins::OptimSubproblemAlgorithm::Factorization
        });
    if !plan.factorization {
        if let Some(value) = fields.exact(PrecondBandWidth) {
            preflight(
                solver,
                value,
                numeric_extension,
                resident_extension,
                runtime,
            )?;
            super::super::super::preflight_option_scalar(solver.name(), value, PrecondBandWidth)?;
            if matches!(value, Value::GpuTensor(_)) {
                let value = crate::dispatcher::gather_if_needed_in_context_async(
                    value,
                    cooperation.runtime(),
                )
                .await?;
                validate_numeric(solver, PrecondBandWidth, &value)?;
                plan.direct = infinite_bandwidth(&value);
                plan.bandwidth = Some(value);
                cooperation.completed(0).await?;
            } else {
                validate_numeric(solver, PrecondBandWidth, value)?;
                plan.direct = infinite_bandwidth(value);
            }
        }
    }
    if plan.fmincon && plan.multiply && (plan.factorization || plan.direct) {
        return Err(crate::builtins::math::optim::fmincon::invalid("trust-region-reflective HessianMultiplyFcn requires cg without infinite PrecondBandWidth"));
    }
    Ok(plan)
}

fn preflight(
    solver: OptimSolver,
    value: &Value,
    numeric_extension: Option<&'static BuiltinExtensionDescriptor>,
    resident_extension: Option<&'static BuiltinExtensionDescriptor>,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    super::super::super::preflight_host_residency(
        solver.name(),
        value,
        resident_extension,
        runtime,
    )?;
    if super::super::super::is_typed_integer_value(value) {
        if let Some(extension) = numeric_extension {
            crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
                runtime,
                extension,
                solver.name(),
            )?;
        }
    }
    Ok(())
}

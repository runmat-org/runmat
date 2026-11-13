//! Private typed native-field invocation. Selection requires the actual
//! native carrier and canonical issued method, never a class-name heuristic.
use super::*;
use crate::object::indexing::ObjectSubscriptPath;
use crate::object::protocol::{ResolvedObjectMethod, ResolvedObjectMethodTarget};
use runmat_types::CallableIdentity;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Target {
    Variable,
    Expression,
    Constraint,
    Problem,
    Equation,
}
impl Target {
    fn entry(self) -> &'static runmat_builtins::BuiltinCatalogEntry {
        match self {
            Self::Variable => &runmat_builtins::OPTIMVAR_SUBSASGN_CATALOG_ENTRY,
            Self::Expression => &runmat_builtins::OPTIMEXPR_SUBSASGN_CATALOG_ENTRY,
            Self::Constraint => &runmat_builtins::OPTIMCONSTRAINT_SUBSASGN_CATALOG_ENTRY,
            Self::Problem => &runmat_builtins::OPTIMPROBLEM_SUBSASGN_CATALOG_ENTRY,
            Self::Equation => &runmat_builtins::EQNPROBLEM_SUBSASGN_CATALOG_ENTRY,
        }
    }
    pub(crate) fn select(object: &ObjectInstance, method: &ResolvedObjectMethod) -> Option<Self> {
        let target = if super::super::optimvar::is_reference(object) {
            Self::Variable
        } else if object.selected_class().is_none()
            && object.class_name().is(standard::OPTIMIZATION_PROBLEM)
            && object.optimization_problem_payload().is_some()
        {
            Self::Problem
        } else if object.selected_class().is_none()
            && object
                .class_name()
                .is(runmat_builtins::EQUATION_PROBLEM_CLASS)
            && object.equation_problem_payload().is_some()
        {
            Self::Equation
        } else if object.selected_class().is_none() {
            match object.optimization_model_payload()? {
                Payload::Expression(_) if is_expression(object) => Self::Expression,
                Payload::Constraint(_)
                    if object.class_name().is(standard::OPTIMIZATION_CONSTRAINT) =>
                {
                    Self::Constraint
                }
                _ => return None,
            }
        } else {
            return None;
        };
        matches!(&method.target,
            ResolvedObjectMethodTarget::Named(CallableIdentity::Builtin(id))
                if id.0 == target.entry().identity.name
                    && method.class == *object.class_name()
                    && method.declaring_class == *object.class_name()
                    && method.convention == crate::object::protocol::ObjectProtocolCallingConvention::StandardSubstruct
                    && method.fallback == runmat_types::CallableFallbackPolicy::None)
        .then_some(target)
    }
    pub(crate) fn assign<'a>(
        self,
        runtime: Option<&'a RuntimeContext>,
        receiver: &'a Value,
        path: &'a ObjectSubscriptPath,
        rhs: &'a Value,
        issuance: &'a crate::class_registry::BuiltinClassIssuance,
        method: &'a ResolvedObjectMethod,
    ) -> crate::execution::RuntimeContinuationFuture<'a, RootedValue> {
        // This replaces ordinary builtin-body admission for the same prepared
        // native method; no per-element continuation or caller grant is added.
        let scoped = crate::context::RuntimeScopeFactory::new(runtime.cloned(), move || {
            self.assign_in_scope(runtime, receiver, path, rhs, issuance, method)
        });
        crate::execution::rooted_continuation::admit_rooted_future_with(
            runtime.cloned(),
            self.entry().identity.name,
            crate::builtin::builtin_control_error,
            move |_| scoped.construct(),
        )
    }
    async fn assign_in_scope(
        self,
        runtime: Option<&RuntimeContext>,
        receiver: &Value,
        path: &ObjectSubscriptPath,
        rhs: &Value,
        issuance: &crate::class_registry::BuiltinClassIssuance,
        method: &ResolvedObjectMethod,
    ) -> BuiltinResult<RootedValue> {
        self.validate_original(runtime, receiver, issuance, method)?;
        let entry = self.entry();
        let invalid = if matches!(self, Self::Variable) {
            &runmat_builtins::OPTIMVAR_ERROR_INVALID_INPUT
        } else if matches!(self, Self::Equation) {
            &runmat_builtins::EQNPROBLEM_ERROR_INVALID_INPUT
        } else if matches!(self, Self::Problem) {
            &runmat_builtins::OPTIMPROBLEM_ERROR_INVALID_INPUT
        } else {
            &runmat_builtins::OPTIMMODEL_ERROR_INVALID_INPUT
        };
        crate::output_count::validate_bounded_outputs_in(
            entry.descriptor,
            invalid,
            entry.identity.name,
            runtime,
        )?;
        let policy = Policy {
            identity: entry.identity,
            invalid,
        };
        if matches!(self, Self::Variable) {
            let mut cooperation = super::super::optimvar::cooperation(runtime).await?;
            let _inputs = policy.roots([receiver, rhs].into_iter(), runtime)?;
            let result = super::super::optimvar::protocol::assign_path(
                receiver,
                path,
                rhs,
                &mut cooperation,
            )
            .await?;
            self.validate_original(runtime, receiver, issuance, method)?;
            return Ok(result);
        }
        let mut cooperation = policy.cooperation(runtime).await?;
        let _inputs = policy.roots([receiver, rhs].into_iter(), cooperation.runtime())?;
        let result = match self {
            Self::Expression => {
                protocol::assign(receiver, path, rhs, policy, &mut cooperation).await
            }
            Self::Constraint => {
                bindings::protocol::comparison_assign_path(
                    receiver,
                    path,
                    rhs,
                    policy,
                    &mut cooperation,
                )
                .await
            }
            Self::Problem => problem::assign(receiver, path, rhs, policy, &mut cooperation).await,
            Self::Equation => equation::assign(receiver, path, rhs, policy, &mut cooperation).await,
            Self::Variable => unreachable!("variable route selected before model policy"),
        }?;
        self.validate_original(runtime, receiver, issuance, method)?;
        Ok(result)
    }
    fn validate_original(
        self,
        runtime: Option<&RuntimeContext>,
        receiver: &Value,
        issuance: &crate::class_registry::BuiltinClassIssuance,
        method: &ResolvedObjectMethod,
    ) -> BuiltinResult<()> {
        let issued = match runtime {
            Some(runtime) => issuance.is_current_in_context(runtime),
            None => issuance.is_current_ordinary(),
        };
        if !issued
            || !crate::object::protocol::resolved_method_is_current(method)
            || !matches!(receiver, Value::Object(object) if Self::select(object, method) == Some(self))
        {
            return Err(crate::runtime_error::semantic_error(
                "StaleObjectProtocolResolution",
                "native contextual-index protocol lost its original issuance",
            ));
        }
        Ok(())
    }
}

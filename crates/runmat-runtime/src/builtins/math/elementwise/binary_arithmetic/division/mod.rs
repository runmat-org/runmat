//! Shared element-wise division semantics.

mod host;
mod provider;
mod fallback;

#[cfg(test)]
use fallback::operand_is_scalar;
use fallback::{
    gather_operand, preferred_source, restore_residency, source_handle, validate_integer_admission,
};

use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor};
use runmat_value::Value;

use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[derive(Clone, Copy)]
pub(super) enum PublicOperandOrder {
    Numerical,
    Reversed,
}

#[derive(Clone, Copy)]
pub(super) struct DivisionContext {
    pub entry: &'static runmat_builtins::BuiltinCatalogEntry,
    pub public_order: PublicOperandOrder,
    pub identity: BuiltinCatalogIdentity,
    pub invalid_input: &'static BuiltinErrorDescriptor,
    pub size_mismatch: &'static BuiltinErrorDescriptor,
    pub internal: &'static BuiltinErrorDescriptor,
}

impl DivisionContext {
    fn public_operands<'a>(
        self,
        numerator: &'a Value,
        denominator: &'a Value,
    ) -> (&'a Value, &'a Value) {
        match self.public_order {
            PublicOperandOrder::Numerical => (numerator, denominator),
            PublicOperandOrder::Reversed => (denominator, numerator),
        }
    }

    fn error(self, descriptor: &'static BuiltinErrorDescriptor) -> RuntimeError {
        let mut builder = build_runtime_error(descriptor.message).with_builtin(self.identity.name);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    fn error_with_detail(
        self,
        descriptor: &'static BuiltinErrorDescriptor,
        detail: impl AsRef<str>,
    ) -> RuntimeError {
        let mut builder =
            build_runtime_error(format!("{}: {}", descriptor.message, detail.as_ref()))
                .with_builtin(self.identity.name);
        if let Some(identifier) = descriptor.identifier {
            builder = builder.with_identifier(identifier);
        }
        builder.build()
    }

    fn internal_error(self, detail: impl std::fmt::Display) -> RuntimeError {
        build_runtime_error(format!("{}: {detail}", self.identity.name))
            .with_builtin(self.identity.name)
            .build()
    }

    fn provider_error(
        self,
        source: impl Into<runmat_accelerate_api::ProviderExecutionError>,
    ) -> RuntimeError {
        let source = source.into();
        let error = self.provider_contract_error(&source);
        crate::runtime_error::with_provider_execution_source(error, source)
    }

    fn provider_error_after_effect(
        self,
        source: impl Into<runmat_accelerate_api::ProviderExecutionError>,
    ) -> RuntimeError {
        let source = source.into().after_effect();
        let error = self.provider_contract_error(&source);
        crate::runtime_error::with_provider_execution_source(error, source)
    }

    fn provider_contract_error(self, detail: impl std::fmt::Display) -> RuntimeError {
        let mut error = self.internal_error(detail);
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    }
}

pub(super) fn execute_host(
    context: DivisionContext,
    numerator: Value,
    denominator: Value,
) -> BuiltinResult<Value> {
    host::execute(context, numerator, denominator)
}

pub(super) async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    context: DivisionContext,
    numerator: Value,
    denominator: Value,
) -> BuiltinResult<Value> {
    validate_integer_admission(context, &numerator, &denominator)?;
    let originals = [source_handle(&numerator), source_handle(&denominator)]
        .into_iter()
        .flatten()
        .cloned()
        .collect::<Vec<_>>();
    let source = preferred_source(&numerator, &denominator);
    let resident = match (&numerator, &denominator) {
        (Value::GpuTensor(numerator), Value::GpuTensor(denominator)) => {
            provider::try_pair(context, numerator, denominator).await?
        }
        (Value::GpuTensor(numerator), denominator) => {
            provider::try_resident_numerator(context, numerator, denominator).await?
        }
        (numerator, Value::GpuTensor(denominator)) => {
            provider::try_resident_denominator(context, numerator, denominator).await?
        }
        _ => None,
    };
    if let Some(output) = resident {
        return super::provider_support::resident_output_from_sources(
            output,
            [source_handle(&numerator), source_handle(&denominator)]
                .into_iter()
                .flatten(),
        );
    }
    let has_explicit_source = originals
        .iter()
        .any(runmat_accelerate_api::handle_is_explicit);
    let (public_left, public_right) = context.public_operands(&numerator, &denominator);
    let fallback = super::provider_support::HostFallbackAdmission::new(
        runtime,
        context.entry,
        public_left,
        public_right,
    )?;
    let numerator = gather_operand(context, numerator).await?;
    let denominator = gather_operand(context, denominator).await?;
    let host = execute_host(context, numerator, denominator)?;
    match source {
        Some(_) if has_explicit_source => fallback.finish(host),
        Some(source) => restore_residency(context, &source, host, &originals),
        None => Ok(host),
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    #[test]
    fn left_division_keeps_public_catalog_order_while_numerics_are_reversed() {
        let numerator = Value::Num(12.0);
        let denominator = Value::Num(3.0);
        let (left, right) =
            super::super::ldivide::DIVISION_CONTEXT.public_operands(&numerator, &denominator);
        assert!(std::ptr::eq(left, &denominator));
        assert!(std::ptr::eq(right, &numerator));
        let result = execute_host(
            super::super::ldivide::DIVISION_CONTEXT,
            numerator,
            denominator,
        )
        .expect("asymmetric left division witness");
        let Value::Num(quotient) = result else {
            panic!("scalar quotient");
        };
        assert_eq!(quotient, 4.0);
    }

    #[test]
    fn division_scalar_admission_requires_one_element_for_host_and_resident() {
        let owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
        for (shape, scalar) in [(vec![1, 1], true), (vec![0, 1], false), (vec![2, 1], false)] {
            let host = runmat_value::Tensor::new(vec![0.0; shape.iter().product()], shape.clone())
                .unwrap();
            let resident = owner
                .allocate_zeroed(
                    shape,
                    runmat_accelerate_api::NumericElementType::F64,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                )
                .unwrap();
            assert_eq!(operand_is_scalar(&Value::Tensor(host)), scalar);
            assert_eq!(
                operand_is_scalar(&Value::GpuTensor(resident.handle)),
                scalar
            );
        }
    }
}

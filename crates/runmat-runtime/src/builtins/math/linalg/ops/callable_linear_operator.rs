//! Context-bound callable vector operation for asynchronous solvers.
//!
//! The callback's result remains in its canonical rooted sequence;
//! temporary numeric scratch admission cannot fund an escaping user result.

use runmat_types::FunctionId;
use runmat_value::{NumericDType, Value};

use crate::call::descriptor::{CallableDescriptor, CallableTarget, FunctionNameResolver};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPolicy, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::RetainedValueReadLease;
use crate::execution::{RootedValue, RootedValueSequence};
use crate::runtime_error::{semantic_error, semantic_error_with_source};
use crate::RuntimeError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::linalg) enum VectorClass {
    RealSingle,
    RealDouble,
    ComplexSingle,
    ComplexDouble,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::builtins::math::linalg) struct VectorContract {
    pub(in crate::builtins::math::linalg) len: usize,
    pub(in crate::builtins::math::linalg) class: VectorClass,
}

impl VectorContract {
    pub(in crate::builtins::math::linalg) fn check(
        self,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        if self.len == 0 {
            return Err(invalid("operator vector dimension must be nonzero"));
        }
        let valid = match (self.class, value) {
            (VectorClass::RealSingle, Value::Tensor(tensor)) => {
                tensor.numeric_dtype() == NumericDType::F32
                    && tensor.shape() == [self.len, 1]
                    && tensor.as_f32_slice().is_some()
            }
            (VectorClass::RealDouble, Value::Tensor(tensor)) => {
                tensor.numeric_dtype() == NumericDType::F64
                    && tensor.shape() == [self.len, 1]
                    && tensor.as_f64_slice().is_some()
            }
            (VectorClass::ComplexSingle, Value::ComplexTensor(tensor)) => {
                tensor.numeric_dtype() == NumericDType::F32
                    && tensor.shape() == [self.len, 1]
                    && tensor.as_f32_slice().is_some()
            }
            (VectorClass::ComplexDouble, Value::ComplexTensor(tensor)) => {
                tensor.numeric_dtype() == NumericDType::F64
                    && tensor.shape() == [self.len, 1]
                    && tensor.as_f64_slice().is_some()
            }
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(invalid(
                "operator callback must accept and return an n-by-1 vector of the selected floating class",
            ))
        }
    }
}

/// A selected callable for repeated, unbounded vector applications. Closure
/// captures have one shared physical owner and one external root lease for the
/// entire solve; selecting the target is never repeated between iterations.
/// Managed metadata/argument admission is deliberately not implied by this
/// owner and is rejected before descriptor construction.
pub(in crate::builtins::math::linalg) struct RepeatedCallableVector {
    descriptor: CallableDescriptor,
    _closure_roots: Option<RetainedValueReadLease>,
    contract: VectorContract,
}

impl RepeatedCallableVector {
    pub(in crate::builtins::math::linalg) fn select(
        runtime: &RuntimeContext,
        target: Value,
        contract: VectorContract,
    ) -> Result<Self, RuntimeError> {
        let policy = HostAllocationReceipt::policy(Some(runtime))
            .map_err(|error| admission(runtime, error))?;
        if policy == HostAllocationPolicy::Managed {
            return Err(semantic_error_with_source(
                "RunMat:CallableLinearOperator:DescriptorUnavailable",
                "repeatable callable metadata requires an admitted descriptor owner",
                CallableDescriptorAdmissionError::ManagedMetadata,
            ));
        }
        if !matches!(
            target,
            Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle(_)
                | Value::Closure(_)
        ) {
            return Err(invalid("operator must be a function handle or closure"));
        }
        let rooted = RootedValue::new(target)
            .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
        rooted.with_owned(|target| {
            let descriptor =
                CallableDescriptor::from_feval_value(target, Vec::new(), 1, &SemanticResolver);
            let closure_roots = match &descriptor.target {
                CallableTarget::BoundLexical { closure, .. }
                | CallableTarget::CapturedResolved { closure, .. } => {
                    Some(closure.retain_for_call(Some(runtime))?)
                }
                _ => None,
            };
            Ok(Self {
                descriptor,
                _closure_roots: closure_roots,
                contract,
            })
        })
    }

    pub(in crate::builtins::math::linalg) async fn apply(
        &self,
        runtime: &RuntimeContext,
        input: Value,
    ) -> Result<RootedValueSequence, RuntimeError> {
        self.contract.check(&input)?;
        let rooted = RootedValue::new(input)
            .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
        let target = self.descriptor.target.clone();
        let metadata = self.descriptor.metadata.clone();
        let output = rooted
            .with_owned_async(|input| async move {
                let descriptor = CallableDescriptor {
                    target,
                    args: vec![input],
                    requested_outputs: 1,
                    metadata,
                };
                crate::call::descriptor::execute_callable_descriptor_in_context(runtime, descriptor)
                    .await
            })
            .await?;
        let [value] = output.value().as_slice() else {
            return Err(invalid("operator callback must return exactly one value"));
        };
        self.contract.check(value)?;
        Ok(output)
    }
}

struct SemanticResolver;

impl FunctionNameResolver for SemanticResolver {
    fn resolve_function(&self, name: &str) -> Option<FunctionId> {
        crate::user_functions::resolve_semantic_function_by_name(name).map(FunctionId)
    }
}

fn invalid(message: &str) -> RuntimeError {
    semantic_error("CallableLinearOperator", message)
}

fn admission(runtime: &RuntimeContext, error: HostAllocationError) -> RuntimeError {
    crate::builtin::builtin_control_error(
        Some(&runtime.cancellation_context()),
        "callable linear operator",
        error,
    )
}

#[derive(Debug)]
enum CallableDescriptorAdmissionError {
    ManagedMetadata,
}

impl std::fmt::Display for CallableDescriptorAdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ManagedMetadata => {
                formatter.write_str("callable descriptor backing is not admitted")
            }
        }
    }
}

impl std::error::Error for CallableDescriptorAdmissionError {}

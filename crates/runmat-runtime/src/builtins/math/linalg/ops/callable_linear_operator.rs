//! Context-bound callable vector operation for asynchronous solvers.
//!
//! The callback's result remains in its canonical rooted sequence;
//! temporary numeric scratch admission cannot fund an escaping user result.

use runmat_types::FunctionId;
use runmat_value::{NumericDType, Value};

use crate::call::descriptor::{FunctionNameResolver, RepeatableCallable};
use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValueSequence, RootedValues};
use crate::runtime_error::semantic_error;
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
    fn infer(len: usize, value: &Value) -> Result<Self, RuntimeError> {
        let class = match value {
            Value::Num(_) => VectorClass::RealDouble,
            Value::Complex(..) => VectorClass::ComplexDouble,
            Value::Tensor(t) if t.as_f32_slice().is_some() => VectorClass::RealSingle,
            Value::Tensor(t) if t.as_f64_slice().is_some() => VectorClass::RealDouble,
            Value::ComplexTensor(t) if t.as_f32_slice().is_some() => VectorClass::ComplexSingle,
            Value::ComplexTensor(t) if t.as_f64_slice().is_some() => VectorClass::ComplexDouble,
            _ => {
                return Err(invalid(
                    "operator callback requires native floating vectors",
                ))
            }
        };
        let contract = Self { len, class };
        contract.check(value)?;
        Ok(contract)
    }
    pub(in crate::builtins::math::linalg) fn check(
        self,
        value: &Value,
    ) -> Result<(), RuntimeError> {
        if self.len == 0 {
            return Err(invalid("operator vector dimension must be nonzero"));
        }
        let valid = match (self.class, value) {
            (VectorClass::RealDouble, Value::Num(_))
            | (VectorClass::ComplexDouble, Value::Complex(_, _)) => self.len == 1,
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

/// A selected callable for repeated vector applications. The callable and its
/// capture graph have one retained owner for the entire solve; each input uses
/// a separately admitted one-element argument carrier so callbacks retain the
/// input's existing physical claim instead of a solver-wide worst-case charge.
pub(in crate::builtins::math::linalg) struct RepeatedCallableVector {
    selected: RepeatableCallable,
    contract: VectorContract,
}

impl RepeatedCallableVector {
    /// One successful, rooted class-discovery evaluation. Its result is not
    /// automatically an iteration product: the caller must prove exact input
    /// binding before reuse. An error is terminal, never an inference retry.
    pub(in crate::builtins::math::linalg) async fn select_with_probe(
        runtime: &RuntimeContext,
        target: Value,
        input: Value,
        len: usize,
    ) -> Result<(Self, RootedValueSequence), RuntimeError> {
        VectorContract::infer(len, &input)?;
        let selected = RepeatableCallable::select(runtime, target, &SemanticResolver)?;
        let output = selected
            .invoke_rooted(runtime, root_argument(runtime, input)?, 1)
            .await?;
        let [value] = output.value().as_slice() else {
            return Err(invalid("operator callback must return exactly one value"));
        };
        let contract = VectorContract::infer(len, value)?;
        Ok((Self { selected, contract }, output))
    }
    pub(in crate::builtins::math::linalg) fn select(
        runtime: &RuntimeContext,
        target: Value,
        contract: VectorContract,
    ) -> Result<Self, RuntimeError> {
        if !matches!(
            target,
            Value::ScopedFunctionHandle(_)
                | Value::FunctionHandle(_)
                | Value::ExternalFunctionHandle(_)
                | Value::MethodFunctionHandle(_)
                | Value::BoundFunctionHandle(_)
                | Value::Closure(_)
        ) {
            return Err(invalid("operator must be a function handle or closure"));
        }
        Ok(Self {
            selected: RepeatableCallable::select(runtime, target, &SemanticResolver)?,
            contract,
        })
    }

    pub(in crate::builtins::math::linalg) async fn apply(
        &self,
        runtime: &RuntimeContext,
        input: Value,
    ) -> Result<RootedValueSequence, RuntimeError> {
        self.contract.check(&input)?;
        let arguments = root_argument(runtime, input)?;
        let output = self.selected.invoke_rooted(runtime, arguments, 1).await?;
        let [value] = output.value().as_slice() else {
            return Err(invalid("operator callback must return exactly one value"));
        };
        self.contract.check(value)?;
        Ok(output)
    }
}

fn root_argument(runtime: &RuntimeContext, input: Value) -> Result<RootedValues, RuntimeError> {
    use runmat_value::OutputListStorage;

    let raw = OutputListStorage::requested_bytes(1)
        .map_err(HostAllocationError::Numeric)
        .and_then(|bytes| u64::try_from(bytes).map_err(|_| HostAllocationError::Overflow))
        .map_err(|error| admission(runtime, error))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(raw, 1)
        .map_err(|error| admission(runtime, error))?;
    let plan = HostAllocationPlan::new(0, retained).map_err(|error| admission(runtime, error))?;
    let Some(receipt) = HostAllocationReceipt::admit(Some(runtime), plan)
        .map_err(|error| admission(runtime, error))?
    else {
        return RootedValues::new(vec![input])
            .map_err(crate::execution::ExecutionServiceError::into_runtime_error);
    };
    let permit = receipt
        .output_permit(runtime)
        .map_err(|error| admission(runtime, error))?;
    let mut input = Some(input);
    let storage = OutputListStorage::try_initialize(1, &permit, |_| {
        Ok::<_, std::convert::Infallible>(input.take().expect("one argument"))
    })
    .map_err(|error| match error {
        runmat_value::OutputListInitializationError::Allocation(error) => {
            admission(runtime, HostAllocationError::Numeric(error))
        }
        runmat_value::OutputListInitializationError::Element(never) => match never {},
    })?;
    receipt
        .checkpoint(runtime)
        .map_err(|error| admission(runtime, error))?;
    RootedValues::from_output_storage_in_context(storage, runtime)
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

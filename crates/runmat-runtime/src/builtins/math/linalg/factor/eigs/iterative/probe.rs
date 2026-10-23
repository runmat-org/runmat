//! One required initial callback product, bound to the exact selected native input class.
//! Solver basis construction consumes this transcript instead of replaying it.
use super::{class::Class, request::Request, scalar::Scalar};
use crate::{
    builtins::math::linalg::ops::callable_linear_operator::RepeatedCallableVector,
    context::{
        host_allocation::{HostAllocationPlan, HostAllocationReceipt},
        RuntimeContext,
    },
    execution::{RootedValue, RootedValueSequence},
    BuiltinResult,
};
use runmat_value::{ComplexElement, Value};
pub(super) enum Product {
    ReusableFirstProduct,
    DiscoveryOnly,
}
// Both successful evaluation owners live on the caller stack, outside the
// proof enum. No large enum, heap box, optional fake owner, or root alias.
pub(super) struct Probe {
    pub selected: RepeatedCallableVector,
    pub input: RootedValue,
    pub output: RootedValueSequence,
    pub product: Product,
    pub class: Class,
}
async fn input<T: Scalar>(
    request: &Request<'_>,
    runtime: &RuntimeContext,
) -> BuiltinResult<RootedValue> {
    let fail = |e| super::admission(Some(runtime), e);
    let scratch = HostAllocationPlan::numeric_workspace_bytes(
        T::DTYPE,
        request
            .n
            .checked_mul(T::LANES)
            .ok_or_else(|| super::invalid("probe shape overflow"))?,
    )
    .map_err(fail)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(scratch, T::output_bytes(request.n).map_err(fail)?)
            .map_err(fail)?,
    )
    .map_err(fail)?;
    let mut work = super::work::Work::new(Some(runtime), receipt.as_ref())?;
    let mut native =
        super::storage::Native::<T>::new(request.n, Some(runtime), receipt.as_ref()).await?;
    super::basis::initial(native.get_mut(), request.start, &mut work).await?;
    let value = if let Some(receipt) = receipt.as_ref() {
        crate::builtins::math::linalg::factor::iterative::copy::admitted::<T>(
            receipt,
            runtime,
            native.get_mut(),
            "eigs",
        )
        .await?
    } else {
        crate::builtins::math::linalg::factor::iterative::copy::unbounded::<T>(
            Some(runtime),
            native.get_mut(),
            "eigs",
        )
        .await?
    };
    work.check()?;
    RootedValue::in_context(value, runtime)
}

async fn inverse_argument<T: Scalar>(
    input: &RootedValue,
    b: &Value,
    n: usize,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    let fail = |e| super::admission(Some(runtime), e);
    let scratch = HostAllocationPlan::numeric_workspace_bytes(
        T::DTYPE,
        n.checked_mul(T::LANES)
            .ok_or_else(|| super::invalid("inverse probe shape overflow"))?,
    )
    .map_err(fail)?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(scratch, T::output_bytes(n).map_err(fail)?).map_err(fail)?,
    )
    .map_err(fail)?;
    let work = super::work::Work::new(Some(runtime), receipt.as_ref())?;
    let mut native = super::storage::Native::<T>::new(n, Some(runtime), receipt.as_ref()).await?;
    let input = T::input(input.value())
        .ok_or_else(|| super::invalid("initial callback input class differs"))?;
    let operator =
        crate::builtins::math::linalg::ops::linear_operator::LinearOperator::borrow_cooperatively(
            b, runtime,
        )
        .await
        .map_err(|e| super::operator_error(Some(runtime), e))?;
    operator
        .apply_converted(input.as_slice(), native.get_mut(), T::native, Some(runtime))
        .await
        .map_err(|e| super::operator_error(Some(runtime), e))?;
    work.check()?;
    let value = if let Some(receipt) = receipt.as_ref() {
        crate::builtins::math::linalg::factor::iterative::copy::admitted::<T>(
            receipt,
            runtime,
            native.get_mut(),
            "eigs",
        )
        .await?
    } else {
        crate::builtins::math::linalg::factor::iterative::copy::unbounded::<T>(
            Some(runtime),
            native.get_mut(),
            "eigs",
        )
        .await?
    };
    work.check()?;
    Ok(value)
}
async fn select_native<T: Scalar>(
    a: &Value,
    request: &Request<'_>,
    runtime: &RuntimeContext,
) -> BuiltinResult<Probe> {
    if matches!(request.sigma,super::super::Sigma::Near(value) if T::native(value).is_none()) {
        return Err(super::invalid(
            "numeric sigma exceeds selected native range",
        ));
    }
    let input = input::<T>(request, runtime).await?;
    let argument = if matches!(
        request.sigma,
        super::super::Sigma::Near(_) | super::super::Sigma::SmallestAbs
    ) {
        if let Some(b) = request.b {
            inverse_argument::<T>(&input, b, request.n, runtime).await?
        } else {
            input.value().clone()
        }
    } else {
        input.value().clone()
    };
    let (selected, output) =
        RepeatedCallableVector::select_with_probe(runtime, a.clone(), argument, request.n).await?;
    let [value] = output.value().as_slice() else {
        return Err(super::invalid("probe must return one vector"));
    };
    let inferred = Class::of(value)
        .ok_or_else(|| super::invalid("probe must return a native floating vector"))?;
    let preliminary = Class::request(a, request);
    if preliminary.single() && !inferred.single() {
        return Err(super::invalid(
            "Afun must return single when B or StartVector is single",
        ));
    }
    if preliminary.complex() && !inferred.complex() {
        return Err(super::invalid(
            "Afun must preserve the selected complex input domain",
        ));
    }
    let product = if preliminary == inferred && !request.cholesky && !request.definite {
        Product::ReusableFirstProduct
    } else {
        // Successful discovery is not a failed-call retry. Its effects and
        // spend remain observable; this result cannot be relabeled as a
        // product of a different native input or normalization.
        Product::DiscoveryOnly
    };
    Ok(Probe {
        selected,
        input,
        output,
        product,
        class: preliminary.combine(inferred),
    })
}
pub(super) async fn select(
    a: &Value,
    request: &Request<'_>,
    runtime: &RuntimeContext,
    class: Class,
) -> BuiltinResult<Probe> {
    match class {
        Class::RealF32 => select_native::<f32>(a, request, runtime).await,
        Class::RealF64 => select_native::<f64>(a, request, runtime).await,
        Class::ComplexF32 => select_native::<ComplexElement<f32>>(a, request, runtime).await,
        Class::ComplexF64 => select_native::<ComplexElement<f64>>(a, request, runtime).await,
    }
}

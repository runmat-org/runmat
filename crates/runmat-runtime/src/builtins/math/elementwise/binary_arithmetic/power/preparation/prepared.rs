//! Delivered numerical, compound and legacy routes keep their original
//! authority checks. Only the selected body exists before publication.
use super::publication::Confirmation;
use super::selection::Pair;
use super::*;
use crate::builtins::math::elementwise::binary_arithmetic::preparation::managed as common_managed;
use crate::builtins::math::host_route;
use futures::future::Either;

pub(super) fn execute<'a, F, Fut, P, Published>(
    runtime: Option<&'a RuntimeContext>,
    values: Vec<Value>,
    compound: Option<AdmittedBuiltinCompound>,
    numerical_output: Option<PreparedHostNumericalOutput>,
    direct: Option<AdmittedBuiltinCompound>,
    body: F,
    publish: P,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    F: FnOnce(Value, Value) -> Fut + 'a,
    Fut: Future<Output = BuiltinResult<RootedValue>> + 'a,
    P: FnOnce(RootedValue) -> Published + 'a,
    Published: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    if let Some(output) = numerical_output {
        Either::Left(numerical(
            runtime,
            values,
            output,
            compound.is_some() || direct.is_some(),
            publish,
        ))
    } else if let Some(compound) = compound.or(direct) {
        Either::Right(Either::Left(compound_body(
            runtime, values, compound, publish,
        )))
    } else {
        Either::Right(Either::Right(legacy(runtime, values, body, publish)))
    }
}
fn numerical<'a, P, Published>(
    runtime: Option<&'a RuntimeContext>,
    mut values: Vec<Value>,
    output: PreparedHostNumericalOutput,
    contradictory: bool,
    publish: P,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    P: FnOnce(RootedValue) -> Published + 'a,
    Published: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    let result = if contradictory {
        Err(common_managed::invalid())
    } else {
        let right = values.remove(1);
        let left = values.remove(0);
        let route = host_route::canonical(
            runtime,
            crate::context::VmFusionHostInstruction::ElementwiseBinary,
            [&left, &right].into_iter(),
        );
        runtime
            .ok_or_else(common_managed::invalid)
            .and_then(|runtime| managed::execute(runtime, output, &left, &right))
            .map(|(base, domain)| {
                (
                    base,
                    route.map(|route| Confirmation {
                        route,
                        evidence: runmat_accelerate_api::NumericalDomainEvidence::RealPower(domain),
                    }),
                )
            })
    };
    crate::execution::after_result(std::future::ready(result), move |(base, confirmation)| {
        publication::publish(runtime, base, publish, confirmation)
    })
}
fn compound_body<'a, P, Published>(
    runtime: Option<&'a RuntimeContext>,
    mut values: Vec<Value>,
    compound: AdmittedBuiltinCompound,
    publish: P,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    P: FnOnce(RootedValue) -> Published + 'a,
    Published: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    let right = values.remove(1);
    let left = values.remove(0);
    let future = match compound {
        AdmittedBuiltinCompound::RealPower(compound) => Either::Left(
            super::super::compound::execute(runtime, *compound, left, right),
        ),
        AdmittedBuiltinCompound::RealPowerScalar(compound) => Either::Right(Either::Left(
            super::super::scalar_compound::execute(runtime, *compound, left, right),
        )),
        AdmittedBuiltinCompound::RealUnary(_) => {
            let mut error =
                super::super::builtin_error("power: received another admitted compound authority");
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            Either::Right(Either::Right(std::future::ready(Err(error))))
        }
    };
    crate::execution::after_result(future, move |base| {
        publication::publish(runtime, base, publish, None)
    })
}
struct Legacy {
    pair: Pair,
    confirmation: Option<Confirmation>,
}
fn prepare_legacy(
    runtime: Option<&RuntimeContext>,
    mut values: Vec<Value>,
) -> BuiltinResult<Legacy> {
    let route = host_route::canonical(
        runtime,
        crate::context::VmFusionHostInstruction::ElementwiseBinary,
        values.iter(),
    );
    if let (Some(_), Some(runtime)) = (&route, runtime) {
        common_managed::require_unbounded_host_output(runtime)?;
    }
    if route.is_none() {
        // A missing recipe is not a certified no-attempt result. Check each
        // actual original owner before legacy probes or scalar staging.
        let mut owners: Vec<&'static dyn runmat_accelerate_api::AccelProvider> = Vec::new();
        for value in &values {
            let Value::GpuTensor(source) = value else {
                continue;
            };
            let Some(owner) =
                crate::builtins::common::gpu_helpers::exact_provider_for_handle(source)
            else {
                continue;
            };
            if owners
                .iter()
                .any(|previous| runmat_accelerate_api::same_provider_instance(*previous, owner))
            {
                continue;
            }
            owners.push(owner);
            crate::context::provider_operation::unavailable_provider_route::<()>(runtime, owner)?;
        }
    }
    let evidence = route.as_ref().map(|_| classify(&values)).transpose()?;
    let right = values.remove(1);
    let left = values.remove(0);
    let confirmation = route
        .zip(evidence)
        .map(|(route, evidence)| Confirmation { route, evidence });
    Ok(Legacy {
        pair: Pair { left, right },
        confirmation,
    })
}
fn legacy<'a, F, Fut, P, Published>(
    runtime: Option<&'a RuntimeContext>,
    values: Vec<Value>,
    body: F,
    publish: P,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    F: FnOnce(Value, Value) -> Fut + 'a,
    Fut: Future<Output = BuiltinResult<RootedValue>> + 'a,
    P: FnOnce(RootedValue) -> Published + 'a,
    Published: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    crate::execution::after_result(
        futures::future::lazy(move |_| prepare_legacy(runtime, values)),
        move |Legacy {
                  pair: Pair { left, right },
                  confirmation,
              }| {
            crate::execution::after_result(body(left, right), move |base| {
                publication::publish(runtime, base, publish, confirmation)
            })
        },
    )
}

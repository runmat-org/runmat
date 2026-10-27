//! Selected POWER futures only. Cold metadata and preceding phases are gone
//! before body, publication or confirmation is polled.
use super::publication::Confirmation;
use super::selection::{Branch, Pair, Request, Selected};
use super::*;
use crate::builtins::math::host_route::NumericHostRoute;
use futures::{future::Either, FutureExt};

pub(super) fn execute<'a, F, Fut, P, Published>(
    runtime: Option<&'a RuntimeContext>,
    selected: Selected,
    body: F,
    publish: P,
) -> impl Future<Output = BuiltinResult<RootedValue>> + 'a
where
    F: FnOnce(Value, Value) -> Fut + 'a,
    Fut: Future<Output = BuiltinResult<RootedValue>> + 'a,
    P: FnOnce(RootedValue) -> Published + 'a,
    Published: Future<Output = BuiltinResult<RootedValue>> + 'a,
{
    let Selected {
        pair,
        branch,
        inputs,
    } = selected;
    let future = match branch {
        Branch::Ordinary => Either::Left(Either::Left(crate::execution::after_result(
            body(pair.left, pair.right),
            move |base| publication::publish(runtime, base, publish, None),
        ))),
        Branch::Host(route) => Either::Left(Either::Right(host(
            runtime.expect("host route runtime"),
            pair,
            route,
            body,
            publish,
        ))),
        Branch::Admitted(admission) => Either::Right(crate::execution::after_result(
            futures::future::lazy(move |_| selection::request(runtime, admission, pair)),
            move |Request { request, direct }| {
                execute_request_numerical(
                    runtime,
                    request,
                    move |values, compound, numerical_output| {
                        prepared::execute(
                            runtime,
                            values,
                            compound,
                            numerical_output,
                            direct,
                            body,
                            publish,
                        )
                    },
                )
            },
        )),
    };
    // Preserve roots on error as well as success, through service adoption,
    // output-template publication and final numerical confirmation.
    crate::execution::after_result(
        future.map(Ok::<_, crate::RuntimeError>),
        move |result| async move {
            drop(inputs);
            result
        },
    )
}
fn host<'a, F, Fut, P, Published>(
    runtime: &'a RuntimeContext,
    pair: Pair,
    route: NumericHostRoute,
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
        futures::future::lazy(move |_| {
            let output = route.take_numerical_output(
                runtime,
                crate::context::VmFusionHostInstruction::ElementwiseBinary,
                BuiltinBindingIdentity {
                    builtin: POWER_CATALOG_ENTRY.identity,
                    variant: "default",
                },
            )?;
            Ok::<_, crate::RuntimeError>((pair, route, output))
        }),
        move |(Pair { left, right }, route, numerical_output)| {
            let future = match numerical_output {
                Some(output) => Either::Left(std::future::ready(
                    managed::execute(runtime, output, &left, &right).map(|(base, domain)| {
                        (
                            base,
                            runmat_accelerate_api::NumericalDomainEvidence::RealPower(domain),
                        )
                    }),
                )),
                None => match classify_operands(&left, &right) {
                    Err(error) => Either::Left(std::future::ready(Err(error))),
                    Ok(evidence) => Either::Right(
                        body(left, right).map(move |r| r.map(|base| (base, evidence))),
                    ),
                },
            };
            crate::execution::after_result(future, move |(base, evidence)| {
                publication::publish(
                    Some(runtime),
                    base,
                    publish,
                    Some(Confirmation { route, evidence }),
                )
            })
        },
    )
}

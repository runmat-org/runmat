//! Per-invocation value-domain selection retires before actual POWER execution.
use crate::context::preparation::{
    execute_request_numerical, AdmittedBuiltinCompound, AdmittedBuiltinPreparation,
    BuiltinPreparationAdmission, PreparedHostNumericalOutput,
};
use crate::context::RuntimeContext;
use crate::execution::{RootedValue, RootedValueLease};
use crate::BuiltinResult;
use runmat_builtins::{BuiltinBindingIdentity, BuiltinPreparationFormId, POWER_CATALOG_ENTRY};
use runmat_value::{NumericDType, Value};
use std::future::Future;
#[path = "preparation/domain.rs"]
mod domain;
mod execution;
mod prepared;
mod publication;
mod selection;
use domain::{classify, classify_operands};
mod managed;

pub(super) fn execute<'a, F, Fut, P, Published>(
    runtime: Option<&'a RuntimeContext>,
    input_count: usize,
    left: Value,
    right: Value,
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
        futures::future::lazy(move |_| selection::select(runtime, input_count, left, right)),
        move |selected| execution::execute(runtime, selected, body, publish),
    )
}

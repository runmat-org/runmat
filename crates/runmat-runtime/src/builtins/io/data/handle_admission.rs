//! Complete-call selection for a sealed Dataset-family receiver.

use crate::context::host_effect::{
    select_host_effect, DatasetHandleInput, DatasetHandleKey, DatasetHandleOperation,
    DatasetHandleRole, HostEffectOutput, HostEffectReceipt, HostEffectRequest, HostEffectWorkload,
};
use crate::context::RuntimeContext;
use crate::data::authority::{AuthorizedHandle, HandleRole};
use crate::{build_runtime_error, BuiltinResult};
use runmat_execution::Digest;
use runmat_value::Value;
use std::future::Future;

pub(super) fn select(
    runtime: Option<&RuntimeContext>,
    owner: &AuthorizedHandle,
    operation: DatasetHandleOperation,
    arguments: &[&Value],
) -> BuiltinResult<Option<HostEffectReceipt>> {
    let facts = arguments
        .iter()
        .map(|&value| input(value))
        .collect::<Vec<_>>();
    select_facts(runtime, owner, operation, &facts)
}

fn select_facts(
    runtime: Option<&RuntimeContext>,
    owner: &AuthorizedHandle,
    operation: DatasetHandleOperation,
    facts: &[DatasetHandleInput],
) -> BuiltinResult<Option<HostEffectReceipt>> {
    let role = match &owner.role {
        HandleRole::Dataset => DatasetHandleRole::Dataset,
        HandleRole::Array(name) => DatasetHandleRole::Array(name),
        HandleRole::Transaction(id) => DatasetHandleRole::Transaction(id),
    };
    if !operation.accepts_role(role) {
        return Err(
            build_runtime_error("dataset handle operation has a foreign sealed role")
                .with_identifier("RunMat:Placement:InvalidDecision")
                .build(),
        );
    }
    let reference = owner.reference();
    let domain_facts =
        Digest::sha256(serde_json::to_vec(facts).expect("typed data handle input facts serialize"));
    select_host_effect(
        runtime,
        HostEffectRequest {
            operation: operation.public_name(),
            workload: HostEffectWorkload::DatasetHandle {
                operation,
                root: &owner.root,
                key: DatasetHandleKey {
                    host_identity: &reference.host_identity,
                    handle: reference.handle,
                    generation: reference.generation,
                },
                role,
                inputs: facts,
                // Every existing receiver binding creates a Value, including
                // logical acknowledgements when the caller discards them.
                output: HostEffectOutput::ReturnedValue,
            },
            domain_facts,
            filesystem: &owner.filesystem,
        },
    )
}

pub(super) async fn execute_facts<T>(
    runtime: Option<&RuntimeContext>,
    owner: &AuthorizedHandle,
    operation: DatasetHandleOperation,
    facts: &[DatasetHandleInput],
    effect: impl Future<Output = BuiltinResult<T>>,
) -> BuiltinResult<T> {
    let receipt = select_facts(runtime, owner, operation, facts)?;
    super::execution::execute(runtime, receipt, effect).await
}

pub(super) fn input(value: &Value) -> DatasetHandleInput {
    use runmat_value::Value;

    let scalar_digest = match value {
        Value::String(text) => Some(Digest::sha256(
            serde_json::to_vec(&("string", text)).expect("typed string control serializes"),
        )),
        Value::CharArray(chars) => Some(Digest::sha256(
            serde_json::to_vec(&("char", &chars.shape, &chars.data))
                .expect("typed character control serializes"),
        )),
        Value::Bool(boolean) => Some(Digest::sha256(
            serde_json::to_vec(&("logical", boolean)).expect("typed logical control serializes"),
        )),
        Value::Int(integer) => Some(Digest::sha256(
            serde_json::to_vec(&("integer", integer)).expect("typed integer control serializes"),
        )),
        Value::Num(number) => Some(Digest::sha256(
            serde_json::to_vec(&("double_bits", number.to_bits()))
                .expect("typed double control serializes"),
        )),
        Value::Complex(real, imaginary) => Some(Digest::sha256(
            serde_json::to_vec(&("complex_double_bits", real.to_bits(), imaginary.to_bits()))
                .expect("typed complex control serializes"),
        )),
        _ => None,
    };
    DatasetHandleInput {
        fact: crate::value_fact::value_fact(value),
        scalar_digest,
    }
}

pub(super) async fn execute<T>(
    runtime: Option<&RuntimeContext>,
    owner: &AuthorizedHandle,
    operation: DatasetHandleOperation,
    arguments: &[&Value],
    effect: impl Future<Output = BuiltinResult<T>>,
) -> BuiltinResult<T> {
    let receipt = select(runtime, owner, operation, arguments)?;
    super::execution::execute(runtime, receipt, effect).await
}

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
    let mut budget = super::handle_identity::HashBudget::new();
    let facts = arguments
        .iter()
        .map(|&value| input_with_budget(value, &mut budget))
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

#[cfg(test)]
pub(super) fn input(value: &Value) -> DatasetHandleInput {
    let mut budget = super::handle_identity::HashBudget::new();
    input_with_budget(value, &mut budget)
}

pub(super) fn input_with_budget(
    value: &Value,
    budget: &mut super::handle_identity::HashBudget,
) -> DatasetHandleInput {
    let identity_digest = super::handle_identity::exact_digest_with(budget, value);
    DatasetHandleInput {
        fact: if identity_digest.is_some() {
            crate::value_fact::value_fact(value)
        } else {
            runmat_types::ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue)
        },
        identity_digest,
    }
}

pub(super) fn inputs(arguments: &[&Value]) -> Vec<DatasetHandleInput> {
    let mut budget = super::handle_identity::HashBudget::new();
    arguments
        .iter()
        .map(|value| input_with_budget(value, &mut budget))
        .collect()
}

#[derive(serde::Serialize)]
struct QueueIdentity {
    base_sequence: u64,
    revision: u64,
    writes: usize,
    resizes: usize,
    fills: usize,
    creates: usize,
    deletes: usize,
    attributes: usize,
}

pub(super) fn queue_input(tx: &crate::data::PendingTxn) -> DatasetHandleInput {
    let identity = QueueIdentity {
        base_sequence: tx.base_sequence,
        revision: tx.revision,
        writes: tx.writes.len(),
        resizes: tx.resizes.len(),
        fills: tx.fills.len(),
        creates: tx.create_arrays.len(),
        deletes: tx.delete_arrays.len(),
        attributes: tx.attrs.len(),
    };
    DatasetHandleInput {
        fact: runmat_types::ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
        identity_digest: Some(Digest::sha256(
            serde_json::to_vec(&identity).expect("typed transaction queue identity serializes"),
        )),
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

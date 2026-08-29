use crate::bytecode::program::ExecutionContext;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub(super) async fn agree(
    execution: &ExecutionContext,
    id: runmat_types::CollectiveId,
    value: Value,
    label: &str,
) -> Result<(Value, runmat_execution::SpmdTaskContext), RuntimeError> {
    let collective = execution
        .runtime
        .service_ports()
        .require_collective(label)
        .map_err(super::super::capability_error)?
        .clone();
    let context = collective.context().clone();
    let payload = runmat_runtime::execution::value_codec::encode_inline_value(&value)
        .map_err(super::super::value_codec_error)?;
    let digest = payload.logical_digest().map_err(|error| {
        crate::interpreter::errors::mex("CodistributedValueDigest", &error.to_string())
    })?;
    let sequence = collective.next_sequence(id)?;
    let response = collective
        .execute(runmat_execution::CollectiveRequest {
            context: context.clone(),
            id,
            sequence,
            invocation: runmat_execution::CollectiveInvocation::AssertEqual { digest },
        })
        .await?;
    let runmat_execution::CollectiveResponse::Agreement { equal } = response else {
        return Err(crate::interpreter::errors::mex(
            "CodistributedCoordination",
            "codistributed agreement returned an incompatible response",
        ));
    };
    if !equal {
        return Err(crate::interpreter::errors::mex(
            "CodistributedWorkerDisagreement",
            &format!("workers supplied different {label} values"),
        ));
    }
    Ok((value, context))
}

pub(super) async fn broadcast_from(
    execution: &ExecutionContext,
    id: runmat_types::CollectiveId,
    root: runmat_types::LabRank,
    value: Value,
) -> Result<(Value, runmat_execution::SpmdTaskContext), RuntimeError> {
    let collective = execution
        .runtime
        .service_ports()
        .require_collective("designated-worker codistributed construction")
        .map_err(super::super::capability_error)?
        .clone();
    let context = collective.context().clone();
    let payload = (context.rank == root)
        .then(|| runmat_runtime::execution::value_codec::encode_inline_value(&value))
        .transpose()
        .map_err(super::super::value_codec_error)?;
    let sequence = collective.next_sequence(id)?;
    let response = collective
        .execute(runmat_execution::CollectiveRequest {
            context: context.clone(),
            id,
            sequence,
            invocation: runmat_execution::CollectiveInvocation::Broadcast {
                root,
                value: payload,
            },
        })
        .await?;
    let runmat_execution::CollectiveResponse::Value { value } = response else {
        return Err(crate::interpreter::errors::mex(
            "CodistributedCoordination",
            "designated-worker broadcast returned an incompatible response",
        ));
    };
    Ok((
        runmat_runtime::execution::value_codec::decode_inline_value(&value)
            .map_err(super::super::value_codec_error)?,
        context,
    ))
}

pub(super) async fn build(
    execution: &ExecutionContext,
    id: runmat_types::CollectiveId,
    local_part: &Value,
    codistributor: Option<&Value>,
    validate_across_workers: bool,
) -> Result<
    (
        Vec<runmat_execution::DistributedBuildContribution>,
        runmat_execution::SpmdTaskContext,
    ),
    RuntimeError,
> {
    let collective = execution
        .runtime
        .service_ports()
        .require_collective("codistributed.build")
        .map_err(super::super::capability_error)?
        .clone();
    let context = collective.context().clone();
    let contribution = runmat_execution::DistributedBuildContribution {
        value: runmat_runtime::value_fact::value_fact(local_part),
        local_shape: runmat_runtime::parallel::distribution::value_shape(local_part).await?,
        codistributor: codistributor
            .map(runmat_runtime::execution::value_codec::encode_inline_value)
            .transpose()
            .map_err(super::super::value_codec_error)?,
    };
    let sequence = collective.next_sequence(id)?;
    let response = collective
        .execute(runmat_execution::CollectiveRequest {
            context: context.clone(),
            id,
            sequence,
            invocation: runmat_execution::CollectiveInvocation::DistributedBuild {
                contribution: Box::new(contribution),
                validate_across_workers,
            },
        })
        .await?;
    let runmat_execution::CollectiveResponse::DistributedBuild { contributions } = response else {
        return Err(crate::interpreter::errors::mex(
            "CodistributedCoordination",
            "codistributed.build returned an incompatible coordination response",
        ));
    };
    if contributions.len() != context.gang.labs.0 as usize {
        return Err(crate::interpreter::errors::mex(
            "CodistributedCoordination",
            "codistributed.build did not receive one contribution per worker",
        ));
    }
    if validate_across_workers {
        let first = &contributions[0];
        if contributions.iter().skip(1).any(|candidate| {
            candidate.value.kind != first.value.kind
                || candidate.value.storage != first.value.storage
                || candidate.value.layout != first.value.layout
                || candidate.codistributor != first.codistributor
        }) {
            return Err(crate::interpreter::errors::mex(
                "CodistributedWorkerDisagreement",
                "codistributed.build workers supplied incompatible classes, storage, layouts, or codistributors",
            ));
        }
    }
    Ok((contributions, context))
}

use crate::bytecode::program::ExecutionContext;
use runmat_runtime::RuntimeError;
use runmat_value::Value;

pub(super) struct Executor<'a> {
    service: &'a dyn runmat_runtime::context::RuntimeDistributedService,
    bytecode: &'a crate::Bytecode,
    execution: &'a ExecutionContext,
}

impl<'a> Executor<'a> {
    pub(super) fn new(
        service: &'a dyn runmat_runtime::context::RuntimeDistributedService,
        bytecode: &'a crate::Bytecode,
        execution: &'a ExecutionContext,
    ) -> Self {
        Self {
            service,
            bytecode,
            execution,
        }
    }

    pub(super) async fn create(
        &self,
        id: runmat_types::DistributedValueId,
        owner: runmat_types::DistributedOwner,
        scheme: runmat_types::DistributionScheme,
        arguments: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        let [input] = super::arguments::decode(arguments)?;
        let contract = contract(
            self.bytecode,
            id,
            owner,
            runmat_types::DistributedConstruction::Fixed {
                scheme: scheme.clone(),
            },
        )?;
        let pool = self
            .execution
            .runtime
            .execution()
            .ensure_pool(runmat_execution::PoolRequest::automatic())
            .map_err(super::super::execution_error)?;
        self.service
            .create(contract, input, scheme, pool)
            .await
            .map(|handle| Value::Distributed(Box::new(handle)))
    }

    pub(super) async fn codistributed(
        &self,
        id: runmat_types::DistributedValueId,
        owner: runmat_types::DistributedOwner,
        overload: crate::BytecodeCodistributedOverload,
        coordination: Option<runmat_types::CollectiveId>,
        arguments: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        let construction = match overload {
            crate::BytecodeCodistributedOverload::ReplicatedInputDefault => {
                runmat_types::DistributedConstruction::ReplicatedInputDefault
            }
            crate::BytecodeCodistributedOverload::CodistributorOrDesignatedWorker => {
                runmat_types::DistributedConstruction::CodistributorOrDesignatedWorker
            }
            crate::BytecodeCodistributedOverload::DesignatedWorkerWithCodistributor => {
                runmat_types::DistributedConstruction::DesignatedWorkerWithCodistributor
            }
        };
        let contract = contract(self.bytecode, id, owner, construction)?;
        if contract.coordination != coordination {
            return Err(crate::interpreter::errors::mex(
                "DistributedContractMissing",
                "codistributed coordination identity disagrees with its semantic contract",
            ));
        }
        let pool = self
            .execution
            .runtime
            .execution()
            .ensure_pool(runmat_execution::PoolRequest::automatic())
            .map_err(super::super::execution_error)?;
        match overload {
            crate::BytecodeCodistributedOverload::ReplicatedInputDefault => {
                let [input] = super::arguments::decode(arguments)?;
                if let Some(coordination) = coordination {
                    let (input, context) = super::coordination::agree(
                        self.execution,
                        coordination,
                        input,
                        "codistributed replicated input",
                    )
                    .await?;
                    create_worker_value(self.service, contract, input, None, context).await
                } else {
                    create_client_value(self.service, contract, input, None, pool).await
                }
            }
            crate::BytecodeCodistributedOverload::CodistributorOrDesignatedWorker => {
                let [input, selector] = super::arguments::decode(arguments)?;
                if runmat_runtime::parallel::codistributor::is_codistributor(&selector) {
                    if let Some(coordination) = coordination {
                        let (input, context) = super::coordination::agree(
                            self.execution,
                            coordination,
                            input,
                            "codistributed replicated input",
                        )
                        .await?;
                        let (selector, selector_context) = super::coordination::agree(
                            self.execution,
                            coordination,
                            selector,
                            "codistributor",
                        )
                        .await?;
                        ensure_same_context(&selector_context, &context)?;
                        create_worker_value(self.service, contract, input, Some(selector), context)
                            .await
                    } else {
                        create_client_value(self.service, contract, input, Some(selector), pool)
                            .await
                    }
                } else {
                    let rank =
                        runmat_runtime::parallel::codistributor::designated_worker(&selector)?;
                    let coordination = require_worker_coordination(coordination)?;
                    let (input, context) = super::coordination::broadcast_from(
                        self.execution,
                        coordination,
                        rank,
                        input,
                    )
                    .await?;
                    create_worker_value(self.service, contract, input, None, context).await
                }
            }
            crate::BytecodeCodistributedOverload::DesignatedWorkerWithCodistributor => {
                let [input, worker, codistributor] = super::arguments::decode(arguments)?;
                let rank = runmat_runtime::parallel::codistributor::designated_worker(&worker)?;
                if !runmat_runtime::parallel::codistributor::is_codistributor(&codistributor) {
                    return Err(crate::interpreter::errors::mex(
                        "CodistributorRequired",
                        "codistributed requires a codistributor1d or codistributor2dbc object",
                    ));
                }
                let coordination = require_worker_coordination(coordination)?;
                let (input, context) =
                    super::coordination::broadcast_from(self.execution, coordination, rank, input)
                        .await?;
                let (codistributor, codistributor_context) = super::coordination::agree(
                    self.execution,
                    coordination,
                    codistributor,
                    "codistributor",
                )
                .await?;
                ensure_same_context(&codistributor_context, &context)?;
                create_worker_value(self.service, contract, input, Some(codistributor), context)
                    .await
            }
        }
    }

    pub(super) async fn build(
        &self,
        id: runmat_types::DistributedValueId,
        owner: runmat_types::DistributedOwner,
        has_codistributor: bool,
        validation: crate::BytecodeDistributedBuildValidation,
        coordination: runmat_types::CollectiveId,
        mut arguments: Vec<Value>,
    ) -> Result<Value, RuntimeError> {
        let construction = runmat_types::DistributedConstruction::LocalParts {
            has_codistributor,
            validation: match validation {
                crate::BytecodeDistributedBuildValidation::ValidateAcrossWorkers => {
                    runmat_types::DistributedBuildValidation::ValidateAcrossWorkers
                }
                crate::BytecodeDistributedBuildValidation::NoCommunication => {
                    runmat_types::DistributedBuildValidation::NoCommunication
                }
                crate::BytecodeDistributedBuildValidation::RuntimeOption => {
                    runmat_types::DistributedBuildValidation::RuntimeOption
                }
            },
        };
        let contract = contract(self.bytecode, id, owner, construction)?;
        if contract.coordination != Some(coordination) {
            return Err(crate::interpreter::errors::mex(
                "DistributedContractMissing",
                "codistributed.build coordination identity disagrees with its semantic contract",
            ));
        }
        let validate_across_workers = match validation {
            crate::BytecodeDistributedBuildValidation::ValidateAcrossWorkers => true,
            crate::BytecodeDistributedBuildValidation::NoCommunication => false,
            crate::BytecodeDistributedBuildValidation::RuntimeOption => {
                let option = arguments.pop().ok_or_else(|| {
                    crate::interpreter::errors::mex(
                        "InvalidDistributedInstruction",
                        "codistributed.build is missing its validation option",
                    )
                })?;
                runmat_runtime::parallel::codistributor::build_validation_option(&option)?
            }
        };
        let codistributor = if has_codistributor {
            Some(arguments.pop().ok_or_else(|| {
                crate::interpreter::errors::mex(
                    "InvalidDistributedInstruction",
                    "codistributed.build is missing its codistributor operand",
                )
            })?)
        } else {
            None
        };
        let [local_part] = super::arguments::decode(arguments)?;
        let (contributions, context) = super::coordination::build(
            self.execution,
            coordination,
            &local_part,
            codistributor.as_ref(),
            validate_across_workers,
        )
        .await?;
        let local_shapes = contributions
            .iter()
            .map(|contribution| contribution.local_shape.clone())
            .collect::<Vec<_>>();
        let canonical_codistributor = contributions
            .first()
            .and_then(|contribution| contribution.codistributor.as_ref())
            .map(runmat_runtime::execution::value_codec::decode_inline_value)
            .transpose()
            .map_err(super::super::value_codec_error)?;
        let (global_shape, scheme) = runmat_runtime::parallel::codistributor::resolve_local_parts(
            canonical_codistributor.as_ref(),
            &local_shapes,
            context.gang.labs,
            validate_across_workers,
        )?;
        let layouts = runmat_runtime::parallel::distribution::partition_layouts(
            &global_shape,
            &scheme,
            context.gang.labs,
        )?;
        self.service
            .build_worker(contract, local_part, global_shape, scheme, layouts, context)
            .await
            .map(|handle| Value::Distributed(Box::new(handle)))
    }
}

fn contract(
    bytecode: &crate::Bytecode,
    id: runmat_types::DistributedValueId,
    owner: runmat_types::DistributedOwner,
    construction: runmat_types::DistributedConstruction,
) -> Result<runmat_types::DistributedValueContract, RuntimeError> {
    bytecode
        .distributed_values
        .iter()
        .find(|contract| {
            contract.id == id && contract.owner == owner && contract.construction == construction
        })
        .cloned()
        .ok_or_else(|| {
            crate::interpreter::errors::mex(
                "DistributedContractMissing",
                "distributed instruction has no matching compiler-owned semantic contract",
            )
        })
}

async fn create_client_value(
    service: &dyn runmat_runtime::context::RuntimeDistributedService,
    contract: runmat_types::DistributedValueContract,
    input: Value,
    codistributor: Option<Value>,
    pool: runmat_execution::PoolSnapshot,
) -> Result<Value, RuntimeError> {
    let input = if let Value::Distributed(handle) = input {
        service.materialize(*handle).await?
    } else {
        input
    };
    let shape = runmat_runtime::parallel::distribution::value_shape(&input).await?;
    let labs = runmat_types::LabCount(pool.workers);
    let scheme = if let Some(codistributor) = codistributor {
        runmat_runtime::parallel::codistributor::resolve(&codistributor, &shape, labs)?
    } else {
        runmat_runtime::parallel::codistributor::default_scheme(&shape, labs)?
    };
    service
        .create(contract, input, scheme, pool)
        .await
        .map(|handle| Value::Distributed(Box::new(handle)))
}

async fn create_worker_value(
    service: &dyn runmat_runtime::context::RuntimeDistributedService,
    contract: runmat_types::DistributedValueContract,
    input: Value,
    codistributor: Option<Value>,
    context: runmat_execution::SpmdTaskContext,
) -> Result<Value, RuntimeError> {
    let shape = runmat_runtime::parallel::distribution::value_shape(&input).await?;
    let scheme = if let Some(codistributor) = codistributor {
        runmat_runtime::parallel::codistributor::resolve(&codistributor, &shape, context.gang.labs)?
    } else {
        runmat_runtime::parallel::codistributor::default_scheme(&shape, context.gang.labs)?
    };
    service
        .create_worker(contract, input, scheme, context)
        .await
        .map(|handle| Value::Distributed(Box::new(handle)))
}

fn require_worker_coordination(
    coordination: Option<runmat_types::CollectiveId>,
) -> Result<runmat_types::CollectiveId, RuntimeError> {
    coordination.ok_or_else(|| {
        crate::interpreter::errors::mex(
            "CodistributedWorkerContextRequired",
            "designated-worker codistributed construction requires an SPMD worker context",
        )
    })
}

fn ensure_same_context(
    actual: &runmat_execution::SpmdTaskContext,
    expected: &runmat_execution::SpmdTaskContext,
) -> Result<(), RuntimeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(crate::interpreter::errors::mex(
            "CodistributedCoordination",
            "codistributed coordination changed worker context",
        ))
    }
}

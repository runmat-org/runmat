use std::collections::BTreeMap;

use runmat_hir::FunctionId;
use runmat_types::{
    CapabilityRequirement, CapabilitySet, CollectiveContract, CollectiveOperation,
    DistributedValueContract, LabCount, ParallelAccess, ParallelManifest, ParallelVariableContract,
    ParallelVariableRole, ProgramFunctionId, RegionValueId, SpmdContract, SpmdLabRequirement,
    ValueFact, ValueKindFact,
};

use crate::parallel::{MirCollectiveOp, MirDistributedOp, MirSpmdHeader};
use crate::{
    MirBody, MirDiagnostic, MirDiagnosticSeverity, MirLocalId, MirLocalKind, MirOperand, MirRvalue,
    MirStmt, MirStmtKind, MirTerminatorKind,
};

use super::super::AnalysisStore;
use super::{facts, legality, region};

pub(super) fn classify_body(
    body: &MirBody,
    function: ProgramFunctionId,
    store: &AnalysisStore,
    summaries: &BTreeMap<FunctionId, super::super::inference::FunctionSummary>,
    manifest: &mut ParallelManifest,
    diagnostics: &mut Vec<MirDiagnostic>,
) {
    for header_block in &body.blocks {
        let MirTerminatorKind::Spmd {
            region: id,
            header,
            body_block,
            exit_block,
        } = &header_block.terminator.kind
        else {
            continue;
        };
        if region::is_nested_parallel_header(body, header_block.id) {
            continue;
        }
        let blocks = region::body_blocks(body, header_block.id, *body_block, *exit_block);
        let mut legal = legality::validate_control_flow(
            body,
            &blocks,
            header_block.id,
            *exit_block,
            "spmd",
            diagnostics,
        );
        if region::contains_parallel_region(body, &blocks) {
            diagnostics.push(
                MirDiagnostic::new(
                    "RM-MIR0020",
                    MirDiagnosticSeverity::Error,
                    "an SPMD body cannot contain another parallel region",
                    header_block.terminator.span,
                )
                .with_primary_label(
                    "nested parallel regions do not share a gang or resource budget",
                )
                .with_help("move the nested parallel work outside this SPMD block")
                .with_category("spmd-legality"),
            );
            legal = false;
        }
        if !legal {
            continue;
        }

        let access = region::accesses(body, &blocks);
        let header_position = header_block.statements.len();
        let variables = VariableContractContext {
            body,
            store,
            function,
            block: header_block.id,
            position: header_position,
        };
        let captures = access
            .reads
            .iter()
            .copied()
            .filter_map(|local| {
                variables.contract(
                    local,
                    true,
                    access.writes.contains(&local),
                    ParallelVariableRole::Broadcast,
                )
            })
            .collect::<Vec<_>>();
        let outputs = access
            .writes
            .iter()
            .copied()
            .filter(|local| {
                !matches!(
                    body.locals.get(local.0).map(|local| &local.kind),
                    Some(MirLocalKind::Temporary)
                )
            })
            .filter_map(|local| {
                variables.contract(
                    local,
                    access.reads.contains(&local),
                    true,
                    ParallelVariableRole::Private,
                )
            })
            .collect::<Vec<_>>();

        let mut capabilities = CapabilitySet::default();
        capabilities
            .0
            .insert(CapabilityRequirement::ParallelRuntime);
        let mut effects = runmat_types::EffectSet::default();
        for block in body
            .blocks
            .iter()
            .filter(|block| blocks.contains(&block.id))
        {
            for (position, statement) in block.statements.iter().enumerate() {
                let (statement_effects, statement_capabilities) =
                    super::super::inference::statement_contract(statement, summaries);
                effects.0.extend(statement_effects.0);
                capabilities.0.extend(statement_capabilities.0);
                visit_statement_rvalues(statement, &mut |value| {
                    collect_parallel_operation(
                        value, store, function, block.id, position, manifest,
                    );
                });
            }
        }
        manifest.spmd_regions.push(SpmdContract {
            id: *id,
            labs: lab_requirement(header),
            captures,
            outputs,
            effects,
            capabilities,
        });
    }
}

pub(super) fn classify_distributed_body(
    body: &MirBody,
    function: ProgramFunctionId,
    store: &AnalysisStore,
    manifest: &mut ParallelManifest,
) {
    for block in &body.blocks {
        for (position, statement) in block.statements.iter().enumerate() {
            visit_statement_rvalues(statement, &mut |value| {
                let MirRvalue::Distributed(MirDistributedOp::Create {
                    id,
                    owner,
                    input,
                    scheme,
                }) = value
                else {
                    return;
                };
                manifest.distributed_values.push(DistributedValueContract {
                    id: *id,
                    value: facts::operand_fact(store, function, block.id, position, input),
                    scheme: scheme.clone(),
                    owner: *owner,
                    materializable: true,
                });
            });
        }
    }
}

struct VariableContractContext<'a> {
    body: &'a MirBody,
    store: &'a AnalysisStore,
    function: ProgramFunctionId,
    block: crate::BasicBlockId,
    position: usize,
}

impl VariableContractContext<'_> {
    fn contract(
        &self,
        local: MirLocalId,
        read: bool,
        write: bool,
        role: ParallelVariableRole,
    ) -> Option<ParallelVariableContract> {
        let value = RegionValueId {
            function: self.function,
            local: u32::try_from(local.0).ok()?,
        };
        let fact = facts::value_fact(self.store, self.function, self.block, self.position, local);
        let access = match (read, write) {
            (true, true) => ParallelAccess::ReadWrite,
            (false, true) => ParallelAccess::Write,
            _ => ParallelAccess::Read,
        };
        if matches!(self.body.locals.get(local.0)?.kind, MirLocalKind::Temporary) {
            return None;
        }
        Some(ParallelVariableContract {
            value,
            role,
            access,
            transferable: facts::transferable(&fact),
            fact,
        })
    }
}

fn collect_parallel_operation(
    value: &MirRvalue,
    store: &AnalysisStore,
    function: ProgramFunctionId,
    block: crate::BasicBlockId,
    position: usize,
    manifest: &mut ParallelManifest,
) {
    if let MirRvalue::Collective(operation) = value {
        manifest.collectives.push(collective_contract(
            operation, store, function, block, position,
        ));
    }
}

fn collective_contract(
    operation: &MirCollectiveOp,
    store: &AnalysisStore,
    function: ProgramFunctionId,
    block: crate::BasicBlockId,
    position: usize,
) -> CollectiveContract {
    let fact =
        |operand: &MirOperand| facts::operand_fact(store, function, block, position, operand);
    let unknown = || ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue);
    match operation {
        MirCollectiveOp::Barrier { id } => contract(*id, CollectiveOperation::Barrier),
        MirCollectiveOp::Broadcast { id, input, root } => CollectiveContract {
            id: *id,
            operation: CollectiveOperation::Broadcast,
            input: input.as_ref().map(fact),
            output: Some(input.as_ref().map_or_else(unknown, fact)),
            root: Some(fact(root)),
            source: None,
            destination: None,
            tag: None,
        },
        MirCollectiveOp::Gather { id, input, root } => CollectiveContract {
            id: *id,
            operation: CollectiveOperation::Gather,
            input: Some(fact(input)),
            output: Some(fact(input)),
            root: Some(fact(root)),
            source: None,
            destination: None,
            tag: None,
        },
        MirCollectiveOp::Scatter { id, input, root } => CollectiveContract {
            id: *id,
            operation: CollectiveOperation::Scatter,
            input: Some(fact(input)),
            output: Some(fact(input)),
            root: Some(fact(root)),
            source: None,
            destination: None,
            tag: None,
        },
        MirCollectiveOp::AllGather { id, input } => CollectiveContract {
            id: *id,
            operation: CollectiveOperation::AllGather,
            input: Some(fact(input)),
            output: Some(fact(input)),
            root: None,
            source: None,
            destination: None,
            tag: None,
        },
        MirCollectiveOp::Reduce {
            id,
            input,
            root,
            operator,
        } => CollectiveContract {
            id: *id,
            operation: CollectiveOperation::Reduce {
                operator: *operator,
            },
            input: Some(fact(input)),
            output: Some(fact(input)),
            root: Some(fact(root)),
            source: None,
            destination: None,
            tag: None,
        },
        MirCollectiveOp::AllReduce {
            id,
            input,
            operator,
        } => CollectiveContract {
            id: *id,
            operation: CollectiveOperation::AllReduce {
                operator: *operator,
            },
            input: Some(fact(input)),
            output: Some(fact(input)),
            root: None,
            source: None,
            destination: None,
            tag: None,
        },
        MirCollectiveOp::Send {
            id,
            input,
            destination,
            tag,
        } => CollectiveContract {
            id: *id,
            operation: CollectiveOperation::Send,
            input: Some(fact(input)),
            output: None,
            root: None,
            source: None,
            destination: Some(fact(destination)),
            tag: tag.as_ref().map(fact),
        },
        MirCollectiveOp::Receive {
            id,
            source,
            tag,
            requested_outputs,
        } => CollectiveContract {
            id: *id,
            operation: CollectiveOperation::Receive {
                requested_outputs: *requested_outputs,
            },
            input: None,
            output: Some(unknown()),
            root: None,
            source: source.as_ref().map(fact),
            destination: None,
            tag: tag.as_ref().map(fact),
        },
        MirCollectiveOp::SendReceive {
            id,
            destination,
            source,
            input,
            tag,
        } => CollectiveContract {
            id: *id,
            operation: CollectiveOperation::SendReceive,
            input: Some(fact(input)),
            output: Some(unknown()),
            root: None,
            source: Some(fact(source)),
            destination: Some(fact(destination)),
            tag: tag.as_ref().map(fact),
        },
        MirCollectiveOp::Probe { id, source, tag } => CollectiveContract {
            id: *id,
            operation: CollectiveOperation::Probe,
            input: None,
            output: Some(ValueFact::scalar(ValueKindFact::Logical)),
            root: None,
            source: source.as_ref().map(fact),
            destination: None,
            tag: tag.as_ref().map(fact),
        },
    }
}

fn contract(id: runmat_types::CollectiveId, operation: CollectiveOperation) -> CollectiveContract {
    CollectiveContract {
        id,
        operation,
        input: None,
        output: None,
        root: None,
        source: None,
        destination: None,
        tag: None,
    }
}

fn visit_statement_rvalues(statement: &MirStmt, operation: &mut impl FnMut(&MirRvalue)) {
    match &statement.kind {
        MirStmtKind::Assign { value, .. }
        | MirStmtKind::MultiAssign { value, .. }
        | MirStmtKind::Expr(value) => visit_rvalue(value, operation),
        MirStmtKind::PlaceMutation(_)
        | MirStmtKind::WorkspaceEffect { .. }
        | MirStmtKind::EnvironmentEffect(_) => {}
    }
}

fn visit_rvalue(value: &MirRvalue, operation: &mut impl FnMut(&MirRvalue)) {
    operation(value);
    if let MirRvalue::ShortCircuit { right_temps, .. } = value {
        for statement in right_temps {
            visit_statement_rvalues(statement, operation);
        }
    }
}

fn lab_requirement(header: &MirSpmdHeader<MirRvalue>) -> SpmdLabRequirement {
    match header {
        MirSpmdHeader::Default => SpmdLabRequirement::Default,
        MirSpmdHeader::One(value) => constant_count(value)
            .map(|labs| SpmdLabRequirement::Exact { labs })
            .unwrap_or(SpmdLabRequirement::Default),
        MirSpmdHeader::Two(minimum, maximum) => range(minimum, maximum),
        MirSpmdHeader::Three(_, minimum, maximum) => range(minimum, maximum),
    }
}

fn range(minimum: &MirRvalue, maximum: &MirRvalue) -> SpmdLabRequirement {
    constant_count(minimum)
        .zip(constant_count(maximum))
        .map(|(minimum, maximum)| SpmdLabRequirement::Range { minimum, maximum })
        .unwrap_or(SpmdLabRequirement::Default)
}

fn constant_count(value: &MirRvalue) -> Option<LabCount> {
    let MirRvalue::Use(MirOperand::Constant(constant)) = value else {
        return None;
    };
    match constant {
        crate::MirConstant::Number(value) => value.parse::<u32>().ok().map(LabCount),
        crate::MirConstant::IntegerLiteral(value) => integer_literal_count(value),
        _ => None,
    }
}

fn integer_literal_count(value: &runmat_hir::IntegerLiteral) -> Option<LabCount> {
    use runmat_hir::IntegerLiteralClass;

    let bits = value.bits();
    let nonnegative = match value.class() {
        IntegerLiteralClass::Int8 => bits < (1 << 7),
        IntegerLiteralClass::Int16 => bits < (1 << 15),
        IntegerLiteralClass::Int32 => bits < (1 << 31),
        IntegerLiteralClass::Int64 => bits < (1 << 63),
        IntegerLiteralClass::UInt8
        | IntegerLiteralClass::UInt16
        | IntegerLiteralClass::UInt32
        | IntegerLiteralClass::UInt64 => true,
    };
    nonnegative
        .then(|| u32::try_from(bits).ok().map(LabCount))
        .flatten()
}

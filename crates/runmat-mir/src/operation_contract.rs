use runmat_types::{CapabilitySet, EffectSet, ProgramFunctionId, ProgramPointId};
use serde::{Deserialize, Serialize};

/// Stable identity of one semantic operation in canonical MIR.
///
/// The identity is shared by analysis and native lowering.  It is deliberately
/// independent of cumulative flow state: two equal effects at adjacent sites
/// still have two distinct contracts.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirOperationSite {
    pub point: ProgramPointId,
    pub phase: MirOperationPhase,
    pub ordinal: u32,
    pub construct: crate::MirConstructKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirOperationPhase {
    Rvalue,
    Statement,
    TerminatorRvalue,
    Terminator,
}

/// Complete operation-local semantic requirements certified by MIR analysis.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirOperationContract {
    pub site: MirOperationSite,
    pub effects: EffectSet,
    pub capabilities: CapabilitySet,
}

pub fn operation_sites(
    function: ProgramFunctionId,
    body: &crate::MirBody,
) -> Option<Vec<MirOperationSite>> {
    let mut sites = Vec::new();
    for block in &body.blocks {
        let block_id = u32::try_from(block.id.0).ok()?;
        for (position, statement) in block.statements.iter().enumerate() {
            let point = ProgramPointId {
                function,
                block: block_id,
                position: u32::try_from(position).ok()?,
            };
            if let Some(value) = statement_rvalue(&statement.kind) {
                sites.push(MirOperationSite {
                    point,
                    phase: MirOperationPhase::Rvalue,
                    ordinal: 0,
                    construct: crate::rvalue_construct_kind(value),
                });
            }
            sites.push(MirOperationSite {
                point,
                phase: MirOperationPhase::Statement,
                ordinal: 0,
                construct: crate::statement_construct_kind(&statement.kind),
            });
        }
        let point = ProgramPointId {
            function,
            block: block_id,
            position: u32::try_from(block.statements.len()).ok()?,
        };
        for (ordinal, value) in terminator_rvalues(&block.terminator.kind)
            .into_iter()
            .enumerate()
        {
            sites.push(MirOperationSite {
                point,
                phase: MirOperationPhase::TerminatorRvalue,
                ordinal: u32::try_from(ordinal).ok()?,
                construct: crate::rvalue_construct_kind(&value),
            });
        }
        sites.push(MirOperationSite {
            point,
            phase: MirOperationPhase::Terminator,
            ordinal: 0,
            construct: crate::terminator_construct_kind(&block.terminator.kind),
        });
    }
    sites.sort();
    sites
        .windows(2)
        .all(|pair| pair[0] != pair[1])
        .then_some(sites)
}

pub fn statement_rvalue(statement: &crate::MirStmtKind) -> Option<&crate::MirRvalue> {
    match statement {
        crate::MirStmtKind::PreparedAssign { value, .. }
        | crate::MirStmtKind::Assign { value, .. }
        | crate::MirStmtKind::MultiAssign { value, .. }
        | crate::MirStmtKind::SequenceAssign { value, .. }
        | crate::MirStmtKind::Expr(value) => Some(value),
        crate::MirStmtKind::PrepareFixedAssignment(_)
        | crate::MirStmtKind::PlaceMutation(_)
        | crate::MirStmtKind::StatementOutput(_)
        | crate::MirStmtKind::CaptureSequence { .. }
        | crate::MirStmtKind::WorkspaceEffect { .. }
        | crate::MirStmtKind::EnvironmentEffect(_) => None,
    }
}

/// Canonical evaluation order for MIR values embedded in a terminator.
pub fn terminator_rvalues(terminator: &crate::MirTerminatorKind) -> Vec<crate::MirRvalue> {
    fn operand(value: &crate::MirOperand, values: &mut Vec<crate::MirRvalue>) {
        if !matches!(value, crate::MirOperand::Local(_)) {
            values.push(crate::MirRvalue::Use(value.clone()));
        }
    }
    let mut values = Vec::new();
    match terminator {
        crate::MirTerminatorKind::Goto(_)
        | crate::MirTerminatorKind::TryCatch { .. }
        | crate::MirTerminatorKind::Unreachable => {}
        crate::MirTerminatorKind::Branch { cond, .. } => operand(cond, &mut values),
        crate::MirTerminatorKind::Switch { discr, cases, .. } => {
            operand(discr, &mut values);
            for (case, _) in cases {
                operand(case, &mut values);
            }
        }
        crate::MirTerminatorKind::For { iterable, .. } => values.push(iterable.clone()),
        crate::MirTerminatorKind::ParFor {
            iterable,
            maximum_workers,
            ..
        } => {
            values.push(iterable.clone());
            values.extend(maximum_workers.as_deref().cloned());
        }
        crate::MirTerminatorKind::Spmd { header, .. } => match header.as_ref() {
            crate::parallel::MirSpmdHeader::Default => {}
            crate::parallel::MirSpmdHeader::One(a) => values.push(a.clone()),
            crate::parallel::MirSpmdHeader::Two(a, b) => {
                values.push(a.clone());
                values.push(b.clone());
            }
            crate::parallel::MirSpmdHeader::Three(a, b, c) => {
                values.push(a.clone());
                values.push(b.clone());
                values.push(c.clone());
            }
        },
        crate::MirTerminatorKind::Return(outputs) => {
            for output in outputs {
                operand(output, &mut values);
            }
        }
        crate::MirTerminatorKind::Await { future, .. } => operand(future, &mut values),
    }
    values
}

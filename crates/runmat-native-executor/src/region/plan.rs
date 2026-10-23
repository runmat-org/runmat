use std::collections::{BTreeMap, BTreeSet};

use runmat_native_codegen::{
    NativeFunction, NativeOperation, NativeRegionBoundaryKind, NativeSitePhase,
};
use runmat_runtime::numeric_region::{
    NumericBinaryOperation, NumericRegionNode, NumericRegionProgram, NumericUnaryOperation,
};
use runmat_types::{ProgramPointId, RegionContract, RegionId, RegionValueId};

/// Retain the executor's existing immutable plan collection, never deep-copy
/// the program, traversal, output or skipped-site metadata at a checkpoint.
#[derive(Clone)]
pub(crate) struct OptimizedRegionPlanLease {
    plans: std::sync::Arc<Vec<OptimizedRegionPlan>>,
    index: usize,
}

impl OptimizedRegionPlanLease {
    pub(crate) fn new(plans: std::sync::Arc<Vec<OptimizedRegionPlan>>, index: usize) -> Self {
        assert!(index < plans.len());
        Self { plans, index }
    }

    #[cfg(test)]
    pub(crate) fn from_test_plan(plan: OptimizedRegionPlan) -> Self {
        Self::new(std::sync::Arc::new(vec![plan]), 0)
    }
}

impl std::ops::Deref for OptimizedRegionPlanLease {
    type Target = OptimizedRegionPlan;
    fn deref(&self) -> &Self::Target {
        &self.plans[self.index]
    }
}

#[derive(Clone)]
pub(crate) struct OptimizedRegionPlan {
    pub region: RegionId,
    pub entry: SiteIdentity,
    /// Exact first compiler-authored site outside this traversal, including
    /// phase and occurrence ordinal. A point alone cannot complete feedback.
    pub exit_site: SiteIdentity,
    pub traversal_sites: Vec<SiteIdentity>,
    /// Arithmetic occurrences in the exact order of the verified program's
    /// Unary/Binary nodes; structural alias/assignment sites are not invented
    /// as numerical operations.
    pub operation_sites: Vec<SiteIdentity>,
    pub inputs: Vec<RegionValueId>,
    pub outputs: Vec<RegionOutput>,
    pub program: NumericRegionProgram,
    pub skipped_sites: BTreeSet<SiteIdentity>,
    pub arithmetic_operations: u32,
}

#[derive(Clone)]
pub(crate) struct RegionOutput {
    pub value: Option<RegionValueId>,
    pub ssa: runmat_native_codegen::NativeValueId,
    pub source: RegionOutputSource,
}

#[derive(Clone, Copy)]
pub(crate) enum RegionOutputSource {
    Input(usize),
    Computed(usize),
}

struct LinearRegion<'a> {
    instructions: Vec<&'a runmat_native_codegen::NativeInstruction>,
    sites: Vec<SiteIdentity>,
    exit_site: SiteIdentity,
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct SiteIdentity {
    pub point: ProgramPointId,
    pub phase: NativeSitePhase,
    pub ordinal: u32,
}

pub(crate) fn derive_plans(
    functions: &[NativeFunction],
    regions: &[RegionContract],
) -> Vec<OptimizedRegionPlan> {
    // Portable ordinals identify points within one product, not session-wide
    // function ownership. An ambiguous ordinal must decline optimization;
    // selecting whichever body appeared last could run a sibling revision.
    let mut by_portable: BTreeMap<_, Option<&NativeFunction>> = BTreeMap::new();
    for function in functions {
        by_portable
            .entry(function.portable_id)
            .and_modify(|owner| *owner = None)
            .or_insert(Some(function));
    }
    let mut plans = regions
        .iter()
        .filter_map(|contract| {
            derive_plan(by_portable.get(&contract.id.function)?.as_ref()?, contract)
        })
        .collect::<Vec<_>>();
    plans.sort_by_key(|plan| plan.region);
    plans
}

fn derive_plan(
    function: &NativeFunction,
    contract: &RegionContract,
) -> Option<OptimizedRegionPlan> {
    if !contract.effects.0.is_empty() || contract.exits.len() != 1 {
        return None;
    }
    let entry_boundary = function
        .blocks
        .iter()
        .flat_map(|block| &block.region_boundaries)
        .find(|boundary| {
            boundary.region == contract.id && boundary.kind == NativeRegionBoundaryKind::Entry
        })?;
    let exit_boundary = function
        .blocks
        .iter()
        .flat_map(|block| &block.region_boundaries)
        .find(|boundary| {
            boundary.region == contract.id && boundary.kind == NativeRegionBoundaryKind::Exit
        })?;
    let trace = linear_region(function, contract.entry, contract.exits[0])?;
    let entry = trace
        .instructions
        .first()
        .map(|instruction| site_identity(&instruction.site))?;
    if entry.point != contract.entry || entry.phase != NativeSitePhase::Rvalue {
        return None;
    }
    let inputs = entry_boundary
        .live_values
        .iter()
        .map(|binding| binding.value)
        .collect::<Vec<_>>();
    let mut local_nodes = inputs
        .iter()
        .enumerate()
        .map(|(index, value)| (value.local as usize, index))
        .collect::<BTreeMap<_, _>>();
    let mut nodes = inputs
        .iter()
        .enumerate()
        .map(|(index, _)| NumericRegionNode::Input(index))
        .collect::<Vec<_>>();
    let mut ssa_nodes = entry_boundary
        .live_values
        .iter()
        .enumerate()
        .map(|(index, binding)| (binding.ssa, index))
        .collect::<BTreeMap<_, _>>();
    let mut arithmetic_operations = 0_u32;
    let mut operation_sites = Vec::new();
    let mut skipped_sites = BTreeSet::new();
    let mut region_sites = BTreeSet::new();
    let mut pending_assignment = None;
    for (index, instruction) in trace.instructions.into_iter().enumerate() {
        region_sites.insert(site_identity(&instruction.site));
        if index != 0 {
            skipped_sites.insert(site_identity(&instruction.site));
        }
        match &instruction.operation {
            NativeOperation::Rvalue { value, .. } => {
                let expression = expression_node(value, &local_nodes, &mut nodes)?;
                if matches!(
                    value,
                    runmat_mir::MirRvalue::Unary(..) | runmat_mir::MirRvalue::Binary(..)
                ) {
                    arithmetic_operations = arithmetic_operations.saturating_add(1);
                    operation_sites.push(site_identity(&instruction.site));
                }
                if instruction.outputs.len() != 1 {
                    return None;
                }
                ssa_nodes.insert(instruction.outputs[0].value, expression);
                if pending_assignment.replace((value, expression)).is_some() {
                    return None;
                }
            }
            NativeOperation::Statement(runmat_mir::MirStmtKind::Assign { place, value }) => {
                let runmat_mir::MirPlace::Local(local) = place else {
                    return None;
                };
                let (expected, expression) = pending_assignment.take()?;
                if expected != value {
                    return None;
                }
                if instruction.outputs.len() != 1 {
                    return None;
                }
                ssa_nodes.insert(instruction.outputs[0].value, expression);
                local_nodes.insert(local.0, expression);
            }
            _ => return None,
        }
    }
    if arithmetic_operations == 0 || pending_assignment.is_some() {
        return None;
    }
    let mut outputs = externally_consumed_values(function, &region_sites)
        .into_iter()
        .filter(|ssa| ssa_nodes.contains_key(ssa))
        .map(|ssa| (ssa, None))
        .collect::<BTreeMap<_, _>>();
    outputs.extend(
        exit_boundary
            .live_values
            .iter()
            .map(|binding| (binding.ssa, Some(binding.value))),
    );
    let mut computed_nodes = Vec::new();
    let mut computed_indices = BTreeMap::new();
    let outputs = outputs
        .into_iter()
        .map(|(ssa, value)| {
            let node = *ssa_nodes.get(&ssa)?;
            let source = match nodes.get(node)? {
                NumericRegionNode::Input(input) => RegionOutputSource::Input(*input),
                _ => {
                    let next = computed_nodes.len();
                    let computed = *computed_indices.entry(node).or_insert_with(|| {
                        computed_nodes.push(node);
                        next
                    });
                    RegionOutputSource::Computed(computed)
                }
            };
            Some(RegionOutput { value, ssa, source })
        })
        .collect::<Option<Vec<_>>>()?;
    let program = NumericRegionProgram {
        nodes,
        outputs: computed_nodes,
    };
    program.validate(inputs.len()).ok()?;
    Some(OptimizedRegionPlan {
        region: contract.id,
        entry,
        exit_site: trace.exit_site,
        traversal_sites: trace.sites,
        operation_sites,
        inputs,
        outputs,
        program,
        skipped_sites,
        arithmetic_operations,
    })
}

fn site_identity(site: &runmat_native_codegen::NativeMirSite) -> SiteIdentity {
    SiteIdentity {
        point: site.point,
        phase: site.phase,
        ordinal: site.ordinal,
    }
}

fn externally_consumed_values(
    function: &NativeFunction,
    region_sites: &BTreeSet<SiteIdentity>,
) -> BTreeSet<runmat_native_codegen::NativeValueId> {
    let mut values = BTreeSet::new();
    for block in &function.blocks {
        for instruction in &block.instructions {
            if region_sites.contains(&site_identity(&instruction.site)) {
                continue;
            }
            values.extend(instruction.inputs.iter().copied());
            if let Some(frame) = &instruction.frame_state {
                values.extend(frame.locals.iter().map(|local| local.value));
            }
        }
        values.extend(
            block
                .terminator
                .frame_state
                .locals
                .iter()
                .map(|local| local.value),
        );
        collect_terminator_values(&block.terminator.kind, &mut values);
    }
    values
}

fn collect_terminator_values(
    terminator: &runmat_native_codegen::NativeTerminatorKind,
    values: &mut BTreeSet<runmat_native_codegen::NativeValueId>,
) {
    use runmat_native_codegen::NativeTerminatorKind;
    match terminator {
        NativeTerminatorKind::Goto { edge } => collect_edge_values(edge, values),
        NativeTerminatorKind::Branch {
            condition,
            then_edge,
            else_edge,
        } => {
            values.insert(*condition);
            collect_edge_values(then_edge, values);
            collect_edge_values(else_edge, values);
        }
        NativeTerminatorKind::Switch {
            discriminant,
            cases,
            otherwise,
        } => {
            values.insert(*discriminant);
            for (case, edge) in cases {
                values.insert(*case);
                collect_edge_values(edge, values);
            }
            collect_edge_values(otherwise, values);
        }
        NativeTerminatorKind::For {
            iterable,
            body,
            exit,
            ..
        }
        | NativeTerminatorKind::ParFor {
            iterable,
            body,
            exit,
            ..
        } => {
            values.insert(*iterable);
            if let NativeTerminatorKind::ParFor {
                maximum_workers: Some(maximum_workers),
                ..
            } = terminator
            {
                values.insert(*maximum_workers);
            }
            collect_edge_values(body, values);
            collect_edge_values(exit, values);
        }
        NativeTerminatorKind::Spmd {
            header, body, exit, ..
        } => {
            values.extend(header.iter().copied());
            collect_edge_values(body, values);
            collect_edge_values(exit, values);
        }
        NativeTerminatorKind::TryCatch {
            try_edge,
            catch_edge,
            ..
        } => {
            collect_edge_values(try_edge, values);
            collect_edge_values(catch_edge, values);
        }
        NativeTerminatorKind::Return { values: returned } => {
            values.extend(returned.iter().copied());
        }
        NativeTerminatorKind::Await { future, resume, .. } => {
            values.insert(*future);
            collect_edge_values(resume, values);
        }
        NativeTerminatorKind::Unreachable => {}
    }
}

fn collect_edge_values(
    edge: &runmat_native_codegen::NativeEdge,
    values: &mut BTreeSet<runmat_native_codegen::NativeValueId>,
) {
    values.extend(edge.arguments.iter().filter_map(|argument| {
        let runmat_native_codegen::NativeEdgeArgument::Value(value) = argument else {
            return None;
        };
        Some(*value)
    }));
}

fn expression_node(
    value: &runmat_mir::MirRvalue,
    locals: &BTreeMap<usize, usize>,
    nodes: &mut Vec<NumericRegionNode>,
) -> Option<usize> {
    use runmat_mir::MirRvalue;
    match value {
        MirRvalue::Use(operand) => operand_node(operand, locals, nodes),
        MirRvalue::Unary(operator, operand) => {
            let operation = match operator {
                runmat_types::OperatorKind::UnaryPlus => NumericUnaryOperation::Plus,
                runmat_types::OperatorKind::UnaryMinus => NumericUnaryOperation::Minus,
                _ => return None,
            };
            let input = operand_node(operand, locals, nodes)?;
            let index = nodes.len();
            nodes.push(NumericRegionNode::Unary { operation, input });
            Some(index)
        }
        MirRvalue::Binary(left, operator, right) => {
            let operation = match operator {
                runmat_types::OperatorKind::Add => NumericBinaryOperation::Add,
                runmat_types::OperatorKind::Subtract => NumericBinaryOperation::Subtract,
                runmat_types::OperatorKind::ElementwiseMultiply => NumericBinaryOperation::Multiply,
                runmat_types::OperatorKind::ElementwiseDivide => NumericBinaryOperation::Divide,
                runmat_types::OperatorKind::ElementwiseLeftDivide => {
                    NumericBinaryOperation::LeftDivide
                }
                _ => return None,
            };
            let left = operand_node(left, locals, nodes)?;
            let right = operand_node(right, locals, nodes)?;
            let index = nodes.len();
            nodes.push(NumericRegionNode::Binary {
                operation,
                left,
                right,
            });
            Some(index)
        }
        _ => None,
    }
}

fn operand_node(
    operand: &runmat_mir::MirOperand,
    locals: &BTreeMap<usize, usize>,
    nodes: &mut Vec<NumericRegionNode>,
) -> Option<usize> {
    match operand {
        runmat_mir::MirOperand::Local(local) => locals.get(&local.0).copied(),
        runmat_mir::MirOperand::Constant(runmat_mir::MirConstant::Number(value)) => {
            let value = value.parse().ok()?;
            let index = nodes.len();
            nodes.push(NumericRegionNode::Constant(value));
            Some(index)
        }
        _ => None,
    }
}

fn linear_region<'a>(
    function: &'a NativeFunction,
    entry: ProgramPointId,
    exit: ProgramPointId,
) -> Option<LinearRegion<'a>> {
    let blocks = function
        .blocks
        .iter()
        .map(|block| (block.id.0, block))
        .collect::<BTreeMap<_, _>>();
    let mut block = entry.block;
    let mut position = entry.position;
    let mut visited = BTreeSet::new();
    let mut instructions = Vec::new();
    let mut sites = Vec::new();
    loop {
        if !visited.insert(block) {
            return None;
        }
        let current = blocks.get(&block)?;
        for instruction in &current.instructions {
            if instruction.site.point.position < position {
                continue;
            }
            if instruction.site.point == exit {
                return Some(LinearRegion {
                    instructions,
                    sites,
                    exit_site: site_identity(&instruction.site),
                });
            }
            instructions.push(instruction);
            sites.push(site_identity(&instruction.site));
        }
        if current.terminator.site.point == exit {
            return Some(LinearRegion {
                instructions,
                sites,
                exit_site: site_identity(&current.terminator.site),
            });
        }
        let runmat_native_codegen::NativeTerminatorKind::Goto { edge } = &current.terminator.kind
        else {
            return None;
        };
        sites.push(site_identity(&current.terminator.site));
        block = edge.target.0;
        position = 0;
    }
}

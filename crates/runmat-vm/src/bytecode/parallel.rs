use runmat_mir::{BasicBlockId, MirAssembly, MirLocalId, MirTerminatorKind};
use runmat_types::{
    ParallelManifest, ParallelRegionId, ParallelVariableContract, ParallelVariableRole,
    ParforContract, ProgramFunctionId, ProgramPointId,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::{Bytecode, BytecodeRegionBoundary};

/// Required VM executable for one analyzed `parfor` region.
///
/// The semantic contract remains authoritative for variable roles and facts.
/// This record binds those identities to the immutable VM frame and control
/// boundaries generated for the same program revision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BytecodeParforRegion {
    pub contract: ParforContract,
    pub header: BytecodeRegionBoundary,
    pub body: BytecodeRegionBoundary,
    pub exit: BytecodeRegionBoundary,
    pub loop_slot: usize,
    pub variables: Vec<BytecodeParallelVariable>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BytecodeParallelVariable {
    pub contract: ParallelVariableContract,
    pub slot: usize,
}

impl BytecodeParforRegion {
    /// Values that must cross from the coordinator into a worker frame.
    pub fn input_variables(&self) -> impl Iterator<Item = &BytecodeParallelVariable> {
        self.variables.iter().filter(|variable| {
            matches!(
                variable.contract.role,
                ParallelVariableRole::Broadcast
                    | ParallelVariableRole::Sliced { .. }
                    | ParallelVariableRole::Reduction { .. }
            )
        })
    }

    /// Values whose worker-local updates must be assembled by the coordinator.
    pub fn output_variables(&self) -> impl Iterator<Item = &BytecodeParallelVariable> {
        self.variables.iter().filter(|variable| {
            matches!(
                variable.contract.role,
                ParallelVariableRole::Sliced { .. } | ParallelVariableRole::Reduction { .. }
            )
        })
    }

    pub(crate) fn rebind_owner(&mut self, owner: ProgramFunctionId) {
        self.contract.id.0.function = owner;
        self.contract.loop_variable.function = owner;
        for variable in &mut self.contract.variables {
            variable.value.function = owner;
            if let ParallelVariableRole::Sliced { access } = &mut variable.role {
                match &mut access.offset {
                    runmat_types::ParallelSliceOffset::Add(
                        runmat_types::ParallelSliceOffsetOperand::Broadcast(value),
                    )
                    | runmat_types::ParallelSliceOffset::Subtract(
                        runmat_types::ParallelSliceOffsetOperand::Broadcast(value),
                    ) => value.function = owner,
                    _ => {}
                }
            }
        }
        self.header.point.function = owner;
        self.body.point.function = owner;
        self.exit.point.function = owner;
    }
}

impl Bytecode {
    /// Installs the mandatory VM execution records for every analyzed parallel
    /// region. Unlike optional optimizer regions, a parallel contract cannot
    /// survive without exact bytecode and frame boundaries.
    pub fn install_parallel_regions(
        &mut self,
        mir: &MirAssembly,
        manifest: &ParallelManifest,
    ) -> Result<(), String> {
        manifest
            .validate()
            .map_err(|error| format!("{}: {}", error.path, error.message))?;
        let mir_regions = mir
            .bodies
            .values()
            .flat_map(|body| &body.blocks)
            .filter_map(|block| match &block.terminator.kind {
                MirTerminatorKind::ParFor { region, .. } => Some(*region),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let contract_regions = manifest
            .parfor_regions
            .iter()
            .map(|contract| contract.id)
            .collect::<BTreeSet<_>>();
        if let Some(region) = mir_regions.difference(&contract_regions).next() {
            return Err(format!(
                "parallel region {region:?} did not pass semantic classification and cannot be compiled"
            ));
        }
        if let Some(region) = contract_regions.difference(&mir_regions).next() {
            return Err(format!(
                "parallel contract {region:?} has no owning MIR terminator"
            ));
        }
        let layout = self
            .layout
            .as_ref()
            .ok_or_else(|| "parallel bytecode requires the canonical VM layout".to_string())?;
        let mut mapped = Vec::with_capacity(manifest.parfor_regions.len());
        for contract in &manifest.parfor_regions {
            let function = function_id(contract.id)?;
            let resume_points = self
                .compiled_resume_points(layout, function)
                .ok_or_else(|| {
                    format!(
                        "parallel region {:?} has no compiled resume-point map",
                        contract.id
                    )
                })?;
            mapped.push(map_parfor_region(mir, layout, resume_points, contract)?);
        }
        mapped.sort_by_key(|region| region.contract.id);

        self.parfor_regions.clear();
        for function in self.function_registry.functions.values_mut() {
            function.parfor_regions.clear();
        }
        for function in self.bound_functions.values_mut() {
            function.parfor_regions.clear();
        }

        for region in mapped {
            let function = function_id(region.contract.id)?;
            let is_entrypoint = layout
                .entrypoints
                .values()
                .any(|entrypoint| entrypoint.target == function);
            if is_entrypoint {
                self.parfor_regions.push(region.clone());
            }
            if let Some(bytecode) = self.function_registry.functions.get_mut(&function) {
                bytecode.parfor_regions.push(region.clone());
            }
            if let Some(bytecode) = self.bound_functions.get_mut(&function) {
                bytecode.parfor_regions.push(region.clone());
            }
            if !is_entrypoint
                && !self.function_registry.functions.contains_key(&function)
                && !self.bound_functions.contains_key(&function)
            {
                return Err(format!(
                    "parallel region {:?} has no compiled owning function",
                    region.contract.id
                ));
            }
        }
        self.parfor_regions.sort_by_key(|region| region.contract.id);
        for function in self.function_registry.functions.values_mut() {
            function
                .parfor_regions
                .sort_by_key(|region| region.contract.id);
        }
        for function in self.bound_functions.values_mut() {
            function
                .parfor_regions
                .sort_by_key(|region| region.contract.id);
        }
        Ok(())
    }

    fn compiled_resume_points<'a>(
        &'a self,
        layout: &'a crate::VmAssemblyLayout,
        function: runmat_hir::FunctionId,
    ) -> Option<&'a BTreeMap<ProgramPointId, usize>> {
        let is_entrypoint = layout
            .entrypoints
            .values()
            .any(|entrypoint| entrypoint.target == function);
        if is_entrypoint {
            return layout
                .functions
                .get(&function)
                .map(|function| &function.resume_points);
        }
        self.function_registry
            .functions
            .get(&function)
            .or_else(|| self.bound_functions.get(&function))
            .map(|function| &function.resume_points)
    }
}

fn map_parfor_region(
    mir: &MirAssembly,
    layout: &crate::VmAssemblyLayout,
    resume_points: &BTreeMap<ProgramPointId, usize>,
    contract: &ParforContract,
) -> Result<BytecodeParforRegion, String> {
    let function = function_id(contract.id)?;
    let body = mir.bodies.get(&function).ok_or_else(|| {
        format!(
            "parallel region {:?} has no MIR owning function",
            contract.id
        )
    })?;
    let (header, binding, body_block, exit_block) = body
        .blocks
        .iter()
        .find_map(|block| match &block.terminator.kind {
            MirTerminatorKind::ParFor {
                region,
                binding,
                body_block,
                exit_block,
                ..
            } if *region == contract.id => Some((block, *binding, *body_block, *exit_block)),
            _ => None,
        })
        .ok_or_else(|| format!("parallel region {:?} has no MIR terminator", contract.id))?;
    let function_layout = layout.functions.get(&function).ok_or_else(|| {
        format!(
            "parallel region {:?} has no VM function layout",
            contract.id
        )
    })?;
    let boundary =
        |block: BasicBlockId, position: usize| -> Result<BytecodeRegionBoundary, String> {
            let point = program_point(contract.id, block, position)?;
            let pc = resume_points.get(&point).copied().ok_or_else(|| {
                format!(
                    "parallel region {:?} boundary {:?} has no bytecode PC",
                    contract.id, point
                )
            })?;
            Ok(BytecodeRegionBoundary { point, pc })
        };
    let header = boundary(header.id, header.statements.len())?;
    let body_boundary = boundary(body_block, 0)?;
    let exit = boundary(exit_block, 0)?;
    if header.pc == body_boundary.pc || header.pc == exit.pc || body_boundary.pc == exit.pc {
        return Err(format!(
            "parallel region {:?} has overlapping control boundaries",
            contract.id
        ));
    }
    let loop_slot = slot_for(function_layout, binding, contract.id)?;
    let variables = contract
        .variables
        .iter()
        .map(|variable| {
            let local = MirLocalId(
                usize::try_from(variable.value.local)
                    .map_err(|_| "parallel local identity exceeds this target".to_string())?,
            );
            Ok(BytecodeParallelVariable {
                contract: variable.clone(),
                slot: slot_for(function_layout, local, contract.id)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if variables
        .iter()
        .find(|variable| variable.contract.value == contract.loop_variable)
        .map(|variable| variable.slot)
        != Some(loop_slot)
    {
        return Err(format!(
            "parallel region {:?} loop variable does not map to its loop slot",
            contract.id
        ));
    }
    Ok(BytecodeParforRegion {
        contract: contract.clone(),
        header,
        body: body_boundary,
        exit,
        loop_slot,
        variables,
    })
}

fn slot_for(
    layout: &crate::VmFunctionLayout,
    local: MirLocalId,
    region: ParallelRegionId,
) -> Result<usize, String> {
    layout
        .mir_local_slots
        .get(&local)
        .map(|slot| slot.0)
        .ok_or_else(|| format!("parallel region {region:?} local {local:?} has no VM slot"))
}

fn function_id(region: ParallelRegionId) -> Result<runmat_hir::FunctionId, String> {
    usize::try_from(region.0.function.0)
        .map(runmat_hir::FunctionId)
        .map_err(|_| "parallel function identity exceeds this target".to_string())
}

fn program_point(
    region: ParallelRegionId,
    block: BasicBlockId,
    position: usize,
) -> Result<ProgramPointId, String> {
    Ok(ProgramPointId {
        function: ProgramFunctionId(region.0.function.0),
        block: u32::try_from(block.0)
            .map_err(|_| "parallel block identity exceeds its portable schema".to_string())?,
        position: u32::try_from(position)
            .map_err(|_| "parallel position exceeds its portable schema".to_string())?,
    })
}

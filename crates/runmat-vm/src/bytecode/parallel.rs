use runmat_mir::{BasicBlockId, MirAssembly, MirLocalId, MirTerminatorKind};
use runmat_types::{
    ParallelManifest, ParallelRegionId, ParallelVariableContract, ParallelVariableRole,
    ParforContract, ProgramFunctionId, ProgramPointId, SpmdContract,
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

/// Required VM executable for one analyzed SPMD region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BytecodeSpmdRegion {
    pub contract: SpmdContract,
    pub header: BytecodeRegionBoundary,
    pub body: BytecodeRegionBoundary,
    pub exit: BytecodeRegionBoundary,
    pub captures: Vec<BytecodeParallelVariable>,
    pub outputs: Vec<BytecodeParallelVariable>,
}

impl BytecodeSpmdRegion {
    pub(crate) fn rebind_owner(&mut self, owner: ProgramFunctionId) {
        self.contract.id.0.function = owner;
        rebind_variables(&mut self.contract.captures, owner);
        rebind_variables(&mut self.contract.outputs, owner);
        self.header.point.function = owner;
        self.body.point.function = owner;
        self.exit.point.function = owner;
    }
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
        rebind_variables(&mut self.contract.variables, owner);
        self.header.point.function = owner;
        self.body.point.function = owner;
        self.exit.point.function = owner;
    }
}

fn rebind_variables(variables: &mut [ParallelVariableContract], owner: ProgramFunctionId) {
    for variable in variables {
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
        let mir_parfor_regions = mir
            .bodies
            .values()
            .flat_map(|body| &body.blocks)
            .filter_map(|block| match &block.terminator.kind {
                MirTerminatorKind::ParFor { region, .. } => Some(*region),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let contract_parfor_regions = manifest
            .parfor_regions
            .iter()
            .map(|contract| contract.id)
            .collect::<BTreeSet<_>>();
        if let Some(region) = mir_parfor_regions
            .difference(&contract_parfor_regions)
            .next()
        {
            return Err(format!(
                "parallel region {region:?} did not pass semantic classification and cannot be compiled"
            ));
        }
        if let Some(region) = contract_parfor_regions
            .difference(&mir_parfor_regions)
            .next()
        {
            return Err(format!(
                "parallel contract {region:?} has no owning MIR terminator"
            ));
        }
        let layout = self
            .layout
            .as_ref()
            .ok_or_else(|| "parallel bytecode requires the canonical VM layout".to_string())?;
        let mir_spmd_regions = mir
            .bodies
            .values()
            .flat_map(|body| &body.blocks)
            .filter_map(|block| match &block.terminator.kind {
                MirTerminatorKind::Spmd { region, .. } => Some(*region),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        let contract_spmd_regions = manifest
            .spmd_regions
            .iter()
            .map(|contract| contract.id)
            .collect::<BTreeSet<_>>();
        if let Some(region) = mir_spmd_regions.difference(&contract_spmd_regions).next() {
            return Err(format!(
                "SPMD region {region:?} did not pass semantic classification and cannot be compiled"
            ));
        }
        if let Some(region) = contract_spmd_regions.difference(&mir_spmd_regions).next() {
            return Err(format!(
                "SPMD contract {region:?} has no owning MIR terminator"
            ));
        }

        let mut mapped_parfor = Vec::with_capacity(manifest.parfor_regions.len());
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
            mapped_parfor.push(map_parfor_region(mir, layout, resume_points, contract)?);
        }
        mapped_parfor.sort_by_key(|region| region.contract.id);
        let mut mapped_spmd = Vec::with_capacity(manifest.spmd_regions.len());
        for contract in &manifest.spmd_regions {
            let function = function_id(contract.id)?;
            let resume_points = self
                .compiled_resume_points(layout, function)
                .ok_or_else(|| {
                    format!(
                        "SPMD region {:?} has no compiled resume-point map",
                        contract.id
                    )
                })?;
            mapped_spmd.push(map_spmd_region(mir, layout, resume_points, contract)?);
        }
        mapped_spmd.sort_by_key(|region| region.contract.id);

        self.parfor_regions.clear();
        self.spmd_regions.clear();
        self.distributed_values.clear();
        self.collective_contracts.clear();
        for function in self.function_registry.functions.values_mut() {
            function.parfor_regions.clear();
            function.spmd_regions.clear();
            function.distributed_values.clear();
            function.collective_contracts.clear();
        }
        for function in self.bound_functions.values_mut() {
            function.parfor_regions.clear();
            function.spmd_regions.clear();
            function.distributed_values.clear();
            function.collective_contracts.clear();
        }

        for region in mapped_parfor {
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
        for region in mapped_spmd {
            let function = function_id(region.contract.id)?;
            let is_entrypoint = layout
                .entrypoints
                .values()
                .any(|entrypoint| entrypoint.target == function);
            if is_entrypoint {
                self.spmd_regions.push(region.clone());
            }
            if let Some(bytecode) = self.function_registry.functions.get_mut(&function) {
                bytecode.spmd_regions.push(region.clone());
            }
            if let Some(bytecode) = self.bound_functions.get_mut(&function) {
                bytecode.spmd_regions.push(region.clone());
            }
            if !is_entrypoint
                && !self.function_registry.functions.contains_key(&function)
                && !self.bound_functions.contains_key(&function)
            {
                return Err(format!(
                    "SPMD region {:?} has no compiled owning function",
                    region.contract.id
                ));
            }
        }
        for contract in &manifest.distributed_values {
            let function = program_function_id(contract.id.function)?;
            let is_entrypoint = layout
                .entrypoints
                .values()
                .any(|entrypoint| entrypoint.target == function);
            if is_entrypoint {
                self.distributed_values.push(contract.clone());
            }
            if let Some(bytecode) = self.function_registry.functions.get_mut(&function) {
                bytecode.distributed_values.push(contract.clone());
            }
            if let Some(bytecode) = self.bound_functions.get_mut(&function) {
                bytecode.distributed_values.push(contract.clone());
            }
            if !is_entrypoint
                && !self.function_registry.functions.contains_key(&function)
                && !self.bound_functions.contains_key(&function)
            {
                return Err(format!(
                    "distributed value {:?} has no compiled owning function",
                    contract.id
                ));
            }
        }
        for contract in &manifest.collectives {
            let function = function_id(contract.id.region)?;
            let is_entrypoint = layout
                .entrypoints
                .values()
                .any(|entrypoint| entrypoint.target == function);
            if is_entrypoint {
                self.collective_contracts.push(contract.clone());
            }
            if let Some(bytecode) = self.function_registry.functions.get_mut(&function) {
                bytecode.collective_contracts.push(contract.clone());
            }
            if let Some(bytecode) = self.bound_functions.get_mut(&function) {
                bytecode.collective_contracts.push(contract.clone());
            }
            if !is_entrypoint
                && !self.function_registry.functions.contains_key(&function)
                && !self.bound_functions.contains_key(&function)
            {
                return Err(format!(
                    "collective {:?} has no compiled owning function",
                    contract.id
                ));
            }
        }
        self.parfor_regions.sort_by_key(|region| region.contract.id);
        self.spmd_regions.sort_by_key(|region| region.contract.id);
        self.distributed_values.sort_by_key(|contract| contract.id);
        self.collective_contracts
            .sort_by_key(|contract| contract.id);
        for function in self.function_registry.functions.values_mut() {
            function
                .parfor_regions
                .sort_by_key(|region| region.contract.id);
            function
                .spmd_regions
                .sort_by_key(|region| region.contract.id);
            function
                .distributed_values
                .sort_by_key(|contract| contract.id);
            function
                .collective_contracts
                .sort_by_key(|contract| contract.id);
        }
        for function in self.bound_functions.values_mut() {
            function
                .parfor_regions
                .sort_by_key(|region| region.contract.id);
            function
                .spmd_regions
                .sort_by_key(|region| region.contract.id);
            function
                .distributed_values
                .sort_by_key(|contract| contract.id);
            function
                .collective_contracts
                .sort_by_key(|contract| contract.id);
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

fn map_spmd_region(
    mir: &MirAssembly,
    layout: &crate::VmAssemblyLayout,
    resume_points: &BTreeMap<ProgramPointId, usize>,
    contract: &SpmdContract,
) -> Result<BytecodeSpmdRegion, String> {
    let function = function_id(contract.id)?;
    let body = mir
        .bodies
        .get(&function)
        .ok_or_else(|| format!("SPMD region {:?} has no MIR owning function", contract.id))?;
    let (header, body_block, exit_block) = body
        .blocks
        .iter()
        .find_map(|block| match &block.terminator.kind {
            MirTerminatorKind::Spmd {
                region,
                body_block,
                exit_block,
                ..
            } if *region == contract.id => Some((block, *body_block, *exit_block)),
            _ => None,
        })
        .ok_or_else(|| format!("SPMD region {:?} has no MIR terminator", contract.id))?;
    let function_layout = layout
        .functions
        .get(&function)
        .ok_or_else(|| format!("SPMD region {:?} has no VM function layout", contract.id))?;
    let boundary =
        |block: BasicBlockId, position: usize| -> Result<BytecodeRegionBoundary, String> {
            let point = program_point(contract.id, block, position)?;
            let pc = resume_points.get(&point).copied().ok_or_else(|| {
                format!(
                    "SPMD region {:?} boundary {:?} has no bytecode PC",
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
            "SPMD region {:?} has overlapping control boundaries",
            contract.id
        ));
    }
    let map_variables = |variables: &[ParallelVariableContract]| {
        variables
            .iter()
            .map(|variable| {
                let local = MirLocalId(
                    usize::try_from(variable.value.local)
                        .map_err(|_| "SPMD local identity exceeds this target".to_string())?,
                );
                Ok(BytecodeParallelVariable {
                    contract: variable.clone(),
                    slot: slot_for(function_layout, local, contract.id)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()
    };
    Ok(BytecodeSpmdRegion {
        contract: contract.clone(),
        header,
        body: body_boundary,
        exit,
        captures: map_variables(&contract.captures)?,
        outputs: map_variables(&contract.outputs)?,
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
    program_function_id(region.0.function)
}

fn program_function_id(function: ProgramFunctionId) -> Result<runmat_hir::FunctionId, String> {
    usize::try_from(function.0)
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

use runmat_types::{ProgramPointId, RegionContract, RegionId};
use serde::{Deserialize, Serialize};

use super::program::Bytecode;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BytecodeRegionBoundary {
    pub point: ProgramPointId,
    pub pc: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BytecodeRegion {
    pub id: RegionId,
    pub entry: BytecodeRegionBoundary,
    pub exits: Vec<BytecodeRegionBoundary>,
}

impl Bytecode {
    /// Transactionally install the complete analyzed region inventory after
    /// resolving every MIR program point to an exact empty-stack VM boundary.
    pub fn install_regions(&mut self, contracts: &[RegionContract]) -> Result<(), String> {
        let mut contracts = contracts.iter().collect::<Vec<_>>();
        contracts.sort_by_key(|contract| contract.id);
        if contracts.windows(2).any(|pair| pair[0].id == pair[1].id) {
            return Err("region inventory contains duplicate identities".to_string());
        }
        let mut mapped = Vec::with_capacity(contracts.len());
        for contract in contracts {
            contract
                .validate()
                .map_err(|error| format!("{}: {}", error.path, error.message))?;
            let function = runmat_hir::FunctionId(
                usize::try_from(contract.id.function.0)
                    .map_err(|_| "region function identity exceeds this target".to_string())?,
            );
            let region = if let Some(function) = self.function_registry.functions.get(&function) {
                map_bytecode_region(contract, &function.resume_points)?
            } else {
                let points = self
                    .layout
                    .as_ref()
                    .and_then(|layout| layout.functions.get(&function))
                    .map(|layout| &layout.resume_points)
                    .ok_or_else(|| {
                        format!(
                            "region {:?} references function without VM layout",
                            contract.id
                        )
                    })?;
                map_bytecode_region(contract, points)?
            };
            mapped.push((function, region));
        }

        self.regions = mapped.iter().map(|(_, region)| region.clone()).collect();
        for function in self.function_registry.functions.values_mut() {
            function.regions.clear();
        }
        for function in self.bound_functions.values_mut() {
            function.regions.clear();
        }
        for (function, region) in mapped {
            if let Some(bytecode) = self.function_registry.functions.get_mut(&function) {
                bytecode.regions.push(region.clone());
            }
            if let Some(bytecode) = self.bound_functions.get_mut(&function) {
                bytecode.regions.push(region);
            }
        }
        Ok(())
    }
}

fn map_bytecode_region(
    contract: &RegionContract,
    points: &std::collections::BTreeMap<ProgramPointId, usize>,
) -> Result<BytecodeRegion, String> {
    let entry = BytecodeRegionBoundary {
        point: contract.entry,
        pc: points.get(&contract.entry).copied().ok_or_else(|| {
            format!(
                "region {:?} entry {:?} has no bytecode boundary",
                contract.id, contract.entry
            )
        })?,
    };
    let exits = contract
        .exits
        .iter()
        .map(|point| {
            Ok(BytecodeRegionBoundary {
                point: *point,
                pc: points.get(point).copied().ok_or_else(|| {
                    format!(
                        "region {:?} exit {:?} has no bytecode boundary",
                        contract.id, point
                    )
                })?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(BytecodeRegion {
        id: contract.id,
        entry,
        exits,
    })
}

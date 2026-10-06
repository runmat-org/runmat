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
    /// Transactionally install every analyzed region that resolves to exact
    /// empty-stack VM boundaries in this bytecode product. Regions are
    /// optional optimization candidates; a composed or transformed product
    /// that cannot represent one must execute normally without it.
    pub fn install_regions(&mut self, contracts: &[RegionContract]) -> Result<(), String> {
        self.install_regions_scoped(contracts, None)
    }

    /// Install this assembly's regions without rewriting retained foreign function metadata.
    /// The root bytecode still receives synthetic-entrypoint regions; registry
    /// functions receive regions only when the compiler declares their ownership.
    pub fn install_regions_for_functions(
        &mut self,
        contracts: &[RegionContract],
        owned_functions: &std::collections::BTreeSet<runmat_hir::FunctionId>,
    ) -> Result<(), String> {
        self.install_regions_scoped(contracts, Some(owned_functions))
    }

    fn install_regions_scoped(
        &mut self,
        contracts: &[RegionContract],
        owned_functions: Option<&std::collections::BTreeSet<runmat_hir::FunctionId>>,
    ) -> Result<(), String> {
        let owns = |function: runmat_hir::FunctionId| {
            owned_functions.is_none_or(|ids| ids.contains(&function))
        };
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
            // Region contracts describe this executable's MIR assembly. Prefer
            // its immutable layout over the composed session registry: a
            // persisted function may legitimately reuse a unit-local function
            // identity while carrying resume points for a different body.
            let layout_points = self
                .layout
                .as_ref()
                .and_then(|layout| layout.functions.get(&function))
                .map(|layout| &layout.resume_points);
            let registry_points = self
                .function_registry
                .functions
                .get(&function)
                .filter(|_| owns(function))
                .map(|function| &function.resume_points);
            let Some(points) = layout_points
                .filter(|points| contains_region_boundaries(contract, points))
                .or_else(|| {
                    registry_points.filter(|points| contains_region_boundaries(contract, points))
                })
            else {
                continue;
            };
            let region = map_bytecode_region(contract, points)?;
            mapped.push((function, region));
        }

        self.regions = mapped.iter().map(|(_, region)| region.clone()).collect();
        for function in self.function_registry.functions.values_mut() {
            if owns(function.function) {
                function.regions.clear();
            }
        }
        for function in self.bound_functions.values_mut() {
            if owns(function.function) {
                function.regions.clear();
            }
        }
        for (function, region) in mapped {
            if !owns(function) {
                continue;
            }
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

fn contains_region_boundaries(
    contract: &RegionContract,
    points: &std::collections::BTreeMap<ProgramPointId, usize>,
) -> bool {
    points.contains_key(&contract.entry)
        && contract.exits.iter().all(|exit| points.contains_key(exit))
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

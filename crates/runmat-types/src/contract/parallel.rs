use crate::{
    CapabilitySet, CollectiveId, DistributedValueId, LabCount, LabRank, OperatorKind,
    ParallelRegionId, RegionValueId, SchemaValidationError, ValueFact,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PARALLEL_MANIFEST_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelVariableRole {
    Loop,
    Broadcast,
    Sliced { dimensions: Vec<u32> },
    Reduction { operator: OperatorKind },
    Temporary,
    Private,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelVariableContract {
    pub value: RegionValueId,
    pub role: ParallelVariableRole,
    pub access: ParallelAccess,
    pub fact: ValueFact,
    pub transferable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelRandomnessPolicy {
    Inherit,
    DeterministicSubstreams,
    Nondeterministic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParforContract {
    pub id: ParallelRegionId,
    pub loop_variable: RegionValueId,
    pub iterable: ValueFact,
    pub variables: Vec<ParallelVariableContract>,
    pub maximum_workers: Option<LabCount>,
    pub capabilities: CapabilitySet,
    pub randomness: ParallelRandomnessPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpmdContract {
    pub id: ParallelRegionId,
    pub minimum_labs: LabCount,
    pub maximum_labs: Option<LabCount>,
    pub captures: Vec<ParallelVariableContract>,
    pub capabilities: CapabilitySet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum DistributionScheme {
    Replicated,
    Block { dimension: u32 },
    Cyclic { dimension: u32 },
    Custom { partitioner: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistributedValueContract {
    pub id: DistributedValueId,
    pub value: ValueFact,
    pub scheme: DistributionScheme,
    pub owner_region: ParallelRegionId,
    pub materializable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CollectiveContract {
    pub id: CollectiveId,
    pub operation: CollectiveOperation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum CollectiveOperation {
    Barrier,
    Broadcast {
        input: DistributedValueId,
        output: DistributedValueId,
        root: LabRank,
    },
    Gather {
        input: DistributedValueId,
        output: DistributedValueId,
        root: LabRank,
    },
    Scatter {
        input: DistributedValueId,
        output: DistributedValueId,
        root: LabRank,
    },
    AllGather {
        input: DistributedValueId,
        output: DistributedValueId,
    },
    Reduce {
        input: DistributedValueId,
        output: DistributedValueId,
        root: LabRank,
        operator: OperatorKind,
    },
    AllReduce {
        input: DistributedValueId,
        output: DistributedValueId,
        operator: OperatorKind,
    },
    Send {
        input: DistributedValueId,
        peer: LabRank,
    },
    Receive {
        output: DistributedValueId,
        peer: LabRank,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParallelManifest {
    pub schema_version: u16,
    pub parfor_regions: Vec<ParforContract>,
    pub spmd_regions: Vec<SpmdContract>,
    pub distributed_values: Vec<DistributedValueContract>,
    pub collectives: Vec<CollectiveContract>,
}

impl ParallelManifest {
    pub fn empty() -> Self {
        Self {
            schema_version: PARALLEL_MANIFEST_SCHEMA_VERSION,
            parfor_regions: Vec::new(),
            spmd_regions: Vec::new(),
            distributed_values: Vec::new(),
            collectives: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), SchemaValidationError> {
        if self.schema_version != PARALLEL_MANIFEST_SCHEMA_VERSION {
            return Err(SchemaValidationError::new(
                "parallel.schema_version",
                format!(
                    "unsupported version {}; expected {}",
                    self.schema_version, PARALLEL_MANIFEST_SCHEMA_VERSION
                ),
            ));
        }
        ensure_sorted("parallel.parfor_regions", &self.parfor_regions, |value| {
            value.id
        })?;
        ensure_sorted("parallel.spmd_regions", &self.spmd_regions, |value| {
            value.id
        })?;
        ensure_sorted(
            "parallel.distributed_values",
            &self.distributed_values,
            |value| value.id,
        )?;
        ensure_sorted("parallel.collectives", &self.collectives, |value| value.id)?;
        let parfor_ids = self
            .parfor_regions
            .iter()
            .map(|region| region.id)
            .collect::<BTreeSet<_>>();
        let spmd_ids = self
            .spmd_regions
            .iter()
            .map(|region| region.id)
            .collect::<BTreeSet<_>>();
        if !parfor_ids.is_disjoint(&spmd_ids) {
            return Err(SchemaValidationError::new(
                "parallel.regions",
                "one parallel region identity cannot represent both parfor and spmd",
            ));
        }
        let region_ids = parfor_ids
            .union(&spmd_ids)
            .copied()
            .collect::<BTreeSet<_>>();
        for region in &self.parfor_regions {
            if region.loop_variable.function != region.id.0.function {
                return Err(SchemaValidationError::new(
                    "parallel.parfor_regions.loop_variable",
                    "loop variable must belong to the parallel region function",
                ));
            }
            validate_variables(
                "parallel.parfor_regions.variables",
                region.id,
                &region.variables,
            )?;
            let loop_variables = region
                .variables
                .iter()
                .filter(|variable| matches!(&variable.role, ParallelVariableRole::Loop))
                .collect::<Vec<_>>();
            if loop_variables.len() != 1 || loop_variables[0].value != region.loop_variable {
                return Err(SchemaValidationError::new(
                    "parallel.parfor_regions.loop_variable",
                    "variables must contain exactly one matching loop classification",
                ));
            }
            if region.maximum_workers.is_some_and(|count| count.0 == 0) {
                return Err(SchemaValidationError::new(
                    "parallel.parfor_regions.maximum_workers",
                    "maximum worker count must be non-zero",
                ));
            }
        }
        for region in &self.spmd_regions {
            if region.minimum_labs.0 == 0 {
                return Err(SchemaValidationError::new(
                    "parallel.spmd_regions",
                    "minimum lab count must be non-zero",
                ));
            }
            if region
                .maximum_labs
                .is_some_and(|maximum| maximum.0 < region.minimum_labs.0)
            {
                return Err(SchemaValidationError::new(
                    "parallel.spmd_regions.maximum_labs",
                    "maximum lab count must not be below minimum",
                ));
            }
            validate_variables(
                "parallel.spmd_regions.captures",
                region.id,
                &region.captures,
            )?;
            if region
                .captures
                .iter()
                .any(|variable| matches!(&variable.role, ParallelVariableRole::Loop))
            {
                return Err(SchemaValidationError::new(
                    "parallel.spmd_regions.captures",
                    "spmd captures cannot use the parfor loop classification",
                ));
            }
        }
        let distributed_ids = self
            .distributed_values
            .iter()
            .map(|value| value.id)
            .collect::<BTreeSet<_>>();
        for distributed in &self.distributed_values {
            if !region_ids.contains(&distributed.owner_region)
                || distributed.id.function != distributed.owner_region.0.function
            {
                return Err(SchemaValidationError::new(
                    "parallel.distributed_values.owner_region",
                    "owner must name a declared parallel region in the same function",
                ));
            }
            if let DistributionScheme::Custom { partitioner } = &distributed.scheme {
                super::schema::validate_token(
                    "parallel.distributed_values.partitioner",
                    partitioner,
                    256,
                )?;
            }
        }
        for collective in &self.collectives {
            if !region_ids.contains(&collective.id.region) {
                return Err(SchemaValidationError::new(
                    "parallel.collectives.region",
                    "collective must belong to a declared parallel region",
                ));
            }
            if collective
                .operation
                .values()
                .iter()
                .any(|value| !distributed_ids.contains(value))
            {
                return Err(SchemaValidationError::new(
                    "parallel.collectives.values",
                    "collective inputs and outputs must name declared distributed values",
                ));
            }
        }
        Ok(())
    }
}

impl Default for ParallelManifest {
    fn default() -> Self {
        Self::empty()
    }
}

fn validate_variables(
    path: &str,
    region: ParallelRegionId,
    variables: &[ParallelVariableContract],
) -> Result<(), SchemaValidationError> {
    ensure_sorted(path, variables, |value| value.value)?;
    if variables
        .iter()
        .any(|value| value.value.function != region.0.function)
    {
        return Err(SchemaValidationError::new(
            path,
            "classified variables must belong to the parallel region function",
        ));
    }
    for variable in variables {
        if let ParallelVariableRole::Sliced { dimensions } = &variable.role {
            if dimensions.is_empty() || dimensions.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(SchemaValidationError::new(
                    path,
                    "sliced dimensions must be sorted, unique, and non-empty",
                ));
            }
        }
    }
    Ok(())
}

impl CollectiveOperation {
    fn values(&self) -> Vec<DistributedValueId> {
        match self {
            Self::Barrier => Vec::new(),
            Self::Broadcast { input, output, .. }
            | Self::Gather { input, output, .. }
            | Self::Scatter { input, output, .. }
            | Self::AllGather { input, output }
            | Self::Reduce { input, output, .. }
            | Self::AllReduce { input, output, .. } => vec![*input, *output],
            Self::Send { input, .. } => vec![*input],
            Self::Receive { output, .. } => vec![*output],
        }
    }
}

fn ensure_sorted<T, K: Ord + Copy>(
    path: &str,
    values: &[T],
    key: impl Fn(&T) -> K,
) -> Result<(), SchemaValidationError> {
    if values.windows(2).any(|pair| key(&pair[0]) >= key(&pair[1])) {
        return Err(SchemaValidationError::new(
            path,
            "entries must be sorted and unique by identity",
        ));
    }
    Ok(())
}

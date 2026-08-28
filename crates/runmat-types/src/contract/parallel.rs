use crate::{
    CapabilitySet, CollectiveId, DistributedValueId, LabCount, OperatorKind, ParallelRegionId,
    RegionValueId, SchemaValidationError, ValueFact,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

mod spmd;
pub use spmd::SpmdLabRequirement;

pub const PARALLEL_MANIFEST_SCHEMA_VERSION: u16 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum ParallelIndexConstant {
    Signed(i64),
    Unsigned(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum ParallelSliceOffsetOperand {
    Constant(ParallelIndexConstant),
    Broadcast(RegionValueId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "operand")]
pub enum ParallelSliceOffset {
    None,
    Add(ParallelSliceOffsetOperand),
    Subtract(ParallelSliceOffsetOperand),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "dimension")]
pub enum ParallelSliceAxis {
    /// One-subscript MATLAB linear indexing, such as `value(index)`.
    Linear,
    /// Multisubscript indexing with the loop variable in this one-based
    /// dimension, such as `value(:, index)`.
    Dimension(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ParallelSliceAccess {
    pub axis: ParallelSliceAxis,
    pub offset: ParallelSliceOffset,
}

impl ParallelSliceAccess {
    pub const fn linear() -> Self {
        Self {
            axis: ParallelSliceAxis::Linear,
            offset: ParallelSliceOffset::None,
        }
    }

    pub const fn dimension(dimension: u32) -> Self {
        Self {
            axis: ParallelSliceAxis::Dimension(dimension),
            offset: ParallelSliceOffset::None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum ParallelVariableRole {
    Loop,
    Broadcast,
    Sliced { access: ParallelSliceAccess },
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
#[serde(deny_unknown_fields)]
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
#[serde(deny_unknown_fields)]
pub struct ParforContract {
    pub id: ParallelRegionId,
    pub loop_variable: RegionValueId,
    pub iterable: ValueFact,
    pub variables: Vec<ParallelVariableContract>,
    pub maximum_workers: Option<LabCount>,
    pub effects: crate::EffectSet,
    pub capabilities: CapabilitySet,
    pub randomness: ParallelRandomnessPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpmdContract {
    pub id: ParallelRegionId,
    pub labs: SpmdLabRequirement,
    pub captures: Vec<ParallelVariableContract>,
    pub outputs: Vec<ParallelVariableContract>,
    pub effects: crate::EffectSet,
    pub capabilities: CapabilitySet,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    rename_all = "snake_case",
    tag = "kind",
    content = "value",
    deny_unknown_fields
)]
pub enum DistributionScheme {
    Replicated,
    Block { dimension: u32 },
    Cyclic { dimension: u32 },
    Custom { partitioner: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributedValueContract {
    pub id: DistributedValueId,
    pub value: ValueFact,
    pub scheme: DistributionScheme,
    pub owner: crate::DistributedOwner,
    pub materializable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectiveContract {
    pub id: CollectiveId,
    pub operation: CollectiveOperation,
    pub input: Option<ValueFact>,
    pub output: Option<ValueFact>,
    pub root: Option<ValueFact>,
    pub source: Option<ValueFact>,
    pub destination: Option<ValueFact>,
    pub tag: Option<ValueFact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveOperation {
    Barrier,
    Broadcast,
    Gather,
    Scatter,
    AllGather,
    Reduce { operator: OperatorKind },
    AllReduce { operator: OperatorKind },
    Send,
    Receive { requested_outputs: u8 },
    SendReceive,
    Probe,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
            region.labs.validate()?;
            validate_variables(
                "parallel.spmd_regions.captures",
                region.id,
                &region.captures,
            )?;
            validate_variables("parallel.spmd_regions.outputs", region.id, &region.outputs)?;
            if region
                .captures
                .iter()
                .chain(&region.outputs)
                .any(|variable| matches!(&variable.role, ParallelVariableRole::Loop))
            {
                return Err(SchemaValidationError::new(
                    "parallel.spmd_regions.captures",
                    "spmd captures cannot use the parfor loop classification",
                ));
            }
        }
        for distributed in &self.distributed_values {
            let owner_valid = match distributed.owner {
                crate::DistributedOwner::Client(function) => function == distributed.id.function,
                crate::DistributedOwner::Region(region) => {
                    region_ids.contains(&region) && region.0.function == distributed.id.function
                }
            };
            if !owner_valid {
                return Err(SchemaValidationError::new(
                    "parallel.distributed_values.owner",
                    "owner must name the same client function or a declared parallel region in that function",
                ));
            }
            if let DistributionScheme::Custom { partitioner } = &distributed.scheme {
                super::schema::validate_token(
                    "parallel.distributed_values.partitioner",
                    partitioner,
                    256,
                )?;
            }
            if matches!(
                distributed.scheme,
                DistributionScheme::Block { dimension: 0 }
                    | DistributionScheme::Cyclic { dimension: 0 }
            ) {
                return Err(SchemaValidationError::new(
                    "parallel.distributed_values.scheme",
                    "distribution dimensions use one-based nonzero identities",
                ));
            }
        }
        for collective in &self.collectives {
            if !spmd_ids.contains(&collective.id.region) {
                return Err(SchemaValidationError::new(
                    "parallel.collectives.region",
                    "collective must belong to a declared SPMD region",
                ));
            }
            collective.validate()?;
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
        if let ParallelVariableRole::Sliced { access } = &variable.role {
            if matches!(access.axis, ParallelSliceAxis::Dimension(0)) {
                return Err(SchemaValidationError::new(
                    path,
                    "sliced dimensions use one-based nonzero identities",
                ));
            }
            let offset_value = match access.offset {
                ParallelSliceOffset::Add(ParallelSliceOffsetOperand::Broadcast(value))
                | ParallelSliceOffset::Subtract(ParallelSliceOffsetOperand::Broadcast(value)) => {
                    Some(value)
                }
                _ => None,
            };
            if let Some(offset_value) = offset_value {
                if offset_value.function != region.0.function
                    || !variables.iter().any(|value| {
                        value.value == offset_value
                            && matches!(value.role, ParallelVariableRole::Broadcast)
                    })
                {
                    return Err(SchemaValidationError::new(
                        path,
                        "sliced offsets must name a broadcast from the same region function",
                    ));
                }
            }
        }
    }
    Ok(())
}

impl CollectiveContract {
    fn validate(&self) -> Result<(), SchemaValidationError> {
        let actual = (
            self.input.is_some(),
            self.output.is_some(),
            self.root.is_some(),
            self.source.is_some(),
            self.destination.is_some(),
        );
        let expected = match self.operation {
            CollectiveOperation::Barrier => (false, false, false, false, false),
            CollectiveOperation::Broadcast => (self.input.is_some(), true, true, false, false),
            CollectiveOperation::Gather
            | CollectiveOperation::Scatter
            | CollectiveOperation::Reduce { .. } => (true, true, true, false, false),
            CollectiveOperation::AllGather | CollectiveOperation::AllReduce { .. } => {
                (true, true, false, false, false)
            }
            CollectiveOperation::Send => (true, false, false, false, true),
            CollectiveOperation::Receive { .. } | CollectiveOperation::Probe => {
                (false, true, false, self.source.is_some(), false)
            }
            CollectiveOperation::SendReceive => (true, true, false, true, true),
        };
        if actual != expected {
            return Err(SchemaValidationError::new(
                "parallel.collectives",
                "collective operand facts do not match the operation contract",
            ));
        }
        if self.tag.is_some()
            && !matches!(
                self.operation,
                CollectiveOperation::Send
                    | CollectiveOperation::Receive { .. }
                    | CollectiveOperation::SendReceive
                    | CollectiveOperation::Probe
            )
        {
            return Err(SchemaValidationError::new(
                "parallel.collectives.tag",
                "only point-to-point operations may declare a tag operand",
            ));
        }
        Ok(())
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

use crate::MirOperand;
use runmat_types::{
    DistributedBuildValidation, DistributedOwner, DistributedValueId, DistributionScheme,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirDistributedOp {
    Create {
        id: DistributedValueId,
        owner: DistributedOwner,
        input: MirOperand,
        scheme: DistributionScheme,
    },
    Codistributed {
        id: DistributedValueId,
        owner: DistributedOwner,
        input: MirOperand,
        overload: MirCodistributedOverload,
        coordination: Option<runmat_types::CollectiveId>,
    },
    Build {
        id: DistributedValueId,
        owner: DistributedOwner,
        local_part: MirOperand,
        codistributor: Option<MirOperand>,
        validation: MirDistributedBuildValidation,
        coordination: runmat_types::CollectiveId,
    },
    LocalPart {
        value: MirOperand,
    },
    Materialize {
        value: MirOperand,
    },
    Codistributor {
        value: MirOperand,
    },
    GlobalIndices {
        value: MirOperand,
        dimension: MirOperand,
        lab: Option<MirOperand>,
        requested_outputs: u8,
    },
    Redistribute {
        value: MirOperand,
        codistributor: MirOperand,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirCodistributedOverload {
    ReplicatedInputDefault,
    CodistributorOrDesignatedWorker {
        operand: MirOperand,
    },
    DesignatedWorkerWithCodistributor {
        worker: MirOperand,
        codistributor: MirOperand,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirDistributedBuildValidation {
    ValidateAcrossWorkers,
    NoCommunication,
    RuntimeOption(MirOperand),
}

impl MirDistributedBuildValidation {
    pub const fn contract(&self) -> DistributedBuildValidation {
        match self {
            Self::ValidateAcrossWorkers => DistributedBuildValidation::ValidateAcrossWorkers,
            Self::NoCommunication => DistributedBuildValidation::NoCommunication,
            Self::RuntimeOption(_) => DistributedBuildValidation::RuntimeOption,
        }
    }
}

impl MirDistributedOp {
    pub fn primary_input(&self) -> &MirOperand {
        match self {
            Self::Create { input, .. }
            | Self::Codistributed { input, .. }
            | Self::Build {
                local_part: input, ..
            }
            | Self::LocalPart { value: input }
            | Self::Materialize { value: input }
            | Self::Codistributor { value: input }
            | Self::GlobalIndices { value: input, .. }
            | Self::Redistribute { value: input, .. } => input,
        }
    }

    pub fn operands(&self) -> impl Iterator<Item = &MirOperand> {
        let (primary, secondary, tertiary) = match self {
            Self::Redistribute {
                value,
                codistributor,
            } => (value, Some(codistributor), None),
            Self::Codistributed {
                input, overload, ..
            } => match overload {
                MirCodistributedOverload::ReplicatedInputDefault => (input, None, None),
                MirCodistributedOverload::CodistributorOrDesignatedWorker { operand } => {
                    (input, Some(operand), None)
                }
                MirCodistributedOverload::DesignatedWorkerWithCodistributor {
                    worker,
                    codistributor,
                } => (input, Some(worker), Some(codistributor)),
            },
            Self::Build {
                local_part,
                codistributor,
                validation,
                ..
            } => (
                local_part,
                codistributor.as_ref(),
                match validation {
                    MirDistributedBuildValidation::RuntimeOption(operand) => Some(operand),
                    MirDistributedBuildValidation::ValidateAcrossWorkers
                    | MirDistributedBuildValidation::NoCommunication => None,
                },
            ),
            Self::GlobalIndices {
                value,
                dimension,
                lab,
                ..
            } => (value, Some(dimension), lab.as_ref()),
            _ => (self.primary_input(), None, None),
        };
        [Some(primary), secondary, tertiary].into_iter().flatten()
    }

    pub fn for_each_operand_mut(&mut self, mut operation: impl FnMut(&mut MirOperand)) {
        match self {
            Self::Create { input, .. }
            | Self::LocalPart { value: input }
            | Self::Materialize { value: input }
            | Self::Codistributor { value: input } => operation(input),
            Self::Codistributed {
                input, overload, ..
            } => {
                operation(input);
                match overload {
                    MirCodistributedOverload::ReplicatedInputDefault => {}
                    MirCodistributedOverload::CodistributorOrDesignatedWorker { operand } => {
                        operation(operand)
                    }
                    MirCodistributedOverload::DesignatedWorkerWithCodistributor {
                        worker,
                        codistributor,
                    } => {
                        operation(worker);
                        operation(codistributor);
                    }
                }
            }
            Self::Build {
                local_part,
                codistributor,
                validation,
                ..
            } => {
                operation(local_part);
                if let Some(codistributor) = codistributor {
                    operation(codistributor);
                }
                if let MirDistributedBuildValidation::RuntimeOption(option) = validation {
                    operation(option);
                }
            }
            Self::GlobalIndices {
                value,
                dimension,
                lab,
                ..
            } => {
                operation(value);
                operation(dimension);
                if let Some(lab) = lab {
                    operation(lab);
                }
            }
            Self::Redistribute {
                value,
                codistributor,
            } => {
                operation(value);
                operation(codistributor);
            }
        }
    }

    pub fn input_mut(&mut self) -> &mut MirOperand {
        match self {
            Self::Create { input, .. }
            | Self::Codistributed { input, .. }
            | Self::Build {
                local_part: input, ..
            }
            | Self::LocalPart { value: input }
            | Self::Materialize { value: input }
            | Self::Codistributor { value: input }
            | Self::GlobalIndices { value: input, .. }
            | Self::Redistribute { value: input, .. } => input,
        }
    }
}

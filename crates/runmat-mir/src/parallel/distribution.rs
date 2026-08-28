use crate::MirOperand;
use runmat_types::{DistributedOwner, DistributedValueId, DistributionScheme};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirDistributedOp {
    Create {
        id: DistributedValueId,
        owner: DistributedOwner,
        input: MirOperand,
        scheme: DistributionScheme,
    },
    LocalPart {
        value: MirOperand,
    },
    Materialize {
        value: MirOperand,
    },
    Redistribute {
        value: MirOperand,
        scheme: DistributionScheme,
    },
}

impl MirDistributedOp {
    pub fn input(&self) -> &MirOperand {
        match self {
            Self::Create { input, .. }
            | Self::LocalPart { value: input }
            | Self::Materialize { value: input }
            | Self::Redistribute { value: input, .. } => input,
        }
    }

    pub fn input_mut(&mut self) -> &mut MirOperand {
        match self {
            Self::Create { input, .. }
            | Self::LocalPart { value: input }
            | Self::Materialize { value: input }
            | Self::Redistribute { value: input, .. } => input,
        }
    }
}

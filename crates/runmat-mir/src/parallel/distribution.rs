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
    Codistributor {
        value: MirOperand,
    },
    Redistribute {
        value: MirOperand,
        codistributor: MirOperand,
    },
}

impl MirDistributedOp {
    pub fn primary_input(&self) -> &MirOperand {
        match self {
            Self::Create { input, .. }
            | Self::LocalPart { value: input }
            | Self::Materialize { value: input }
            | Self::Codistributor { value: input }
            | Self::Redistribute { value: input, .. } => input,
        }
    }

    pub fn operands(&self) -> impl Iterator<Item = &MirOperand> {
        let (primary, secondary) = match self {
            Self::Redistribute {
                value,
                codistributor,
            } => (value, Some(codistributor)),
            _ => (self.primary_input(), None),
        };
        [Some(primary), secondary].into_iter().flatten()
    }

    pub fn input_mut(&mut self) -> &mut MirOperand {
        match self {
            Self::Create { input, .. }
            | Self::LocalPart { value: input }
            | Self::Materialize { value: input }
            | Self::Codistributor { value: input }
            | Self::Redistribute { value: input, .. } => input,
        }
    }
}

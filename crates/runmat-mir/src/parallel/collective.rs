use crate::MirOperand;
use runmat_types::{CollectiveId, OperatorKind};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MirCollectiveOp {
    Barrier {
        id: CollectiveId,
    },
    Broadcast {
        id: CollectiveId,
        input: MirOperand,
        root: MirOperand,
    },
    Gather {
        id: CollectiveId,
        input: MirOperand,
        root: MirOperand,
    },
    Scatter {
        id: CollectiveId,
        input: MirOperand,
        root: MirOperand,
    },
    AllGather {
        id: CollectiveId,
        input: MirOperand,
    },
    Reduce {
        id: CollectiveId,
        input: MirOperand,
        root: MirOperand,
        operator: OperatorKind,
    },
    AllReduce {
        id: CollectiveId,
        input: MirOperand,
        operator: OperatorKind,
    },
    Send {
        id: CollectiveId,
        input: MirOperand,
        destination: MirOperand,
        tag: Option<MirOperand>,
    },
    Receive {
        id: CollectiveId,
        source: Option<MirOperand>,
        tag: Option<MirOperand>,
    },
    Probe {
        id: CollectiveId,
        source: Option<MirOperand>,
        tag: Option<MirOperand>,
    },
}

impl MirCollectiveOp {
    pub fn id_mut(&mut self) -> &mut CollectiveId {
        match self {
            Self::Barrier { id }
            | Self::Broadcast { id, .. }
            | Self::Gather { id, .. }
            | Self::Scatter { id, .. }
            | Self::AllGather { id, .. }
            | Self::Reduce { id, .. }
            | Self::AllReduce { id, .. }
            | Self::Send { id, .. }
            | Self::Receive { id, .. }
            | Self::Probe { id, .. } => id,
        }
    }

    pub fn for_each_operand_mut(&mut self, mut operation: impl FnMut(&mut MirOperand)) {
        match self {
            Self::Barrier { .. } => {}
            Self::Broadcast { input, root, .. }
            | Self::Gather { input, root, .. }
            | Self::Scatter { input, root, .. }
            | Self::Reduce { input, root, .. } => {
                operation(input);
                operation(root);
            }
            Self::AllGather { input, .. } | Self::AllReduce { input, .. } => operation(input),
            Self::Send {
                input,
                destination,
                tag,
                ..
            } => {
                operation(input);
                operation(destination);
                if let Some(tag) = tag {
                    operation(tag);
                }
            }
            Self::Receive { source, tag, .. } | Self::Probe { source, tag, .. } => {
                if let Some(source) = source {
                    operation(source);
                }
                if let Some(tag) = tag {
                    operation(tag);
                }
            }
        }
    }

    pub fn input(&self) -> Option<&MirOperand> {
        match self {
            Self::Broadcast { input, .. }
            | Self::Gather { input, .. }
            | Self::Scatter { input, .. }
            | Self::AllGather { input, .. }
            | Self::Reduce { input, .. }
            | Self::AllReduce { input, .. }
            | Self::Send { input, .. } => Some(input),
            Self::Barrier { .. } | Self::Receive { .. } | Self::Probe { .. } => None,
        }
    }

    /// Runtime operands in deterministic source order: value first, followed
    /// by root/peer and then tag where the operation defines them.
    pub fn operands(&self) -> Vec<&MirOperand> {
        match self {
            Self::Barrier { .. } => Vec::new(),
            Self::Broadcast { input, root, .. }
            | Self::Gather { input, root, .. }
            | Self::Scatter { input, root, .. }
            | Self::Reduce { input, root, .. } => vec![input, root],
            Self::AllGather { input, .. } | Self::AllReduce { input, .. } => vec![input],
            Self::Send {
                input,
                destination,
                tag,
                ..
            } => std::iter::once(input)
                .chain(std::iter::once(destination))
                .chain(tag.iter())
                .collect(),
            Self::Receive { source, tag, .. } | Self::Probe { source, tag, .. } => {
                source.iter().chain(tag.iter()).collect()
            }
        }
    }

    pub fn operator(&self) -> Option<OperatorKind> {
        match self {
            Self::Reduce { operator, .. } | Self::AllReduce { operator, .. } => Some(*operator),
            _ => None,
        }
    }
}

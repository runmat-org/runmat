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
        input: Option<MirOperand>,
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
    Cat {
        id: CollectiveId,
        input: MirOperand,
        dimension: MirOperand,
        root: Option<MirOperand>,
    },
    FunctionalReduce {
        id: CollectiveId,
        reducer: MirOperand,
        input: MirOperand,
        root: Option<MirOperand>,
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
        requested_outputs: u8,
    },
    SendReceive {
        id: CollectiveId,
        destination: MirOperand,
        source: MirOperand,
        input: MirOperand,
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
            | Self::Cat { id, .. }
            | Self::FunctionalReduce { id, .. }
            | Self::Send { id, .. }
            | Self::Receive { id, .. }
            | Self::SendReceive { id, .. }
            | Self::Probe { id, .. } => id,
        }
    }

    pub fn for_each_operand_mut(&mut self, mut operation: impl FnMut(&mut MirOperand)) {
        match self {
            Self::Barrier { .. } => {}
            Self::Broadcast { input, root, .. } => {
                if let Some(input) = input {
                    operation(input);
                }
                operation(root);
            }
            Self::Gather { input, root, .. }
            | Self::Scatter { input, root, .. }
            | Self::Reduce { input, root, .. } => {
                operation(input);
                operation(root);
            }
            Self::AllGather { input, .. } | Self::AllReduce { input, .. } => operation(input),
            Self::Cat {
                input,
                dimension,
                root,
                ..
            } => {
                operation(input);
                operation(dimension);
                if let Some(root) = root {
                    operation(root);
                }
            }
            Self::FunctionalReduce {
                reducer,
                input,
                root,
                ..
            } => {
                operation(reducer);
                operation(input);
                if let Some(root) = root {
                    operation(root);
                }
            }
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
            Self::SendReceive {
                destination,
                source,
                input,
                tag,
                ..
            } => {
                operation(destination);
                operation(source);
                operation(input);
                if let Some(tag) = tag {
                    operation(tag);
                }
            }
        }
    }

    pub fn for_each_operand(&self, mut operation: impl FnMut(&MirOperand)) {
        match self {
            Self::Barrier { .. } => {}
            Self::Broadcast { input, root, .. } => {
                if let Some(input) = input {
                    operation(input);
                }
                operation(root);
            }
            Self::Gather { input, root, .. }
            | Self::Scatter { input, root, .. }
            | Self::Reduce { input, root, .. } => {
                operation(input);
                operation(root);
            }
            Self::AllGather { input, .. } | Self::AllReduce { input, .. } => operation(input),
            Self::Cat {
                input,
                dimension,
                root,
                ..
            } => {
                operation(input);
                operation(dimension);
                if let Some(root) = root {
                    operation(root);
                }
            }
            Self::FunctionalReduce {
                reducer,
                input,
                root,
                ..
            } => {
                operation(reducer);
                operation(input);
                if let Some(root) = root {
                    operation(root);
                }
            }
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
            Self::SendReceive {
                destination,
                source,
                input,
                tag,
                ..
            } => {
                operation(destination);
                operation(source);
                operation(input);
                if let Some(tag) = tag {
                    operation(tag);
                }
            }
        }
    }

    pub fn input(&self) -> Option<&MirOperand> {
        match self {
            Self::Broadcast { input, .. } => input.as_ref(),
            Self::Gather { input, .. }
            | Self::Scatter { input, .. }
            | Self::AllGather { input, .. }
            | Self::Reduce { input, .. }
            | Self::AllReduce { input, .. }
            | Self::Cat { input, .. }
            | Self::FunctionalReduce { input, .. }
            | Self::Send { input, .. }
            | Self::SendReceive { input, .. } => Some(input),
            Self::Barrier { .. } | Self::Receive { .. } | Self::Probe { .. } => None,
        }
    }

    /// Runtime operands in deterministic source order: value first, followed
    /// by root/peer and then tag where the operation defines them.
    pub fn operands(&self) -> Vec<&MirOperand> {
        match self {
            Self::Barrier { .. } => Vec::new(),
            Self::Broadcast { input, root, .. } => {
                input.iter().chain(std::iter::once(root)).collect()
            }
            Self::Gather { input, root, .. }
            | Self::Scatter { input, root, .. }
            | Self::Reduce { input, root, .. } => vec![input, root],
            Self::AllGather { input, .. } | Self::AllReduce { input, .. } => vec![input],
            Self::Cat {
                input,
                dimension,
                root,
                ..
            } => std::iter::once(input)
                .chain(std::iter::once(dimension))
                .chain(root.iter())
                .collect(),
            Self::FunctionalReduce {
                reducer,
                input,
                root,
                ..
            } => std::iter::once(reducer)
                .chain(std::iter::once(input))
                .chain(root.iter())
                .collect(),
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
            Self::SendReceive {
                destination,
                source,
                input,
                tag,
                ..
            } => vec![destination, source, input]
                .into_iter()
                .chain(tag.iter())
                .collect(),
        }
    }

    pub fn operator(&self) -> Option<OperatorKind> {
        match self {
            Self::Reduce { operator, .. } | Self::AllReduce { operator, .. } => Some(*operator),
            _ => None,
        }
    }
}

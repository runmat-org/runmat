use crate::{NativeEdge, NativeTerminatorKind};

pub(super) fn edges(terminator: &NativeTerminatorKind) -> Vec<&NativeEdge> {
    match terminator {
        NativeTerminatorKind::Goto { edge } => vec![edge],
        NativeTerminatorKind::Branch {
            then_edge,
            else_edge,
            ..
        }
        | NativeTerminatorKind::For {
            body: then_edge,
            exit: else_edge,
            ..
        }
        | NativeTerminatorKind::ParFor {
            body: then_edge,
            exit: else_edge,
            ..
        }
        | NativeTerminatorKind::Spmd {
            body: then_edge,
            exit: else_edge,
            ..
        }
        | NativeTerminatorKind::TryCatch {
            try_edge: then_edge,
            catch_edge: else_edge,
            ..
        } => vec![then_edge, else_edge],
        NativeTerminatorKind::Switch {
            cases, otherwise, ..
        } => cases
            .iter()
            .map(|(_, edge)| edge)
            .chain(std::iter::once(otherwise))
            .collect(),
        NativeTerminatorKind::Await { resume, .. } => vec![resume],
        NativeTerminatorKind::Return { .. } | NativeTerminatorKind::Unreachable => Vec::new(),
    }
}

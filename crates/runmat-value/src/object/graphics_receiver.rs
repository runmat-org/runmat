//! Opaque identity carried by a live native graphics object.
//!
//! The plot registry issues one shared lease for each selected object. A
//! numeric route and generation are diagnostic coordinates only; consumers
//! must compare this exact lease against the current registry owner before
//! reading or mutating a graphics object.

use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphicsReceiverKind {
    Figure {
        figure: u32,
    },
    Axes {
        figure: u32,
        index: usize,
    },
    /// The numeric child key is a diagnostic coordinate, not object authority.
    PlotChild {
        id: u64,
    },
}

#[derive(Debug, Eq, PartialEq)]
struct Lease {
    kind: GraphicsReceiverKind,
    generation: u64,
}

#[derive(Clone, Debug)]
pub struct GraphicsReceiverRef(Arc<Lease>);

impl PartialEq for GraphicsReceiverRef {
    fn eq(&self, other: &Self) -> bool {
        self.shares_lease(other)
    }
}

impl Eq for GraphicsReceiverRef {}

impl GraphicsReceiverRef {
    /// Issuance alone does not grant authority. The plotting registry must
    /// retain the matching lease and compare pointer identity at every use.
    pub fn issued(kind: GraphicsReceiverKind, generation: u64) -> Self {
        Self(Arc::new(Lease { kind, generation }))
    }

    pub fn kind(&self) -> GraphicsReceiverKind {
        self.0.kind
    }

    pub fn generation(&self) -> u64 {
        self.0.generation
    }

    pub fn shares_lease(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_coordinates_do_not_transfer_graphics_object_authority() {
        let kind = GraphicsReceiverKind::Figure { figure: 7 };
        let selected = GraphicsReceiverRef::issued(kind, 11);
        assert!(selected.shares_lease(&selected.clone()));
        assert!(!selected.shares_lease(&GraphicsReceiverRef::issued(kind, 11)));
        assert_eq!(selected.generation(), 11);
    }
}

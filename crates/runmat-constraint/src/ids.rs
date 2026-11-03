//! Typed handles into a [`System`](crate::System).
//!
//! Ids are dense indices in creation order. They are stable for the life of a system and are
//! the keys the CAD layer uses to map diagnostics back to source spans.

use std::fmt;

macro_rules! id_type {
    ($(#[$doc:meta])* $name:ident, $prefix:literal) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub(crate) u32);

        impl $name {
            /// The dense index of this id (creation order, starting at zero).
            pub fn index(self) -> usize {
                self.0 as usize
            }

            /// Rebuilds an id from a dense index. Using an index the system never issued gives
            /// a build error, never undefined behaviour.
            pub fn from_index(index: usize) -> Self {
                Self(index as u32)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "{}"), self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!($prefix, "{}"), self.0)
            }
        }
    };
}

id_type!(
    /// A scalar unknown (a coordinate, a radius, a quaternion component, …).
    VarId,
    "v"
);
id_type!(
    /// A geometric entity: a 2D point or curve, a rigid body, or a mate connector.
    EntityId,
    "e"
);
id_type!(
    /// A constraint, user-written or internal (arc radius equality, quaternion normalisation).
    ConstraintId,
    "c"
);

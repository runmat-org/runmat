//! The item model the engine runs over, supplied by the host (circuit, board, CAD).

use crate::selector::PseudoArg;
use crate::value::{AttrValue, Dim};
use std::collections::BTreeMap;
use std::fmt::Debug;

/// A typed item graph: a forest of items with kinds, names, classes, attributes and
/// host-defined pseudo-classes. Item ids are totally ordered, which is what makes every result
/// of the engine independent of hash or insertion order.
pub trait ItemGraph {
    type Id: Copy + Ord + Debug;

    /// Every item. Order does not matter; the engine sorts.
    fn items(&self) -> Vec<Self::Id>;
    /// The item's own kind (`pad`, `track`, `net`, `component`).
    fn kind(&self, id: Self::Id) -> &str;
    /// Whether a type selector matches. Override to give kinds a hierarchy (`copper` matching
    /// `pad`, `track` and `via`); the default is equality with [`ItemGraph::kind`].
    fn is_kind(&self, id: Self::Id, kind: &str) -> bool {
        self.kind(id) == kind
    }
    /// The name an `#id` selector matches (refdes, net name, layer name).
    fn name(&self, id: Self::Id) -> Option<&str>;
    fn has_class(&self, id: Self::Id, class: &str) -> bool;
    fn attr(&self, id: Self::Id, name: &str) -> Option<AttrValue>;
    fn parent(&self, id: Self::Id) -> Option<Self::Id>;
    fn children(&self, id: Self::Id) -> Vec<Self::Id>;
    /// Host pseudo-classes such as `:net-class(power)` or `:layer(outer)`. `None` means the
    /// pseudo-class is not known to this model; it then never matches (and [`Schema`]
    /// validation reports it at compile time).
    fn pseudo(&self, id: Self::Id, name: &str, args: &[PseudoArg]) -> Option<bool>;
}

/// The declared type of an attribute, for compile-time selector checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttrType {
    Str,
    Bool,
    /// A number with this dimension.
    Num(Dim),
    List,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PseudoSpec {
    pub min_args: usize,
    pub max_args: usize,
}

/// What a model accepts, so a selector naming an unknown kind, attribute or pseudo-class, or
/// comparing a length attribute against volts, is a diagnostic at rule-compile time.
#[derive(Clone, Debug, Default)]
pub struct Schema {
    pub kinds: BTreeMap<String, ()>,
    pub attrs: BTreeMap<String, AttrType>,
    pub pseudos: BTreeMap<String, PseudoSpec>,
}

impl Schema {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn kind(mut self, k: &str) -> Self {
        self.kinds.insert(k.to_string(), ());
        self
    }
    pub fn kinds(mut self, ks: &[&str]) -> Self {
        for k in ks {
            self.kinds.insert(k.to_string(), ());
        }
        self
    }
    pub fn attr(mut self, name: &str, ty: AttrType) -> Self {
        self.attrs.insert(name.to_string(), ty);
        self
    }
    pub fn pseudo(mut self, name: &str, min_args: usize, max_args: usize) -> Self {
        self.pseudos
            .insert(name.to_string(), PseudoSpec { min_args, max_args });
        self
    }
}

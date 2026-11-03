//! `runmat-rules`: the selector, precedence and explain engine shared by RunMat circuit design
//! rules and CAD `check` packs.
//!
//! - [`selector`]: CSS-like selectors over a typed item graph: type, `#id`, `.class`,
//!   attribute tests (`=`, `!=`, `^=`, `$=`, `*=`, `~=`, and `<`, `<=`, `>`, `>=` on
//!   quantities with units), descendant and child combinators, host pseudo-classes
//!   (`:net-class(power)`, `:layer(outer)`), `:not`/`:is`/`:where`/`:has`, and set operations
//!   (`,` union, `&` intersection, ` - ` difference).
//! - [`cascade`]: rules with declarations, the precedence order (inline, then
//!   `(specificity, priority, source order)`, then inheritance, then default), explain with
//!   overridden candidates, and incremental re-evaluation.
//! - [`pair`]: two-sided rules (clearance between `a` and `b`) with a signature memo.
//! - [`graph`]: the [`ItemGraph`] trait hosts implement, and a [`Schema`] for compile-time
//!   validation.
//! - [`value`]: attribute values and quantities with SI dimensions.
//!
//! # Determinism
//!
//! Only ordered maps; items are visited in id order; ties are broken by a total order. The
//! same graph and rules always give the same matches, values and explanations.
//!
//! # Example
//!
//! ```
//! use runmat_rules::*;
//! # struct G;
//! # impl ItemGraph for G {
//! #     type Id = u32;
//! #     fn items(&self) -> Vec<u32> { vec![0, 1] }
//! #     fn kind(&self, id: u32) -> &str { if id == 0 { "net" } else { "track" } }
//! #     fn name(&self, id: u32) -> Option<&str> { (id == 0).then_some("VBUS") }
//! #     fn has_class(&self, _: u32, c: &str) -> bool { c == "power" }
//! #     fn attr(&self, _: u32, _: &str) -> Option<AttrValue> { None }
//! #     fn parent(&self, id: u32) -> Option<u32> { (id == 1).then_some(0) }
//! #     fn children(&self, id: u32) -> Vec<u32> { if id == 0 { vec![1] } else { vec![] } }
//! #     fn pseudo(&self, _: u32, _: &str, _: &[PseudoArg]) -> Option<bool> { None }
//! # }
//! let g = G;
//! let u = BuiltinUnits;
//! let mut c: Cascade<f64, u32> = Cascade::new();
//! c.push_rule(Rule::new("all", Selector::parse("track", &u).unwrap()).set("width", 0.15));
//! c.push_rule(Rule::new("power", Selector::parse("net.power > track", &u).unwrap()).set("width", 0.5));
//! c.rebuild(&g);
//! let e = c.explain(&g, 1, "width");
//! assert_eq!(e.effective.unwrap().value, 0.5);
//! assert_eq!(e.overridden[0].rule_id, "all");
//! ```

pub mod cascade;
pub mod graph;
pub mod pair;
pub mod selector;
pub mod value;

pub use cascade::{
    Candidate, Cascade, Declaration, Explanation, LostTo, Origin, PropertySpec, Resolved, Rule,
    RuleHandle, SourceKey, SourceOrder,
};
pub use graph::{AttrType, ItemGraph, PseudoSpec, Schema};
pub use pair::{PairCandidate, PairCascade, PairExplanation, PairResolved, PairRule, PairWinner};
pub use selector::{
    AttrOp, Combinator, Complex, Compound, Expr, Literal, PseudoArg, Selector, SelectorError,
    Simple, Span, Specificity,
};
pub use value::{parse_quantity, AttrValue, BuiltinUnits, Dim, Quantity, UnitDef, UnitResolver};

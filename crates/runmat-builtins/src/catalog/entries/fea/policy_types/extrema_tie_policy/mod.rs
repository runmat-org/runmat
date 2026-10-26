//! Catalog identities of the `fea.ExtremaTiePolicy` typed policy class.

use crate::*;

mod all_locations;
mod first_canonical;
#[cfg(test)]
mod tests;

pub(super) const ROLE: &str = concat!(
    "An extrema-tie policy decides which locations a minimum, maximum, absolute maximum, ",
    "or envelope reports when several are exactly tied. The rule is deterministic and ",
    "independent of parallel execution.",
);
pub(super) const STUDY: &str = concat!(
    "Pass the policy as the third argument of `fea.ReductionPolicy.new(spatial, series, ",
    "extremaTies)`.",
);
pub(super) const FAQ: BuiltinDocumentationFaq = BuiltinDocumentationFaq {
    question: "Are near-ties merged?",
    answer: "No. Only exactly equal values are ties.",
};

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[
    &first_canonical::FEA_EXTREMA_TIE_POLICY_FIRST_CANONICAL_CATALOG_ENTRY,
    &all_locations::FEA_EXTREMA_TIE_POLICY_ALL_LOCATIONS_CATALOG_ENTRY,
];

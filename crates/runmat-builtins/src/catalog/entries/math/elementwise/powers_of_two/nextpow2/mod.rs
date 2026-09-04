mod documentation;
mod errors;
mod integer;
mod signatures;

use super::super::support::{provider_unary_numeric_catalog_entry, UnaryNumericCatalogSpec};
use crate::*;
use documentation::DOCUMENTATION;
pub use errors::*;
pub use integer::*;
use signatures::NEXTPOW2_DESCRIPTOR;

const BINDINGS: [BuiltinBindingDeclaration; 1] = REQUIRED_DEFAULT_BINDING;

pub const NEXTPOW2_CATALOG_ENTRY: BuiltinCatalogEntry =
    provider_unary_numeric_catalog_entry(UnaryNumericCatalogSpec {
        identity: BuiltinCatalogIdentity { name: "nextpow2" },
        documentation: DOCUMENTATION,
        descriptor: &NEXTPOW2_DESCRIPTOR,
        inference_rule: BuiltinInferenceRule::Math(MathInferenceRule::PowerOfTwo(
            PowerOfTwoInferenceRule::NextExponent,
        )),
        bindings: &BINDINGS,
        extensions: &[],
        integer_capabilities: &NEXTPOW2_INTEGER_CAPABILITIES,
        integer_audit: None,
        fusion: BuiltinFusionPolicy::Candidate,
    });

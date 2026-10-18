use runmat_types::{CallInference, CallRequest, ValueFact};
use serde::Serialize;

use crate::BuiltinCatalogEntry;

pub(crate) type BuiltinInferenceHandler = fn(&CallRequest, &BuiltinCatalogEntry) -> CallInference;
pub(crate) type BuiltinFusionFactHandler = fn(&CallRequest) -> Option<ValueFact>;

/// Connects one catalog entry directly to inference owned by that identity.
///
/// The executable pointer is omitted from serialized catalog data because a
/// process address is not a stable contract. The surrounding entry supplies
/// the stable builtin identity, while generated products fingerprint source.
#[derive(Clone, Copy, Serialize)]
pub struct IdentityInferenceRule {
    #[serde(skip)]
    handler: BuiltinInferenceHandler,
    #[serde(skip)]
    fusion_fact: Option<BuiltinFusionFactHandler>,
    #[serde(skip)]
    structural_dimension_constructor: bool,
}

impl IdentityInferenceRule {
    pub(crate) const fn new(handler: BuiltinInferenceHandler) -> Self {
        Self {
            handler,
            fusion_fact: None,
            structural_dimension_constructor: false,
        }
    }

    pub(crate) const fn with_fusion_fact(
        handler: BuiltinInferenceHandler,
        fusion_fact: BuiltinFusionFactHandler,
    ) -> Self {
        Self {
            handler,
            fusion_fact: Some(fusion_fact),
            structural_dimension_constructor: false,
        }
    }

    /// The identity's own inference treats structural dimensions (or a
    /// `like` prototype) as result shape. This grants analysis only; it does
    /// not admit the call to executable fusion.
    pub(crate) const fn with_structural_dimensions(handler: BuiltinInferenceHandler) -> Self {
        Self {
            handler,
            fusion_fact: None,
            structural_dimension_constructor: true,
        }
    }

    pub fn has_structural_dimensions(self) -> bool {
        self.structural_dimension_constructor
    }

    pub(crate) fn infer(self, request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
        (self.handler)(request, entry)
    }

    pub(crate) fn fusion_fact(self, request: &CallRequest) -> Option<ValueFact> {
        self.fusion_fact.and_then(|handler| handler(request))
    }

    pub(crate) fn has_fusion_fact(self) -> bool {
        self.fusion_fact.is_some()
    }
}

impl std::fmt::Debug for IdentityInferenceRule {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("IdentityInferenceRule")
    }
}

impl PartialEq for IdentityInferenceRule {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.handler, other.handler)
            && self.structural_dimension_constructor == other.structural_dimension_constructor
            && match (self.fusion_fact, other.fusion_fact) {
                (Some(left), Some(right)) => std::ptr::fn_addr_eq(left, right),
                (None, None) => true,
                _ => false,
            }
    }
}

impl Eq for IdentityInferenceRule {}

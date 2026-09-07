mod domain;
mod hyperbolic;
mod prototype;
mod trigonometric;

use domain::{inverse_hyperbolic_literal_domain, inverse_trigonometric_literal_domain};
use prototype::apply_inverse_trigonometric_like;

use crate::{BuiltinCatalogEntry, InverseHyperbolicFunction, InverseTrigonometricFunction};
use runmat_types::{CallInference, CallRequest};

pub(in crate::catalog::inference) fn infer_inverse_trigonometric(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: InverseTrigonometricFunction,
) -> CallInference {
    trigonometric::infer_inverse_trigonometric(request, entry, function)
}

pub(in crate::catalog::inference) fn infer_inverse_hyperbolic(
    request: &CallRequest,
    entry: &BuiltinCatalogEntry,
    function: InverseHyperbolicFunction,
) -> CallInference {
    hyperbolic::infer_inverse_hyperbolic(request, entry, function)
}

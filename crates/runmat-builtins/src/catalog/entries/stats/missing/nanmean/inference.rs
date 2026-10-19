use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

use super::super::nan_reduction_support as support;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    support::infer(
        request,
        entry,
        support::Alias {
            rejects_integer_data: false,
            defaults_integer_to_double: true,
            normalization_before_dimension: false,
            pairwise_min: false,
            single_output_dynamic: false,
            output_class_control: true,
            complex_error_code: None,
        },
    )
}

use crate::BuiltinCatalogEntry;
use runmat_types::{CallInference, CallRequest};

use super::super::nan_reduction_support as support;

pub(super) fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    support::infer(
        request,
        entry,
        support::Alias {
            rejects_integer_data: true,
            defaults_integer_to_double: false,
            normalization_before_dimension: true,
            pairwise_min: false,
            single_output_dynamic: true,
            output_class_control: false,
            complex_error_code: Some("RM.STD.COMPLEX_UNSUPPORTED"),
        },
    )
}

use super::shared::{array_method_documentation, array_method_entry};
use super::{inference, shared};
use crate::*;

fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::sum(request, entry)
}

array_method_entry!(DLARRAY_SUM_CATALOG_ENTRY, "dlarray.sum", DLARRAY_SUM_DESCRIPTOR,
    array_method_documentation!("dlarray.sum", "Reduce real floating dlarray Data to a scalar.",
        "The supported autodiff form is a full-array scalar reduction, optionally selected with 'all'. Dimension-specific reductions are not yet part of this host tape.",
        "A = dlarray(single([1 2 3])); B = sum(A, 'all'); assert(isequal(B.Data, single(6)));",
        "assert(strcmp(class(B.Data), 'single')); assert(isequal(size(B.Data), [1 1])); assert(isequal(B.Data, single(6)));"), infer);

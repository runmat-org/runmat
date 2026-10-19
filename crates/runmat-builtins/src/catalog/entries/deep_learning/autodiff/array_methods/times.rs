use super::shared::{array_method_documentation, array_method_entry};
use super::{inference, shared};
use crate::*;

fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::binary(request, entry, inference::BinaryGeometry::Elementwise)
}

array_method_entry!(DLARRAY_TIMES_CATALOG_ENTRY, "dlarray.times", DLARRAY_TIMES_DESCRIPTOR,
    array_method_documentation!("dlarray.times", "Multiply real floating dlarray values elementwise.",
        "Elementwise multiplication with scalar expansion or equal-shaped host operands. The reverse-mode tape records this operation inside dlfeval.",
        "A = dlarray(single([1 2 3])); B = A .* A; assert(isequal(B.Data, single([1 4 9])));",
        "assert(strcmp(class(B.Data), 'single')); assert(isequal(size(B.Data), [1 3])); assert(isequal(B.Data, single([1 4 9])));"), infer);

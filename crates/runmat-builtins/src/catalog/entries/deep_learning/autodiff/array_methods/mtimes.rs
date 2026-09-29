use super::shared::{array_method_documentation, array_method_entry};
use super::{inference, shared};
use crate::*;

fn infer(
    request: &runmat_types::CallRequest,
    entry: &crate::BuiltinCatalogEntry,
) -> runmat_types::CallInference {
    inference::binary(request, entry, inference::BinaryGeometry::Matrix)
}

array_method_entry!(DLARRAY_MTIMES_CATALOG_ENTRY, "dlarray.mtimes", DLARRAY_MTIMES_DESCRIPTOR,
    array_method_documentation!("dlarray.mtimes", "Matrix-multiply real floating dlarray values.",
        "Two-dimensional host matrix multiplication with compatible inner dimensions, or multiplication by a scalar in either operand order. The output remains a dlarray and participates in the host reverse-mode tape.",
        "A = dlarray(single([1 2; 3 4])); B = A * single([1; 2]); assert(isequal(B.Data, single([5; 11])));",
        "assert(strcmp(class(B.Data), 'single')); assert(isequal(size(B.Data), [2 1])); assert(isequal(B.Data, single([5; 11])));"), infer);

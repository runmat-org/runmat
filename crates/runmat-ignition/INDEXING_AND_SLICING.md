# Indexing and Slicing in Ignition

Ignition centralizes MATLAB’s indexing/slicing rules in the VM to ensure uniform semantics.

## Gather (IndexSlice)
- Inputs: number of dims, how many numeric scalars were pushed, bitmasks for `:` and `end` per dim. Indices are popped (numeric scalars/ranges/logic) followed by the base.
- Scalar indices drop a dimension; vector/range/logic produce extents. Shapes are normalized so that in 2‑D:
  - `(I, scalar)` → shape `[|I|, 1]`
  - `(scalar, J)` → shape `[1, |J|]`
- Logical masks must match the dimension length; masks are transformed to indices.
- `end` arithmetic is resolved per dimension using the base shape.

## Scatter (StoreSlice)
- RHS must either match the selected extents exactly or broadcast (e.g., column vector across selected columns). Scalar RHS broadcasts along all selected elements.
- Errors: shape mismatch, index out of bounds, unsupported index type.

## Cells and expansion
- `C{...}` returns elements of the cell content; vector indexing expands to multiple elements, used for argument expansion or slice targets where permitted.
- Function returns expand in multi‑assign or argument positions; `CallFunctionMulti` is used to route multiple outputs unambiguously.

## End arithmetic helpers
- Separate instructions (`IndexRangeEnd`, `Index1DRangeEnd`, `IndexSliceEx`) support `end‑k` forms with proper bounds.

See `vm.rs` for exact code paths and `tests/indexing_properties.rs` for invariants and edge cases.

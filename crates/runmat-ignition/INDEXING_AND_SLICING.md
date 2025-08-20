# Indexing and Slicing

Ignition implements MATLAB-compatible indexing/slicing in the VM to ensure one source of truth for semantics. The compiler recognizes patterns and emits specialized instructions that keep the VM simple and fast.

Indexing categories:

- Numeric-only scalar indices → `Index(n)` / `StoreIndex(n)`
- Mixed `:` (colon), `end`, ranges, vectors, logical masks → `IndexSlice` / `StoreSlice`
- `end - k` arithmetic on numeric positions → `IndexSliceEx`
- Ranges using per-dimension `end` arithmetic → `IndexRangeEnd` (N-D) and `Index1DRangeEnd` (1-D)
- Cell content indexing → `IndexCell` / `IndexCellExpand`

All indexing is 1-based and column-major. Out-of-bounds and shape-mismatch conditions are normalized via mex identifiers.

## Stack conventions

- For gather (read): the compiler pushes the base value first, then any numeric index values. The VM pops numeric indices (reversing their order) and then pops the base.
- For scatter (write): the compiler pushes base, then numeric indices, then the RHS value. The VM pops RHS, then numerics, then base.

## Gather: Index(n)

`Index(n)` is used when all indices are numeric scalars (no `:`, `end`, vectors, logical). The VM:

1. Pops `n` numeric indices and reverses to restore left-to-right order
2. Pops the base
3. If base is an object → routes to `subsref(obj, '()', {indices})`
4. Else uses `runmat_runtime::perform_indexing(base, &indices)`

Error cases:
- Non-tensor/non-object with `n > 1`
- Out-of-bounds → `MATLAB:IndexOutOfBounds` or `MATLAB:SubscriptOutOfBounds`

## Gather: IndexSlice(dims, numeric_count, colon_mask, end_mask)

Use when any dimension involves `:`, `end`, ranges/vectors, or logical masks. The VM constructs per-dimension selectors:

- Colon → full range in that dimension
- End → scalar index equal to the dimension length
- Numeric (scalar) → use as 1-based index
- Numeric (vector) → materialize list of indices (1-based)
- Logical mask → dimension length must match; non-zeros become indices

Shape rules:
- Each selector contributes either 1 (scalar) or its length
- In 2-D, shapes are normalized to match MATLAB:
  - `(I, scalar)` → `[len(I), 1]`
  - `(scalar, J)` → `[1, len(J)]`
- A single result value returns as a scalar `Value::Num` or `Value::String`

Fast 2-D paths (tensors):
- `A(:, j)` returns full column quickly
- `A(i, :)` returns full row quickly
- `A(:, J)` returns `[rows, |J|]`
- `A(I, :)` returns `[|I|, cols]`

String arrays mirror tensor semantics, but return `Value::String` for scalar results and `Value::StringArray` for arrays.

Errors:
- Out-of-bounds → `MATLAB:IndexOutOfBounds`
- Logical mask shape mismatch → `MATLAB:IndexShape`
- Slicing non-tensors/strings → `MATLAB:SliceNonTensor`

### 1-D specialization

For `dims == 1`, the VM supports:
- `A(:)` → all linear elements
- `A(end)` → last element
- Logical mask the same length as `numel(A)` → keep non-zeros
- Vector of indices → gather in order

## Gather with end arithmetic: IndexSliceEx

`IndexSliceEx(dims, numeric_count, colon_mask, end_mask, end_offsets)` applies `end - k` to specific numeric positions. Each `(pos, k)` pairs the position within the numeric indices list (not dimension index) with an offset `k`. The VM maps numeric positions to actual dimensions by skipping colon and plain `end` dims.

## Gather with ranged end arithmetic: IndexRangeEnd / Index1DRangeEnd

Use when the end of a range depends on a dimension length: `i:j:end-k` per dimension.

- `IndexRangeEnd` parameters:
  - `dims`, `numeric_count`, `colon_mask`, `end_mask`
  - `range_dims`: which dims are ranges
  - `range_has_step`: whether each range has a step
  - `end_offsets`: `k` for `end-k`
- Stack order: base, then for each range in increasing dimension order push `start[, step]`, then numeric scalar indices
- The VM computes concrete indices per dim, honoring sign of the step; step 0 → `MATLAB:IndexStepZero`
- `Index1DRangeEnd` is a compact form for the common 1-D case

## Scatter: StoreIndex / StoreSlice

`StoreIndex(n)` supports scalar numeric indices only. `StoreSlice` parallels `IndexSlice` with the same selector construction.

Broadcasting rules (tensors):
- RHS can be scalar → broadcast to all selected elements
- RHS can be a tensor whose per-dimension lengths are either 1 or equal to the selection extent in that dimension
- Column-/row-fast paths update entire columns or rows efficiently

String array writes:
- RHS can be a `String`, `StringArray`, or numeric converted to string (for convenience)

Errors:
- Out-of-bounds → `MATLAB:IndexOutOfBounds`
- Shape mismatch for broadcasting → `MATLAB:ShapeMismatch`
- Non-tensor/strings → `MATLAB:SliceNonTensor`

## Scatter with end arithmetic: StoreSliceEx / StoreRangeEnd / StoreSlice1DRangeEnd

`StoreSliceEx` applies `end - k` to numeric positions before performing a generic scatter.

`StoreRangeEnd` mirrors `IndexRangeEnd` for writes: the VM builds per-dimension index lists (including ranged `end-k`) and then performs broadcasting-aware scatter into the selected positions.

`StoreSlice1DRangeEnd` is a compact 1-D writer: base, `start[, step]`, `rhs` and an `offset` for `end-k`.

## Cells: IndexCell and IndexCellExpand

`IndexCell(n)` supports:
- 1-D `C{i}`
- 2-D `C{i,j}`
- For objects: routes to `subsref(obj, '{}', {indices})`

`IndexCellExpand(n, out_count)` expands contents into a comma-list in column-major order. If indices are omitted with `expand_all`, all cell elements expand. This is used for argument expansion at call sites and for building vectors via `PackToRow/PackToCol` when assigning into slices.

Errors:
- Unsupported index type → `MATLAB:CellIndexType`
- Out-of-bounds → `MATLAB:CellSubscriptOutOfBounds`

## Interactions with function calls

- A user function call used as an argument may be expanded into multiple inputs by compiling it via `CallFunctionMulti` and then packed using `PackToRow`/`PackToCol`
- Cell expansion at call-sites is expressed via `Call*ExpandMulti` with `ArgSpec` entries describing which arguments expand and how many indices are consumed

See `vm.rs` for full instruction handlers and `INSTR_SET.md` for exact stack layouts.

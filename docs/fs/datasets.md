---
title: "Datasets API"
category: "Filesystem"
section: "13.1"
last_updated: "May 28, 2026"
---

# Datasets API

RunMat's dataset API lets you scale large, long-lived arrays by storing them as named datasets that can be reopened across sessions, read and written in slices, and backed by local or remote storage. The API is built around chunked array storage, so workflows can grow toward petabyte-scale datasets without rewriting whole files for every update.

Applications keep using path-based MATLAB calls, while RunMat maps slices onto chunks and lets the active filesystem provider move those chunks through local disk, browser storage, or remote object storage.

Use a dataset when ordinary workspace variables or MAT-file snapshots become too coarse-grained: training data, simulation output, experiment results, generated caches, shared project data, or any array that is written incrementally and read back later.

## Creating A Dataset

Datasets start with a schema. The schema names each array and gives its shape. Chunk shape is optional, but it is worth setting when the array will be read or written in predictable blocks.

```matlab
schema.arrays.samples = struct( ...
    "dtype", "f64", ...
    "shape", [1000000 64], ...
    "chunk", [4096 64]);

schema.arrays.labels = struct( ...
    "dtype", "f64", ...
    "shape", [1000000 1], ...
    "chunk", [4096 1]);

ds = data.create("training.data", schema);
```

This creates a `training.data` directory with two arrays:

- `samples`, a `1000000 x 64` matrix.
- `labels`, a `1000000 x 1` vector.

The dataset can be opened later from the same path:

```matlab
ds = data.open("training.data");
samples = Dataset.array(ds, "samples");
labels = Dataset.array(ds, "labels");
```

## Writing Batches

Arrays are updated through `DataArray.write`. Passing a slice writes only that region of the array.

```matlab
batchRows = [1 4096];

DataArray.write(samples, { batchRows, ":" }, sampleBatch);
DataArray.write(labels, { batchRows, ":" }, labelBatch);
```

Slice entries use one-based MATLAB indexing:

| Slice entry | Meaning |
| --- | --- |
| `":"` | The full dimension. |
| `17` | One element in that dimension. |
| `[start end]` | Inclusive range. |

The right-hand value must match the shape selected by the slice. A write to `{ [1 4096], ":" }` for a `1000000 x 64` array expects a `4096 x 64` value.

## Reading Batches

Reading uses the same slice format.

```matlab
batchRows = [4097 8192];

X = DataArray.read(samples, { batchRows, ":" });
y = DataArray.read(labels, { batchRows, ":" });
```

Calling `DataArray.read(samples)` reads the full array. For large datasets, prefer slice reads that match the region being processed.

## Metadata

Datasets can store attributes alongside the arrays. Attributes are useful for provenance, units, source paths, processing state, and application-level labels.

```matlab
Dataset.set_attr(ds, "source", "runmat://project/raw/training");
Dataset.set_attr(ds, "normalized", true);

attrs = Dataset.attrs(ds);
source = Dataset.get_attr(ds, "source", "unknown");
```

Each metadata update advances the dataset version. `Dataset.version(ds)` returns the current manifest version token.

## Choosing Chunks

Chunk shape controls how an array is divided for storage. It should follow the common access pattern.

For the training dataset above, rows are written and read in batches. A chunk shape of `[4096 64]` means each batch of 4096 rows lines up with one chunk of `samples`; `[4096 1]` does the same for `labels`.

Other common choices:

| Access pattern | Chunk shape |
| --- | --- |
| Row batches | `[batchSize columns]` |
| Matrix tiles | `[tileRows tileCols]` |
| Time windows | `[windowLength channels]` |
| Full-vector reads | A large one-dimensional chunk, up to the expected read size. |

Larger chunks reduce bookkeeping and are efficient for large contiguous reads. Smaller chunks reduce how much data must be rewritten for small updates. If unsure, choose the shape of the block the program naturally reads or writes.

## Transactions

Use a transaction when several changes should land together. This is common when a batch write needs to update multiple arrays and metadata in one commit.

```matlab
tx = Dataset.begin(ds);

DataTransaction.write(tx, "samples", { [8193 12288], ":" }, sampleBatch);
DataTransaction.write(tx, "labels", { [8193 12288], ":" }, labelBatch);
DataTransaction.set_attr(tx, "lastBatch", 3);

commit(tx);
```

At commit time, RunMat checks that the dataset has not changed since the transaction began. If another writer committed first, the commit fails with a manifest conflict instead of overwriting newer state.

For explicit concurrency control, commit with the version token the code expects:

```matlab
version = Dataset.version(ds);

tx = Dataset.begin(ds);
DataTransaction.set_attr(tx, "stage", "validated");
DataTransaction.commit(tx, struct("if_manifest", version));
```

Transactions are process-local until commit. If the process exits before commit, staged changes are discarded.

## Storage And Scale

A dataset is a directory. Treat the directory as the unit to copy, move, import, export, or delete.

```text
training.data/
  manifest.json
  arrays/
    samples/
    labels/
```

The manifest records the array names, shapes, chunk shapes, attributes, and version. The array directories contain the stored payloads and chunk indexes. Most code should use the dataset API instead of reading those files directly.

Because datasets use the active RunMat filesystem provider, the same code works against local disk, sandboxed storage, browser-backed storage, or a remote project filesystem. 

On remote storage, chunked writes can upload the touched chunks directly through provider-managed upload targets, so large dataset updates do not have to move as one monolithic file, and benefit from high-throughput parallel uploads to object storage.

Combined with RunMat Server's remote filesystem, datasets can be used to share large arrays between team members and projects.

## Dataset Operations

Path-level operations manage whole datasets.

| Function | Use |
| --- | --- |
| `data.create(path, schema)` | Create a dataset. |
| `data.open(path)` | Open an existing dataset. |
| `data.exists(path)` | Check whether a dataset exists. |
| `data.list(prefix)` | List child `.data` datasets under a directory. |
| `data.inspect(path)` | Return path, ID, array count, and update time. |
| `data.copy(fromPath, toPath)` | Copy a dataset directory. |
| `data.move(fromPath, toPath)` | Rename a dataset directory. |
| `data.delete(path)` | Remove a dataset directory. |
| `data.import(path, "data", sourcePath)` | Copy another RunMat dataset into `path`. |
| `data.export(path, "data", targetPath)` | Copy a dataset to another path. |

`data.import` and `data.export` currently support RunMat's `"data"` format.

## Common Methods

| Method | Use |
| --- | --- |
| `Dataset.path(ds)` | Return the dataset path. |
| `Dataset.id(ds)` | Return the dataset ID. |
| `Dataset.version(ds)` | Return the current version token. |
| `Dataset.arrays(ds)` | Return array names. |
| `Dataset.has_array(ds, name)` | Check whether an array exists. |
| `Dataset.array(ds, name)` | Return an array handle. |
| `Dataset.attrs(ds)` | Return dataset attributes. |
| `Dataset.set_attr(ds, key, value)` | Set one attribute. |
| `Dataset.refresh(ds)` | Reopen the handle from current storage state. |
| `DataArray.shape(arr)` | Return array shape. |
| `DataArray.chunk_shape(arr)` | Return chunk shape. |
| `DataArray.read(arr, sliceSpec)` | Read a full array or slice. |
| `DataArray.write(arr, sliceSpec, values)` | Write a full array or slice. |
| `DataTransaction.commit(tx)` | Apply staged transaction changes. |
| `DataTransaction.abort(tx)` | Discard staged transaction changes. |

## Boundaries

Datasets are for persistent numeric arrays and metadata. They are not a table format, a general object database, or a replacement for every MAT-file workflow.

For short scripts that need to save and reload a few variables, `save` and `load` are usually the right tool. For arrays that are long-lived, named, sliced, shared, or updated in batches, use a dataset.

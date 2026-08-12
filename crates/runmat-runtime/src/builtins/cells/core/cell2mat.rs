//! MATLAB-compatible `cell2mat` builtin implemented for the modern RunMat runtime.

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{
    CharArray, ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray, NumericStorage,
    Tensor, Value,
};

use crate::builtins::cells::type_resolvers::cell2mat_type;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, FusionError,
    FusionExprContext, FusionKernelTemplate, GpuOpKind, ReductionNaN, ResidencyPolicy, ScalarType,
    ShapeRequirements,
};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::cells::core::cell2mat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cell2mat",
    op_kind: GpuOpKind::Custom("cell-flatten"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Current RunMat behavior gathers resident cell contents and returns a host array. MATLAB documents resident gpuArray cell contents as remaining resident since R2025a; preserving that residency requires a future provider-backed cell concatenation path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::cells::core::cell2mat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cell2mat",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: Some(FusionKernelTemplate {
        scalar_precisions: &[ScalarType::F32, ScalarType::F64],
        wgsl_body: |_ctx: &FusionExprContext| {
            Err(FusionError::Message(
                "cell2mat terminates fusion; contents are materialised on the host.",
            ))
        },
    }),
    reduction: None,
    emits_nan: false,
    notes: "Acts as a fusion sink; fusion planner stops GPU fusion before calling cell2mat.",
};

const BUILTIN_NAME: &str = "cell2mat";

const CELL2MAT_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Dense array assembled from cell contents.",
}];

const CELL2MAT_INPUTS: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input cell array.",
}];

const CELL2MAT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "A = cell2mat(C)",
    inputs: &CELL2MAT_INPUTS,
    outputs: &CELL2MAT_OUTPUT,
}];

const CELL2MAT_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2MAT.INVALID_INPUT",
    identifier: Some("RunMat:cell2mat:InvalidInput"),
    when: "The input is not a cell array.",
    message: "cell2mat: expected a cell array input",
};

const CELL2MAT_ERROR_INVALID_CONTENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2MAT.INVALID_CONTENTS",
    identifier: Some("RunMat:cell2mat:InvalidContents"),
    when: "Cell contents cannot be concatenated to a dense array.",
    message: "cell2mat: cell contents are not compatible for concatenation",
};

const CELL2MAT_ERROR_SIZE_EXCEEDED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2MAT.SIZE_EXCEEDED",
    identifier: Some("RunMat:cell2mat:SizeExceeded"),
    when: "Resulting output exceeds supported platform limits.",
    message: "cell2mat: resulting matrix exceeds platform limits",
};

const CELL2MAT_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CELL2MAT.INTERNAL",
    identifier: None,
    when: "Internal cell2mat allocation or conversion failed.",
    message: "cell2mat: internal error",
};

const CELL2MAT_ERRORS: [BuiltinErrorDescriptor; 4] = [
    CELL2MAT_ERROR_INVALID_INPUT,
    CELL2MAT_ERROR_INVALID_CONTENTS,
    CELL2MAT_ERROR_SIZE_EXCEEDED,
    CELL2MAT_ERROR_INTERNAL,
];

pub const CELL2MAT_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CELL2MAT_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CELL2MAT_ERRORS,
};

const CELL2MAT_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "C contents",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::AllowedExceptWith64BitInteger,
        notes: "All eight integer classes are documented cell contents. Since R2025a, unlike integer classes may be concatenated; scalar double contents may coexist except with int64 or uint64.",
    }];

pub const CELL2MAT_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "A = cell2mat(C) for real integer cell contents",
        inputs: &CELL2MAT_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Saturate,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Same-class contents remain exact. For mixed integer classes the leftmost nonempty integer class selects the output class and later contents saturate into it. Scalar doubles use the same assignment conversion except with int64/uint64. Resident contents currently gather to a host result; resident output preservation remains an explicit compatibility gap.",
    }];

fn cell2mat_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "cell2mat",
    category = "cells/core",
    summary = "Convert cell arrays of blocks into dense arrays.",
    keywords = "cell2mat,cell,matrix,concatenation",
    accel = "gather",
    type_resolver(cell2mat_type),
    descriptor(crate::builtins::cells::core::cell2mat::CELL2MAT_DESCRIPTOR),
    integer_capabilities(crate::builtins::cells::core::cell2mat::CELL2MAT_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::cells::core::cell2mat"
)]
async fn cell2mat_builtin(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::Cell(ca) => cell_array_to_matrix(&ca).await,
        other => Err(cell2mat_error_with_message(
            format!("{}, got {other:?}", CELL2MAT_ERROR_INVALID_INPUT.message),
            &CELL2MAT_ERROR_INVALID_INPUT,
        )),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElementKind {
    Numeric,
    Complex,
    TypedComplexInteger,
    Logical,
    Char,
}

#[derive(Clone)]
struct CellEntry {
    kind: ElementKind,
    shape: Vec<usize>,
    data: EntryData,
}

#[derive(Clone)]
enum EntryData {
    Numeric(NumericStorage),
    Complex(Vec<(f64, f64)>),
    TypedComplexInteger(IntegerComplexStorage),
    Logical(Vec<u8>),
    Char(Vec<char>),
}

impl EntryData {
    fn len(&self) -> usize {
        match self {
            EntryData::Numeric(storage) => storage.len(),
            EntryData::Complex(data) => data.len(),
            EntryData::TypedComplexInteger(storage) => storage.len(),
            EntryData::Logical(data) => data.len(),
            EntryData::Char(data) => data.len(),
        }
    }
}

impl CellEntry {
    fn len(&self) -> usize {
        self.data.len()
    }
}

async fn cell_array_to_matrix(ca: &runmat_value::CellArray) -> BuiltinResult<Value> {
    if ca.data.is_empty() {
        // Mirror MATLAB's behaviour: empty cell array -> 0x0 double matrix.
        let tensor = Tensor::new(Vec::new(), vec![0, 0]).map_err(|e| {
            cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
        })?;
        return Ok(Value::Tensor(tensor));
    }

    let cell_shape = ca.shape.clone();
    let rank = cell_shape.len();

    let mut entries: Vec<CellEntry> = Vec::with_capacity(ca.data.len());
    let mut detected_kind: Option<ElementKind> = None;

    for ptr in &ca.data {
        let gathered = gather_if_needed_async(ptr).await?;
        let entry = parse_cell_entry(gathered)?;
        if let Some(kind) = detected_kind {
            if kind != entry.kind {
                return Err(cell2mat_error_with_message(
                    "cell2mat: all cell contents must share the same fundamental type",
                    &CELL2MAT_ERROR_INVALID_CONTENTS,
                ));
            }
        } else {
            detected_kind = Some(entry.kind);
        }
        entries.push(entry);
    }

    let element_kind = detected_kind.unwrap_or(ElementKind::Numeric);
    validate_entry_kinds(&entries, element_kind)?;

    let multi_indices: Vec<Vec<usize>> = (0..entries.len())
        .map(|linear| linear_to_multi_row_major(linear, &cell_shape))
        .collect();

    let mut block_sizes: Vec<Vec<usize>> = cell_shape
        .iter()
        .map(|&extent| vec![0usize; extent])
        .collect();
    let mut extra_shape: Option<Vec<usize>> = None;

    for (entry, indices) in entries.iter().zip(multi_indices.iter()) {
        let tile_dims = extend_shape(&entry.shape, rank);
        for dim in 0..rank {
            let size = tile_dims[dim];
            let Some(slot) = block_sizes
                .get_mut(dim)
                .and_then(|v| v.get_mut(indices[dim]))
            else {
                return Err(cell2mat_error_with_message(
                    "cell2mat: cell index is out of bounds for the provided shape",
                    &CELL2MAT_ERROR_INVALID_CONTENTS,
                ));
            };
            if *slot == 0 {
                *slot = size;
            } else if *slot != size {
                return Err(cell2mat_error_with_message(
                    "cell2mat: all cells in the same row and column must agree on block sizes",
                    &CELL2MAT_ERROR_INVALID_CONTENTS,
                ));
            }
        }

        let entry_extra = if entry.shape.len() > rank {
            entry.shape[rank..].to_vec()
        } else {
            Vec::new()
        };
        if let Some(existing) = &extra_shape {
            if existing.len() != entry_extra.len()
                || existing.iter().zip(&entry_extra).any(|(a, b)| *a != *b)
            {
                return Err(cell2mat_error_with_message(
                    "cell2mat: higher-dimensional extents must match across all cells",
                    &CELL2MAT_ERROR_INVALID_CONTENTS,
                ));
            }
        } else {
            extra_shape = Some(entry_extra);
        }
    }

    let extra_dims = extra_shape.unwrap_or_default();
    let mut result_shape = Vec::with_capacity(rank + extra_dims.len());
    for sizes in &block_sizes {
        let sum = sizes
            .iter()
            .try_fold(0usize, |acc, &v| acc.checked_add(v))
            .ok_or_else(|| {
                cell2mat_error_with_message(
                    "cell2mat: resulting matrix is too large to represent on this platform",
                    &CELL2MAT_ERROR_SIZE_EXCEEDED,
                )
            })?;
        result_shape.push(sum);
    }
    result_shape.extend(extra_dims.iter().copied());

    if result_shape.is_empty() {
        result_shape = vec![0, 0];
    }

    if element_kind == ElementKind::Char && result_shape.len() > 2 {
        return Err(cell2mat_error_with_message(
            "cell2mat: character cell contents must form a 2-D character array",
            &CELL2MAT_ERROR_INVALID_CONTENTS,
        ));
    }

    let total_elems = total_len(&result_shape).ok_or_else(|| {
        cell2mat_error_with_message(
            "cell2mat: resulting matrix is too large to represent on this platform",
            &CELL2MAT_ERROR_SIZE_EXCEEDED,
        )
    })?;

    let prefix_offsets: Vec<Vec<usize>> =
        block_sizes.iter().map(|sizes| prefix_sums(sizes)).collect();

    match element_kind {
        ElementKind::Numeric => {
            let prototype = numeric_prototype(&entries)?;
            let storage = prototype.zeros_like(total_elems);
            let mut tensor =
                Tensor::from_numeric_storage(storage, result_shape.clone()).map_err(|e| {
                    cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
                })?;
            copy_numeric(
                &entries,
                &multi_indices,
                &result_shape,
                &prefix_offsets,
                rank,
                &mut tensor,
            )?;
            Ok(Value::Tensor(tensor))
        }
        ElementKind::Complex => {
            let mut data = vec![(0.0f64, 0.0f64); total_elems];
            copy_complex(
                &entries,
                &multi_indices,
                &result_shape,
                &prefix_offsets,
                rank,
                &mut data,
            )?;
            let tensor = ComplexTensor::new(data, result_shape).map_err(|e| {
                cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
            })?;
            Ok(Value::ComplexTensor(tensor))
        }
        ElementKind::TypedComplexInteger => {
            let prototype = typed_complex_integer_prototype(&entries)?;
            let mut storage = IntegerComplexStorage::new(
                prototype.real.zeros_like(total_elems),
                prototype.imag.zeros_like(total_elems),
            )
            .map_err(|e| {
                cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
            })?;
            copy_typed_complex_integer(
                &entries,
                &multi_indices,
                &result_shape,
                &prefix_offsets,
                rank,
                &mut storage,
            )?;
            let tensor = ComplexTensor::new_integer(storage, result_shape).map_err(|e| {
                cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
            })?;
            Ok(Value::ComplexTensor(tensor))
        }
        ElementKind::Logical => {
            let mut data = vec![0u8; total_elems];
            copy_logical(
                &entries,
                &multi_indices,
                &result_shape,
                &prefix_offsets,
                rank,
                &mut data,
            )?;
            let logical = LogicalArray::new(data, result_shape).map_err(|e| {
                cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
            })?;
            Ok(Value::LogicalArray(logical))
        }
        ElementKind::Char => {
            let rows = result_shape.first().copied().unwrap_or(0);
            let cols = result_shape.get(1).copied().unwrap_or(1);
            let char_data = copy_chars(&entries, &multi_indices, rows, cols, &block_sizes)?;
            let array = CharArray::new(char_data, rows, cols).map_err(|e| {
                cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
            })?;
            Ok(Value::CharArray(array))
        }
    }
}

fn validate_entry_kinds(entries: &[CellEntry], expected: ElementKind) -> BuiltinResult<()> {
    for entry in entries {
        if entry.len() == 0 {
            continue;
        }
        if entry.kind != expected {
            return Err(cell2mat_error_with_message(
                "cell2mat: all non-empty cell contents must share the same fundamental type",
                &CELL2MAT_ERROR_INVALID_CONTENTS,
            ));
        }
    }
    Ok(())
}

fn typed_complex_integer_prototype(entries: &[CellEntry]) -> BuiltinResult<&IntegerComplexStorage> {
    let Some(prototype) = entries.iter().find_map(|entry| match &entry.data {
        EntryData::TypedComplexInteger(storage) => Some(storage),
        _ => None,
    }) else {
        return Err(cell2mat_error_with_message(
            "cell2mat: typed complex integer cell contents are missing storage",
            &CELL2MAT_ERROR_INTERNAL,
        ));
    };

    if entries.iter().any(|entry| {
        matches!(
            &entry.data,
            EntryData::TypedComplexInteger(storage) if storage.class_name() != prototype.class_name()
        )
    }) {
        return Err(cell2mat_error_with_message(
            "cell2mat: typed complex integer cell contents must share the same integer class",
            &CELL2MAT_ERROR_INVALID_CONTENTS,
        ));
    }

    Ok(prototype)
}

fn numeric_prototype(entries: &[CellEntry]) -> BuiltinResult<&NumericStorage> {
    let integer_prototype = entries.iter().find_map(|entry| match &entry.data {
        EntryData::Numeric(storage)
            if !storage.is_empty() && storage.clone().into_integer_storage().is_ok() =>
        {
            Some(storage)
        }
        _ => None,
    });
    let Some(prototype) = integer_prototype.or_else(|| {
        entries
            .iter()
            .find_map(|entry| match &entry.data {
                EntryData::Numeric(storage) if !storage.is_empty() => Some(storage),
                _ => None,
            })
            .or_else(|| {
                entries.iter().find_map(|entry| match &entry.data {
                    EntryData::Numeric(storage) => Some(storage),
                    _ => None,
                })
            })
    }) else {
        return Err(cell2mat_error_with_message(
            "cell2mat: numeric cell contents are missing storage",
            &CELL2MAT_ERROR_INTERNAL,
        ));
    };

    let prototype_integer = prototype.clone().into_integer_storage().ok();
    let has_scalar_double = entries.iter().any(|entry| {
        matches!(&entry.data, EntryData::Numeric(NumericStorage::F64(values)) if values.len() == 1)
    });
    let integer_compatible = entries.iter().all(|entry| match &entry.data {
        EntryData::Numeric(storage) if storage.clone().into_integer_storage().is_ok() => true,
        EntryData::Numeric(NumericStorage::F64(values)) => values.len() == 1,
        _ => true,
    });
    let integer_scalar_double_allowed = prototype_integer.as_ref().is_some_and(|storage| {
        !has_scalar_double || !matches!(storage, IntegerStorage::I64(_) | IntegerStorage::U64(_))
    });
    if !(prototype_integer.is_some() && integer_compatible && integer_scalar_double_allowed)
        && entries.iter().any(|entry| {
            matches!(
                &entry.data,
                EntryData::Numeric(storage) if storage.numeric_dtype() != prototype.numeric_dtype()
            )
        })
    {
        return Err(cell2mat_error_with_message(
            "cell2mat: floating-point numeric contents must share a class; scalar doubles cannot concatenate with int64 or uint64",
            &CELL2MAT_ERROR_INVALID_CONTENTS,
        ));
    }

    Ok(prototype)
}

fn copy_numeric(
    entries: &[CellEntry],
    indices: &[Vec<usize>],
    result_shape: &[usize],
    prefix_offsets: &[Vec<usize>],
    rank: usize,
    output: &mut Tensor,
) -> BuiltinResult<()> {
    let total_rank = result_shape.len();
    let dest_strides = column_major_strides(result_shape);

    for (entry, multi) in entries.iter().zip(indices.iter()) {
        if entry.len() == 0 {
            continue;
        }
        let EntryData::Numeric(storage) = &entry.data else {
            continue;
        };
        let padded_shape = extend_shape(&entry.shape, total_rank);
        let base_offsets = compute_base_offsets(multi, prefix_offsets, total_rank, rank)?;

        for linear in 0..storage.len() {
            let local_index = linear_to_multi_column_major(linear, &padded_shape);
            let dest_linear = accumulate_linear(&base_offsets, &local_index, &dest_strides);
            output
                .set_numeric_assignment_at(
                    dest_linear,
                    storage
                        .value_at(linear)
                        .expect("numeric cell storage is in bounds"),
                )
                .map_err(|e| {
                    cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
                })?;
        }
    }
    Ok(())
}

fn copy_complex(
    entries: &[CellEntry],
    indices: &[Vec<usize>],
    result_shape: &[usize],
    prefix_offsets: &[Vec<usize>],
    rank: usize,
    output: &mut [(f64, f64)],
) -> BuiltinResult<()> {
    let total_rank = result_shape.len();
    let dest_strides = column_major_strides(result_shape);

    for (entry, multi) in entries.iter().zip(indices.iter()) {
        if entry.len() == 0 {
            continue;
        }
        let EntryData::Complex(ref data) = entry.data else {
            continue;
        };
        let padded_shape = extend_shape(&entry.shape, total_rank);
        let base_offsets = compute_base_offsets(multi, prefix_offsets, total_rank, rank)?;

        for (linear, value) in data.iter().enumerate() {
            let local_index = linear_to_multi_column_major(linear, &padded_shape);
            let dest_linear = accumulate_linear(&base_offsets, &local_index, &dest_strides);
            output[dest_linear] = *value;
        }
    }
    Ok(())
}

fn copy_typed_complex_integer(
    entries: &[CellEntry],
    indices: &[Vec<usize>],
    result_shape: &[usize],
    prefix_offsets: &[Vec<usize>],
    rank: usize,
    output: &mut IntegerComplexStorage,
) -> BuiltinResult<()> {
    let total_rank = result_shape.len();
    let dest_strides = column_major_strides(result_shape);

    for (entry, multi) in entries.iter().zip(indices.iter()) {
        if entry.len() == 0 {
            continue;
        }
        let EntryData::TypedComplexInteger(storage) = &entry.data else {
            continue;
        };
        let padded_shape = extend_shape(&entry.shape, total_rank);
        let base_offsets = compute_base_offsets(multi, prefix_offsets, total_rank, rank)?;

        for linear in 0..storage.len() {
            let local_index = linear_to_multi_column_major(linear, &padded_shape);
            let dest_linear = accumulate_linear(&base_offsets, &local_index, &dest_strides);
            let real = storage
                .real
                .value_at(linear)
                .expect("typed complex storage is in bounds");
            let imag = storage
                .imag
                .value_at(linear)
                .expect("typed complex storage is in bounds");
            output.real.set_value(dest_linear, real).map_err(|e| {
                cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
            })?;
            output.imag.set_value(dest_linear, imag).map_err(|e| {
                cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
            })?;
        }
    }
    Ok(())
}

fn copy_logical(
    entries: &[CellEntry],
    indices: &[Vec<usize>],
    result_shape: &[usize],
    prefix_offsets: &[Vec<usize>],
    rank: usize,
    output: &mut [u8],
) -> BuiltinResult<()> {
    let total_rank = result_shape.len();
    let dest_strides = column_major_strides(result_shape);

    for (entry, multi) in entries.iter().zip(indices.iter()) {
        if entry.len() == 0 {
            continue;
        }
        let EntryData::Logical(ref data) = entry.data else {
            continue;
        };
        let padded_shape = extend_shape(&entry.shape, total_rank);
        let base_offsets = compute_base_offsets(multi, prefix_offsets, total_rank, rank)?;

        for (linear, value) in data.iter().enumerate() {
            let local_index = linear_to_multi_column_major(linear, &padded_shape);
            let dest_linear = accumulate_linear(&base_offsets, &local_index, &dest_strides);
            output[dest_linear] = *value;
        }
    }
    Ok(())
}

fn copy_chars(
    entries: &[CellEntry],
    indices: &[Vec<usize>],
    rows: usize,
    cols: usize,
    block_sizes: &[Vec<usize>],
) -> BuiltinResult<Vec<char>> {
    let mut output = vec!['\0'; rows.saturating_mul(cols)];
    let row_prefix = block_sizes
        .first()
        .map(|sizes| prefix_sums(sizes))
        .unwrap_or_else(|| vec![0]);
    let col_prefix = block_sizes
        .get(1)
        .map(|sizes| prefix_sums(sizes))
        .unwrap_or_else(|| vec![0]);

    for (entry, multi) in entries.iter().zip(indices.iter()) {
        if entry.len() == 0 {
            continue;
        }
        let EntryData::Char(ref data) = entry.data else {
            continue;
        };
        let shape = extend_shape(&entry.shape, 2);
        let row_offset = row_prefix
            .get(multi.first().copied().unwrap_or(0))
            .copied()
            .unwrap_or(0);
        let col_offset = col_prefix
            .get(multi.get(1).copied().unwrap_or(0))
            .copied()
            .unwrap_or(0);

        for (linear, value) in data.iter().enumerate() {
            let local_idx = linear_to_multi_row_major(linear, &shape);
            let dest_row = row_offset + local_idx.first().copied().unwrap_or(0);
            let dest_col = col_offset + local_idx.get(1).copied().unwrap_or(0);
            let dest_linear = dest_row
                .checked_mul(cols)
                .and_then(|v| v.checked_add(dest_col))
                .ok_or_else(|| {
                    cell2mat_error_with_message(
                        "cell2mat: resulting character array exceeds supported size",
                        &CELL2MAT_ERROR_SIZE_EXCEEDED,
                    )
                })?;
            output[dest_linear] = *value;
        }
    }

    Ok(output)
}

fn compute_base_offsets(
    multi: &[usize],
    prefix_offsets: &[Vec<usize>],
    total_rank: usize,
    rank: usize,
) -> BuiltinResult<Vec<usize>> {
    let mut base = vec![0usize; total_rank];
    for dim in 0..rank.min(prefix_offsets.len()) {
        let idx = multi.get(dim).copied().unwrap_or(0);
        let offset = prefix_offsets[dim].get(idx).copied().ok_or_else(|| {
            cell2mat_error_with_message(
                "cell2mat: internal offset calculation failed",
                &CELL2MAT_ERROR_SIZE_EXCEEDED,
            )
        })?;
        base[dim] = offset;
    }
    Ok(base)
}

fn parse_cell_entry(value: Value) -> BuiltinResult<CellEntry> {
    match value {
        Value::Tensor(t) => {
            let shape = normalize_shape(t.shape.clone());
            let storage = t.into_numeric_storage().map_err(|e| {
                cell2mat_error_with_message(format!("cell2mat: {e}"), &CELL2MAT_ERROR_INTERNAL)
            })?;
            Ok(CellEntry {
                kind: ElementKind::Numeric,
                shape,
                data: EntryData::Numeric(storage),
            })
        }
        Value::Num(n) => Ok(CellEntry {
            kind: ElementKind::Numeric,
            shape: vec![1, 1],
            data: EntryData::Numeric(NumericStorage::F64(vec![n])),
        }),
        Value::Int(i) => Ok(CellEntry {
            kind: ElementKind::Numeric,
            shape: vec![1, 1],
            data: EntryData::Numeric(NumericStorage::from(IntegerStorage::from_scalar(i))),
        }),
        Value::Bool(b) => Ok(CellEntry {
            kind: ElementKind::Logical,
            shape: vec![1, 1],
            data: EntryData::Logical(vec![if b { 1 } else { 0 }]),
        }),
        Value::LogicalArray(la) => Ok(CellEntry {
            kind: ElementKind::Logical,
            shape: normalize_shape(la.shape.clone()),
            data: EntryData::Logical(la.data.clone()),
        }),
        Value::Complex(re, im) => Ok(CellEntry {
            kind: ElementKind::Complex,
            shape: vec![1, 1],
            data: EntryData::Complex(vec![(re, im)]),
        }),
        Value::ComplexTensor(ct) => {
            let shape = normalize_shape(ct.shape.clone());
            if let Some(storage) = ct.integer_storage() {
                Ok(CellEntry {
                    kind: ElementKind::TypedComplexInteger,
                    shape,
                    data: EntryData::TypedComplexInteger(storage.clone()),
                })
            } else {
                Ok(CellEntry {
                    kind: ElementKind::Complex,
                    shape,
                    data: EntryData::Complex(ct.materialize_f64()),
                })
            }
        }
        Value::CharArray(ca) => Ok(CellEntry {
            kind: ElementKind::Char,
            shape: vec![ca.rows, ca.cols],
            data: EntryData::Char(ca.data.clone()),
        }),
        Value::Cell(_) => Err(cell2mat_error_with_message(
            "cell2mat: nested cell arrays are not supported",
            &CELL2MAT_ERROR_INVALID_CONTENTS,
        )),
        Value::String(_) | Value::StringArray(_) => Err(cell2mat_error_with_message(
            "cell2mat: string inputs are not supported; convert to char arrays first",
            &CELL2MAT_ERROR_INVALID_CONTENTS,
        )),
        Value::GpuTensor(_) => Err(cell2mat_error_with_message(
            "cell2mat: unexpected GPU tensor after gather; please report this issue",
            &CELL2MAT_ERROR_INVALID_CONTENTS,
        )),
        other => Err(cell2mat_error_with_message(
            format!("cell2mat: unsupported cell element type: {other:?}"),
            &CELL2MAT_ERROR_INVALID_CONTENTS,
        )),
    }
}

fn extend_shape(shape: &[usize], min_len: usize) -> Vec<usize> {
    if shape.len() >= min_len {
        shape.to_vec()
    } else {
        let mut extended = shape.to_vec();
        extended.resize(min_len, 1);
        extended
    }
}

fn normalize_shape(mut shape: Vec<usize>) -> Vec<usize> {
    if shape.is_empty() {
        shape.push(1);
        shape.push(1);
    }
    shape
}

fn prefix_sums(values: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(values.len());
    let mut accum = 0usize;
    for &v in values {
        out.push(accum);
        accum = accum.saturating_add(v);
    }
    out
}

fn column_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut accum = 1usize;
    for &dim in shape {
        strides.push(accum);
        accum = accum.saturating_mul(dim.max(1));
    }
    strides
}

fn total_len(shape: &[usize]) -> Option<usize> {
    if shape.is_empty() {
        return Some(0);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn linear_to_multi_column_major(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = Vec::with_capacity(shape.len());
    for &dim in shape {
        if dim == 0 {
            coords.push(0);
        } else {
            coords.push(linear % dim);
            linear /= dim;
        }
    }
    coords
}

fn linear_to_multi_row_major(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut coords = vec![0usize; shape.len()];
    for (idx, &dim) in shape.iter().enumerate().rev() {
        if dim == 0 {
            coords[idx] = 0;
        } else {
            coords[idx] = linear % dim;
            linear /= dim;
        }
    }
    coords
}

fn accumulate_linear(base_offsets: &[usize], local_index: &[usize], strides: &[usize]) -> usize {
    local_index
        .iter()
        .enumerate()
        .map(|(dim, &idx)| (base_offsets[dim] + idx) * strides[dim])
        .sum()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_value::{IntValue, IntegerStorage};

    fn cell2mat_builtin(value: Value) -> BuiltinResult<Value> {
        block_on(super::cell2mat_builtin(value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn descriptor_signatures_cover_cell2mat_forms() {
        let labels: Vec<&str> = CELL2MAT_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert_eq!(labels, vec!["A = cell2mat(C)"]);
    }

    fn scalar_cell(values: &[f64], rows: usize, cols: usize) -> Value {
        let cells: Vec<Value> = values.iter().map(|&v| Value::Num(v)).collect();
        crate::make_cell(cells, rows, cols).expect("cell")
    }

    fn append_same_class(left: &IntegerStorage, right: &IntegerStorage) -> IntegerStorage {
        macro_rules! append {
            ($left:expr, $right:expr, $variant:ident) => {{
                let mut values = $left.clone();
                values.extend_from_slice($right);
                IntegerStorage::$variant(values)
            }};
        }

        match (left, right) {
            (IntegerStorage::I8(left), IntegerStorage::I8(right)) => append!(left, right, I8),
            (IntegerStorage::I16(left), IntegerStorage::I16(right)) => append!(left, right, I16),
            (IntegerStorage::I32(left), IntegerStorage::I32(right)) => append!(left, right, I32),
            (IntegerStorage::I64(left), IntegerStorage::I64(right)) => append!(left, right, I64),
            (IntegerStorage::U8(left), IntegerStorage::U8(right)) => append!(left, right, U8),
            (IntegerStorage::U16(left), IntegerStorage::U16(right)) => append!(left, right, U16),
            (IntegerStorage::U32(left), IntegerStorage::U32(right)) => append!(left, right, U32),
            (IntegerStorage::U64(left), IntegerStorage::U64(right)) => append!(left, right, U64),
            _ => panic!("test inputs must have matching integer classes"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn simple_numeric_cell() {
        let cell = scalar_cell(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn native_single_cells_preserve_class_and_block_layout() {
        let first =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![1.25, 2.5]), vec![2, 1]).unwrap();
        let second =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![3.75, 4.5]), vec![2, 1]).unwrap();
        let cell =
            crate::make_cell(vec![Value::Tensor(first), Value::Tensor(second)], 1, 2).unwrap();
        let Value::Tensor(output) = cell2mat_builtin(cell).expect("cell2mat") else {
            panic!("expected tensor");
        };

        assert_eq!(output.shape, vec![2, 2]);
        assert_eq!(
            output.into_numeric_storage(),
            Ok(NumericStorage::F32(vec![1.25, 2.5, 3.75, 4.5]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn block_concatenation() {
        let row1_left = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor"));
        let row1_right =
            Value::Tensor(Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3]).expect("tensor"));
        let row2_left = Value::Tensor(Tensor::new(vec![6.0, 7.0], vec![1, 2]).expect("tensor"));
        let row2_right =
            Value::Tensor(Tensor::new(vec![8.0, 9.0, 10.0], vec![1, 3]).expect("tensor"));
        let cell = crate::make_cell(vec![row1_left, row1_right, row2_left, row2_right], 2, 2)
            .expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 5]);
                assert_eq!(
                    t.materialize_f64(),
                    vec![1.0, 6.0, 2.0, 7.0, 3.0, 8.0, 4.0, 9.0, 5.0, 10.0]
                );
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_cell() {
        let a = Value::Bool(true);
        let b = Value::Bool(false);
        let c = Value::Bool(true);
        let d = Value::Bool(false);
        let cell = crate::make_cell(vec![a, b, c, d], 2, 2).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::LogicalArray(la) => {
                assert_eq!(la.shape, vec![2, 2]);
                assert_eq!(la.data, vec![1, 1, 0, 0]);
            }
            other => panic!("expected logical array result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn complex_cell() {
        let values = vec![Value::Complex(1.0, 2.0), Value::Complex(3.0, 4.0)];
        let cell = crate::make_cell(values, 1, 2).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 2]);
                assert_eq!(ct.materialize_f64(), vec![(1.0, 2.0), (3.0, 4.0)]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn typed_complex_integer_cells_preserve_exact_components() {
        let first = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![5]),
            )
            .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor");
        let second = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![1_u64 << 63]),
                IntegerStorage::U64(vec![6]),
            )
            .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor");
        let cell = crate::make_cell(
            vec![Value::ComplexTensor(first), Value::ComplexTensor(second)],
            1,
            2,
        )
        .expect("cell");

        let result = cell2mat_builtin(cell).expect("cell2mat");
        assert!(matches!(
            result,
            Value::ComplexTensor(tensor)
                if tensor.shape == vec![1, 2]
                    && tensor.integer_storage().as_ref().is_some_and(|storage|
                        storage.real == IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63])
                            && storage.imag == IntegerStorage::U64(vec![5, 6]))
        ));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn typed_complex_integer_cells_reject_mixed_classes() {
        let u64_value = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::U64(vec![1]), IntegerStorage::U64(vec![2]))
                .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor");
        let i64_value = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I64(vec![1]), IntegerStorage::I64(vec![2]))
                .expect("storage"),
            vec![1, 1],
        )
        .expect("tensor");
        let cell = crate::make_cell(
            vec![
                Value::ComplexTensor(u64_value),
                Value::ComplexTensor(i64_value),
            ],
            1,
            2,
        )
        .expect("cell");

        let err = cell2mat_builtin(cell).expect_err("mixed integer classes must reject");
        assert!(err
            .to_string()
            .contains("must share the same integer class"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn typed_integer_cells_preserve_every_native_class_exactly() {
        let cases = [
            (IntegerStorage::I8(vec![-7]), IntegerStorage::I8(vec![4])),
            (
                IntegerStorage::I16(vec![-700]),
                IntegerStorage::I16(vec![4]),
            ),
            (
                IntegerStorage::I32(vec![-70_000]),
                IntegerStorage::I32(vec![4]),
            ),
            (
                IntegerStorage::I64(vec![i64::MIN + 1]),
                IntegerStorage::I64(vec![4]),
            ),
            (IntegerStorage::U8(vec![7]), IntegerStorage::U8(vec![4])),
            (IntegerStorage::U16(vec![700]), IntegerStorage::U16(vec![4])),
            (
                IntegerStorage::U32(vec![70_000]),
                IntegerStorage::U32(vec![4]),
            ),
            (
                IntegerStorage::U64(vec![u64::MAX]),
                IntegerStorage::U64(vec![1_u64 << 63]),
            ),
        ];

        for (left, right) in cases {
            let cell = crate::make_cell(
                vec![
                    Value::Tensor(Tensor::new_integer(left.clone(), vec![1, 1]).expect("left")),
                    Value::Tensor(Tensor::new_integer(right.clone(), vec![1, 1]).expect("right")),
                ],
                1,
                2,
            )
            .expect("cell");
            let result = cell2mat_builtin(cell).expect("cell2mat");
            let Value::Tensor(output) = result else {
                panic!("expected typed integer tensor");
            };
            assert_eq!(output.shape, vec![1, 2]);
            assert_eq!(
                output.integer_storage(),
                Some(&append_same_class(&left, &right))
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn typed_integer_cells_preserve_block_shape_and_saturate_mixed_classes() {
        let first =
            Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 7]), vec![2, 1]).expect("first");
        let second = Tensor::new_integer(IntegerStorage::U64(vec![1_u64 << 63, 3]), vec![2, 1])
            .expect("second");
        let cell = crate::make_cell(vec![Value::Tensor(first), Value::Tensor(second)], 1, 2)
            .expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        assert!(matches!(
            result,
            Value::Tensor(output)
                if output.shape == vec![2, 2]
                    && output.integer_storage()
                        == Some(&IntegerStorage::U64(vec![u64::MAX, 7, 1_u64 << 63, 3]))
        ));

        let scalar_cell = crate::make_cell(
            vec![
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Int(IntValue::U64(1_u64 << 63)),
            ],
            1,
            2,
        )
        .expect("scalar cell");
        let scalar_result = cell2mat_builtin(scalar_cell).expect("cell2mat");
        assert!(matches!(
            scalar_result,
            Value::Tensor(output)
                if output.integer_storage()
                    == Some(&IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]))
        ));

        let empty = Tensor::new_integer(IntegerStorage::U64(Vec::new()), vec![0, 0])
            .expect("empty typed tensor");
        let empty_cell = crate::make_cell(vec![Value::Tensor(empty)], 1, 1).expect("empty cell");
        let empty_result = cell2mat_builtin(empty_cell).expect("cell2mat");
        assert!(matches!(
            empty_result,
            Value::Tensor(output)
                if output.shape == vec![0, 0]
                    && output.integer_storage() == Some(&IntegerStorage::U64(Vec::new()))
        ));

        let mixed = crate::make_cell(
            vec![
                Value::Int(IntValue::I8(-7)),
                Value::Int(IntValue::U64(u64::MAX)),
            ],
            1,
            2,
        )
        .expect("cell");
        let Value::Tensor(output) = cell2mat_builtin(mixed).expect("mixed integer cell") else {
            panic!("expected integer tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::I8(vec![-7, i8::MAX]))
        );

        let unsigned_left = crate::make_cell(
            vec![Value::Int(IntValue::U8(3)), Value::Int(IntValue::I64(-9))],
            1,
            2,
        )
        .expect("cell");
        let Value::Tensor(output) = cell2mat_builtin(unsigned_left).expect("mixed integer cell")
        else {
            panic!("expected integer tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U8(vec![3, 0]))
        );

        let scalar_double =
            crate::make_cell(vec![Value::Num(300.2), Value::Int(IntValue::U8(5))], 1, 2)
                .expect("cell");
        let Value::Tensor(output) = cell2mat_builtin(scalar_double).expect("scalar double") else {
            panic!("expected integer tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U8(vec![u8::MAX, 5]))
        );

        let wide_with_double =
            crate::make_cell(vec![Value::Int(IntValue::U64(1)), Value::Num(2.0)], 1, 2)
                .expect("cell");
        assert!(cell2mat_builtin(wide_with_double).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_cell() {
        let a = Value::CharArray(CharArray::new("hi".chars().collect(), 1, 2).unwrap());
        let b = Value::CharArray(CharArray::new("BY".chars().collect(), 1, 2).unwrap());
        let cell = crate::make_cell(vec![a, b], 2, 1).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 2);
                assert_eq!(arr.cols, 2);
                assert_eq!(arr.data, vec!['h', 'i', 'B', 'Y']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn mismatched_block_sizes_error() {
        let a = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![3.0], vec![1, 1]).unwrap());
        let cell = crate::make_cell(vec![a, b], 1, 2).expect("cell");
        let err = cell2mat_builtin(cell).unwrap_err().to_string();
        assert!(err.contains("block sizes"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn higher_dimensional_tiling() {
        let a = Value::Tensor(Tensor::new(vec![1.0; 8], vec![2, 2, 2]).unwrap());
        let b = Value::Tensor(Tensor::new(vec![2.0; 4], vec![2, 1, 2]).unwrap());
        let cell = crate::make_cell(vec![a, b], 1, 2).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 3, 2]);
                assert_eq!(t.materialize_f64().len(), 12);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn empty_cell_returns_empty_double() {
        let cell = crate::make_cell(Vec::new(), 0, 0).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![0, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cell2mat_gpu_cells_are_gathered() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cell =
                crate::make_cell(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell");
            let result = cell2mat_builtin(cell).expect("cell2mat");
            match result {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![2, 2]);
                    assert_eq!(t.materialize_f64(), tensor.materialize_f64());
                }
                other => panic!("expected tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cell2mat_wgpu_cells_are_gathered() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("tensor");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let cell = crate::make_cell(vec![Value::GpuTensor(handle.clone())], 1, 1).expect("cell");
        let result = cell2mat_builtin(cell).expect("cell2mat");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2]);
                assert_eq!(t.materialize_f64(), tensor.materialize_f64());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }
}

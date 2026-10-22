//! Sparse's own argument admission precedes its intentional dense readback.
use super::*;
use runmat_builtins::SparseConstructionForm;
use runmat_types::{NumericDomain, ValueKindFact};

enum Operand<T> {
    Host(T),
    Resident(runmat_accelerate_api::GpuTensorHandle),
}
impl<T> Operand<T> {
    fn count(&self, host_count: impl FnOnce(&T) -> usize) -> Option<usize> {
        match self {
            Self::Host(value) => Some(host_count(value)),
            Self::Resident(handle) => handle
                .shape()
                .iter()
                .try_fold(1usize, |count, extent| count.checked_mul(*extent)),
        }
    }
    async fn resolve(self, parse: impl FnOnce(Value) -> BuiltinResult<T>) -> BuiltinResult<T> {
        match self {
            Self::Host(value) => Ok(value),
            Self::Resident(handle) => {
                let value = Value::GpuTensor(handle);
                let host = gpu_helpers::gather_value_async(&value).await?;
                parse(host)
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Role {
    Conversion,
    Size,
    Triplet,
}

fn operand<T>(
    value: Value,
    role: Role,
    parse: impl FnOnce(Value) -> BuiltinResult<T>,
) -> BuiltinResult<Operand<T>> {
    let Value::GpuTensor(handle) = value else {
        return parse(value).map(Operand::Host);
    };
    let fact = crate::value_fact::value_fact(&Value::GpuTensor(handle.clone()));
    let numeric = fact.numeric();
    let supported = matches!(fact.kind, ValueKindFact::Logical)
        || numeric.is_some_and(|numeric| {
            numeric.domain == NumericDomain::Real
                || (matches!(role, Role::Conversion) && numeric.class.integer_class().is_none())
        });
    if !supported {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sparse: unsupported resident argument class or complexity",
        ));
    }
    if matches!(role, Role::Conversion) && handle.shape().len() != 2 {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            format!(
                "sparse: input must be a 2-D matrix, got {}-D tensor",
                handle.shape().len()
            ),
        ));
    }
    if matches!(role, Role::Size) && fact.shape.element_count() != Some(1) {
        return Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sparse: dimensions and nzmax must be scalar sizes",
        ));
    }
    Ok(Operand::Resident(handle))
}

pub(super) struct PreparedSparse(Construction);
enum Construction {
    Conversion(Operand<SparseTensor>),
    Dimensions(Operand<usize>, Operand<usize>, MatrixClass),
    Triplets {
        rows: Operand<Vec<usize>>,
        columns: Operand<Vec<usize>>,
        values: Operand<TripletValues>,
        dimensions: Option<(Operand<usize>, Operand<usize>)>,
        nzmax: Option<Operand<usize>>,
    },
}

pub(super) fn prepare(args: crate::call::BuiltinArguments) -> BuiltinResult<PreparedSparse> {
    prepare_form(args).map(PreparedSparse)
}
fn prepare_form(args: crate::call::BuiltinArguments) -> BuiltinResult<Construction> {
    let count = args.len();
    let typename = count == 3 && keyword_of(&args[2])?.is_some();
    let form = SparseConstructionForm::for_arguments(count, typename);
    if form == Some(SparseConstructionForm::Triplets) {
        validate_triplet_subscript_classes(&args[0], &args[1])?;
    }
    let mut args = args.into_iter();
    match form {
        Some(SparseConstructionForm::Conversion) => Ok(Construction::Conversion(operand(
            args.next().expect("one input"),
            Role::Conversion,
            sparse_from_value,
        )?)),
        Some(SparseConstructionForm::Dimensions) if count == 2 => Ok(Construction::Dimensions(
            operand(args.next().expect("m"), Role::Size, |v| {
                parse_size_arg(&v, "m")
            })?,
            operand(args.next().expect("n"), Role::Size, |v| {
                parse_size_arg(&v, "n")
            })?,
            MatrixClass::Floating(NumericDType::F64),
        )),
        Some(SparseConstructionForm::Dimensions) => {
            let rows = operand(args.next().expect("m"), Role::Size, |v| {
                parse_size_arg(&v, "m")
            })?;
            let cols = operand(args.next().expect("n"), Role::Size, |v| {
                parse_size_arg(&v, "n")
            })?;
            let class = sparse_storage_class(&args.next().expect("typename"), NAME)?;
            Ok(Construction::Dimensions(rows, cols, class))
        }
        Some(SparseConstructionForm::Triplets) => {
            let rows = operand(args.next().expect("i"), Role::Triplet, |v| {
                triplet_subscripts(&v, "row")
            })?;
            let columns = operand(args.next().expect("j"), Role::Triplet, |v| {
                triplet_subscripts(&v, "column")
            })?;
            let values = operand(args.next().expect("v"), Role::Triplet, |v| {
                triplet_values(&v)
            })?;
            let dimensions = if count >= 5 {
                Some((
                    operand(args.next().expect("m"), Role::Size, |v| {
                        parse_size_arg(&v, "m")
                    })?,
                    operand(args.next().expect("n"), Role::Size, |v| {
                        parse_size_arg(&v, "n")
                    })?,
                ))
            } else {
                None
            };
            let nzmax = args
                .next()
                .map(|v| operand(v, Role::Size, |v| parse_size_arg(&v, "nzmax")))
                .transpose()?;
            if let (Some(row_count), Some(column_count), Some(value_count)) = (
                rows.count(Vec::len),
                columns.count(Vec::len),
                values.count(TripletValues::len),
            ) {
                let target = row_count.max(column_count).max(value_count);
                if [row_count, column_count, value_count]
                    .iter()
                    .any(|&count| count != target && !(count == 1 && target > 1))
                {
                    return Err(sparse_error(
                        &SPARSE_ERROR_INVALID_INPUT,
                        "sparse: i, j, and v must have the same number of elements",
                    ));
                }
            }
            if let Some((row_limit, col_limit)) = &dimensions {
                for (indices, limit) in [(&rows, row_limit), (&columns, col_limit)] {
                    if let (Operand::Host(indices), Operand::Host(limit)) = (indices, limit) {
                        if indices.iter().any(|index| index > limit) {
                            return Err(sparse_error(
                                &SPARSE_ERROR_INVALID_INDEX,
                                "sparse: subscript exceeds matrix dimensions",
                            ));
                        }
                    }
                }
            }
            Ok(Construction::Triplets {
                rows,
                columns,
                values,
                dimensions,
                nzmax,
            })
        }
        _ => Err(sparse_error(
            &SPARSE_ERROR_INVALID_INPUT,
            "sparse: expected sparse(A), sparse(m,n), or sparse(i,j,v[,m,n[,nzmax]])",
        )),
    }
}

impl PreparedSparse {
    pub(super) async fn execute(self) -> BuiltinResult<SparseTensor> {
        match self.0 {
            Construction::Conversion(value) => value.resolve(sparse_from_value).await,
            Construction::Dimensions(rows, columns, class) => {
                let rows = rows.resolve(|v| parse_size_arg(&v, "m")).await?;
                let columns = columns.resolve(|v| parse_size_arg(&v, "n")).await?;
                match class {
                    MatrixClass::Floating(NumericDType::F64) => {
                        Ok(SparseTensor::zeros(rows, columns))
                    }
                    MatrixClass::Floating(NumericDType::F32) => {
                        Ok(SparseTensor::zeros_f32(rows, columns))
                    }
                    MatrixClass::Logical => Ok(SparseTensor::zeros_logical(rows, columns)),
                    MatrixClass::Floating(_) => Err(sparse_error(
                        &SPARSE_ERROR_INVALID_INPUT,
                        "sparse: unsupported storage class",
                    )),
                }
            }
            Construction::Triplets {
                rows,
                columns,
                values,
                dimensions,
                nzmax,
            } => {
                let rows = rows.resolve(|v| triplet_subscripts(&v, "row")).await?;
                let columns = columns
                    .resolve(|v| triplet_subscripts(&v, "column"))
                    .await?;
                let values = values.resolve(|v| triplet_values(&v)).await?;
                let dimensions = match dimensions {
                    Some((rows, cols)) => Some((
                        rows.resolve(|v| parse_size_arg(&v, "m")).await?,
                        cols.resolve(|v| parse_size_arg(&v, "n")).await?,
                    )),
                    None => None,
                };
                if let Some(nzmax) = nzmax {
                    nzmax.resolve(|v| parse_size_arg(&v, "nzmax")).await?;
                }
                sparse_from_parsed_triplets(rows, columns, values, dimensions)
            }
        }
    }
}

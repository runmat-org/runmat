//! Bit-preserving prepared numeric headers. Dense/sparse storage retains its
//! actual representation; serialization does not materialize a dense mirror.
use super::*;

pub(in super::super) enum ArchivedMatrix {
    Scalar(u64),
    Dense {
        shape: Vec<usize>,
        values: Vec<f64>,
    },
    Sparse {
        rows: usize,
        columns: usize,
        pointers: Vec<usize>,
        indices: Vec<usize>,
        values: Vec<f64>,
    },
}
impl ArchivedMatrix {
    pub(in super::super) async fn read(
        reader: &mut super::super::checkpoint::json::Reader,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<Self> {
        reader.token(b'{')?;
        let value = if reader.field_is("Scalar") {
            reader.key("Scalar", true)?;
            Self::Scalar(reader.metadata(20, c).await?)
        } else if reader.field_is("Dense") {
            reader.key("Dense", true)?;
            reader.token(b'{')?;
            reader.key("shape", true)?;
            let shape = reader.array(20, c).await?;
            reader.key("values", false)?;
            let values = reader.bits(c).await?;
            reader.token(b'}')?;
            Self::Dense { shape, values }
        } else {
            reader.key("Sparse", true)?;
            reader.token(b'{')?;
            reader.key("rows", true)?;
            let rows = reader.metadata(20, c).await?;
            reader.key("columns", false)?;
            let columns = reader.metadata(20, c).await?;
            reader.key("pointers", false)?;
            let pointers = reader.array(20, c).await?;
            reader.key("indices", false)?;
            let indices = reader.array(20, c).await?;
            reader.key("values", false)?;
            let values = reader.bits(c).await?;
            reader.token(b'}')?;
            Self::Sparse {
                rows,
                columns,
                pointers,
                indices,
                values,
            }
        };
        reader.token(b'}')?;
        Ok(value)
    }
    pub(in super::super) fn into_value(self) -> BuiltinResult<Value> {
        let invalid = |detail: String| super::super::checkpoint::error(detail);
        match self {
            Self::Scalar(value) => Ok(Value::Num(f64::from_bits(value))),
            Self::Dense { shape, values } => {
                if shape.iter().try_fold(1usize, |n, &dim| n.checked_mul(dim)) != Some(values.len())
                {
                    return Err(invalid(
                        "checkpoint dense header has an invalid extent".to_owned(),
                    ));
                }
                Tensor::new(values, shape)
                    .map(Value::Tensor)
                    .map_err(invalid)
            }
            Self::Sparse {
                rows,
                columns,
                pointers,
                indices,
                values,
            } => {
                if columns.checked_add(1) != Some(pointers.len())
                    || pointers.first() != Some(&0)
                    || pointers.last() != Some(&values.len())
                    || indices.len() != values.len()
                {
                    return Err(invalid(
                        "checkpoint sparse header has an invalid extent".to_owned(),
                    ));
                }
                for column in 0..columns {
                    let (first, end) = (pointers[column], pointers[column + 1]);
                    if first > end
                        || end > values.len()
                        || indices[first..end].iter().any(|&row| row >= rows)
                        || indices[first..end]
                            .windows(2)
                            .any(|pair| pair[0] >= pair[1])
                    {
                        return Err(invalid("checkpoint sparse column is invalid".to_owned()));
                    }
                }
                SparseTensor::new(rows, columns, pointers, indices, values)
                    .map(Value::SparseTensor)
                    .map_err(invalid)
            }
        }
    }
}

pub(in super::super) struct BorrowedMatrix<'a>(pub &'a DoubleArray);
impl BorrowedMatrix<'_> {
    pub(in super::super) async fn write(
        &self,
        writer: &mut super::super::checkpoint::json::Writer,
        c: &mut crate::execution::value_codec::selected_json::Cooperation<'_>,
    ) -> BuiltinResult<()> {
        writer.token(b"{")?;
        match &self.0 .0 {
            Storage::Scalar(value) => {
                writer.key("Scalar", true)?;
                writer.metadata(&value.to_bits(), c).await?;
            }
            Storage::Dense(value) => {
                writer.key("Dense", true)?;
                writer.token(b"{")?;
                writer.key("shape", true)?;
                writer.array(value.shape(), c).await?;
                writer.key("values", false)?;
                writer
                    .array(
                        value
                            .as_f64_slice()
                            .expect("prepared double backing")
                            .iter()
                            .map(|value| value.to_bits()),
                        c,
                    )
                    .await?;
                writer.token(b"}")?;
            }
            Storage::Sparse(value) => {
                writer.key("Sparse", true)?;
                writer.token(b"{")?;
                writer.key("rows", true)?;
                writer.metadata(&value.rows, c).await?;
                writer.key("columns", false)?;
                writer.metadata(&value.cols, c).await?;
                writer.key("pointers", false)?;
                writer.array(&value.col_ptrs, c).await?;
                writer.key("indices", false)?;
                writer.array(&value.row_indices, c).await?;
                writer.key("values", false)?;
                writer
                    .array(
                        value
                            .as_f64_slice()
                            .expect("prepared double backing")
                            .iter()
                            .map(|value| value.to_bits()),
                        c,
                    )
                    .await?;
                writer.token(b"}")?;
            }
        }
        writer.token(b"}")
    }
}

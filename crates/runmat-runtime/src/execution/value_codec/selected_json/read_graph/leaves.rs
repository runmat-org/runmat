//! Incremental numeric/text graph leaves. Reconstruction uses the existing
//! typed payloads; no tensor/provider/object is materialized by this reader.
use super::*;
pub(super) async fn payload(
    reader: &mut Reader,
    budget: &mut Budget,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<ValuePayload> {
    reader.token(b'{')?;
    reader.key("form", true)?;
    if reader.string_limited(16, c).await? != "inline" {
        return Err(error(
            "sessionless capture contains an external service reference",
        ));
    }
    reader.key("value", false)?;
    let value = inline(reader, budget, c).await?;
    reader.token(b'}')?;
    Ok(ValuePayload::Inline(Box::new(value)))
}
async fn inline(
    reader: &mut Reader,
    budget: &mut Budget,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<InlineValue> {
    reader.token(b'{')?;
    reader.key("type", true)?;
    let kind = reader.string_limited(64, c).await?;
    if kind == "null" {
        reader.token(b'}')?;
        return Ok(InlineValue::Null);
    }
    reader.key("value", false)?;
    budget.backing(match kind.as_str() {
        "logical" | "i8" | "u8" => 1,
        "i16" | "u16" => 2,
        "i32" | "u32" => 4,
        "f64_bits" | "i64" | "u64" => 8,
        "complex_f64_bits" => 16,
        _ => 0,
    })?;
    let value = match kind.as_str() {
        "logical" => InlineValue::Logical(reader.metadata(5, c).await?),
        "f64_bits" => InlineValue::F64Bits(reader.metadata(20, c).await?),
        "i8" => InlineValue::I8(reader.metadata(4, c).await?),
        "i16" => InlineValue::I16(reader.metadata(6, c).await?),
        "i32" => InlineValue::I32(reader.metadata(11, c).await?),
        "i64" => InlineValue::I64(reader.metadata(20, c).await?),
        "u8" => InlineValue::U8(reader.metadata(3, c).await?),
        "u16" => InlineValue::U16(reader.metadata(5, c).await?),
        "u32" => InlineValue::U32(reader.metadata(10, c).await?),
        "u64" => InlineValue::U64(reader.metadata(20, c).await?),
        "complex_f64_bits" => {
            reader.token(b'{')?;
            reader.key("real", true)?;
            let real = reader.metadata(20, c).await?;
            reader.key("imaginary", false)?;
            let imaginary = reader.metadata(20, c).await?;
            reader.token(b'}')?;
            InlineValue::ComplexF64Bits { real, imaginary }
        }
        "string" => InlineValue::String(runmat_execution::value::StringElement(
            budget.optional_text(reader, c).await?,
        )),
        "char" => {
            reader.token(b'{')?;
            reader.key("encoding", true)?;
            let encoding = reader.metadata(64, c).await?;
            reader.key("shape", false)?;
            let shape = budget.shape(reader, c).await?;
            reader.key("code_units", false)?;
            let code_units = budget.array(reader, 5, 1, 2, c).await?;
            reader.token(b'}')?;
            InlineValue::Char {
                encoding,
                shape,
                code_units,
            }
        }
        "string_array" => {
            reader.token(b'{')?;
            reader.key("shape", true)?;
            let shape = budget.shape(reader, c).await?;
            reader.key("values", false)?;
            reader.token(b'[')?;
            let mut values = Vec::new();
            while !reader.is(b']') {
                budget.elements(1)?;
                if !values.is_empty() {
                    reader.token(b',')?;
                }
                values.try_reserve(1).map_err(|source| {
                    with_source("selected string-array allocation failed", source)
                })?;
                values.push(runmat_execution::value::StringElement(
                    budget.optional_text(reader, c).await?,
                ));
            }
            reader.token(b']')?;
            reader.token(b'}')?;
            InlineValue::StringArray { shape, values }
        }
        "dense" => {
            reader.token(b'{')?;
            reader.key("element_type", true)?;
            let element_type = reader.metadata(32, c).await?;
            reader.key("shape", false)?;
            let shape = budget.shape(reader, c).await?;
            let mut elements = 1u64;
            for &dimension in &shape {
                elements = elements.checked_mul(dimension).ok_or_else(|| {
                    with_source(
                        "selected dense shape element extent overflow",
                        runmat_execution::ContractError::Invalid {
                            field: "shape",
                            reason: "element count overflow".into(),
                        },
                    )
                })?;
                c.completed(1).await?;
            }
            budget.elements(elements)?;
            reader.key("little_endian_data", false)?;
            let little_endian_data = budget.bytes(reader, c).await?;
            reader.token(b'}')?;
            InlineValue::Dense(DenseValue {
                element_type,
                shape,
                little_endian_data,
            })
        }
        "sparse" => {
            reader.token(b'{')?;
            reader.key("element_type", true)?;
            let element_type = reader.metadata(32, c).await?;
            reader.key("rows", false)?;
            let rows = reader.metadata(20, c).await?;
            reader.key("columns", false)?;
            let columns = reader.metadata(20, c).await?;
            reader.key("column_offsets", false)?;
            let column_offsets = budget.array(reader, 20, 0, 8, c).await?;
            reader.key("row_indices", false)?;
            let row_indices = budget.array(reader, 20, 1, 8, c).await?;
            reader.key("little_endian_data", false)?;
            let little_endian_data = budget.bytes(reader, c).await?;
            reader.token(b'}')?;
            InlineValue::Sparse(SparseValue {
                element_type,
                rows,
                columns,
                column_offsets,
                row_indices,
                little_endian_data,
            })
        }
        "exception" => {
            InlineValue::Exception(super::exception::exception(reader, budget, 0, c).await?)
        }
        _ => {
            return Err(error(
                "recursive or durable value is hidden in a graph leaf",
            ))
        }
    };
    reader.token(b'}')?;
    Ok(value)
}
pub(super) async fn texts(
    reader: &mut Reader,
    budget: &mut Budget,
    c: &mut Cooperation<'_>,
) -> BuiltinResult<Vec<String>> {
    reader.token(b'[')?;
    let mut values = Vec::new();
    while !reader.is(b']') {
        budget.elements(1)?;
        if !values.is_empty() {
            reader.token(b',')?;
        }
        values
            .try_reserve(1)
            .map_err(|source| with_source("selected text-array allocation failed", source))?;
        values.push(budget.text(reader, c).await?);
    }
    reader.token(b']')?;
    Ok(values)
}

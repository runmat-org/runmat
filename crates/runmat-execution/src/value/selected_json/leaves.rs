//! The already-admitted graph-leaf grammar, with exact stored numeric bits.
use super::*;
impl<'a> SelectedGraphJsonCursor<'a> {
    pub(super) fn payload(
        &mut self,
        value: &'a ValuePayload,
    ) -> Result<(), SelectedGraphJsonError> {
        match value {
            ValuePayload::Inline(value) => self.object([
                ("form", Task::Text("inline")),
                ("value", Task::Inline(value)),
            ]),
            // Sessionless portable code cannot turn old service handles into
            // fresh resident/distributed/reference authority on restore.
            _ => Err(Self::invalid(
                "sessionless capture contains an external service reference",
            )),
        }
    }
    fn tagged(&mut self, tag: &'static str, value: Task<'a>) -> Result<(), SelectedGraphJsonError> {
        self.object([("type", Task::Text(tag)), ("value", value)])
    }
    pub(super) fn inline(&mut self, value: &'a InlineValue) -> Result<(), SelectedGraphJsonError> {
        match value {
            InlineValue::Null => self.object([("type", Task::Text("null"))]),
            InlineValue::Logical(value) => self.tagged("logical", Task::Atom(value)),
            InlineValue::F64Bits(value) => self.tagged("f64_bits", Task::Atom(value)),
            InlineValue::I8(value) => self.tagged("i8", Task::Atom(value)),
            InlineValue::I16(value) => self.tagged("i16", Task::Atom(value)),
            InlineValue::I32(value) => self.tagged("i32", Task::Atom(value)),
            InlineValue::I64(value) => self.tagged("i64", Task::Atom(value)),
            InlineValue::U8(value) => self.tagged("u8", Task::Atom(value)),
            InlineValue::U16(value) => self.tagged("u16", Task::Atom(value)),
            InlineValue::U32(value) => self.tagged("u32", Task::Atom(value)),
            InlineValue::U64(value) => self.tagged("u64", Task::Atom(value)),
            InlineValue::ComplexF64Bits { real, imaginary } => {
                self.push(Task::Raw(b"}"))?;
                self.object([
                    ("real", Task::Atom(real)),
                    ("imaginary", Task::Atom(imaginary)),
                ])?;
                self.push(Task::Raw(b"{\"type\":\"complex_f64_bits\",\"value\":"))
            }
            InlineValue::String(value) => self.tagged("string", Task::OptionalText(value.as_str())),
            InlineValue::Char {
                encoding,
                shape,
                code_units,
            } => {
                self.push(Task::Raw(b"}"))?;
                self.object([
                    ("encoding", Task::Atom(encoding)),
                    ("shape", Task::Array(Array::U64(shape))),
                    ("code_units", Task::Array(Array::U16(code_units))),
                ])?;
                self.push(Task::Raw(b"{\"type\":\"char\",\"value\":"))
            }
            InlineValue::StringArray { shape, values } => {
                self.push(Task::Raw(b"}"))?;
                self.object([
                    ("shape", Task::Array(Array::U64(shape))),
                    ("values", Task::Array(Array::Strings(values))),
                ])?;
                self.push(Task::Raw(b"{\"type\":\"string_array\",\"value\":"))
            }
            InlineValue::Dense(value) => {
                self.push(Task::Raw(b"}"))?;
                self.object([
                    ("element_type", Task::Atom(&value.element_type)),
                    ("shape", Task::Array(Array::U64(&value.shape))),
                    (
                        "little_endian_data",
                        Task::Array(Array::U8(&value.little_endian_data)),
                    ),
                ])?;
                self.push(Task::Raw(b"{\"type\":\"dense\",\"value\":"))
            }
            InlineValue::Sparse(value) => {
                self.push(Task::Raw(b"}"))?;
                self.object([
                    ("element_type", Task::Atom(&value.element_type)),
                    ("rows", Task::Atom(&value.rows)),
                    ("columns", Task::Atom(&value.columns)),
                    (
                        "column_offsets",
                        Task::Array(Array::U64(&value.column_offsets)),
                    ),
                    ("row_indices", Task::Array(Array::U64(&value.row_indices))),
                    (
                        "little_endian_data",
                        Task::Array(Array::U8(&value.little_endian_data)),
                    ),
                ])?;
                self.push(Task::Raw(b"{\"type\":\"sparse\",\"value\":"))
            }
            InlineValue::Exception(value) => self.tagged("exception", Task::Exception(value)),
            _ => Err(Self::invalid(
                "recursive or durable value is hidden in a graph leaf",
            )),
        }
    }
}

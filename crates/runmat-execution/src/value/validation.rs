use super::{
    DenseValue, ExceptionValue, InlineValue, RegisteredData, StructField, ValueLimits,
    ValuePayload, ValueRef, ValueRefKind,
};
use crate::{schema::VALUE_PAYLOAD_SCHEMA_V1, ContractError};

pub(super) fn validate(payload: &ValuePayload, limits: ValueLimits) -> Result<(), ContractError> {
    ValidationBudget {
        limits,
        nodes: 0,
        inline_bytes: 0,
        elements: 0,
        text_bytes: 0,
    }
    .payload(payload, 0)
}

struct ValidationBudget {
    limits: ValueLimits,
    nodes: u64,
    inline_bytes: u64,
    elements: u64,
    text_bytes: u64,
}

impl ValidationBudget {
    fn payload(&mut self, payload: &ValuePayload, depth: u16) -> Result<(), ContractError> {
        self.node(depth)?;
        match payload {
            ValuePayload::Inline(value) => self.inline(value, depth),
            ValuePayload::Object(reference) => self.reference(reference),
        }
    }

    fn node(&mut self, depth: u16) -> Result<(), ContractError> {
        if depth > self.limits.max_depth {
            return Err(ContractError::Limit {
                field: "value depth",
                limit: self.limits.max_depth.into(),
            });
        }
        self.nodes = self.nodes.saturating_add(1);
        if self.nodes > self.limits.max_nodes {
            return Err(ContractError::Limit {
                field: "value nodes",
                limit: self.limits.max_nodes,
            });
        }
        Ok(())
    }

    fn inline(&mut self, value: &InlineValue, depth: u16) -> Result<(), ContractError> {
        match value {
            InlineValue::Null => {}
            InlineValue::Logical(_) | InlineValue::I8(_) | InlineValue::U8(_) => self.bytes(1)?,
            InlineValue::I16(_) | InlineValue::U16(_) => self.bytes(2)?,
            InlineValue::I32(_) | InlineValue::U32(_) => self.bytes(4)?,
            InlineValue::F64Bits(_) | InlineValue::I64(_) | InlineValue::U64(_) => self.bytes(8)?,
            InlineValue::ComplexF64Bits { .. } => self.bytes(16)?,
            InlineValue::String(value) => self.text(value)?,
            InlineValue::Char { shape, code_points } => {
                let elements = checked_product(shape)?;
                if elements != code_points.len() as u64 {
                    return Err(ContractError::invalid(
                        "char payload",
                        "shape does not match character count",
                    ));
                }
                self.elements(elements)?;
                self.bytes((code_points.len() as u64).saturating_mul(4))?;
                if code_points
                    .iter()
                    .any(|scalar| char::from_u32(*scalar).is_none())
                {
                    return Err(ContractError::invalid(
                        "char payload",
                        "contains an invalid Unicode scalar",
                    ));
                }
            }
            InlineValue::StringArray { shape, values } => {
                let elements = checked_product(shape)?;
                if elements != values.len() as u64 {
                    return Err(ContractError::invalid(
                        "string-array payload",
                        "shape does not match string count",
                    ));
                }
                self.elements(elements)?;
                for value in values {
                    self.text(value)?;
                }
            }
            InlineValue::Dense(value) => self.dense(value)?,
            InlineValue::Sparse(value) => self.sparse(value)?,
            InlineValue::Cell { shape, values } => {
                let elements = checked_product(shape)?;
                if elements != values.len() as u64 {
                    return Err(ContractError::invalid(
                        "cell payload",
                        "shape does not match cell count",
                    ));
                }
                self.elements(elements)?;
                for value in values {
                    self.payload(value, depth + 1)?;
                }
            }
            InlineValue::OutputList(values) => {
                self.elements(values.len() as u64)?;
                for value in values {
                    self.payload(value, depth + 1)?;
                }
            }
            InlineValue::Struct(fields) => self.fields(fields, depth)?,
            InlineValue::Symbolic(value) | InlineValue::ImmutableValueClass(value) => {
                self.registered(value, depth)?;
            }
            InlineValue::Exception(value) => self.exception(value, depth)?,
            InlineValue::Callable(value) => {
                self.text(&value.owner_identity)?;
                self.text(&value.qualified_name)?;
                value.validate_identity()?;
                for capture in &value.captures {
                    self.payload(capture, depth + 1)?;
                }
            }
        }
        Ok(())
    }

    fn dense(&mut self, value: &DenseValue) -> Result<(), ContractError> {
        let elements = checked_product(&value.shape)?;
        self.elements(elements)?;
        let expected = elements
            .checked_mul(value.element_type.byte_width())
            .ok_or_else(|| ContractError::invalid("dense payload", "byte length overflow"))?;
        if expected != value.little_endian_data.len() as u64 {
            return Err(ContractError::invalid(
                "dense payload",
                "shape and element type do not match data length",
            ));
        }
        self.bytes(expected)
    }

    fn sparse(&mut self, value: &super::SparseValue) -> Result<(), ContractError> {
        if value.column_offsets.len() as u64 != value.columns.saturating_add(1)
            || value.column_offsets.first().copied() != Some(0)
            || value
                .column_offsets
                .windows(2)
                .any(|pair| pair[0] > pair[1])
        {
            return Err(ContractError::invalid(
                "sparse payload",
                "column offsets are not canonical CSC offsets",
            ));
        }
        let nonzero = value.row_indices.len() as u64;
        if value.column_offsets.last().copied() != Some(nonzero)
            || value.row_indices.iter().any(|row| *row >= value.rows)
            || value.little_endian_data.len() as u64
                != nonzero.saturating_mul(value.element_type.byte_width())
        {
            return Err(ContractError::invalid(
                "sparse payload",
                "indices or data length do not match nonzero count",
            ));
        }
        self.elements(nonzero)?;
        self.bytes(value.little_endian_data.len() as u64)
    }

    fn fields(&mut self, fields: &[StructField], depth: u16) -> Result<(), ContractError> {
        self.named_fields(
            fields.len(),
            fields
                .iter()
                .map(|field| (field.name.as_str(), &field.value)),
            "struct fields",
            depth,
        )
    }

    fn registered(&mut self, value: &RegisteredData, depth: u16) -> Result<(), ContractError> {
        self.text(&value.type_identity)?;
        if value.schema_version == 0 {
            return Err(ContractError::invalid(
                "registered value",
                "schema version must be non-zero",
            ));
        }
        self.named_fields(
            value.fields.len(),
            value
                .fields
                .iter()
                .map(|field| (field.name.as_str(), &field.value)),
            "registered fields",
            depth,
        )
    }

    fn exception(&mut self, value: &ExceptionValue, depth: u16) -> Result<(), ContractError> {
        self.text(&value.identifier)?;
        self.text(&value.message)?;
        for frame in &value.stack {
            self.text(frame)?;
        }
        self.elements(value.causes.len() as u64)?;
        for cause in &value.causes {
            self.node(depth + 1)?;
            self.exception(cause, depth + 1)?;
        }
        Ok(())
    }

    fn reference(&mut self, reference: &ValueRef) -> Result<(), ContractError> {
        if reference.schema_version != VALUE_PAYLOAD_SCHEMA_V1 {
            return Err(ContractError::UnsupportedSchema {
                actual: reference.schema_version,
                supported: VALUE_PAYLOAD_SCHEMA_V1,
            });
        }
        self.text(&reference.media_type)?;
        self.text(&reference.value_schema)?;
        self.text(&reference.authorization_scope)?;
        match (reference.kind, &reference.resident_fence) {
            (ValueRefKind::ResidentObject, None) => Err(ContractError::invalid(
                "resident value",
                "resident references require a worker/node/process fence",
            )),
            (ValueRefKind::ResidentObject, Some(fence)) => {
                if let Some(device) = &fence.device_identity {
                    self.text(device)?;
                }
                Ok(())
            }
            (_, Some(_)) => Err(ContractError::invalid(
                "value reference",
                "only resident references may carry a resident fence",
            )),
            (_, None) => Ok(()),
        }
    }

    fn bytes(&mut self, bytes: u64) -> Result<(), ContractError> {
        self.inline_bytes = self.inline_bytes.saturating_add(bytes);
        if self.inline_bytes > self.limits.max_inline_bytes {
            return Err(ContractError::Limit {
                field: "inline value bytes",
                limit: self.limits.max_inline_bytes,
            });
        }
        Ok(())
    }

    fn elements(&mut self, elements: u64) -> Result<(), ContractError> {
        self.elements = self.elements.saturating_add(elements);
        if self.elements > self.limits.max_elements {
            return Err(ContractError::Limit {
                field: "value elements",
                limit: self.limits.max_elements,
            });
        }
        Ok(())
    }

    fn text(&mut self, value: &str) -> Result<(), ContractError> {
        self.text_bytes = self.text_bytes.saturating_add(value.len() as u64);
        if self.text_bytes > self.limits.max_text_bytes {
            return Err(ContractError::Limit {
                field: "value text bytes",
                limit: self.limits.max_text_bytes,
            });
        }
        Ok(())
    }

    fn named_fields<'a>(
        &mut self,
        count: usize,
        fields: impl IntoIterator<Item = (&'a str, &'a ValuePayload)>,
        label: &'static str,
        depth: u16,
    ) -> Result<(), ContractError> {
        if count > self.limits.max_fields as usize {
            return Err(ContractError::Limit {
                field: label,
                limit: self.limits.max_fields.into(),
            });
        }
        let mut previous: Option<&str> = None;
        for (name, value) in fields {
            self.text(name)?;
            if previous.is_some_and(|previous| previous >= name) {
                return Err(ContractError::invalid(
                    label,
                    "field names must be unique and sorted",
                ));
            }
            previous = Some(name);
            self.payload(value, depth + 1)?;
        }
        Ok(())
    }
}

fn checked_product(shape: &[u64]) -> Result<u64, ContractError> {
    shape.iter().try_fold(1_u64, |total, extent| {
        total
            .checked_mul(*extent)
            .ok_or_else(|| ContractError::invalid("shape", "element count overflow"))
    })
}

use super::{
    DenseValue, ExceptionValue, InlineValue, RegisteredData, StructField, ValueLimits,
    ValuePayload, ValueRef, ValueRefKind, ValueTransportContext,
};
use crate::{schema::VALUE_PAYLOAD_SCHEMA_VERSION, ContractError};

mod durable;

pub(super) fn validate(payload: &ValuePayload, limits: ValueLimits) -> Result<(), ContractError> {
    ValidationBudget {
        limits,
        transport: None,
        durable: false,
        program_sources: None,
        nodes: 0,
        inline_bytes: 0,
        elements: 0,
        text_bytes: 0,
    }
    .payload(payload, 0)
}

pub(super) fn validate_for_program_sources(
    payload: &ValuePayload,
    limits: ValueLimits,
    source_map: Option<&crate::ExecutableSourceMapV2>,
) -> Result<(), ContractError> {
    ValidationBudget {
        limits,
        transport: None,
        durable: false,
        program_sources: Some(source_map),
        nodes: 0,
        inline_bytes: 0,
        elements: 0,
        text_bytes: 0,
    }
    .payload(payload, 0)
}

pub(super) fn validate_for_durable_storage(
    payload: &ValuePayload,
    limits: ValueLimits,
) -> Result<(), ContractError> {
    ValidationBudget {
        limits,
        transport: None,
        durable: true,
        program_sources: None,
        nodes: 0,
        inline_bytes: 0,
        elements: 0,
        text_bytes: 0,
    }
    .payload(payload, 0)
}

pub(super) fn validate_for_transport(
    payload: &ValuePayload,
    limits: ValueLimits,
    context: ValueTransportContext<'_>,
) -> Result<(), ContractError> {
    ValidationBudget {
        limits,
        transport: Some(context),
        durable: false,
        program_sources: None,
        nodes: 0,
        inline_bytes: 0,
        elements: 0,
        text_bytes: 0,
    }
    .payload(payload, 0)
}

struct ValidationBudget<'a> {
    limits: ValueLimits,
    transport: Option<ValueTransportContext<'a>>,
    durable: bool,
    /// None is ordinary local validation; Some(None) is a portable route
    /// without a source map and must reject source-bearing frames.
    program_sources: Option<Option<&'a crate::ExecutableSourceMapV2>>,
    nodes: u64,
    inline_bytes: u64,
    elements: u64,
    text_bytes: u64,
}

impl ValidationBudget<'_> {
    fn payload(&mut self, payload: &ValuePayload, depth: u16) -> Result<(), ContractError> {
        self.node(depth)?;
        match payload {
            ValuePayload::Inline(value) => self.inline(value, depth),
            ValuePayload::Object(_) | ValuePayload::Distributed(_) | ValuePayload::Composite(_)
                if self.durable =>
            {
                Err(ContractError::invalid(
                    "durable preference value",
                    "execution references are not durable",
                ))
            }
            ValuePayload::Object(reference) => self.reference(reference),
            ValuePayload::Distributed(handle) => handle.validate(),
            ValuePayload::Composite(handle) => handle.validate(),
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
            InlineValue::Null
            | InlineValue::Callable(_)
            | InlineValue::Symbolic(_)
            | InlineValue::ImmutableValueClass(_)
                if self.durable =>
            {
                return Err(ContractError::invalid(
                    "durable storage value",
                    "context-bound or unregistered value is not durable",
                ));
            }
            InlineValue::Null => {}
            InlineValue::Logical(_) | InlineValue::I8(_) | InlineValue::U8(_) => self.bytes(1)?,
            InlineValue::I16(_) | InlineValue::U16(_) => self.bytes(2)?,
            InlineValue::I32(_) | InlineValue::U32(_) => self.bytes(4)?,
            InlineValue::F64Bits(_) | InlineValue::I64(_) | InlineValue::U64(_) => self.bytes(8)?,
            InlineValue::ComplexF64Bits { .. } => self.bytes(16)?,
            InlineValue::String(value) => self.text(value)?,
            InlineValue::Char { shape, code_points } => {
                let elements = if shape.is_empty() {
                    0
                } else {
                    checked_product(shape)?
                };
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
            InlineValue::Struct(fields) => self.fields(fields, depth)?,
            InlineValue::DurableOrderedStruct(fields) if self.durable => {
                self.ordered_fields(fields, depth)?;
            }
            InlineValue::DurableOrderedStruct(_) => {
                return Err(ContractError::invalid(
                    "V2 execution value",
                    "durable ordered structures are not admitted by V2 transport",
                ));
            }
            InlineValue::DurablePlainObject(value) if self.durable => {
                self.plain_object(value, depth)?
            }
            InlineValue::DurableStructArray(value) if self.durable => {
                self.struct_array(value, depth)?
            }
            InlineValue::DurableObjectArray(value) if self.durable => {
                self.object_array(value, depth)?
            }
            InlineValue::DurablePlainObject(_)
            | InlineValue::DurableStructArray(_)
            | InlineValue::DurableObjectArray(_) => {
                return Err(ContractError::invalid(
                    "V2 execution value",
                    "durable preference aggregate tags are not admitted by V2 transport",
                ));
            }
            InlineValue::Symbolic(value) | InlineValue::ImmutableValueClass(value) => {
                self.registered(value, depth)?;
            }
            InlineValue::Exception(value) => self.exception(value, depth)?,
            InlineValue::Callable(value) => {
                self.bytes(match value.owner {
                    super::CallableOwner::Bound(_) => 1 + 32 + 4,
                    super::CallableOwner::Workspace
                    | super::CallableOwner::External
                    | super::CallableOwner::Method => 1,
                })?;
                self.text(&value.qualified_name)?;
                if let Some(path) = &value.source_file {
                    self.text(path)?;
                }
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
        if let Some(runtime) = &value.runtime {
            if let Some(builtin) = &runtime.builtin {
                self.text(builtin)?;
            }
            if let Some(task_id) = &runtime.task_id {
                self.text(task_id)?;
            }
            self.elements(runtime.call_frames.len() as u64)?;
            for frame in &runtime.call_frames {
                self.text(&frame.function)?;
                if let (Some(source_map), Some(source_id)) =
                    (self.program_sources, frame.program_source_id)
                {
                    if source_map.is_none_or(|map| !map.contains_source(source_id)) {
                        return Err(ContractError::invalid(
                            "portable exception source",
                            "source ordinal is absent from the selected artifact map",
                        ));
                    }
                }
            }
            self.elements(runtime.call_stack.len() as u64)?;
            for frame in &runtime.call_stack {
                self.text(frame)?;
            }
            if let Some(phase) = &runtime.phase {
                self.text(phase)?;
            }
        }
        self.elements(value.causes.len() as u64)?;
        for cause in &value.causes {
            self.node(depth + 1)?;
            self.exception(cause, depth + 1)?;
        }
        Ok(())
    }

    fn reference(&mut self, reference: &ValueRef) -> Result<(), ContractError> {
        if reference.schema_version != VALUE_PAYLOAD_SCHEMA_VERSION {
            return Err(ContractError::UnsupportedSchema {
                actual: reference.schema_version,
                supported: VALUE_PAYLOAD_SCHEMA_VERSION,
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
                fence.validate()?;
                match self.transport {
                    None => Ok(()),
                    Some(ValueTransportContext::Portable) => Err(ContractError::invalid(
                        "resident value",
                        "process- or device-resident references cannot cross a portable transport boundary",
                    )),
                    Some(ValueTransportContext::Resident(authority))
                        if authority.authorizes(fence) =>
                    {
                        Ok(())
                    }
                    Some(ValueTransportContext::Resident(_)) => Err(ContractError::invalid(
                        "resident value",
                        "worker, node lease, process generation, attempt, or accelerator lease is stale or different",
                    )),
                }
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

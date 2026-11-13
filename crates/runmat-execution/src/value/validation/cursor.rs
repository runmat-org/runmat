//! The single framed validator used by synchronous and cooperative frontends.

use super::super::{
    CallableOwner, CharacterEncoding, DurableDecompositionPayload, DurableNumericFactor,
    DurableObjectArray, DurablePlainObject, DurableStructArray, ElementType, ExceptionFrameSource,
    ExceptionValue, InlineValue, RegisteredField, SparseValue, StringElement, StructField,
    ValueLimitBudget, ValueLimits, ValuePayload, ValueRef, ValueRefKind, ValueTransportContext,
};
use crate::{schema::VALUE_PAYLOAD_SCHEMA_VERSION, ContractError};
use std::{collections::HashSet, num::NonZeroUsize};

mod budget;
mod durable;
mod exception;
mod policy;
mod sparse;
#[cfg(test)]
mod tests;

/// Result of one bounded validation advance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueValidationProgress {
    /// Validation has more structural work.
    Pending { completed_work: usize },
    /// The complete batch passed validation.
    Complete { completed_work: usize },
}

#[derive(Clone, Copy)]
enum Sources<'a> {
    Local,
    Missing,
    Map(&'a crate::ExecutableSourceMapV2),
}

struct Budget<'a> {
    limits: ValueLimits,
    transport: Option<ValueTransportContext<'a>>,
    durable: bool,
    detached_frames: bool,
    sources: Sources<'a>,
    usage: ValueLimitBudget,
}

/// Incremental canonical validator for one aggregate of portable values.
///
/// Aggregate counters span every member supplied at construction. The cursor
/// contains no executor or cancellation policy; cooperative callers checkpoint
/// `completed_work`, while synchronous callers drain the same cursor.
pub struct ValuePayloadValidationCursor<'a> {
    budget: Budget<'a>,
    tasks: Vec<Task<'a>>,
}

#[derive(Clone, Copy)]
enum Fields<'a> {
    Struct(&'a [StructField]),
    Registered(&'a [RegisteredField]),
}
impl<'a> Fields<'a> {
    fn len(self) -> usize {
        match self {
            Self::Struct(v) => v.len(),
            Self::Registered(v) => v.len(),
        }
    }
    fn get(self, i: usize) -> Option<(&'a str, &'a ValuePayload)> {
        match self {
            Self::Struct(v) => v.get(i).map(|f| (f.name.as_str(), &f.value)),
            Self::Registered(v) => v.get(i).map(|f| (f.name.as_str(), &f.value)),
        }
    }
}

enum AfterShape<'a> {
    Char(&'a [u16]),
    Strings(&'a [StringElement]),
    Dense(ElementType, &'a [u8]),
    Cell(&'a [ValuePayload], u16),
    StructArray(&'a DurableStructArray, u16),
    ObjectArray(&'a DurableObjectArray, u16),
}

enum Task<'a> {
    Payload(&'a ValuePayload, u16),
    Inline(&'a InlineValue, u16),
    Many(&'a [ValuePayload], usize, u16),
    Texts(&'a [StringElement], usize),
    Chars(&'a [u16], usize),
    Shape(&'a [u64], usize, u64, AfterShape<'a>),
    SparseOffsets(&'a SparseValue, usize),
    SparseRows(&'a SparseValue, usize),
    SortedFields(Fields<'a>, usize, u16, &'static str),
    OrderedFields(&'a [StructField], usize, u16, HashSet<&'a str>),
    Dynamic(&'a DurablePlainObject, usize),
    FactorIndices(&'a [u64], usize),
    FactorBlock(&'a DurableNumericFactor, usize, usize, usize, bool),
    FactorDense(&'a DurableNumericFactor, usize, usize),
    FactorMap(&'a DurableNumericFactor, usize, usize),
    FactorScale(&'a DurableNumericFactor, usize),
    StructArray(&'a DurableStructArray, u64, usize, u16, HashSet<&'a str>),
    ObjectArray(&'a DurableObjectArray, usize, u16),
    ExceptionStack(&'a ExceptionValue, usize, u16),
    ExceptionFrames(&'a ExceptionValue, usize, u16),
    ExceptionCallStack(&'a ExceptionValue, usize, u16),
    ExceptionCauses(&'a [ExceptionValue], usize, u16),
}

impl<'a> ValuePayloadValidationCursor<'a> {
    pub(super) fn for_graph_leaves(
        leaves: &[(&'a ValuePayload, u16)],
        limits: ValueLimits,
        map: Option<&'a crate::ExecutableSourceMapV2>,
    ) -> Self {
        let mut cursor = Self::make(
            &[],
            limits,
            false,
            map.map_or(Sources::Missing, Sources::Map),
            None,
            true,
        );
        cursor.tasks.clear();
        cursor.tasks.extend(
            leaves
                .iter()
                .rev()
                .map(|(value, depth)| Task::Payload(value, *depth)),
        );
        cursor
    }
    /// Validate execution payloads under one shared budget.
    pub fn new(values: &'a [ValuePayload], limits: ValueLimits) -> Self {
        Self::make(values, limits, false, Sources::Local, None, true)
    }
    /// Validate durable payloads under one shared budget.
    pub fn for_durable_storage(values: &'a [ValuePayload], limits: ValueLimits) -> Self {
        Self::make(values, limits, true, Sources::Local, None, true)
    }
    /// Validate exception source ordinals against the selected source map.
    pub fn for_program_sources(
        values: &'a [ValuePayload],
        limits: ValueLimits,
        map: Option<&'a crate::ExecutableSourceMapV2>,
    ) -> Self {
        Self::make(
            values,
            limits,
            false,
            map.map_or(Sources::Missing, Sources::Map),
            None,
            false,
        )
    }
    /// The V8 graph source contract also admits detached presentation frames.
    pub fn for_graph_sources(
        values: &'a [ValuePayload],
        limits: ValueLimits,
        map: Option<&'a crate::ExecutableSourceMapV2>,
    ) -> Self {
        Self::make(
            values,
            limits,
            false,
            map.map_or(Sources::Missing, Sources::Map),
            None,
            true,
        )
    }
    /// Validate reference transport authority as well as payload structure.
    pub fn for_transport(
        values: &'a [ValuePayload],
        limits: ValueLimits,
        context: ValueTransportContext<'a>,
    ) -> Self {
        Self::make(values, limits, false, Sources::Local, Some(context), true)
    }
    fn make(
        values: &'a [ValuePayload],
        limits: ValueLimits,
        durable: bool,
        sources: Sources<'a>,
        transport: Option<ValueTransportContext<'a>>,
        detached_frames: bool,
    ) -> Self {
        Self {
            budget: Budget {
                limits,
                transport,
                durable,
                detached_frames,
                sources,
                usage: ValueLimitBudget::new(limits),
            },
            tasks: vec![Task::Many(values, 0, 0)],
        }
    }

    /// Perform at most `maximum` framed validation steps.
    pub fn advance(
        &mut self,
        maximum: NonZeroUsize,
    ) -> Result<ValueValidationProgress, ContractError> {
        let mut done = 0;
        while done < maximum.get() {
            let Some(task) = self.tasks.pop() else {
                return Ok(ValueValidationProgress::Complete {
                    completed_work: done,
                });
            };
            self.step(task)?;
            done += 1;
        }
        if self.tasks.is_empty() {
            Ok(ValueValidationProgress::Complete {
                completed_work: done,
            })
        } else {
            Ok(ValueValidationProgress::Pending {
                completed_work: done,
            })
        }
    }

    fn step(&mut self, task: Task<'a>) -> Result<(), ContractError> {
        match task {
            Task::Payload(v, d) => self.payload(v, d),
            Task::Inline(v, d) => self.inline(v, d),
            Task::Many(v, i, d) => {
                if let Some(x) = v.get(i) {
                    self.tasks.push(Task::Many(v, i + 1, d));
                    self.tasks.push(Task::Payload(x, d));
                }
                Ok(())
            }
            Task::Texts(v, i) => {
                if let Some(x) = v.get(i) {
                    if let Some(text) = x.as_str() {
                        self.budget.add_text(text)?;
                    }
                    self.tasks.push(Task::Texts(v, i + 1));
                }
                Ok(())
            }
            Task::Chars(v, i) => {
                if v.get(i).is_some() {
                    self.tasks.push(Task::Chars(v, i + 1));
                }
                Ok(())
            }
            Task::Shape(v, i, n, after) => self.shape_step(v, i, n, after),
            Task::SparseOffsets(v, i) => self.sparse_offset(v, i),
            Task::SparseRows(v, i) => self.sparse_row(v, i),
            Task::SortedFields(v, i, d, l) => self.sorted_field(v, i, d, l),
            Task::OrderedFields(v, i, d, n) => self.ordered_field(v, i, d, n),
            Task::Dynamic(v, i) => self.dynamic(v, i),
            Task::FactorIndices(v, i) => self.factor_index(v, i),
            Task::FactorBlock(v, b, c, e, d) => self.factor_block(v, b, c, e, d),
            Task::FactorDense(v, b, i) => self.factor_dense(v, b, i),
            Task::FactorMap(v, i, o) => self.factor_map(v, i, o),
            Task::FactorScale(v, i) => self.factor_scale(v, i),
            Task::StructArray(v, n, i, d, s) => self.struct_array_field(v, n, i, d, s),
            Task::ObjectArray(v, i, d) => self.object_array_item(v, i, d),
            Task::ExceptionStack(v, i, d) => self.exception_stack(v, i, d),
            Task::ExceptionFrames(v, i, d) => self.exception_frames(v, i, d),
            Task::ExceptionCallStack(v, i, d) => self.exception_call_stack(v, i, d),
            Task::ExceptionCauses(v, i, d) => self.exception_cause(v, i, d),
        }
    }

    fn payload(&mut self, value: &'a ValuePayload, depth: u16) -> Result<(), ContractError> {
        self.budget.node(depth)?;
        match value {
            ValuePayload::Inline(v) => self.tasks.push(Task::Inline(v, depth)),
            ValuePayload::Object(_) | ValuePayload::Distributed(_) | ValuePayload::Composite(_)
                if self.budget.durable =>
            {
                return Err(invalid(
                    "durable preference value",
                    "execution references are not durable",
                ))
            }
            ValuePayload::Object(v) => self.reference(v)?,
            ValuePayload::Distributed(v) => v.validate()?,
            ValuePayload::Composite(v) => v.validate()?,
        }
        Ok(())
    }

    fn inline(&mut self, v: &'a InlineValue, depth: u16) -> Result<(), ContractError> {
        if self.budget.durable
            && matches!(
                v,
                InlineValue::Null
                    | InlineValue::Callable(_)
                    | InlineValue::Symbolic(_)
                    | InlineValue::ImmutableValueClass(_)
            )
        {
            return Err(invalid(
                "durable storage value",
                "context-bound or unregistered value is not durable",
            ));
        }
        match v {
            InlineValue::Null => {}
            InlineValue::Logical(_) | InlineValue::I8(_) | InlineValue::U8(_) => {
                self.budget.add_bytes(1)?
            }
            InlineValue::I16(_) | InlineValue::U16(_) => self.budget.add_bytes(2)?,
            InlineValue::I32(_) | InlineValue::U32(_) => self.budget.add_bytes(4)?,
            InlineValue::F64Bits(_) | InlineValue::I64(_) | InlineValue::U64(_) => {
                self.budget.add_bytes(8)?
            }
            InlineValue::ComplexF64Bits { .. } => self.budget.add_bytes(16)?,
            InlineValue::String(x) => {
                if let Some(text) = x.as_str() {
                    self.budget.add_text(text)?;
                }
            }
            InlineValue::Char {
                encoding: CharacterEncoding::Utf16CodeUnitsV1,
                shape,
                code_units,
            } => self.shape(shape, AfterShape::Char(code_units)),
            InlineValue::StringArray { shape, values } => {
                self.shape(shape, AfterShape::Strings(values))
            }
            InlineValue::Dense(x) => self.shape(
                &x.shape,
                AfterShape::Dense(x.element_type, &x.little_endian_data),
            ),
            InlineValue::Sparse(x) => self.sparse(x)?,
            InlineValue::Cell { shape, values } => {
                self.shape(shape, AfterShape::Cell(values, depth))
            }
            InlineValue::Struct(x) => self.sorted(Fields::Struct(x), "struct fields", depth)?,
            InlineValue::DurableOrderedStruct(x) if self.budget.durable => {
                self.field_limit(x.len(), "ordered structure fields")?;
                self.tasks.push(Task::OrderedFields(
                    x,
                    0,
                    depth,
                    HashSet::with_capacity(x.len()),
                ));
            }
            InlineValue::DurableOrderedStruct(_) => {
                return Err(invalid(
                    "V2 execution value",
                    "durable ordered structures are not admitted by V2 transport",
                ))
            }
            InlineValue::DurablePlainObject(x) if self.budget.durable => {
                self.plain_object(x, depth)?
            }
            InlineValue::DurableStructArray(x) if self.budget.durable => {
                self.struct_array(x, depth)?
            }
            InlineValue::DurableObjectArray(x) if self.budget.durable => {
                self.object_array(x, depth)?
            }
            InlineValue::DurablePlainObject(_)
            | InlineValue::DurableStructArray(_)
            | InlineValue::DurableObjectArray(_) => {
                return Err(invalid(
                    "V2 execution value",
                    "durable preference aggregate tags are not admitted by V2 transport",
                ))
            }
            InlineValue::Symbolic(x) | InlineValue::ImmutableValueClass(x) => {
                self.budget.add_text(&x.type_identity)?;
                super::super::RegisteredData::validate_identity_parts(
                    &x.type_identity,
                    x.schema_version,
                )?;
                self.sorted(Fields::Registered(&x.fields), "registered fields", depth)?;
            }
            InlineValue::Exception(x) => {
                self.budget.add_text(&x.identifier)?;
                self.budget.add_text(&x.message)?;
                self.budget.add_elements(x.stack.len() as u64)?;
                self.tasks.push(Task::ExceptionStack(x, 0, depth));
            }
            InlineValue::Callable(x) => {
                self.budget
                    .add_bytes(if matches!(x.owner, CallableOwner::Bound(_)) {
                        37
                    } else {
                        1
                    })?;
                self.budget.add_text(&x.qualified_name)?;
                if let Some(p) = &x.source_file {
                    self.budget.add_text(p)?;
                }
                x.validate_identity()?;
                self.tasks.push(Task::Many(&x.captures, 0, depth + 1));
            }
        }
        Ok(())
    }

    fn shape(&mut self, shape: &'a [u64], after: AfterShape<'a>) {
        self.tasks.push(Task::Shape(shape, 0, 1, after));
    }
    fn shape_step(
        &mut self,
        shape: &'a [u64],
        i: usize,
        n: u64,
        after: AfterShape<'a>,
    ) -> Result<(), ContractError> {
        if let Some(x) = shape.get(i) {
            self.budget.add_bytes(8)?;
            let n = n
                .checked_mul(*x)
                .ok_or_else(|| invalid("shape", "element count overflow"))?;
            self.tasks.push(Task::Shape(shape, i + 1, n, after));
            return Ok(());
        }
        match after {
            AfterShape::Char(v) => {
                let n = if shape.is_empty() { 0 } else { n };
                if n != v.len() as u64 {
                    return Err(invalid(
                        "char payload",
                        "shape does not match character count",
                    ));
                }
                self.budget.add_elements(n)?;
                self.budget.add_bytes((v.len() as u64).saturating_mul(2))?;
                self.tasks.push(Task::Chars(v, 0));
            }
            AfterShape::Strings(v) => {
                if n != v.len() as u64 {
                    return Err(invalid(
                        "string-array payload",
                        "shape does not match string count",
                    ));
                }
                self.budget.add_elements(n)?;
                self.tasks.push(Task::Texts(v, 0));
            }
            AfterShape::Dense(t, b) => {
                self.budget.add_elements(n)?;
                let expected = n
                    .checked_mul(t.byte_width())
                    .ok_or_else(|| invalid("dense payload", "byte length overflow"))?;
                if expected != b.len() as u64 {
                    return Err(invalid(
                        "dense payload",
                        "shape and element type do not match data length",
                    ));
                }
                self.budget.add_bytes(expected)?;
            }
            AfterShape::Cell(v, d) => {
                if n != v.len() as u64 {
                    return Err(invalid("cell payload", "shape does not match cell count"));
                }
                self.budget.add_elements(n)?;
                self.tasks.push(Task::Many(v, 0, d + 1));
            }
            AfterShape::StructArray(v, d) => {
                self.budget.add_elements(n)?;
                self.field_limit(v.fields.len(), "structure array fields")?;
                self.tasks.push(Task::StructArray(
                    v,
                    n,
                    0,
                    d,
                    HashSet::with_capacity(v.fields.len()),
                ));
            }
            AfterShape::ObjectArray(v, d) => {
                if n != v.elements.len() as u64 {
                    return Err(invalid(
                        "object array",
                        "element count does not match shape",
                    ));
                }
                self.budget.add_elements(n)?;
                self.tasks.push(Task::ObjectArray(v, 0, d));
            }
        }
        Ok(())
    }

    fn sorted(&mut self, v: Fields<'a>, label: &'static str, d: u16) -> Result<(), ContractError> {
        self.field_limit(v.len(), label)?;
        self.tasks.push(Task::SortedFields(v, 0, d, label));
        Ok(())
    }
    fn sorted_field(
        &mut self,
        v: Fields<'a>,
        i: usize,
        d: u16,
        label: &'static str,
    ) -> Result<(), ContractError> {
        if let Some((name, value)) = v.get(i) {
            self.budget.add_text(name)?;
            if i > 0 && v.get(i - 1).is_some_and(|(p, _)| p >= name) {
                return Err(invalid(label, "field names must be unique and sorted"));
            }
            self.tasks.push(Task::SortedFields(v, i + 1, d, label));
            self.tasks.push(Task::Payload(value, d + 1));
        }
        Ok(())
    }
    fn ordered_field(
        &mut self,
        v: &'a [StructField],
        i: usize,
        d: u16,
        mut names: HashSet<&'a str>,
    ) -> Result<(), ContractError> {
        if let Some(f) = v.get(i) {
            self.budget.add_text(&f.name)?;
            if !names.insert(&f.name) {
                return Err(invalid("ordered structure", "field names must be unique"));
            }
            self.tasks.push(Task::OrderedFields(v, i + 1, d, names));
            self.tasks.push(Task::Payload(&f.value, d + 1));
        }
        Ok(())
    }
}

fn invalid(field: &'static str, reason: &'static str) -> ContractError {
    ContractError::invalid(field, reason)
}

use super::super::*;
use super::text::TextCopy;
use runmat_execution::value::{CallableOwner, CallableValue, RegisteredData, StructField};
use std::rc::Rc;

pub(in crate::execution::value_codec::decode::machine) struct CellTask<'a> {
    shape: super::text::ShapeCopy<'a>,
    values: Vec<Value>,
    children: TakenValues,
    path: String,
}

impl<'a> CellTask<'a> {
    pub(super) fn new(shape: &'a [u64], children: TakenValues, path: String) -> Self {
        let capacity = children.remaining();
        Self {
            shape: super::text::ShapeCopy::new(shape, path.clone()),
            values: Vec::with_capacity(capacity),
            children,
            path,
        }
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        if !self.shape.is_complete() {
            return Ok((self.shape.advance(maximum_work)?, None));
        }
        if self.children.remaining() != 0 {
            let count = self.children.remaining().min(maximum_work);
            for _ in 0..count {
                self.values.push(self.children.next().expect("cell child"));
            }
            return Ok((count, None));
        }
        let shape = std::mem::replace(
            &mut self.shape,
            super::text::ShapeCopy::new(&[], String::new()),
        )
        .finish();
        let cell = CellArray::new_with_shape(std::mem::take(&mut self.values), shape)
            .map_err(|error| ValueCodecError::invalid(&self.path, error))?;
        Ok((1, Some(Value::Cell(cell))))
    }

    pub(super) fn take_children(&mut self) -> TakenValues {
        std::mem::replace(&mut self.children, empty_taken_values())
    }
}

pub(in crate::execution::value_codec::decode::machine) struct StructTask<'a> {
    fields: &'a [StructField],
    index: usize,
    name: Option<TextCopy<'a>>,
    output: StructValue,
    children: TakenValues,
    _path: String,
}

impl<'a> StructTask<'a> {
    pub(super) fn new(fields: &'a [StructField], children: TakenValues, path: String) -> Self {
        Self {
            fields,
            index: 0,
            name: None,
            output: StructValue::with_capacity(fields.len()),
            children,
            _path: path,
        }
    }

    pub(super) fn advance(&mut self, maximum_work: usize) -> (usize, bool) {
        if self.index == self.fields.len() {
            return (1, true);
        }
        let copy = self
            .name
            .get_or_insert_with(|| TextCopy::new(&self.fields[self.index].name));
        let completed = copy.advance(maximum_work);
        if copy.is_complete() {
            let name = self.name.take().expect("active field name").finish();
            let value = self.children.next().expect("one decoded value per field");
            self.output.insert(name, value);
            self.index += 1;
        }
        (completed, self.index == self.fields.len())
    }

    pub(super) fn finish(&mut self) -> Value {
        Value::Struct(std::mem::replace(&mut self.output, StructValue::new()))
    }

    pub(super) fn take_children(&mut self) -> TakenValues {
        std::mem::replace(&mut self.children, empty_taken_values())
    }
}

pub(in crate::execution::value_codec::decode::machine) struct CallableTask<'a> {
    source: &'a CallableValue,
    source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
    name: TextCopy<'a>,
    decoded_name: Option<String>,
    source_file: Option<TextCopy<'a>>,
    decoded_source_file: Option<String>,
    decoded_captures: Vec<Value>,
    captures: TakenValues,
    path: String,
}

impl<'a> CallableTask<'a> {
    pub(super) fn new(
        source: &'a CallableValue,
        captures: TakenValues,
        source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
        path: String,
    ) -> Self {
        let capacity = captures.remaining();
        Self {
            source,
            source_resolver,
            name: TextCopy::new(&source.qualified_name),
            decoded_name: None,
            source_file: source.source_file.as_deref().map(TextCopy::new),
            decoded_source_file: None,
            decoded_captures: Vec::with_capacity(capacity),
            captures,
            path,
        }
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        if self.decoded_name.is_none() {
            let completed = self.name.advance(maximum_work);
            if self.name.is_complete() {
                self.decoded_name =
                    Some(std::mem::replace(&mut self.name, TextCopy::new("")).finish());
            }
            return Ok((completed, None));
        }
        if let Some(copy) = &mut self.source_file {
            let completed = copy.advance(maximum_work);
            if !copy.is_complete() {
                return Ok((completed, None));
            }
            self.decoded_source_file = Some(
                self.source_file
                    .take()
                    .expect("active source-file copy")
                    .finish(),
            );
            return Ok((completed, None));
        }
        if self.captures.remaining() != 0 {
            let count = self.captures.remaining().min(maximum_work);
            for _ in 0..count {
                self.decoded_captures
                    .push(self.captures.next().expect("callable capture"));
            }
            return Ok((count, None));
        }
        let binding = match self.source.owner {
            CallableOwner::Bound(origin) => Some(
                match &self.source_resolver {
                    Some(resolver) => {
                        crate::user_functions::resolve_binding_origin_binding_in(resolver, origin)
                    }
                    None => crate::user_functions::resolve_binding_origin_binding(origin),
                }
                .ok_or_else(|| {
                    ValueCodecError::unsupported(
                        &self.path,
                        "exact bound callable origin is unavailable",
                    )
                })?,
            ),
            _ => None,
        };
        let name = self.decoded_name.take().expect("decoded callable name");
        let source_file = self.decoded_source_file.take();
        let captures = std::mem::take(&mut self.decoded_captures);
        let value = if self.source.form == runmat_execution::value::CallableForm::Closure {
            Value::Closure(Closure::from_data(runmat_value::ClosureData {
                lexical_environment: None,
                function_name: name,
                binding,
                source_role: self.source.source_role,
                source_file: source_file.map(String::into_boxed_str),
                captures: captures.into(),
            }))
        } else {
            match self.source.owner {
                CallableOwner::Workspace => Value::function_handle(name),
                CallableOwner::External => Value::external_function_handle(name),
                CallableOwner::Method => Value::method_function_handle(name),
                CallableOwner::Bound(_) => {
                    let binding = binding.expect("resolved exact origin");
                    Value::bound_function_handle(
                        name,
                        binding.function,
                        binding.authority,
                        self.source.source_role,
                        source_file,
                    )
                }
            }
        };
        Ok((1, Some(value)))
    }

    pub(super) fn take_children(&mut self) -> TakenValues {
        std::mem::replace(&mut self.captures, empty_taken_values())
    }
}

pub(in crate::execution::value_codec::decode::machine) struct ImmutableTask<'a> {
    source: &'a RegisteredData,
    index: usize,
    field_name: Option<TextCopy<'a>>,
    object: Option<ObjectInstance>,
    children: TakenValues,
    path: String,
}

impl<'a> ImmutableTask<'a> {
    pub(super) fn new(
        source: &'a RegisteredData,
        class_identity: runmat_types::ClassIdentity,
        children: TakenValues,
        path: String,
    ) -> Self {
        Self {
            source,
            index: 0,
            field_name: None,
            object: Some({
                let mut object = ObjectInstance::new(class_identity);
                object.properties.reserve(source.fields.len());
                object
            }),
            children,
            path,
        }
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        if self.index < self.source.fields.len() {
            let field = &self.source.fields[self.index];
            let copy = self
                .field_name
                .get_or_insert_with(|| TextCopy::new(&field.name));
            let completed = copy.advance(maximum_work);
            if copy.is_complete() {
                let name = self.field_name.take().expect("active field name").finish();
                let value = self.children.next().expect("one decoded value per field");
                if self
                    .object
                    .as_mut()
                    .expect("class identity was prepared")
                    .properties
                    .insert(name, value)
                    .is_some()
                {
                    return Err(ValueCodecError::invalid(
                        &self.path,
                        "immutable value-class fields must be unique",
                    ));
                }
                self.index += 1;
            }
            return Ok((completed, None));
        }
        let object = self.object.take().expect("class identity was prepared");
        match crate::execution::value_codec::immutable_class::schema_for_class(
            object.class_name.display_name(),
        )? {
            Some(schema) => schema.validate_object(&object, &self.path)?,
            None => crate::parallel::codistributor::validate_definition(&object)
                .map_err(|error| ValueCodecError::invalid(&self.path, error.to_string()))?,
        }
        Ok((1, Some(Value::Object(object))))
    }

    pub(super) fn take_children(&mut self) -> TakenValues {
        std::mem::replace(&mut self.children, empty_taken_values())
    }
}

fn empty_taken_values() -> TakenValues {
    DecodeMachine::empty_taken_values()
}

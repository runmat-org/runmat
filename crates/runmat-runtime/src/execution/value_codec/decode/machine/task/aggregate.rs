use super::super::*;
use super::aggregate_output::{Admission, Array, Quote};
use crate::context::RuntimeContext;
use crate::execution::value_codec::object_output::{ObjectOutput, ObjectText};
use crate::execution::value_codec::shape_copy::ShapeCopy;
use runmat_execution::value::{CallableOwner, CallableValue, RegisteredData};
use std::rc::Rc;

pub(in crate::execution::value_codec::decode::machine) struct CellTask<'a> {
    shape: Option<ShapeCopy<'a>>,
    values: Option<Array<Value>>,
    children: TakenValues<'a>,
    path: ControlPath<'a>,
    admission: Admission<'a>,
}

impl<'a> CellTask<'a> {
    pub(super) fn new(
        shape: &'a [u64],
        children: TakenValues<'a>,
        path: ControlPath<'a>,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        let capacity = children.remaining();
        let mut quote = Quote::default();
        quote.array::<Value>(capacity, &path)?;
        let admission = Admission::new(quote, runtime, &path)?;
        let values = admission.array(capacity, &path)?;
        let shape = ShapeCopy::new(shape, runtime, &path)?;
        Ok(Self {
            shape: Some(shape),
            values: Some(values),
            children,
            path,
            admission,
        })
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        self.admission.checkpoint(&self.path)?;
        let shape = self.shape.as_mut().expect("active cell shape");
        if !shape.is_complete() {
            return Ok((shape.advance(maximum_work, &self.path)?, None));
        }
        if self.children.remaining() != 0 {
            let count = self.children.remaining().min(maximum_work);
            for _ in 0..count {
                self.values
                    .as_mut()
                    .expect("cell destination")
                    .push(self.children.next()?.expect("cell child"));
            }
            return Ok((count, None));
        }
        let shape = self
            .shape
            .take()
            .expect("active cell shape")
            .finish(&self.path)?;
        let values = self.values.take().expect("cell destination").finish();
        let cell = CellArray::from_shared_storage(values, shape)
            .map_err(|error| ValueCodecError::invalid(&self.path, error))?;
        self.admission.checkpoint(&self.path)?;
        Ok((1, Some(Value::Cell(cell))))
    }

    pub(super) fn take_children(&mut self) -> TakenValues<'a> {
        std::mem::replace(&mut self.children, empty_taken_values())
    }
}

pub(super) use super::structure::StructTask;

pub(in crate::execution::value_codec::decode::machine) struct CallableTask<'a> {
    source: &'a CallableValue,
    source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
    name: Option<ObjectText<'a>>,
    decoded_name: Option<runmat_value::Utf8Text>,
    source_file: Option<ObjectText<'a>>,
    decoded_source_file: Option<runmat_value::Utf8Text>,
    decoded_captures: Option<Array<Value>>,
    captures: TakenValues<'a>,
    path: ControlPath<'a>,
    admission: Admission<'a>,
}

impl<'a> CallableTask<'a> {
    pub(super) fn new(
        source: &'a CallableValue,
        captures: TakenValues<'a>,
        source_resolver: Option<Rc<crate::user_functions::ExecutionSourceResolver>>,
        path: ControlPath<'a>,
        runtime: Option<&'a RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        let capacity = captures.remaining();
        let mut quote = Quote::default();
        quote.text(&source.qualified_name, &path)?;
        if let Some(source_file) = &source.source_file {
            quote.text(source_file, &path)?;
        }
        quote.array::<Value>(capacity, &path)?;
        if source.form == runmat_execution::value::CallableForm::Closure {
            quote.add(Closure::resolved_header_bytes(), 1, &path)?;
        } else if matches!(source.owner, CallableOwner::Bound(_)) {
            quote.add(
                runmat_value::BoundFunctionHandle::requested_header_bytes(),
                1,
                &path,
            )?;
        }
        let admission = Admission::new(quote, runtime, &path)?;
        let name = admission.text(&source.qualified_name, &path)?;
        let source_file = source
            .source_file
            .as_deref()
            .map(|s| admission.text(s, &path))
            .transpose()?;
        let decoded_captures = admission.array(capacity, &path)?;
        Ok(Self {
            source,
            source_resolver,
            name: Some(name),
            decoded_name: None,
            source_file,
            decoded_source_file: None,
            decoded_captures: Some(decoded_captures),
            captures,
            path,
            admission,
        })
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        self.admission.checkpoint(&self.path)?;
        if self.decoded_name.is_none() {
            let name = self.name.as_mut().expect("active callable name");
            let completed = name.advance(maximum_work);
            if name.is_complete() {
                self.decoded_name = Some(self.name.take().expect("active callable name").finish());
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
                    .as_mut()
                    .expect("capture destination")
                    .push(self.captures.next()?.expect("callable capture"));
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
        let captures = self
            .decoded_captures
            .take()
            .expect("capture destination")
            .finish();
        let value = if self.source.form == runmat_execution::value::CallableForm::Closure {
            let data = runmat_value::ClosureData {
                lexical_environment: None,
                function_name: name,
                binding,
                source_role: self.source.source_role,
                source_file,
                captures,
            };
            self.admission.construct(&self.path, |permit| {
                let closure = match permit {
                    Some(permit) => Closure::try_from_data_with_exact_header(data, permit)
                        .map_err(|e| match e {
                            runmat_value::ClosureHeaderError::Allocation(e) => {
                                ValueCodecError::allocation(&self.path, e.into())
                            }
                            runmat_value::ClosureHeaderError::CapturesNotShared => {
                                ValueCodecError::invalid(
                                    &self.path,
                                    "decoded captures require their shared owner",
                                )
                            }
                        })?,
                    None => Closure::from_data(data),
                };
                Ok(Value::Closure(closure))
            })?
        } else {
            match self.source.owner {
                CallableOwner::Workspace => {
                    Value::FunctionHandle(runmat_value::NamedFunctionHandle::from_text_owner(name))
                }
                CallableOwner::External => Value::ExternalFunctionHandle(
                    runmat_value::NamedFunctionHandle::from_text_owner(name),
                ),
                CallableOwner::Method => Value::MethodFunctionHandle(
                    runmat_value::NamedFunctionHandle::from_text_owner(name),
                ),
                CallableOwner::Bound(_) => {
                    let binding = binding.expect("resolved exact origin");
                    let data = runmat_value::BoundFunctionHandleData {
                        name,
                        function: binding.function,
                        authority: binding.authority,
                        source_role: self.source.source_role,
                        source_file,
                    };
                    self.admission.construct(&self.path, |permit| {
                        let handle = match permit {
                            Some(permit) => {
                                runmat_value::BoundFunctionHandle::try_from_data_with_exact_header(
                                    data, permit,
                                )
                                .map_err(|e| ValueCodecError::allocation(&self.path, e.into()))?
                            }
                            None => runmat_value::BoundFunctionHandle::from_data(data),
                        };
                        Ok(Value::BoundFunctionHandle(handle))
                    })?
                }
            }
        };
        self.admission.checkpoint(&self.path)?;
        Ok((1, Some(value)))
    }

    pub(super) fn take_children(&mut self) -> TakenValues<'a> {
        std::mem::replace(&mut self.captures, empty_taken_values())
    }
}

pub(in crate::execution::value_codec::decode::machine) struct ImmutableTask<'a> {
    workspace: crate::execution::value_codec::ValidationWorkspaceOwner,
    source: &'a RegisteredData,
    index: usize,
    field_name: Option<ObjectText<'a>>,
    object: Option<ObjectOutput<'a>>,
    children: TakenValues<'a>,
    path: ControlPath<'a>,
}

impl<'a> ImmutableTask<'a> {
    pub(super) fn new(
        source: &'a RegisteredData,
        class_identity: runmat_types::ClassIdentity,
        children: TakenValues<'a>,
        path: ControlPath<'a>,
        runtime: Option<&'a crate::context::RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        let object = ObjectOutput::new(
            class_identity.display_name(),
            source.fields.iter().map(|f| f.name.as_str()),
            &[],
            crate::execution::value_codec::object_output::DynamicMetadataCopy::Incremental,
            runtime,
            &path,
        )?;
        Ok(Self {
            workspace: crate::execution::value_codec::ValidationWorkspaceOwner::for_runtime(
                runtime,
            ),
            source,
            index: 0,
            field_name: None,
            object: Some(object),
            children,
            path,
        })
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        self.object
            .as_ref()
            .expect("object output")
            .checkpoint(&self.path)?;
        if self.index < self.source.fields.len() {
            let field = &self.source.fields[self.index];
            if self.field_name.is_none() {
                self.field_name = Some(
                    self.object
                        .as_ref()
                        .expect("object output")
                        .text(&field.name, &self.path)?,
                );
            }
            let copy = self.field_name.as_mut().expect("active field name");
            let completed = copy.advance(maximum_work);
            if copy.is_complete() {
                let name = self.field_name.take().expect("active field name").finish();
                let value = self.children.next()?.expect("one decoded value per field");
                if self
                    .object
                    .as_mut()
                    .expect("class identity was prepared")
                    .insert(name, value, &self.path)?
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
        let object = self
            .object
            .take()
            .expect("class identity was prepared")
            .finish(&self.path)?;
        match crate::execution::value_codec::immutable_class::schema_for_class(
            object.class_name().display_name(),
        )? {
            Some(schema) => schema.validate_object(&object, &self.path, Some(&self.workspace))?,
            None => crate::parallel::codistributor::validate_definition(&object)
                .map_err(|error| ValueCodecError::invalid(&self.path, error.to_string()))?,
        }
        Ok((1, Some(Value::Object(object))))
    }

    pub(super) fn take_children(&mut self) -> TakenValues<'a> {
        std::mem::replace(&mut self.children, empty_taken_values())
    }
}

fn empty_taken_values<'a>() -> TakenValues<'a> {
    DecodeMachine::empty_taken_values()
}

use super::super::*;
use super::text::{ShapeCopy, TextCopy};
use crate::execution::value_codec::object_output::{ObjectOutput, ObjectText};
use runmat_execution::value::{
    DurableDecompositionPayload, DurableDynamicProperty, DurableObjectArray, DurablePlainObject,
    DurableStructArray, DurableStructuralDecomposition,
};
use runmat_value::{DynamicPropertyDef, StructArrayAdmission};
use std::num::NonZeroUsize;

#[path = "durable/numeric_factor.rs"]
mod numeric_factor;
use numeric_factor::NumericFactorDecode;

pub(in crate::execution::value_codec::decode::machine) struct PlainObjectTask<'a> {
    source: &'a DurablePlainObject,
    object: Option<ObjectOutput<'a>>,
    children: TakenValues<'a>,
    property: usize,
    property_name: Option<ObjectText<'a>>,
    dynamic: usize,
    dynamic_text: usize,
    texts: [Option<runmat_value::Utf8Text>; 4],
    numeric_factor: Option<NumericFactorDecode<'a>>,
    path: ControlPath<'a>,
}

impl<'a> PlainObjectTask<'a> {
    pub(super) fn new(
        source: &'a DurablePlainObject,
        children: TakenValues<'a>,
        path: ControlPath<'a>,
        runtime: Option<&'a crate::context::RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        let is_decomposition = source.class_identity == runmat_types::standard::DECOMPOSITION;
        if source.decomposition.is_some() != is_decomposition {
            return Err(ValueCodecError::invalid(
                &path,
                "decomposition class and closed payload differ",
            ));
        }
        let object = if matches!(
            source.decomposition.as_ref(),
            Some(DurableDecompositionPayload::Data(_))
        ) {
            ObjectOutput::new_numeric_factor(
                source.class_identity.display_name(),
                source.properties.iter().map(|f| f.name.as_str()),
                &source.dynamic_properties,
                runtime,
                &path,
            )?
        } else {
            ObjectOutput::new(
                source.class_identity.display_name(),
                source.properties.iter().map(|f| f.name.as_str()),
                &source.dynamic_properties,
                crate::execution::value_codec::object_output::DynamicMetadataCopy::Incremental,
                runtime,
                &path,
            )?
        };
        Ok(Self {
            source,
            object: Some(object),
            children,
            property: 0,
            property_name: None,
            dynamic: 0,
            dynamic_text: 0,
            texts: [None, None, None, None],
            numeric_factor: match source.decomposition.as_ref() {
                Some(DurableDecompositionPayload::Data(factor)) => {
                    Some(NumericFactorDecode::new(factor, runtime, path.try_clone())?)
                }
                Some(DurableDecompositionPayload::Structural(_)) | None => None,
            },
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
        if self.property < self.source.properties.len() {
            let field = &self.source.properties[self.property];
            if self.property_name.is_none() {
                self.property_name = Some(
                    self.object
                        .as_ref()
                        .expect("object output")
                        .text(&field.name, &self.path)?,
                );
            }
            let copy = self.property_name.as_mut().expect("active property name");
            let completed = copy.advance(maximum_work);
            if copy.is_complete() {
                let name = self.property_name.take().expect("property name").finish();
                let value = self.children.next()?.expect("decoded property value");
                self.object
                    .as_mut()
                    .expect("object")
                    .insert(name, value, &self.path)?;
                self.property += 1;
            }
            return Ok((completed, None));
        }
        if self.dynamic < self.source.dynamic_properties.len() {
            let definition = &self.source.dynamic_properties[self.dynamic];
            let source = dynamic_text(definition, self.dynamic_text);
            let mut copy = match self.property_name.take() {
                Some(copy) => copy,
                None => self
                    .object
                    .as_ref()
                    .expect("object output")
                    .text(source, &self.path)?,
            };
            let completed = copy.advance(maximum_work);
            if !copy.is_complete() {
                self.property_name = Some(copy);
                return Ok((completed, None));
            }
            self.texts[self.dynamic_text] = Some(copy.finish());
            self.dynamic_text += 1;
            if self.dynamic_text == 4 {
                let key = self.texts[0].take().expect("dynamic key");
                let name = self.texts[1].take().expect("dynamic definition name");
                let class_text = self.texts[2].take().expect("defining class");
                let defining_class = self
                    .object
                    .as_ref()
                    .expect("object output")
                    .class(&class_text, &self.path)?;
                let description = self.texts[3].take().expect("description");
                let property = dynamic_property(definition, name, defining_class, description);
                self.object
                    .as_mut()
                    .expect("object")
                    .insert_dynamic(key, property, &self.path)?;
                self.dynamic += 1;
                self.dynamic_text = 0;
            }
            return Ok((completed, None));
        }
        if let Some(factor) = &mut self.numeric_factor {
            let (completed, data) = factor.advance(maximum_work, &self.path)?;
            if let Some(data) = data {
                self.object
                    .as_mut()
                    .expect("object")
                    .attach_numeric_factor(data, &self.path)?;
                self.numeric_factor = None;
            }
            return Ok((completed, None));
        }
        let object = self
            .object
            .take()
            .expect("decoded object")
            .finish(&self.path)?;
        if let Some(expected) = &self.source.decomposition {
            use crate::builtins::math::linalg::factor::decomposition::{
                decomposition_persistence_kind, DecompositionPersistenceKind,
            };
            let actual = decomposition_persistence_kind(&object)
                .map_err(|error| ValueCodecError::runtime(&self.path, error))?;
            let matching = matches!(
                (expected, actual),
                (
                    DurableDecompositionPayload::Data(_),
                    DecompositionPersistenceKind::Factor(_)
                ) | (
                    DurableDecompositionPayload::Structural(
                        DurableStructuralDecomposition::Diagonal,
                    ),
                    DecompositionPersistenceKind::Diagonal,
                ) | (
                    DurableDecompositionPayload::Structural(
                        DurableStructuralDecomposition::Triangular,
                    ),
                    DecompositionPersistenceKind::Triangular,
                )
            );
            if !matching {
                return Err(ValueCodecError::invalid(
                    &self.path,
                    "decomposition payload differs from its public recipe",
                ));
            }
        }
        Ok((1, Some(Value::Object(object))))
    }

    pub(super) fn take_children(&mut self) -> TakenValues<'a> {
        std::mem::replace(&mut self.children, DecodeMachine::empty_taken_values())
    }
}

fn dynamic_text(value: &DurableDynamicProperty, index: usize) -> &str {
    match index {
        0 => &value.name,
        1 => &value.definition_name,
        2 => value.defining_class.display_name(),
        3 => &value.description,
        _ => unreachable!("four dynamic-property strings"),
    }
}

fn dynamic_property(
    value: &DurableDynamicProperty,
    name: runmat_value::Utf8Text,
    defining_class: runmat_value::SharedClassIdentity,
    description: runmat_value::Utf8Text,
) -> DynamicPropertyDef {
    DynamicPropertyDef {
        name,
        defining_class,
        metadata_handle: None,
        get_access: value.get_access,
        set_access: value.set_access,
        dependent: value.dependent,
        hidden: value.hidden,
        transient: value.transient,
        non_copyable: value.non_copyable,
        abort_set: value.abort_set,
        set_observable: value.set_observable,
        get_observable: value.get_observable,
        description,
    }
}

pub(in crate::execution::value_codec::decode::machine) struct StructArrayTask<'a> {
    source: &'a DurableStructArray,
    shape: ShapeCopy<'a>,
    names: Vec<String>,
    name: Option<TextCopy<'a>>,
    values: Vec<Value>,
    admission: Option<StructArrayAdmission>,
    children: TakenValues<'a>,
    path: ControlPath<'a>,
}

impl<'a> StructArrayTask<'a> {
    pub(super) fn new(
        source: &'a DurableStructArray,
        children: TakenValues<'a>,
        path: ControlPath<'a>,
    ) -> Result<Self, ValueCodecError> {
        let capacity = children.remaining();
        Ok(Self {
            source,
            shape: ShapeCopy::new(&source.shape, path.try_clone()?),
            names: Vec::with_capacity(source.fields.len()),
            name: None,
            values: Vec::with_capacity(capacity),
            admission: None,
            children,
            path,
        })
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        if let Some(admission) = &mut self.admission {
            let completed = admission
                .advance(NonZeroUsize::new(maximum_work).expect("machine work is nonzero"))
                .map_err(|error| ValueCodecError::invalid(&self.path, error))?;
            if !admission.is_complete() {
                return Ok((completed, None));
            }
            let admission = self.admission.take().expect("completed struct admission");
            return Ok((
                completed,
                Some(
                    admission
                        .into_validated()
                        .ok()
                        .expect("completed struct admission")
                        .into_value(),
                ),
            ));
        }
        if !self.shape.is_complete() {
            return Ok((self.shape.advance(maximum_work)?, None));
        }
        if self.names.len() < self.source.fields.len() {
            let copy = self
                .name
                .get_or_insert_with(|| TextCopy::new(&self.source.fields[self.names.len()].name));
            let completed = copy.advance(maximum_work);
            if copy.is_complete() {
                self.names
                    .push(self.name.take().expect("field name").finish());
            }
            return Ok((completed, None));
        }
        let count = self.children.remaining().min(maximum_work);
        for _ in 0..count {
            self.values
                .push(self.children.next()?.expect("field-major value"));
        }
        if self.children.remaining() != 0 {
            return Ok((count.max(1), None));
        }
        let shape = self.shape.take_finished();
        self.admission = Some(
            StructArrayAdmission::field_major(
                std::mem::take(&mut self.names),
                std::mem::take(&mut self.values),
                shape,
            )
            .map_err(|error| ValueCodecError::invalid(&self.path, error))?,
        );
        Ok((count.max(1), None))
    }

    pub(super) fn take_children(&mut self) -> TakenValues<'a> {
        std::mem::replace(&mut self.children, DecodeMachine::empty_taken_values())
    }
}

pub(in crate::execution::value_codec::decode::machine) struct ObjectArrayTask<'a> {
    output: Option<crate::execution::object_array::ObjectArrayConstruction<'a>>,
    shape: Option<crate::execution::value_codec::shape_copy::ShapeCopy<'a>>,
    source: &'a DurableObjectArray,
    children: TakenValues<'a>,
    path: ControlPath<'a>,
    runtime: Option<&'a crate::context::RuntimeContext>,
}

impl<'a> ObjectArrayTask<'a> {
    pub(super) fn new(
        source: &'a DurableObjectArray,
        children: TakenValues<'a>,
        path: ControlPath<'a>,
        runtime: Option<&'a crate::context::RuntimeContext>,
    ) -> Result<Self, ValueCodecError> {
        let shape = crate::execution::value_codec::shape_copy::ShapeCopy::new(
            &source.shape,
            runtime,
            &path,
        )?;
        Ok(Self {
            output: None,
            shape: Some(shape),
            source,
            children,
            path,
            runtime,
        })
    }
    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        if let Some(shape) = &mut self.shape {
            if !shape.is_complete() {
                return Ok((shape.advance(maximum_work, &self.path)?, None));
            }
            let shape = self
                .shape
                .take()
                .expect("completed object-array shape")
                .finish(&self.path)?;
            self.output = Some(
                crate::execution::object_array::ObjectArrayConstruction::new_retaining_shape(
                    self.runtime,
                    crate::execution::object_array::Class::Name(
                        self.source.class_identity.display_name(),
                    ),
                    self.children.remaining(),
                    &shape,
                    std::iter::empty(),
                    false,
                )
                .map_err(|error| ValueCodecError::runtime(&self.path, error))?,
            );
        }
        let output = self.output.as_mut().expect("prepared object-array output");
        output
            .checkpoint()
            .map_err(|error| ValueCodecError::runtime(&self.path, error))?;
        let count = self.children.remaining().min(maximum_work);
        for _ in 0..count {
            let value @ Value::Object(_) =
                self.children.next()?.expect("decoded object-array child")
            else {
                return Err(ValueCodecError::invalid(
                    &self.path,
                    "object-array child is not an object",
                ));
            };
            output
                .push(value)
                .map_err(|error| ValueCodecError::runtime(&self.path, error))?;
        }
        if self.children.remaining() != 0 {
            return Ok((count.max(1), None));
        }
        let value = self
            .output
            .take()
            .expect("complete object-array output")
            .finish_value()
            .map_err(|error| ValueCodecError::runtime(&self.path, error))?;
        Ok((count.max(1), Some(value)))
    }
    pub(super) fn take_children(&mut self) -> TakenValues<'a> {
        std::mem::replace(&mut self.children, DecodeMachine::empty_taken_values())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "durable/tests.rs"]
mod tests;

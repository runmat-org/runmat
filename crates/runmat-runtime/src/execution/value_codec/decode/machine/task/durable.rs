use super::super::*;
use super::text::{ShapeCopy, TextCopy};
use runmat_execution::value::{
    DurableDynamicProperty, DurableObjectArray, DurablePlainObject, DurableStructArray,
};
use runmat_value::{DynamicPropertyDef, ObjectArray, StructArray};

pub(in crate::execution::value_codec::decode::machine) struct PlainObjectTask<'a> {
    source: &'a DurablePlainObject,
    children: TakenValues,
    class: TextCopy<'a>,
    object: Option<ObjectInstance>,
    property: usize,
    property_name: Option<TextCopy<'a>>,
    dynamic: usize,
    dynamic_text: usize,
    texts: Vec<String>,
    path: String,
}

impl<'a> PlainObjectTask<'a> {
    pub(super) fn new(source: &'a DurablePlainObject, children: TakenValues, path: String) -> Self {
        Self {
            source,
            children,
            class: TextCopy::new(source.class_identity.display_name()),
            object: None,
            property: 0,
            property_name: None,
            dynamic: 0,
            dynamic_text: 0,
            texts: Vec::new(),
            path,
        }
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        if self.object.is_none() {
            let completed = self.class.advance(maximum_work);
            if self.class.is_complete() {
                let class = std::mem::replace(&mut self.class, TextCopy::new("")).finish();
                let class = runmat_types::ClassIdentity::new(class)
                    .map_err(|error| ValueCodecError::invalid(&self.path, error.to_string()))?;
                let mut object = ObjectInstance::new(class);
                object.properties.reserve(self.source.properties.len());
                if !self.source.dynamic_properties.is_empty() {
                    object.dynamic_properties =
                        Some(Box::new(std::collections::HashMap::with_capacity(
                            self.source.dynamic_properties.len(),
                        )));
                }
                self.object = Some(object);
            }
            return Ok((completed, None));
        }
        if self.property < self.source.properties.len() {
            let field = &self.source.properties[self.property];
            let copy = self
                .property_name
                .get_or_insert_with(|| TextCopy::new(&field.name));
            let completed = copy.advance(maximum_work);
            if copy.is_complete() {
                let name = self.property_name.take().expect("property name").finish();
                let value = self.children.next().expect("decoded property value");
                self.object
                    .as_mut()
                    .expect("object")
                    .properties
                    .insert(name, value);
                self.property += 1;
            }
            return Ok((completed, None));
        }
        if self.dynamic < self.source.dynamic_properties.len() {
            let definition = &self.source.dynamic_properties[self.dynamic];
            let source = dynamic_text(definition, self.dynamic_text);
            let mut copy = self
                .property_name
                .take()
                .unwrap_or_else(|| TextCopy::new(source));
            let completed = copy.advance(maximum_work);
            if !copy.is_complete() {
                self.property_name = Some(copy);
                return Ok((completed, None));
            }
            self.texts.push(copy.finish());
            self.dynamic_text += 1;
            if self.dynamic_text == 4 {
                let mut texts = std::mem::take(&mut self.texts).into_iter();
                let key = texts.next().expect("dynamic key");
                let name = texts.next().expect("dynamic definition name");
                let defining_class =
                    runmat_types::ClassIdentity::new(texts.next().expect("defining class"))
                        .map_err(|error| ValueCodecError::invalid(&self.path, error.to_string()))?;
                let description = texts.next().expect("description");
                let property = dynamic_property(definition, name, defining_class, description);
                self.object
                    .as_mut()
                    .expect("object")
                    .dynamic_properties
                    .get_or_insert_with(Default::default)
                    .insert(key, property);
                self.dynamic += 1;
                self.dynamic_text = 0;
            }
            return Ok((completed, None));
        }
        Ok((
            1,
            Some(Value::Object(self.object.take().expect("decoded object"))),
        ))
    }

    pub(super) fn take_children(&mut self) -> TakenValues {
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
    name: String,
    defining_class: runmat_types::ClassIdentity,
    description: String,
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
    children: TakenValues,
    shape: ShapeCopy<'a>,
    names: Vec<String>,
    name: Option<TextCopy<'a>>,
    values: Vec<Value>,
    path: String,
}

impl<'a> StructArrayTask<'a> {
    pub(super) fn new(source: &'a DurableStructArray, children: TakenValues, path: String) -> Self {
        let capacity = children.remaining();
        Self {
            source,
            children,
            shape: ShapeCopy::new(&source.shape, path.clone()),
            names: Vec::with_capacity(source.fields.len()),
            name: None,
            values: Vec::with_capacity(capacity),
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
                .push(self.children.next().expect("field-major value"));
        }
        if self.children.remaining() != 0 {
            return Ok((count.max(1), None));
        }
        let shape = std::mem::replace(&mut self.shape, ShapeCopy::new(&[], String::new())).finish();
        let value = StructArray::normalize_field_major(
            std::mem::take(&mut self.names),
            std::mem::take(&mut self.values),
            shape,
        )
        .map_err(|error| ValueCodecError::invalid(&self.path, error))?;
        Ok((count.max(1), Some(value)))
    }

    pub(super) fn take_children(&mut self) -> TakenValues {
        std::mem::replace(&mut self.children, DecodeMachine::empty_taken_values())
    }
}

pub(in crate::execution::value_codec::decode::machine) struct ObjectArrayTask<'a> {
    children: TakenValues,
    class: TextCopy<'a>,
    decoded_class: Option<runmat_types::ClassIdentity>,
    shape: ShapeCopy<'a>,
    objects: Vec<ObjectInstance>,
    path: String,
}

impl<'a> ObjectArrayTask<'a> {
    pub(super) fn new(source: &'a DurableObjectArray, children: TakenValues, path: String) -> Self {
        let capacity = children.remaining();
        Self {
            children,
            class: TextCopy::new(source.class_identity.display_name()),
            decoded_class: None,
            shape: ShapeCopy::new(&source.shape, path.clone()),
            objects: Vec::with_capacity(capacity),
            path,
        }
    }

    pub(super) fn advance(
        &mut self,
        maximum_work: usize,
    ) -> Result<(usize, Option<Value>), ValueCodecError> {
        if self.decoded_class.is_none() {
            let completed = self.class.advance(maximum_work);
            if self.class.is_complete() {
                let class = std::mem::replace(&mut self.class, TextCopy::new("")).finish();
                self.decoded_class =
                    Some(runmat_types::ClassIdentity::new(class).map_err(|error| {
                        ValueCodecError::invalid(&self.path, error.to_string())
                    })?);
            }
            return Ok((completed, None));
        }
        if !self.shape.is_complete() {
            return Ok((self.shape.advance(maximum_work)?, None));
        }
        let count = self.children.remaining().min(maximum_work);
        for _ in 0..count {
            let Value::Object(object) = self.children.next().expect("decoded object") else {
                return Err(ValueCodecError::invalid(
                    &self.path,
                    "object-array child is not an object",
                ));
            };
            self.objects.push(object);
        }
        if self.children.remaining() != 0 {
            return Ok((count.max(1), None));
        }
        let shape = std::mem::replace(&mut self.shape, ShapeCopy::new(&[], String::new())).finish();
        let array = ObjectArray::from_objects(
            self.decoded_class.take().expect("class identity"),
            std::mem::take(&mut self.objects),
            shape,
        )
        .map_err(|error| ValueCodecError::invalid(&self.path, error))?;
        Ok((count.max(1), Some(Value::ObjectArray(array))))
    }

    pub(super) fn take_children(&mut self) -> TakenValues {
        std::mem::replace(&mut self.children, DecodeMachine::empty_taken_values())
    }
}

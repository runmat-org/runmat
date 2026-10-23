use super::*;

#[path = "durable/numeric_factor.rs"]
mod numeric_factor;

impl<'a> ValuePayloadValidationCursor<'a> {
    pub(super) fn plain_object(
        &mut self,
        value: &'a DurablePlainObject,
        depth: u16,
    ) -> Result<(), ContractError> {
        self.class(&value.class_identity)?;
        if value.class_identity == runmat_types::standard::DECOMPOSITION
            && value.decomposition.is_none()
        {
            return Err(invalid(
                "numerical factor",
                "decomposition object is missing its closed payload",
            ));
        }
        if let Some(decomposition) = &value.decomposition {
            if value.class_identity != runmat_types::standard::DECOMPOSITION {
                return Err(invalid(
                    "numerical factor",
                    "decomposition payload has a different object class",
                ));
            }
            if let DurableDecompositionPayload::Data(factor) = decomposition {
                self.numeric_factor(factor)?;
            }
        }
        self.tasks.push(Task::Dynamic(value, 0));
        self.sorted(Fields::Struct(&value.properties), "struct fields", depth)
    }

    pub(super) fn dynamic(
        &mut self,
        value: &'a DurablePlainObject,
        index: usize,
    ) -> Result<(), ContractError> {
        if index == 0 {
            self.field_limit(value.dynamic_properties.len(), "dynamic object properties")?;
        }
        if let Some(property) = value.dynamic_properties.get(index) {
            self.budget.add_text(&property.name)?;
            if index > 0 && value.dynamic_properties[index - 1].name >= property.name {
                return Err(invalid(
                    "dynamic object properties",
                    "names must be unique and sorted",
                ));
            }
            self.budget.add_text(&property.definition_name)?;
            self.class(&property.defining_class)?;
            self.budget.add_text(&property.description)?;
            self.tasks.push(Task::Dynamic(value, index + 1));
        }
        Ok(())
    }

    pub(super) fn struct_array(
        &mut self,
        value: &'a DurableStructArray,
        depth: u16,
    ) -> Result<(), ContractError> {
        if value.shape.len() < 2 {
            return Err(invalid(
                "structure array",
                "shape requires at least two dimensions",
            ));
        }
        self.shape(&value.shape, AfterShape::StructArray(value, depth));
        Ok(())
    }

    pub(super) fn struct_array_field(
        &mut self,
        value: &'a DurableStructArray,
        count: u64,
        index: usize,
        depth: u16,
        mut names: HashSet<&'a str>,
    ) -> Result<(), ContractError> {
        if let Some(field) = value.fields.get(index) {
            self.budget.add_text(&field.name)?;
            if !names.insert(&field.name) || field.values.len() as u64 != count {
                return Err(invalid(
                    "structure array",
                    "field names or column lengths are invalid",
                ));
            }
            self.tasks
                .push(Task::StructArray(value, count, index + 1, depth, names));
            self.tasks.push(Task::Many(&field.values, 0, depth + 1));
        }
        Ok(())
    }

    pub(super) fn object_array(
        &mut self,
        value: &'a DurableObjectArray,
        depth: u16,
    ) -> Result<(), ContractError> {
        if value.shape.len() < 2 {
            return Err(invalid(
                "object array",
                "shape requires at least two dimensions",
            ));
        }
        self.class(&value.class_identity)?;
        self.shape(&value.shape, AfterShape::ObjectArray(value, depth));
        Ok(())
    }

    pub(super) fn object_array_item(
        &mut self,
        value: &'a DurableObjectArray,
        index: usize,
        depth: u16,
    ) -> Result<(), ContractError> {
        if let Some(element) = value.elements.get(index) {
            self.budget.node(depth + 1)?;
            if element.class_identity != value.class_identity {
                return Err(invalid(
                    "object array",
                    "element class does not match array class",
                ));
            }
            self.tasks.push(Task::ObjectArray(value, index + 1, depth));
            self.plain_object(element, depth + 1)?;
        }
        Ok(())
    }
}

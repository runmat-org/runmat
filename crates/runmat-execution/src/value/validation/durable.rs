//! Durable aggregate validation; execution V2 admits none of these tags.

use super::{checked_product, ValidationBudget};
use crate::{
    value::{DurableObjectArray, DurablePlainObject, DurableStructArray, StructField},
    ContractError,
};

impl ValidationBudget<'_> {
    pub(super) fn ordered_fields(
        &mut self,
        fields: &[StructField],
        depth: u16,
    ) -> Result<(), ContractError> {
        if fields.len() > self.limits.max_fields as usize {
            return Err(ContractError::Limit {
                field: "ordered structure fields",
                limit: self.limits.max_fields.into(),
            });
        }
        let mut names = std::collections::HashSet::with_capacity(fields.len());
        for field in fields {
            self.text(&field.name)?;
            if !names.insert(field.name.as_str()) {
                return Err(ContractError::invalid(
                    "ordered structure",
                    "field names must be unique",
                ));
            }
            self.payload(&field.value, depth + 1)?;
        }
        Ok(())
    }

    pub(super) fn plain_object(
        &mut self,
        value: &DurablePlainObject,
        depth: u16,
    ) -> Result<(), ContractError> {
        self.class_identity(&value.class_identity)?;
        self.fields(&value.properties, depth)?;
        if value.dynamic_properties.len() > self.limits.max_fields as usize {
            return Err(ContractError::Limit {
                field: "dynamic object properties",
                limit: self.limits.max_fields.into(),
            });
        }
        let mut previous: Option<&str> = None;
        for property in &value.dynamic_properties {
            self.text(&property.name)?;
            if previous.is_some_and(|previous| previous >= property.name.as_str()) {
                return Err(ContractError::invalid(
                    "dynamic object properties",
                    "names must be unique and sorted",
                ));
            }
            previous = Some(&property.name);
            self.class_identity(&property.defining_class)?;
            self.text(&property.description)?;
        }
        Ok(())
    }

    pub(super) fn struct_array(
        &mut self,
        value: &DurableStructArray,
        depth: u16,
    ) -> Result<(), ContractError> {
        if value.shape.len() < 2 {
            return Err(ContractError::invalid(
                "structure array",
                "shape requires at least two dimensions",
            ));
        }
        let count = checked_product(&value.shape)?;
        self.elements(count)?;
        if value.fields.len() > self.limits.max_fields as usize {
            return Err(ContractError::Limit {
                field: "structure array fields",
                limit: self.limits.max_fields.into(),
            });
        }
        let mut names = std::collections::HashSet::with_capacity(value.fields.len());
        for field in &value.fields {
            self.text(&field.name)?;
            if !names.insert(field.name.as_str()) || field.values.len() as u64 != count {
                return Err(ContractError::invalid(
                    "structure array",
                    "field names or column lengths are invalid",
                ));
            }
            for item in &field.values {
                self.payload(item, depth + 1)?;
            }
        }
        Ok(())
    }

    pub(super) fn object_array(
        &mut self,
        value: &DurableObjectArray,
        depth: u16,
    ) -> Result<(), ContractError> {
        if value.shape.len() < 2 {
            return Err(ContractError::invalid(
                "object array",
                "shape requires at least two dimensions",
            ));
        }
        self.class_identity(&value.class_identity)?;
        let count = checked_product(&value.shape)?;
        if count != value.elements.len() as u64 {
            return Err(ContractError::invalid(
                "object array",
                "element count does not match shape",
            ));
        }
        self.elements(count)?;
        for element in &value.elements {
            self.node(depth + 1)?;
            if element.class_identity != value.class_identity {
                return Err(ContractError::invalid(
                    "object array",
                    "element class does not match array class",
                ));
            }
            self.plain_object(element, depth + 1)?;
        }
        Ok(())
    }

    fn class_identity(
        &mut self,
        identity: &runmat_types::ClassIdentity,
    ) -> Result<(), ContractError> {
        runmat_types::ClassIdentity::new(identity.display_name())
            .map_err(|_| ContractError::invalid("class identity", "invalid class name"))?;
        self.text(identity.display_name())
    }
}

use crate::mxarray::MxArrayData;
use crate::{MxArray, MxClassId};

use super::MxApi;

impl MxApi {
    pub fn set_class_name(
        &mut self,
        value: *mut MxArray,
        class_name: String,
    ) -> Result<(), String> {
        let class_name =
            runmat_types::ClassIdentity::new(class_name).map_err(|error| error.to_string())?;
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        let replacement =
            match std::mem::replace(value.data_mut(), MxArrayData::Logical(Vec::new().into())) {
                MxArrayData::Struct { fields, values } => MxArrayData::Object {
                    class_name,
                    properties: fields,
                    values,
                },
                MxArrayData::Object {
                    properties, values, ..
                } => MxArrayData::Object {
                    class_name,
                    properties,
                    values,
                },
                other => {
                    *value.data_mut() = other;
                    return Err("mxSetClassName requires a struct or object array".into());
                }
            };
        *value.data_mut() = replacement;
        value.set_class_id(MxClassId::Object);
        Ok(())
    }

    pub fn get_property(
        &self,
        value: *const MxArray,
        element: usize,
        property: &str,
    ) -> Result<*mut MxArray, String> {
        let value = self.arena.get(value).map_err(|error| error.to_string())?;
        let numel = value.numel();
        let MxArrayData::Object {
            properties, values, ..
        } = value.data()
        else {
            return Err("mxArray is not an object array".into());
        };
        let property = properties
            .iter()
            .position(|candidate| candidate == property)
            .ok_or_else(|| format!("unknown object property '{property}'"))?;
        if element >= numel {
            return Err("object property index exceeds array bounds".into());
        }
        Ok(values[property * numel + element]
            .as_deref()
            .map(|value| std::ptr::from_ref(value).cast_mut())
            .unwrap_or(std::ptr::null_mut()))
    }

    pub fn set_property(
        &mut self,
        value: *mut MxArray,
        element: usize,
        property: &str,
        child: *mut MxArray,
    ) -> Result<(), String> {
        let (numel, property_index) = {
            let value = self.arena.get(value).map_err(|error| error.to_string())?;
            let MxArrayData::Object { properties, .. } = value.data() else {
                return Err("mxArray is not an object array".into());
            };
            if element >= value.numel() {
                return Err("object property index exceeds array bounds".into());
            }
            let property_index = properties
                .iter()
                .position(|candidate| candidate == property)
                .ok_or_else(|| format!("unknown object property '{property}'"))?;
            (value.numel(), property_index)
        };
        let child = if child.is_null() {
            None
        } else {
            Some(self.arena.take(child).map_err(|error| error.to_string())?)
        };
        let value = self
            .arena
            .get_mut(value)
            .map_err(|error| error.to_string())?;
        let MxArrayData::Object { values, .. } = value.data_mut() else {
            unreachable!("validated object array")
        };
        values[property_index * numel + element] = child;
        Ok(())
    }
}

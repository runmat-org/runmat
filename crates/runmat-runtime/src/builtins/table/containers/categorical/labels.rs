//! Borrowed categorical observation labels, using the same code decoder as owned labels.

use super::*;

pub(crate) struct CategoricalLabelView<'a> {
    categories: &'a [String],
    codes: &'a Tensor,
}

pub(crate) fn categorical_label_view(
    object: &ObjectInstance,
) -> BuiltinResult<CategoricalLabelView<'_>> {
    if !object.is_class(CATEGORICAL_CLASS) {
        return Err(invalid_variable("categorical: expected categorical object"));
    }
    let categories = match object.properties.get("Categories") {
        Some(Value::StringArray(array)) => &array.data,
        _ => return Err(invalid_variable("categorical: missing Categories property")),
    };
    let codes = match object.properties.get("Codes") {
        Some(Value::Tensor(tensor)) => tensor,
        _ => return Err(invalid_variable("categorical: missing Codes property")),
    };
    Ok(CategoricalLabelView { categories, codes })
}

impl<'a> CategoricalLabelView<'a> {
    pub(crate) fn len(&self) -> usize {
        self.codes.len()
    }

    /// Undefined or out-of-range codes retain the empty-label behavior of
    /// `categorical_labels`; malformed physical storage remains an error.
    pub(crate) fn label_at(&self, index: usize) -> BuiltinResult<&'a str> {
        let code = self
            .codes
            .numeric_value_at(index)
            .ok_or_else(|| invalid_variable("categorical: invalid numeric code storage"))?;
        Ok(category_label_ref_for_numeric_code(code, self.categories).unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntegerStorage;

    #[test]
    fn borrowed_labels_match_owned_labels_for_integer_and_undefined_codes() {
        let codes = Tensor::from_f32(vec![2.0, f32::NAN, 1.5], vec![1, 3]).unwrap();
        let mut object = ObjectInstance::new(CATEGORICAL_CLASS.to_string());
        object
            .properties
            .insert("Codes".into(), Value::Tensor(codes));
        object.properties.insert(
            "Categories".into(),
            Value::StringArray(
                StringArray::new(vec!["one".into(), "two".into()], vec![1, 2]).unwrap(),
            ),
        );
        let view = categorical_label_view(&object).unwrap();
        let borrowed = (0..view.len())
            .map(|index| view.label_at(index).unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(borrowed, categorical_object_labels(&object).unwrap());
        assert_eq!(borrowed, ["two", "", ""]);

        object.properties.insert(
            "Codes".into(),
            Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).unwrap()),
        );
        assert_eq!(
            categorical_label_view(&object)
                .unwrap()
                .label_at(0)
                .unwrap(),
            "two"
        );
    }
}

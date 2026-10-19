use super::{invalid, selectors, EmptyFieldRule, VariableImportOptions, WhitespaceRule};
use super::{DELIMITED_IMPORT_CLASS, SPREADSHEET_IMPORT_CLASS, VARIABLE_IMPORT_CLASS};
use crate::BuiltinResult;
use runmat_value::{CellArray, ObjectInstance, StringArray, StructValue, Value};

const VARIABLE_STORAGE: &str = "__runmat_variable_options";
const SOURCE_ORIGIN: &str = "__runmat_variable_source_origin";
const MAX_EXACT_SOURCE_ORIGIN: u64 = 9_007_199_254_740_991;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ImportSource {
    DelimitedText,
    Spreadsheet,
}

#[derive(Clone, Debug)]
pub(crate) struct ImportOptions {
    source: ImportSource,
    base: StructValue,
    variables: Vec<VariableImportOptions>,
    source_origin: Option<usize>,
}

impl ImportOptions {
    pub(crate) fn from_legacy(base: StructValue) -> BuiltinResult<Self> {
        let source = match base.fields.get("FileType") {
            Some(value) if selectors::scalar_text(value)?.eq_ignore_ascii_case("spreadsheet") => {
                ImportSource::Spreadsheet
            }
            _ => ImportSource::DelimitedText,
        };
        Self::from_parts(source, base, None)
    }

    pub(crate) fn from_detected(base: StructValue, source_origin: usize) -> BuiltinResult<Self> {
        let mut options = Self::from_legacy(base)?;
        options.source_origin = Some(source_origin);
        Ok(options)
    }

    pub(crate) fn decode(value: &Value) -> BuiltinResult<Option<Self>> {
        match value {
            Value::Struct(value) => Self::from_legacy(value.clone()).map(Some),
            Value::Object(value)
                if value.is_class(DELIMITED_IMPORT_CLASS)
                    || value.is_class(SPREADSHEET_IMPORT_CLASS) =>
            {
                let source = if value.is_class(SPREADSHEET_IMPORT_CLASS) {
                    ImportSource::Spreadsheet
                } else {
                    ImportSource::DelimitedText
                };
                let mut base = StructValue::new();
                let mut stored = None;
                let mut source_origin = None;
                for (key, property) in &value.properties {
                    if key == VARIABLE_STORAGE {
                        stored = Some(property);
                    } else if key == SOURCE_ORIGIN {
                        source_origin = Some(decode_source_origin(property)?);
                    } else {
                        base.insert(key.clone(), property.clone());
                    }
                }
                let mut options = Self::from_parts(source, base, stored)?;
                options.source_origin = source_origin;
                Ok(Some(options))
            }
            _ => Ok(None),
        }
    }

    fn from_parts(
        source: ImportSource,
        base: StructValue,
        stored: Option<&Value>,
    ) -> BuiltinResult<Self> {
        let names = match base.fields.get("VariableNames") {
            Some(value) => selectors::text_list(value)?,
            None => {
                let count = base
                    .fields
                    .get("NumVariables")
                    .map(|value| match value {
                        Value::Num(number)
                            if number.is_finite()
                                && *number >= 0.0
                                && number.fract() == 0.0
                                && *number < usize::MAX as f64 =>
                        {
                            Ok(*number as usize)
                        }
                        _ => Err(invalid("import options: invalid NumVariables")),
                    })
                    .transpose()?
                    .unwrap_or(0);
                (1..=count).map(|index| format!("Var{index}")).collect()
            }
        };
        let types = match base.fields.get("VariableTypes") {
            Some(value) => selectors::text_list(value)?,
            None => Vec::new(),
        };
        if types.len() > names.len() {
            return Err(invalid(
                "import options: VariableTypes exceeds VariableNames",
            ));
        }
        let variables = if let Some(Value::Cell(stored)) = stored {
            if stored.data.len() != names.len() {
                return Err(invalid("import options: variable-option count is stale"));
            }
            stored
                .data
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let Value::Struct(fields) = value else {
                        return Err(invalid("import options: malformed variable option"));
                    };
                    let mut variable = VariableImportOptions::new(
                        names[index].clone(),
                        types.get(index).cloned().unwrap_or_else(|| "auto".into()),
                    );
                    for (name, value) in &fields.fields {
                        variable.apply(name, value)?;
                    }
                    variable.validate()?;
                    Ok(variable)
                })
                .collect::<BuiltinResult<Vec<_>>>()?
        } else if stored.is_some() {
            return Err(invalid("import options: malformed variable-option storage"));
        } else {
            names
                .into_iter()
                .enumerate()
                .map(|(index, name)| {
                    VariableImportOptions::new(
                        name,
                        types.get(index).cloned().unwrap_or_else(|| "auto".into()),
                    )
                })
                .collect()
        };
        Ok(Self {
            source,
            base,
            variables,
            source_origin: None,
        })
    }

    pub(crate) fn base(&self) -> &StructValue {
        &self.base
    }
    pub(crate) fn variables(&self) -> &[VariableImportOptions] {
        &self.variables
    }
    pub(crate) fn source(&self) -> ImportSource {
        self.source
    }

    pub(crate) fn source_origin(&self) -> Option<usize> {
        self.source_origin
    }

    pub(crate) fn to_value(&self) -> BuiltinResult<Value> {
        super::ensure_import_options_classes_registered();
        let class = match self.source {
            ImportSource::DelimitedText => DELIMITED_IMPORT_CLASS,
            ImportSource::Spreadsheet => SPREADSHEET_IMPORT_CLASS,
        };
        let mut object = ObjectInstance::new(class);
        for (key, value) in &self.base.fields {
            object.properties.insert(key.clone(), value.clone());
        }
        if let Some(source_origin) = self.source_origin {
            let source_origin = u64::try_from(source_origin).map_err(|_| {
                invalid("import options: variable source origin exceeds exact metadata range")
            })?;
            if source_origin > MAX_EXACT_SOURCE_ORIGIN {
                return Err(invalid(
                    "import options: variable source origin exceeds exact metadata range",
                ));
            }
            object
                .properties
                .insert(SOURCE_ORIGIN.into(), Value::Num(source_origin as f64));
        }
        let mut variable_storage = Vec::new();
        variable_storage
            .try_reserve_exact(self.variables.len())
            .map_err(|_| invalid("import options: variable metadata exceeds available memory"))?;
        for variable in &self.variables {
            variable_storage.push(Value::Struct(variable_fields(variable)?));
        }
        object.properties.insert(
            VARIABLE_STORAGE.into(),
            Value::Cell(
                CellArray::new(variable_storage, 1, self.variables.len())
                    .map_err(|error| invalid(format!("import options: {error}")))?,
            ),
        );
        Ok(Value::Object(object))
    }

    pub(crate) fn set_options(
        &mut self,
        selection: Option<&Value>,
        pairs: &[(String, Value)],
    ) -> BuiltinResult<()> {
        let indices = match selection {
            Some(selection) => selectors::resolve(selection, &self.names())?,
            None => (0..self.variables.len()).collect(),
        };
        if indices.is_empty() {
            return Err(invalid("setvaropts: variable selection is empty"));
        }
        // Apply to a private copy so one bad option never publishes a partially updated object.
        let mut proposed = self.variables.clone();
        for index in indices {
            for (name, value) in pairs {
                proposed[index].apply(name, value)?;
            }
            proposed[index].validate()?;
        }
        self.variables = proposed;
        Ok(())
    }

    pub(crate) fn set_types(&mut self, selection: &Value, types: &Value) -> BuiltinResult<()> {
        let indices = selectors::resolve(selection, &self.names())?;
        let supplied = selectors::text_list(types)?;
        if supplied.len() != 1 && supplied.len() != indices.len() {
            return Err(invalid(
                "setvartype: number of types must match selected variables",
            ));
        }
        let mut proposed = self.variables.clone();
        for (position, index) in indices.into_iter().enumerate() {
            let type_name = &supplied[if supplied.len() == 1 { 0 } else { position }];
            // Reuse the existing table import type decoder at the execution
            // adapter; the model never guesses a class from a string.
            if type_name.trim().is_empty() {
                return Err(invalid("setvartype: variable type cannot be empty"));
            }
            proposed[index].type_name = type_name.clone();
        }
        let names = proposed
            .iter()
            .map(|variable| variable.type_name.clone())
            .collect::<Vec<_>>();
        self.base.insert(
            "VariableTypes",
            Value::StringArray(
                StringArray::new(names, vec![1, proposed.len()])
                    .map_err(|error| invalid(format!("setvartype: {error}")))?,
            ),
        );
        self.variables = proposed;
        Ok(())
    }

    pub(crate) fn selected(&self, selection: &Value) -> BuiltinResult<Vec<&VariableImportOptions>> {
        selectors::resolve(selection, &self.names())?
            .into_iter()
            .map(|index| Ok(&self.variables[index]))
            .collect()
    }

    fn names(&self) -> Vec<String> {
        self.variables
            .iter()
            .map(|variable| variable.name.clone())
            .collect()
    }

    pub(crate) fn variable_value(variable: &VariableImportOptions) -> BuiltinResult<Value> {
        super::ensure_import_options_classes_registered();
        let mut object = ObjectInstance::new(VARIABLE_IMPORT_CLASS);
        object
            .properties
            .insert("Name".into(), Value::string(variable.name.clone()));
        object
            .properties
            .insert("Type".into(), Value::string(variable.type_name.clone()));
        for (key, value) in variable_fields(variable)?.fields {
            object.properties.insert(key, value);
        }
        Ok(Value::Object(object))
    }
}

fn decode_source_origin(value: &Value) -> BuiltinResult<usize> {
    let Value::Num(number) = value else {
        return Err(invalid("import options: invalid variable source origin"));
    };
    if !number.is_finite()
        || *number < 0.0
        || number.fract() != 0.0
        || *number > MAX_EXACT_SOURCE_ORIGIN as f64
    {
        return Err(invalid("import options: invalid variable source origin"));
    }
    usize::try_from(*number as u64)
        .map_err(|_| invalid("import options: variable source origin exceeds target index range"))
}

fn variable_fields(variable: &VariableImportOptions) -> BuiltinResult<StructValue> {
    let mut fields = StructValue::new();
    if !variable.treat_as_missing.is_empty() {
        let values = variable
            .treat_as_missing
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        fields.insert(
            "TreatAsMissing",
            Value::StringArray(
                StringArray::new(values, vec![1, variable.treat_as_missing.len()])
                    .map_err(|error| invalid(format!("import options: {error}")))?,
            ),
        );
    }
    if let Some(separator) = variable.decimal_separator {
        fields.insert("DecimalSeparator", Value::string(separator.to_string()));
    }
    if let Some(separator) = variable.thousands_separator {
        fields.insert(
            "ThousandsSeparator",
            Value::string(separator.map_or_else(String::new, |value| value.to_string())),
        );
    }
    if variable.empty_field_rule != EmptyFieldRule::Missing {
        fields.insert(
            "EmptyFieldRule",
            Value::String(
                match variable.empty_field_rule {
                    EmptyFieldRule::Missing => "missing",
                    EmptyFieldRule::Auto => "auto",
                    EmptyFieldRule::Error => "error",
                }
                .into(),
            ),
        );
    }
    if variable.whitespace_rule != WhitespaceRule::Trim {
        fields.insert(
            "WhitespaceRule",
            Value::String(
                match variable.whitespace_rule {
                    WhitespaceRule::Trim => "trim",
                    WhitespaceRule::Preserve => "preserve",
                }
                .into(),
            ),
        );
    }
    if let Some(value) = &variable.fill_value {
        fields.insert("FillValue", value.clone());
    }
    if let Some(format) = &variable.input_format {
        fields.insert("InputFormat", Value::string(format.clone()));
    }
    Ok(fields)
}

#[cfg(test)]
mod source_origin_tests {
    use super::*;

    #[test]
    fn serialized_origin_is_exact_and_fits_target_pointer_width() {
        let max_native = u64::try_from(usize::MAX).expect("supported target width");
        let admitted = max_native.min(MAX_EXACT_SOURCE_ORIGIN);
        let mut object = ObjectInstance::new(DELIMITED_IMPORT_CLASS);
        object
            .properties
            .insert(SOURCE_ORIGIN.into(), Value::Num(admitted as f64));
        let decoded = ImportOptions::decode(&Value::Object(object.clone()))
            .expect("admitted metadata")
            .expect("import options object");
        assert_eq!(
            decoded.source_origin(),
            Some(usize::try_from(admitted).unwrap())
        );

        if admitted < MAX_EXACT_SOURCE_ORIGIN {
            object
                .properties
                .insert(SOURCE_ORIGIN.into(), Value::Num((admitted + 1) as f64));
            assert!(ImportOptions::decode(&Value::Object(object)).is_err());
        }
    }

    #[test]
    fn unrecorded_and_explicit_zero_origins_remain_distinct_on_round_trip() {
        let legacy = ImportOptions::from_legacy(StructValue::new()).unwrap();
        assert_eq!(legacy.source_origin(), None);
        let legacy_value = legacy.to_value().unwrap();
        assert_eq!(
            ImportOptions::decode(&legacy_value)
                .unwrap()
                .unwrap()
                .source_origin(),
            None
        );

        let detected = ImportOptions::from_detected(StructValue::new(), 0).unwrap();
        let detected_value = detected.to_value().unwrap();
        assert_eq!(
            ImportOptions::decode(&detected_value)
                .unwrap()
                .unwrap()
                .source_origin(),
            Some(0)
        );
    }
}

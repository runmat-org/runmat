use super::{invalid, selectors, EmptyFieldRule, VariableImportOptions, WhitespaceRule};
use super::{DELIMITED_IMPORT_CLASS, SPREADSHEET_IMPORT_CLASS, VARIABLE_IMPORT_CLASS};
use crate::context::{
    host_allocation::{
        host_allocation_runtime_error, HostAllocationError, HostAllocationPlan,
        HostAllocationPolicy, HostAllocationReceipt,
    },
    RuntimeContext,
};
use crate::execution::{RootedValue, RootedValueLease, RootedValues};
use crate::BuiltinResult;
use runmat_value::{
    CellArray, ExactArrayInitializationError, ExactArrayStorage, SharedShape, StringArray,
    StructValue, Utf8Text, Value,
};

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

    pub(crate) fn decode(
        runtime: Option<&RuntimeContext>,
        value: &Value,
    ) -> BuiltinResult<Option<Self>> {
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
                let mut stored = None;
                let mut source_origin = None;
                for (key, property) in value.properties().iter() {
                    if key == VARIABLE_STORAGE {
                        stored = Some(property);
                    } else if key == SOURCE_ORIGIN {
                        source_origin = Some(decode_source_origin(property)?);
                    }
                }
                let fields = value
                    .properties()
                    .iter()
                    .filter(|(key, _)| *key != VARIABLE_STORAGE && *key != SOURCE_ORIGIN);
                let base = crate::execution::construct_struct_from_fields_in_optional_context(
                    runtime, fields,
                )?;
                let (base, _base_roots) = base.into_value_and_lease();
                let Value::Struct(base) = base else {
                    unreachable!("structure constructor publishes Struct")
                };
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
            if stored.data().len() != names.len() {
                return Err(invalid("import options: variable-option count is stale"));
            }
            stored
                .data()
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
                    for (name, value) in fields.fields.iter() {
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

    pub(crate) fn to_value(&self, runtime: Option<&RuntimeContext>) -> BuiltinResult<Value> {
        super::ensure_import_options_classes_registered();
        let class = match self.source {
            ImportSource::DelimitedText => DELIMITED_IMPORT_CLASS,
            ImportSource::Spreadsheet => SPREADSHEET_IMPORT_CLASS,
        };
        let source_origin = if let Some(source_origin) = self.source_origin {
            let source_origin = u64::try_from(source_origin).map_err(|_| {
                invalid("import options: variable source origin exceeds exact metadata range")
            })?;
            if source_origin > MAX_EXACT_SOURCE_ORIGIN {
                return Err(invalid(
                    "import options: variable source origin exceeds exact metadata range",
                ));
            }
            Some(Value::Num(source_origin as f64))
        } else {
            None
        };
        let variable_storage = variable_storage(runtime, &self.variables)?;
        let properties = self
            .base
            .fields
            .iter()
            .chain(source_origin.iter().map(|value| (SOURCE_ORIGIN, value)))
            .chain(std::iter::once((VARIABLE_STORAGE, &variable_storage)));
        let object = crate::execution::object::construct_object_from_borrowed_fields(
            runtime,
            class.display_name(),
            properties,
        )?;
        Ok(object.into_value_and_lease().0)
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

    pub(crate) fn set_types(
        &mut self,
        runtime: Option<&RuntimeContext>,
        selection: &Value,
        types: &Value,
    ) -> BuiltinResult<()> {
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
        let types = string_array_value(
            runtime,
            proposed.iter().map(|variable| variable.type_name.as_str()),
            proposed.len(),
        )?;
        let fields = self
            .base
            .fields
            .iter()
            .filter(|(name, _)| *name != "VariableTypes")
            .chain(std::iter::once(("VariableTypes", &types)));
        let base =
            crate::execution::construct_struct_from_fields_in_optional_context(runtime, fields)?;
        let (base, _base_roots) = base.into_value_and_lease();
        let Value::Struct(base) = base else {
            unreachable!("structure constructor publishes Struct")
        };
        self.base = base;
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

    pub(crate) fn variable_value(
        runtime: Option<&RuntimeContext>,
        variable: &VariableImportOptions,
    ) -> BuiltinResult<Value> {
        super::ensure_import_options_classes_registered();
        let name = text_value(runtime, &variable.name)?;
        let type_name = text_value(runtime, &variable.type_name)?;
        let fields = variable_fields(runtime, variable)?;
        let (fields, _field_roots) = fields.into_value_and_lease();
        let Value::Struct(fields) = fields else {
            unreachable!("variable-field constructor publishes Struct")
        };
        let properties = [("Name", &name), ("Type", &type_name)]
            .into_iter()
            .chain(fields.fields.iter());
        let object = crate::execution::object::construct_object_from_borrowed_fields(
            runtime,
            VARIABLE_IMPORT_CLASS.display_name(),
            properties,
        )?;
        Ok(object.into_value_and_lease().0)
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

fn variable_fields(
    runtime: Option<&RuntimeContext>,
    variable: &VariableImportOptions,
) -> BuiltinResult<RootedValue> {
    let treat_as_missing = if variable.treat_as_missing.is_empty() {
        None
    } else {
        Some(string_array_value(
            runtime,
            variable.treat_as_missing.iter().map(String::as_str),
            variable.treat_as_missing.len(),
        )?)
    };
    let decimal_separator = variable
        .decimal_separator
        .map(|separator| text_value(runtime, &separator.to_string()))
        .transpose()?;
    let thousands_separator = variable
        .thousands_separator
        .map(|separator| {
            text_value(
                runtime,
                &separator.map_or_else(String::new, |value| value.to_string()),
            )
        })
        .transpose()?;
    let empty_field_rule = (variable.empty_field_rule != EmptyFieldRule::Missing)
        .then(|| {
            text_value(
                runtime,
                match variable.empty_field_rule {
                    EmptyFieldRule::Missing => "missing",
                    EmptyFieldRule::Auto => "auto",
                    EmptyFieldRule::Error => "error",
                },
            )
        })
        .transpose()?;
    let whitespace_rule = (variable.whitespace_rule != WhitespaceRule::Trim)
        .then(|| {
            text_value(
                runtime,
                match variable.whitespace_rule {
                    WhitespaceRule::Trim => "trim",
                    WhitespaceRule::Preserve => "preserve",
                },
            )
        })
        .transpose()?;
    let input_format = variable
        .input_format
        .as_deref()
        .map(|format| text_value(runtime, format))
        .transpose()?;
    let fields = [
        treat_as_missing
            .as_ref()
            .map(|value| ("TreatAsMissing", value)),
        decimal_separator
            .as_ref()
            .map(|value| ("DecimalSeparator", value)),
        thousands_separator
            .as_ref()
            .map(|value| ("ThousandsSeparator", value)),
        empty_field_rule
            .as_ref()
            .map(|value| ("EmptyFieldRule", value)),
        whitespace_rule
            .as_ref()
            .map(|value| ("WhitespaceRule", value)),
        variable
            .fill_value
            .as_ref()
            .map(|value| ("FillValue", value)),
        input_format.as_ref().map(|value| ("InputFormat", value)),
    ];
    crate::execution::construct_struct_from_fields_in_optional_context(
        runtime,
        fields.into_iter().flatten(),
    )
}

fn variable_storage(
    runtime: Option<&RuntimeContext>,
    variables: &[VariableImportOptions],
) -> BuiltinResult<Value> {
    if !managed(runtime)? {
        let values = variables
            .iter()
            .map(|variable| {
                variable_fields(None, variable).map(|value| value.into_value_and_lease().0)
            })
            .collect::<BuiltinResult<Vec<_>>>()?;
        return CellArray::new(values, 1, variables.len())
            .map(Value::Cell)
            .map_err(|error| invalid(format!("import options: {error}")));
    }
    let runtime = runtime.expect("managed output has a selected context");
    let source_roots = root_variable_sources(runtime, variables)?;
    let shape = admitted_shape(runtime, &[1, variables.len()])?;
    let data = RootedValues::try_initialize_in_context(variables.len(), runtime, |index| {
        variable_fields(Some(runtime), &variables[index])
            .map(|value| value.into_value_and_lease().0)
    })?;
    let value = data.with_owned_output_storage(|data| {
        CellArray::from_admitted(data, shape)
            .map(Value::Cell)
            .map_err(|error| invalid(format!("import options: {error}")))
    })?;
    drop(source_roots);
    Ok(value)
}

fn root_variable_sources(
    runtime: &RuntimeContext,
    variables: &[VariableImportOptions],
) -> BuiltinResult<RootedValueLease> {
    let values = variables
        .iter()
        .filter_map(|variable| variable.fill_value.as_ref());
    RootedValueLease::for_values_iterator_in_context(values, runtime)
}

fn string_array_value<'a>(
    runtime: Option<&RuntimeContext>,
    values: impl Iterator<Item = &'a str> + Clone,
    count: usize,
) -> BuiltinResult<Value> {
    if !managed(runtime)? {
        let values = values.map(str::to_owned).collect::<Vec<_>>();
        return StringArray::new(values, vec![1, count])
            .map(Value::StringArray)
            .map_err(|error| invalid(format!("import options: {error}")));
    }
    let runtime = runtime.expect("managed output has a selected context");
    let mut physical = ExactArrayStorage::<runmat_value::StringElement>::requested_bytes(count)
        .map_err(|source| allocation(runtime, source.into()))?;
    physical = physical
        .checked_add(
            SharedShape::requested_bytes_for_rank(2)
                .map_err(|source| allocation(runtime, source.into()))?,
        )
        .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    let mut claims = usize::from(count != 0) + 1;
    for value in values.clone() {
        physical = physical
            .checked_add(
                Utf8Text::requested_bytes_for_capacity(value.len())
                    .map_err(|source| allocation(runtime, source.into()))?,
            )
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
        claims = claims
            .checked_add(usize::from(!value.is_empty()))
            .ok_or_else(|| allocation(runtime, HostAllocationError::Overflow))?;
    }
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
        claims,
    )
    .map_err(|source| allocation(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(|source| allocation(runtime, source))?,
    )
    .map_err(|source| allocation(runtime, source))?
    .ok_or_else(|| allocation(runtime, HostAllocationError::MissingManagedOwner))?;
    let permit = receipt
        .output_permit(runtime)
        .map_err(|source| allocation(runtime, source))?;
    let mut values = values;
    let data = ExactArrayStorage::try_initialize(count, &permit, |_| {
        receipt.checkpoint(runtime)?;
        let source = values.next().ok_or(HostAllocationError::StaleAuthority)?;
        Utf8Text::try_copy_str_exact_with(source, &permit, |_| receipt.checkpoint(runtime))
            .map(runmat_value::StringElement::Text)
            .map_err(|error| match error {
                ExactArrayInitializationError::Allocation(source) => {
                    HostAllocationError::Numeric(source)
                }
                ExactArrayInitializationError::Element(source) => source,
            })
    })
    .map_err(|error| match error {
        ExactArrayInitializationError::Allocation(source) => {
            allocation(runtime, HostAllocationError::Numeric(source))
        }
        ExactArrayInitializationError::Element(source) => allocation(runtime, source),
    })?;
    let shape = SharedShape::new_admitted_from_slice(&[1, count], &permit)
        .map_err(|source| allocation(runtime, source.into()))?;
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation(runtime, source))?;
    StringArray::from_admitted(data, shape)
        .map(Value::StringArray)
        .map_err(|error| invalid(format!("import options: {error}")))
}

fn text_value(runtime: Option<&RuntimeContext>, text: &str) -> BuiltinResult<Value> {
    if !managed(runtime)? {
        return Ok(Value::string(text));
    }
    let runtime = runtime.expect("managed output has a selected context");
    let physical = Utf8Text::requested_bytes_for_capacity(text.len())
        .map_err(|source| allocation(runtime, source.into()))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(physical).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
        usize::from(!text.is_empty()),
    )
    .map_err(|source| allocation(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(|source| allocation(runtime, source))?,
    )
    .map_err(|source| allocation(runtime, source))?
    .ok_or_else(|| allocation(runtime, HostAllocationError::MissingManagedOwner))?;
    let permit = receipt
        .output_permit(runtime)
        .map_err(|source| allocation(runtime, source))?;
    let value = Utf8Text::try_copy_str_exact_with(text, &permit, |_| receipt.checkpoint(runtime))
        .map_err(|error| match error {
        ExactArrayInitializationError::Allocation(source) => {
            allocation(runtime, HostAllocationError::Numeric(source))
        }
        ExactArrayInitializationError::Element(source) => allocation(runtime, source),
    })?;
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation(runtime, source))?;
    Ok(Value::String(value.into()))
}

fn admitted_shape(runtime: &RuntimeContext, shape: &[usize]) -> BuiltinResult<SharedShape> {
    let bytes = SharedShape::requested_bytes_for_rank(shape.len())
        .map_err(|source| allocation(runtime, source.into()))?;
    let retained = HostAllocationPlan::retained_exact_output_bytes(
        u64::try_from(bytes).map_err(|_| allocation(runtime, HostAllocationError::Overflow))?,
        1,
    )
    .map_err(|source| allocation(runtime, source))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(|source| allocation(runtime, source))?,
    )
    .map_err(|source| allocation(runtime, source))?
    .ok_or_else(|| allocation(runtime, HostAllocationError::MissingManagedOwner))?;
    let permit = receipt
        .output_permit(runtime)
        .map_err(|source| allocation(runtime, source))?;
    let shape = SharedShape::new_admitted_from_slice(shape, &permit)
        .map_err(|source| allocation(runtime, source.into()))?;
    receipt
        .checkpoint(runtime)
        .map_err(|source| allocation(runtime, source))?;
    Ok(shape)
}

fn managed(runtime: Option<&RuntimeContext>) -> BuiltinResult<bool> {
    HostAllocationReceipt::policy(runtime)
        .map(|policy| policy == HostAllocationPolicy::Managed)
        .map_err(|source| match runtime {
            Some(runtime) => allocation(runtime, source),
            None => host_allocation_runtime_error(
                None,
                "RunMat:ImportOptions:Allocation",
                "import-options output policy is unavailable",
                source,
            ),
        })
}

fn allocation(runtime: &RuntimeContext, source: HostAllocationError) -> crate::RuntimeError {
    host_allocation_runtime_error(
        Some(runtime),
        "RunMat:ImportOptions:Allocation",
        "import-options output admission failed",
        source,
    )
}

#[cfg(test)]
mod source_origin_tests {
    use super::*;

    #[test]
    fn serialized_origin_is_exact_and_fits_target_pointer_width() {
        let max_native = u64::try_from(usize::MAX).expect("supported target width");
        let admitted = max_native.min(MAX_EXACT_SOURCE_ORIGIN);
        let mut builder = runmat_value::UnboundedObjectBuilder::new(DELIMITED_IMPORT_CLASS);
        builder.insert(SOURCE_ORIGIN.into(), Value::Num(admitted as f64));
        let object = builder.finish();
        let decoded = ImportOptions::decode(None, &Value::Object(object.clone()))
            .expect("admitted metadata")
            .expect("import options object");
        assert_eq!(
            decoded.source_origin(),
            Some(usize::try_from(admitted).unwrap())
        );

        if admitted < MAX_EXACT_SOURCE_ORIGIN {
            let mut builder = object
                .into_unbounded_builder()
                .expect("ordinary test object has no admitted aliases");
            builder.insert(SOURCE_ORIGIN.into(), Value::Num((admitted + 1) as f64));
            let object = builder.finish();
            assert!(ImportOptions::decode(None, &Value::Object(object)).is_err());
        }
    }

    #[test]
    fn unrecorded_and_explicit_zero_origins_remain_distinct_on_round_trip() {
        let legacy = ImportOptions::from_legacy(StructValue::new()).unwrap();
        assert_eq!(legacy.source_origin(), None);
        let legacy_value = legacy.to_value(None).unwrap();
        assert_eq!(
            ImportOptions::decode(None, &legacy_value)
                .unwrap()
                .unwrap()
                .source_origin(),
            None
        );

        let detected = ImportOptions::from_detected(StructValue::new(), 0).unwrap();
        let detected_value = detected.to_value(None).unwrap();
        assert_eq!(
            ImportOptions::decode(None, &detected_value)
                .unwrap()
                .unwrap()
                .source_origin(),
            Some(0)
        );
    }
}

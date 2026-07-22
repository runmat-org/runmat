use super::*;

#[derive(Clone, Default)]
pub(in crate::builtins::table) struct ParquetReadOptions {
    selected_variable_names: Option<Vec<String>>,
    row_groups: Option<Vec<usize>>,
    output_type: ParquetOutputType,
    row_times: Option<String>,
    row_filter: Option<Value>,
}

impl ParquetReadOptions {
    pub(in crate::builtins::table) fn parse(args: &[Value]) -> BuiltinResult<Self> {
        let mut options = Self::default();
        let mut idx = 0usize;
        while idx < args.len() {
            if idx + 1 >= args.len() {
                return Err(invalid_argument(
                    "parquetread: name-value options must be provided in pairs",
                ));
            }
            let name = scalar_text(&args[idx], "parquetread option")?;
            options.apply(&name, &args[idx + 1])?;
            idx += 2;
        }
        Ok(options)
    }

    fn apply(&mut self, name: &str, value: &Value) -> BuiltinResult<()> {
        if name.eq_ignore_ascii_case("SelectedVariableNames") {
            self.selected_variable_names = Some(string_list(value)?);
        } else if name.eq_ignore_ascii_case("RowGroups") {
            self.row_groups = Some(one_based_indices(value, "RowGroups")?);
        } else if name.eq_ignore_ascii_case("OutputType") {
            self.output_type = ParquetOutputType::parse(value)?;
        } else if name.eq_ignore_ascii_case("RowTimes") {
            self.row_times = Some(scalar_text(value, "RowTimes")?);
        } else if name.eq_ignore_ascii_case("RowFilter") {
            self.row_filter = Some(value.clone());
        } else {
            return Err(invalid_argument(format!(
                "parquetread: unsupported option '{name}'"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum ParquetOutputType {
    #[default]
    Table,
    Timetable,
}

impl ParquetOutputType {
    fn parse(value: &Value) -> BuiltinResult<Self> {
        let text = scalar_text(value, "OutputType")?;
        match text.trim().to_ascii_lowercase().as_str() {
            "table" => Ok(Self::Table),
            "timetable" => Ok(Self::Timetable),
            other => Err(invalid_argument(format!(
                "parquetread: unsupported OutputType '{other}'"
            ))),
        }
    }
}

pub(in crate::builtins::table) async fn read_parquet_table(
    path: &Path,
    options: &ParquetReadOptions,
) -> BuiltinResult<Value> {
    read_parquet_table_impl(path, options).await
}

pub(in crate::builtins::table) async fn parquet_file_info(path: &Path) -> BuiltinResult<Value> {
    parquet_file_info_impl(path).await
}

#[cfg(target_arch = "wasm32")]
async fn read_parquet_table_impl(
    _path: &Path,
    _options: &ParquetReadOptions,
) -> BuiltinResult<Value> {
    Err(table_error(
        &TABLE_ERROR_UNSUPPORTED_FILE,
        "parquetread: Parquet decoding is available in native host runtimes",
    ))
}

#[cfg(target_arch = "wasm32")]
async fn parquet_file_info_impl(_path: &Path) -> BuiltinResult<Value> {
    Err(table_error(
        &TABLE_ERROR_UNSUPPORTED_FILE,
        "parquetinfo: Parquet metadata inspection is available in native host runtimes",
    ))
}

#[cfg(not(target_arch = "wasm32"))]
async fn read_parquet_table_impl(
    path: &Path,
    options: &ParquetReadOptions,
) -> BuiltinResult<Value> {
    use ::parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ProjectionMask};
    use arrow_array::{
        types::{Date32Type, Date64Type, Int32Type},
        Array, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int16Array, Int32Array,
        Int64Array, Int8Array, LargeStringArray, RecordBatch, StringArray as ArrowStringArray,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    };
    use arrow_schema::DataType;

    if options.output_type == ParquetOutputType::Timetable {
        return Err(invalid_argument(
            "parquetread: OutputType 'timetable' requires timetable row-time synthesis and is not supported yet",
        ));
    }
    if options.row_times.is_some() {
        return Err(invalid_argument(
            "parquetread: RowTimes is only valid with OutputType 'timetable'",
        ));
    }
    if options.row_filter.is_some() {
        return Err(invalid_argument(
            "parquetread: RowFilter is not supported until rowfilter predicate execution is available",
        ));
    }

    let bytes = bytes::Bytes::from(read_file_bytes(path).await?);
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(|err| {
        table_error(
            &TABLE_ERROR_UNSUPPORTED_FILE,
            format!(
                "parquetread: unable to read parquet schema '{}': {err}",
                path.display()
            ),
        )
    })?;
    let schema = builder.schema().clone();
    let (selected_indices, selected_names) =
        selected_columns(schema.fields().iter().map(|field| field.name()), options)?;
    let selected_projection =
        ProjectionMask::roots(builder.parquet_schema(), selected_indices.to_vec());
    let row_groups = selected_row_groups(builder.metadata().num_row_groups(), options)?;
    let batch_reader = builder
        .with_projection(selected_projection)
        .with_row_groups(row_groups)
        .build()
        .map_err(|err| {
            table_error(
                &TABLE_ERROR_UNSUPPORTED_FILE,
                format!(
                    "parquetread: unable to decode parquet file '{}': {err}",
                    path.display()
                ),
            )
        })?;

    let mut columns = selected_names
        .iter()
        .map(|name| ParquetColumnBuilder::new(name.clone()))
        .collect::<Vec<_>>();

    for batch in batch_reader {
        let batch = batch.map_err(|err| {
            table_error(
                &TABLE_ERROR_UNSUPPORTED_FILE,
                format!(
                    "parquetread: unable to decode record batch '{}': {err}",
                    path.display()
                ),
            )
        })?;
        append_record_batch(&batch, &selected_names, &mut columns)?;
    }

    let values = columns
        .into_iter()
        .map(ParquetColumnBuilder::into_value)
        .collect::<BuiltinResult<Vec<_>>>()?;
    return table_from_columns_with_properties(selected_names, values, None);

    fn append_record_batch(
        batch: &RecordBatch,
        column_names: &[String],
        columns: &mut [ParquetColumnBuilder],
    ) -> BuiltinResult<()> {
        let batch_schema = batch.schema();
        for (name, builder) in column_names.iter().zip(columns.iter_mut()) {
            let (idx, _) = batch_schema.column_with_name(name).ok_or_else(|| {
                invalid_variable(format!(
                    "parquetread: decoded batch is missing projected column '{name}'"
                ))
            })?;
            let array = batch.column(idx).as_ref();
            match array.data_type() {
                DataType::Boolean => {
                    let values = array
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .ok_or_else(|| invalid_variable("parquetread: invalid boolean column"))?;
                    for row in 0..values.len() {
                        builder.push_bool((!values.is_null(row)).then(|| values.value(row)));
                    }
                }
                DataType::Int8 => {
                    push_numeric::<Int8Array>(array, builder, |a, i| a.value(i) as f64)?
                }
                DataType::Int16 => {
                    push_numeric::<Int16Array>(array, builder, |a, i| a.value(i) as f64)?
                }
                DataType::Int32 => {
                    push_numeric::<Int32Array>(array, builder, |a, i| a.value(i) as f64)?
                }
                DataType::Int64 => {
                    push_numeric::<Int64Array>(array, builder, |a, i| a.value(i) as f64)?
                }
                DataType::UInt8 => {
                    push_numeric::<UInt8Array>(array, builder, |a, i| a.value(i) as f64)?
                }
                DataType::UInt16 => {
                    push_numeric::<UInt16Array>(array, builder, |a, i| a.value(i) as f64)?
                }
                DataType::UInt32 => {
                    push_numeric::<UInt32Array>(array, builder, |a, i| a.value(i) as f64)?
                }
                DataType::UInt64 => {
                    push_numeric::<UInt64Array>(array, builder, |a, i| a.value(i) as f64)?
                }
                DataType::Float32 => {
                    push_numeric::<Float32Array>(array, builder, |a, i| a.value(i) as f64)?
                }
                DataType::Float64 => {
                    push_numeric::<Float64Array>(array, builder, |a, i| a.value(i))?
                }
                DataType::Utf8 => {
                    let values = array
                        .as_any()
                        .downcast_ref::<ArrowStringArray>()
                        .ok_or_else(|| invalid_variable("parquetread: invalid string column"))?;
                    for row in 0..values.len() {
                        builder.push_text(
                            (!values.is_null(row)).then(|| values.value(row).to_string()),
                        );
                    }
                }
                DataType::LargeUtf8 => {
                    let values = array
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .ok_or_else(|| {
                            invalid_variable("parquetread: invalid large string column")
                        })?;
                    for row in 0..values.len() {
                        builder.push_text(
                            (!values.is_null(row)).then(|| values.value(row).to_string()),
                        );
                    }
                }
                DataType::Date32 => push_temporal::<arrow_array::PrimitiveArray<Date32Type>>(
                    array,
                    builder,
                    |a, i| days_since_unix_to_datenum(a.value(i) as f64),
                )?,
                DataType::Date64 => push_temporal::<arrow_array::PrimitiveArray<Date64Type>>(
                    array,
                    builder,
                    |a, i| millis_since_unix_to_datenum(a.value(i) as f64),
                )?,
                DataType::Timestamp(unit, _) => match unit {
                    arrow_schema::TimeUnit::Second => {
                        push_temporal::<TimestampSecondArray>(array, builder, |a, i| {
                            seconds_since_unix_to_datenum(a.value(i) as f64)
                        })?
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        push_temporal::<TimestampMillisecondArray>(array, builder, |a, i| {
                            millis_since_unix_to_datenum(a.value(i) as f64)
                        })?
                    }
                    arrow_schema::TimeUnit::Microsecond => {
                        push_temporal::<TimestampMicrosecondArray>(array, builder, |a, i| {
                            seconds_since_unix_to_datenum(a.value(i) as f64 / 1_000_000.0)
                        })?
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        push_temporal::<TimestampNanosecondArray>(array, builder, |a, i| {
                            seconds_since_unix_to_datenum(a.value(i) as f64 / 1_000_000_000.0)
                        })?
                    }
                },
                DataType::Dictionary(_, value_type)
                    if matches!(value_type.as_ref(), DataType::Utf8) =>
                {
                    let values = array
                        .as_any()
                        .downcast_ref::<DictionaryArray<Int32Type>>()
                        .ok_or_else(|| {
                            invalid_variable(
                                "parquetread: only int32-key UTF-8 dictionary columns are supported",
                            )
                        })?;
                    let dictionary = values
                        .values()
                        .as_any()
                        .downcast_ref::<ArrowStringArray>()
                        .ok_or_else(|| {
                            invalid_variable("parquetread: invalid dictionary values")
                        })?;
                    for row in 0..values.len() {
                        if values.is_null(row) {
                            builder.push_text(None);
                        } else {
                            let key = values.keys().value(row) as usize;
                            builder.push_text(Some(dictionary.value(key).to_string()));
                        }
                    }
                }
                other => {
                    return Err(invalid_variable(format!(
                        "parquetread: unsupported parquet column type {other:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn push_numeric<A>(
        array: &dyn Array,
        builder: &mut ParquetColumnBuilder,
        value_at: impl Fn(&A, usize) -> f64,
    ) -> BuiltinResult<()>
    where
        A: Array + 'static,
    {
        let values = array
            .as_any()
            .downcast_ref::<A>()
            .ok_or_else(|| invalid_variable("parquetread: invalid numeric column"))?;
        for row in 0..values.len() {
            builder.push_number((!values.is_null(row)).then(|| value_at(values, row)));
        }
        Ok(())
    }

    fn push_temporal<A>(
        array: &dyn Array,
        builder: &mut ParquetColumnBuilder,
        value_at: impl Fn(&A, usize) -> f64,
    ) -> BuiltinResult<()>
    where
        A: Array + 'static,
    {
        let values = array
            .as_any()
            .downcast_ref::<A>()
            .ok_or_else(|| invalid_variable("parquetread: invalid temporal column"))?;
        for row in 0..values.len() {
            builder.push_datetime((!values.is_null(row)).then(|| value_at(values, row)));
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn parquet_file_info_impl(path: &Path) -> BuiltinResult<Value> {
    use ::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use ::parquet::basic::{Compression, Encoding};

    let bytes = read_file_bytes(path).await?;
    let file_size = bytes.len() as f64;
    let builder =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes)).map_err(|err| {
            table_error(
                &TABLE_ERROR_UNSUPPORTED_FILE,
                format!(
                    "parquetinfo: unable to open parquet file '{}': {err}",
                    path.display()
                ),
            )
        })?;
    let metadata = builder.metadata();
    let file_metadata = metadata.file_metadata();
    let schema_descr = file_metadata.schema_descr();
    let row_group_count = metadata.num_row_groups();
    let column_count = schema_descr.num_columns();

    let mut info = StructValue::new();
    info.insert(
        "Filename",
        Value::String(path.to_string_lossy().to_string()),
    );
    info.insert("FileSize", Value::Num(file_size));
    info.insert("NumRows", Value::Num(file_metadata.num_rows() as f64));
    info.insert("NumRowGroups", Value::Num(row_group_count as f64));
    info.insert("NumVariables", Value::Num(column_count as f64));
    info.insert(
        "VariableNames",
        string_array_value(
            (0..column_count)
                .map(|idx| schema_descr.column(idx).name().to_string())
                .collect(),
            "parquetinfo",
        )?,
    );
    info.insert(
        "VariableTypes",
        string_array_value(
            (0..column_count)
                .map(|idx| schema_descr.column(idx).physical_type().to_string())
                .collect(),
            "parquetinfo",
        )?,
    );
    info.insert("Version", Value::Num(file_metadata.version() as f64));
    info.insert(
        "CreatedBy",
        Value::String(file_metadata.created_by().unwrap_or("").to_string()),
    );
    info.insert("RowGroups", parquet_row_group_info(metadata)?);
    info.insert(
        "Compression",
        parquet_column_property(metadata, compression_label)?,
    );
    info.insert(
        "Encoding",
        parquet_column_property(metadata, encoding_label)?,
    );
    return Ok(Value::Struct(info));

    fn parquet_row_group_info(
        metadata: &::parquet::file::metadata::ParquetMetaData,
    ) -> BuiltinResult<Value> {
        let mut row_groups = StructValue::new();
        for idx in 0..metadata.num_row_groups() {
            let group = metadata.row_group(idx);
            let mut fields = StructValue::new();
            fields.insert("NumRows", Value::Num(group.num_rows() as f64));
            fields.insert("TotalByteSize", Value::Num(group.total_byte_size() as f64));
            fields.insert("NumColumns", Value::Num(group.num_columns() as f64));
            row_groups.insert(format!("RowGroup{}", idx + 1), Value::Struct(fields));
        }
        Ok(Value::Struct(row_groups))
    }

    fn parquet_column_property(
        metadata: &::parquet::file::metadata::ParquetMetaData,
        label: impl Fn(&::parquet::file::metadata::ColumnChunkMetaData) -> String,
    ) -> BuiltinResult<Value> {
        let mut labels = Vec::new();
        if metadata.num_row_groups() == 0 {
            return string_array_value(labels, "parquetinfo");
        }
        let row_group = metadata.row_group(0);
        for idx in 0..row_group.num_columns() {
            labels.push(label(row_group.column(idx)));
        }
        string_array_value(labels, "parquetinfo")
    }

    fn compression_label(column: &::parquet::file::metadata::ColumnChunkMetaData) -> String {
        match column.compression() {
            Compression::UNCOMPRESSED => "UNCOMPRESSED",
            Compression::SNAPPY => "SNAPPY",
            Compression::GZIP(_) => "GZIP",
            Compression::LZO => "LZO",
            Compression::BROTLI(_) => "BROTLI",
            Compression::LZ4 => "LZ4",
            Compression::ZSTD(_) => "ZSTD",
            Compression::LZ4_RAW => "LZ4_RAW",
        }
        .to_string()
    }

    fn encoding_label(column: &::parquet::file::metadata::ColumnChunkMetaData) -> String {
        column
            .encodings()
            .next()
            .unwrap_or(Encoding::PLAIN)
            .to_string()
    }
}

#[derive(Clone)]
enum ParquetColumnBuilder {
    Number { values: Vec<f64> },
    Logical { values: Vec<u8> },
    Text { values: Vec<String> },
    DateTime { values: Vec<f64> },
    Empty,
}

impl ParquetColumnBuilder {
    fn new(_name: String) -> Self {
        Self::Empty
    }

    fn push_number(&mut self, value: Option<f64>) {
        self.push_as(ParquetColumnKind::Number, value.map(ParquetCell::Number));
    }

    fn push_bool(&mut self, value: Option<bool>) {
        self.push_as(ParquetColumnKind::Logical, value.map(ParquetCell::Logical));
    }

    fn push_text(&mut self, value: Option<String>) {
        self.push_as(ParquetColumnKind::Text, value.map(ParquetCell::Text));
    }

    fn push_datetime(&mut self, value: Option<f64>) {
        self.push_as(
            ParquetColumnKind::DateTime,
            value.map(ParquetCell::DateTime),
        );
    }

    fn push_as(&mut self, kind: ParquetColumnKind, cell: Option<ParquetCell>) {
        match self {
            Self::Empty => {
                *self = match kind {
                    ParquetColumnKind::Number => Self::Number { values: Vec::new() },
                    ParquetColumnKind::Logical => Self::Logical { values: Vec::new() },
                    ParquetColumnKind::Text => Self::Text { values: Vec::new() },
                    ParquetColumnKind::DateTime => Self::DateTime { values: Vec::new() },
                };
                self.push_as(kind, cell);
            }
            Self::Number { values, .. } => match cell {
                Some(ParquetCell::Number(value)) => values.push(value),
                None => values.push(f64::NAN),
                _ => unreachable!("parquetread column kind changed within one Arrow array"),
            },
            Self::Logical { values, .. } => match cell {
                Some(ParquetCell::Logical(value)) => values.push(u8::from(value)),
                None => values.push(0),
                _ => unreachable!("parquetread column kind changed within one Arrow array"),
            },
            Self::Text { values, .. } => match cell {
                Some(ParquetCell::Text(value)) => values.push(value),
                None => values.push(String::new()),
                _ => unreachable!("parquetread column kind changed within one Arrow array"),
            },
            Self::DateTime { values, .. } => match cell {
                Some(ParquetCell::DateTime(value)) => values.push(value),
                None => values.push(f64::NAN),
                _ => unreachable!("parquetread column kind changed within one Arrow array"),
            },
        }
    }

    fn into_value(self) -> BuiltinResult<Value> {
        match self {
            Self::Number { values } => Tensor::new(values.clone(), vec![values.len(), 1])
                .map(Value::Tensor)
                .map_err(|err| invalid_variable(format!("parquetread: {err}"))),
            Self::Logical { values } => LogicalArray::new(values.clone(), vec![values.len(), 1])
                .map(Value::LogicalArray)
                .map_err(|err| invalid_variable(format!("parquetread: {err}"))),
            Self::Text { values } => StringArray::new(values.clone(), vec![values.len(), 1])
                .map(Value::StringArray)
                .map_err(|err| invalid_variable(format!("parquetread: {err}"))),
            Self::DateTime { values } => {
                let tensor = Tensor::new(values.clone(), vec![values.len(), 1])
                    .map_err(|err| invalid_variable(format!("parquetread: {err}")))?;
                crate::builtins::datetime::datetime_object_from_serial_tensor(
                    tensor,
                    "yyyy-MM-dd HH:mm:ss",
                )
            }
            Self::Empty => Tensor::new(Vec::new(), vec![0, 1])
                .map(Value::Tensor)
                .map_err(|err| invalid_variable(format!("parquetread: {err}"))),
        }
    }
}

#[derive(Clone, Copy)]
enum ParquetColumnKind {
    Number,
    Logical,
    Text,
    DateTime,
}

enum ParquetCell {
    Number(f64),
    Logical(bool),
    Text(String),
    DateTime(f64),
}

fn selected_columns<'a>(
    names: impl Iterator<Item = &'a String>,
    options: &ParquetReadOptions,
) -> BuiltinResult<(Vec<usize>, Vec<String>)> {
    let all_names = names.cloned().collect::<Vec<_>>();
    let Some(selected) = &options.selected_variable_names else {
        return Ok(((0..all_names.len()).collect(), all_names));
    };
    let mut requested = HashSet::new();
    for name in selected {
        if !requested.insert(name.clone()) {
            return Err(invalid_argument(format!(
                "parquetread: SelectedVariableNames contains duplicate variable '{name}'"
            )));
        }
        if !all_names.iter().any(|candidate| candidate == name) {
            return Err(invalid_argument(format!(
                "parquetread: SelectedVariableNames contains unknown variable '{name}'"
            )));
        }
    }
    let projection_indices = all_names
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| requested.contains(name).then_some(idx))
        .collect();
    Ok((projection_indices, selected.clone()))
}

fn selected_row_groups(
    available: usize,
    options: &ParquetReadOptions,
) -> BuiltinResult<Vec<usize>> {
    let Some(row_groups) = &options.row_groups else {
        return Ok((0..available).collect());
    };
    for index in row_groups {
        if *index >= available {
            return Err(invalid_argument(format!(
                "parquetread: RowGroups index {} exceeds number of row groups {}",
                index + 1,
                available
            )));
        }
    }
    Ok(row_groups.clone())
}

fn one_based_indices(value: &Value, context: &str) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Num(number) => one_based_index_from_number(*number, context).map(|idx| vec![idx]),
        Value::Tensor(tensor) => tensor
            .data
            .iter()
            .map(|value| one_based_index_from_number(*value, context))
            .collect(),
        other => Err(invalid_argument(format!(
            "parquetread: {context} must be a positive integer scalar or numeric vector, got {other:?}"
        ))),
    }
}

fn one_based_index_from_number(value: f64, context: &str) -> BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 || value < 1.0 {
        return Err(invalid_argument(format!(
            "parquetread: {context} entries must be positive integers"
        )));
    }
    Ok(value as usize - 1)
}

fn days_since_unix_to_datenum(days: f64) -> f64 {
    719_529.0 + days
}

fn millis_since_unix_to_datenum(millis: f64) -> f64 {
    seconds_since_unix_to_datenum(millis / 1_000.0)
}

fn seconds_since_unix_to_datenum(seconds: f64) -> f64 {
    719_529.0 + seconds / 86_400.0
}

//! Parsed host controls precede the sole admitted numeric RowGroups readback.
#[cfg(test)]
mod tests;
use super::*;
use runmat_builtins::{ParquetOutputType, ParquetReadControl};

#[derive(Clone, Default)]
pub(in crate::builtins::table) struct ParquetReadOptions {
    pub(super) selected_variable_names: Option<Vec<String>>,
    pub(super) row_groups: Option<Vec<usize>>,
    pub(super) output_type: ParquetOutputType,
    pub(super) row_times: Option<String>,
    pub(super) row_filter: Option<Value>,
}

enum RowGroupInput<'a> {
    Parsed(Vec<usize>),
    Resident(&'a Value),
}

impl<'a> RowGroupInput<'a> {
    fn parse(value: &'a Value) -> BuiltinResult<Self> {
        if matches!(value, Value::GpuTensor(_)) {
            let fact = crate::value_fact::value_fact(value);
            if !fact
                .numeric()
                .is_some_and(|numeric| numeric.domain == runmat_types::NumericDomain::Real)
            {
                return Err(invalid_argument(
                    "parquetread: RowGroups must be real numeric indices",
                ));
            }
            Ok(Self::Resident(value))
        } else {
            one_based_indices(value, "RowGroups").map(Self::Parsed)
        }
    }
}

impl ParquetReadOptions {
    pub(in crate::builtins::table) async fn parse(
        args: crate::call::FunctionArgumentView<'_>,
    ) -> BuiltinResult<Self> {
        if !args.len().is_multiple_of(2) {
            return Err(invalid_argument(
                "parquetread: name-value options must be provided in pairs",
            ));
        }
        let mut options = Self::default();
        let mut row_groups = Vec::new();
        for pair in args.chunks_exact(2) {
            let name = scalar_text(&pair[0], "parquetread option")?;
            let value = pair.get(1).expect("exact option pair");
            match ParquetReadControl::parse(&name) {
                Some(ParquetReadControl::SelectedVariableNames) => {
                    options.selected_variable_names = Some(string_list(value)?)
                }
                Some(ParquetReadControl::RowGroups) => {
                    row_groups.push(RowGroupInput::parse(value)?);
                }
                Some(ParquetReadControl::OutputType) => {
                    let text = scalar_text(value, "OutputType")?;
                    options.output_type =
                        ParquetOutputType::parse_label(&text).ok_or_else(|| {
                            invalid_argument(format!(
                                "parquetread: unsupported OutputType '{}'",
                                text.trim().to_ascii_lowercase()
                            ))
                        })?;
                }
                Some(ParquetReadControl::RowTimes) => {
                    options.row_times = Some(scalar_text(value, "RowTimes")?)
                }
                Some(ParquetReadControl::RowFilter) => options.row_filter = Some(value.clone()),
                None => {
                    return Err(invalid_argument(format!(
                        "parquetread: unsupported option '{name}'"
                    )))
                }
            }
        }
        if let Some(names) = &options.selected_variable_names {
            let mut seen = HashSet::new();
            for name in names {
                if !seen.insert(name) {
                    return Err(invalid_argument(format!(
                        "parquetread: SelectedVariableNames contains duplicate variable '{name}'"
                    )));
                }
            }
        }
        options.require_supported_route()?;
        // Every occurrence is validated, even if a later one overrides it.
        // Do not reparse names or gather unrelated/nested metadata values.
        for input in row_groups {
            options.row_groups = Some(match input {
                RowGroupInput::Parsed(indices) => indices,
                RowGroupInput::Resident(value) => {
                    let gathered = crate::gather_if_needed_async(value)
                        .await
                        .map_err(map_control_flow)?;
                    one_based_indices(&gathered, "RowGroups")?
                }
            });
        }
        Ok(options)
    }

    fn require_supported_route(&self) -> BuiltinResult<()> {
        #[cfg(target_arch = "wasm32")]
        {
            Err(table_error(
                &TABLE_ERROR_UNSUPPORTED_FILE,
                "parquetread: Parquet decoding is available in native host runtimes",
            ))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.output_type == ParquetOutputType::Timetable {
                return Err(invalid_argument("parquetread: OutputType 'timetable' requires timetable row-time synthesis and is not supported yet"));
            }
            if self.row_times.is_some() {
                return Err(invalid_argument(
                    "parquetread: RowTimes is only valid with OutputType 'timetable'",
                ));
            }
            if self.row_filter.is_some() {
                return Err(invalid_argument("parquetread: RowFilter is not supported until rowfilter predicate execution is available"));
            }
            Ok(())
        }
    }
}

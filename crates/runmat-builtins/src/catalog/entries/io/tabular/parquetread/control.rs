/// Parsed public option spelling; shared by runtime adaptation and inference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParquetReadControl {
    SelectedVariableNames,
    RowGroups,
    OutputType,
    RowTimes,
    RowFilter,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ParquetOutputType {
    #[default]
    Table,
    Timetable,
}

// table::scalar_text accepts every character page when its cached row count is one.
pub(in crate::catalog::entries::io::tabular) fn scalar_text_input(
    fact: &runmat_types::ValueFact,
) -> bool {
    if fact.kind == runmat_types::ValueKindFact::Character {
        return match fact.shape.known_dims() {
            Some(dims) if dims.len() == 1 => true,
            Some(dims) => dims
                .first()
                .is_some_and(|rows| rows.is_none_or(|rows| rows == 1)),
            None => true,
        };
    }
    super::super::super::repl_fs::text_input::scalar(fact)
}
impl ParquetOutputType {
    pub fn parse_label(label: &str) -> Option<Self> {
        match label.trim().to_ascii_lowercase().as_str() {
            "table" => Some(Self::Table),
            "timetable" => Some(Self::Timetable),
            _ => None,
        }
    }
}
impl ParquetReadControl {
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "selectedvariablenames" => Some(Self::SelectedVariableNames),
            "rowgroups" => Some(Self::RowGroups),
            "outputtype" => Some(Self::OutputType),
            "rowtimes" => Some(Self::RowTimes),
            "rowfilter" => Some(Self::RowFilter),
            _ => None,
        }
    }
}

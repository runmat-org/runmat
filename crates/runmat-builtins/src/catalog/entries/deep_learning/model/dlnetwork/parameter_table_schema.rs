//! Public column identities for deep-learning parameter tables.
//!
//! These identify the table's external structure, not the names or roles of
//! individual learnables. User-provided layer and parameter labels remain
//! opaque text until a network assignment validates its own canonical keys.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepLearningParameterTableColumn {
    Layer,
    Parameter,
    Value,
}

impl DeepLearningParameterTableColumn {
    pub const ALL: [Self; 3] = [Self::Layer, Self::Parameter, Self::Value];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Layer => "Layer",
            Self::Parameter => "Parameter",
            Self::Value => "Value",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Layer" => Some(Self::Layer),
            "Parameter" => Some(Self::Parameter),
            "Value" => Some(Self::Value),
            _ => None,
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Layer => 0,
            Self::Parameter => 1,
            Self::Value => 2,
        }
    }

    /// Recognize exactly the public parameter-table columns, in any order.
    /// This does not validate column values, row counts, or network keys.
    pub fn has_exact_columns<'a>(names: impl IntoIterator<Item = &'a str>) -> bool {
        let mut seen = [false; 3];
        for name in names {
            let Some(column) = Self::from_name(name) else {
                return false;
            };
            let was_seen = &mut seen[column.index()];
            if *was_seen {
                return false;
            }
            *was_seen = true;
        }
        seen.into_iter().all(|present| present)
    }
}

#[cfg(test)]
mod tests {
    use super::DeepLearningParameterTableColumn as Column;

    #[test]
    fn external_column_schema_is_case_sensitive_and_order_independent() {
        for names in [
            ["Layer", "Parameter", "Value"],
            ["Layer", "Value", "Parameter"],
            ["Parameter", "Layer", "Value"],
            ["Parameter", "Value", "Layer"],
            ["Value", "Layer", "Parameter"],
            ["Value", "Parameter", "Layer"],
        ] {
            assert!(Column::has_exact_columns(names));
        }
        assert!(!Column::has_exact_columns(["Layer", "Parameter"]));
        assert!(!Column::has_exact_columns(["Layer", "Layer", "Value"]));
        assert!(!Column::has_exact_columns([
            "Layer",
            "Parameter",
            "Value",
            "Other"
        ]));
        assert!(!Column::has_exact_columns(["layer", "Parameter", "Value"]));
        assert_eq!(Column::from_name("CustomWeight"), None);
        assert_eq!(
            Column::ALL.map(Column::name),
            ["Layer", "Parameter", "Value"]
        );
    }
}

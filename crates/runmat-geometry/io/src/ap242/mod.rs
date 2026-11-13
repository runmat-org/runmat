mod change_set;
mod material;
mod p21;
mod profile;
mod units;

pub use change_set::evaluate_change_set;
pub use profile::inspect_ap242;
pub use units::{declared_length_unit, inspect_length_units};

pub fn has_ap242_schema(source: &str) -> bool {
    source.split(';').any(|statement| {
        let statement = statement.trim().to_ascii_uppercase();
        statement.starts_with("FILE_SCHEMA") && statement.contains("AP242")
    })
}

pub fn has_ap242_schema_bytes(source: &[u8]) -> bool {
    has_ap242_schema(&String::from_utf8_lossy(source))
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Ap242InspectionError {
    #[error("invalid STEP physical file: {0}")]
    Syntax(String),
    #[error("unsupported AP242 exchange fact: {0}")]
    Unsupported(String),
    #[error("inconsistent AP242 exchange graph: {0}")]
    Inconsistent(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_detection_is_confined_to_the_file_schema_statement() {
        assert!(has_ap242_schema(
            "ISO-10303-21;HEADER;FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF'));ENDSEC;"
        ));
        assert!(!has_ap242_schema(
            "ISO-10303-21;HEADER;FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));ENDSEC;DATA;#1=DESCRIPTIVE_REPRESENTATION_ITEM('note','AP242');"
        ));
    }
}

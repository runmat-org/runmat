//! Length unit declared by an IGES file's global section.
//!
//! The global section (records with `G` in column 73) holds comma-delimited parameters; parameter 1
//! and 2 may redefine the parameter and record delimiters. Parameter 14 is the unit flag and
//! parameter 15 the unit name, which governs when the flag is 3.
use runmat_geometry_core::UnitSystem;

const UNIT_FLAG: usize = 13;
const UNIT_NAME: usize = 14;

/// The single length unit an IGES file declares, or `None` when the global section is missing,
/// malformed, or names a unit outside metres, millimetres and inches.
pub fn iges_declared_length_unit(bytes: &[u8]) -> Option<UnitSystem> {
    let text = String::from_utf8_lossy(bytes);
    let mut global = String::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line.len() >= 73 && line.as_bytes().get(72) == Some(&b'G') {
            global.push_str(line.get(..72)?);
        }
    }
    let parameters = global_parameters(&global)?;
    let flag = parameters.get(UNIT_FLAG)?.trim();
    let flag: i64 = flag.strip_suffix('.').unwrap_or(flag).parse().ok()?;
    match flag {
        1 => Some(UnitSystem::Inch),
        2 => Some(UnitSystem::Millimeter),
        6 => Some(UnitSystem::Meter),
        3 => named_unit(parameters.get(UNIT_NAME)?),
        _ => None,
    }
}

fn named_unit(name: &str) -> Option<UnitSystem> {
    match name.trim().to_ascii_uppercase().as_str() {
        "IN" | "INCH" => Some(UnitSystem::Inch),
        "MM" => Some(UnitSystem::Millimeter),
        "M" => Some(UnitSystem::Meter),
        _ => None,
    }
}

/// Splits the global section into parameter values, decoding Hollerith strings (`nH...`).
fn global_parameters(global: &str) -> Option<Vec<String>> {
    let bytes = global.as_bytes();
    let mut parameter_delimiter = b',';
    let mut record_delimiter = b';';
    let mut values = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let (value, next) = field(bytes, index, parameter_delimiter, record_delimiter)?;
        match values.len() {
            0 if !value.is_empty() => parameter_delimiter = *value.as_bytes().first()?,
            1 if !value.is_empty() => record_delimiter = *value.as_bytes().first()?,
            _ => {}
        }
        values.push(value);
        let terminator = bytes.get(next).copied();
        index = next + 1;
        if terminator == Some(record_delimiter) || terminator.is_none() {
            break;
        }
    }
    Some(values)
}

/// One field starting at `start`: a Hollerith string or plain text up to the next delimiter.
/// Returns the value and the index of the delimiter that ends it.
fn field(bytes: &[u8], start: usize, parameter: u8, record: u8) -> Option<(String, usize)> {
    let mut index = start;
    while bytes.get(index) == Some(&b' ') {
        index += 1;
    }
    let digits_end = index
        + bytes[index..]
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count();
    if digits_end > index && matches!(bytes.get(digits_end), Some(b'H' | b'h')) {
        let length: usize = std::str::from_utf8(&bytes[index..digits_end])
            .ok()?
            .parse()
            .ok()?;
        let value_start = digits_end + 1;
        let value = bytes.get(value_start..value_start + length)?;
        let mut end = value_start + length;
        while bytes.get(end) == Some(&b' ') {
            end += 1;
        }
        return Some((String::from_utf8_lossy(value).into_owned(), end));
    }
    let end = index
        + bytes[index..]
            .iter()
            .take_while(|byte| **byte != parameter && **byte != record)
            .count();
    Some((
        String::from_utf8_lossy(&bytes[index..end])
            .trim()
            .to_string(),
        end,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iges_with_global(global: &str) -> Vec<u8> {
        let mut text = format!("{:<72}S      1\n", "");
        for (row, chunk) in global.as_bytes().chunks(72).enumerate() {
            let chunk = std::str::from_utf8(chunk).unwrap();
            text.push_str(&format!("{chunk:<72}G{:>7}\n", row + 1));
        }
        text.into_bytes()
    }

    fn global(flag: &str, name: &str) -> String {
        format!("1H,,1H;,4HPART,8Hpart.igs,4HTEST,3H1.0,32,38,6,308,15,4HPART,1.0,{flag},{name},1,0.1,13H251025.120000,1E-07,1.0;")
    }

    #[test]
    fn unit_flags_resolve_to_declared_units() {
        assert_eq!(
            iges_declared_length_unit(&iges_with_global(&global("2", "2HMM"))),
            Some(UnitSystem::Millimeter)
        );
        assert_eq!(
            iges_declared_length_unit(&iges_with_global(&global("6", "1HM"))),
            Some(UnitSystem::Meter)
        );
        assert_eq!(
            iges_declared_length_unit(&iges_with_global(&global("1", "4HINCH"))),
            Some(UnitSystem::Inch)
        );
        assert_eq!(
            iges_declared_length_unit(&iges_with_global(&global("3", "2HIN"))),
            Some(UnitSystem::Inch)
        );
    }

    #[test]
    fn unrepresentable_or_missing_units_are_undeclared() {
        // Feet, centimetres and an unknown named unit have no UnitSystem.
        for (flag, name) in [("4", "2HFT"), ("10", "2HCM"), ("3", "4HFOOT")] {
            assert_eq!(
                iges_declared_length_unit(&iges_with_global(&global(flag, name))),
                None
            );
        }
        assert_eq!(iges_declared_length_unit(b"not an iges file"), None);
    }

    #[test]
    fn default_delimiters_and_hollerith_delimiters_are_honoured() {
        let defaults = ",,4HPART,8Hpart.igs,4HTEST,3H1.0,32,38,6,308,15,4HPART,1.0,2,2HMM,1;";
        assert_eq!(
            iges_declared_length_unit(&iges_with_global(defaults)),
            Some(UnitSystem::Millimeter)
        );
        let custom = "1H/,1H#/4HPA,T/8Hpart.igs/4HTEST/3H1.0/32/38/6/308/15/4HPART/1.0/6/1HM/1#";
        assert_eq!(
            iges_declared_length_unit(&iges_with_global(custom)),
            Some(UnitSystem::Meter)
        );
    }
}

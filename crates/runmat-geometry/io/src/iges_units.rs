//! Length unit declared by an IGES file's global section.
//!
//! The global section (records with `G` in column 73) holds comma-delimited parameters; parameter 1
//! and 2 may redefine the parameter and record delimiters. Parameter 14 is the unit flag and
//! parameter 15 the unit name, which governs when the flag is 3.
use runmat_geometry_core::UnitSystem;

const UNIT_FLAG: usize = 13;
const UNIT_NAME: usize = 14;

/// The length unit an IGES file declares (any IGES 5.3 unit flag, or a recognised unit name with
/// flag 3), or `None` when the global section is missing or malformed or the unit is unknown.
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
    let meters_per_unit = match flag {
        3 => named_unit(parameters.get(UNIT_NAME)?)?,
        flag => flag_unit(flag)?,
    };
    UnitSystem::from_meters_per_unit(meters_per_unit)
}

/// IGES 5.3 global parameter 14 unit flags.
fn flag_unit(flag: i64) -> Option<f64> {
    Some(match flag {
        1 => 0.0254,
        2 => 0.001,
        4 => 0.3048,
        5 => 1609.344,
        6 => 1.0,
        7 => 1000.0,
        8 => 2.54e-5,
        9 => 1.0e-6,
        10 => 0.01,
        11 => 2.54e-8,
        _ => return None,
    })
}

/// Unit names that may accompany flag 3 (IGES 5.3 global parameter 15).
fn named_unit(name: &str) -> Option<f64> {
    let flag = match name.trim().to_ascii_uppercase().as_str() {
        "IN" | "INCH" => 1,
        "MM" => 2,
        "FT" => 4,
        "MI" => 5,
        "M" => 6,
        "KM" => 7,
        "MIL" => 8,
        "UM" | "MICRON" => 9,
        "CM" => 10,
        "UIN" => 11,
        _ => return None,
    };
    flag_unit(flag)
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
    fn every_unit_flag_resolves_to_its_size() {
        let meters = |flag: &str, name: &str| {
            iges_declared_length_unit(&iges_with_global(&global(flag, name)))
                .and_then(UnitSystem::meters_per_unit)
        };
        assert_eq!(meters("4", "2HFT"), Some(0.3048));
        assert_eq!(meters("10", "2HCM"), Some(0.01));
        assert_eq!(meters("9", "6HMICRON"), Some(1.0e-6));
        assert_eq!(meters("3", "2HFT"), Some(0.3048));
        // An unknown named unit or flag, or no global section, is unresolvable.
        assert_eq!(meters("3", "4HFOOT"), None);
        assert_eq!(meters("12", "2HMM"), None);
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

use runmat_types::NumericClass;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreadInputType {
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float32,
    Float64,
}

impl FreadInputType {
    pub fn byte_len(self) -> usize {
        match self {
            Self::UInt8 | Self::Int8 => 1,
            Self::UInt16 | Self::Int16 => 2,
            Self::UInt32 | Self::Int32 | Self::Float32 => 4,
            Self::UInt64 | Self::Int64 | Self::Float64 => 8,
        }
    }

    pub fn numeric_class(self) -> NumericClass {
        match self {
            Self::UInt8 => NumericClass::UInt8,
            Self::Int8 => NumericClass::Int8,
            Self::UInt16 => NumericClass::UInt16,
            Self::Int16 => NumericClass::Int16,
            Self::UInt32 => NumericClass::UInt32,
            Self::Int32 => NumericClass::Int32,
            Self::UInt64 => NumericClass::UInt64,
            Self::Int64 => NumericClass::Int64,
            Self::Float32 => NumericClass::Single,
            Self::Float64 => NumericClass::Double,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreadOutputKind {
    Numeric(NumericClass),
    Char,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FreadPrecisionSpec {
    pub input: FreadInputType,
    pub output: FreadOutputKind,
    pub repeat: usize,
}

impl Default for FreadPrecisionSpec {
    fn default() -> Self {
        Self {
            input: FreadInputType::UInt8,
            output: FreadOutputKind::Numeric(NumericClass::Double),
            repeat: 1,
        }
    }
}

pub fn parse_fread_precision(raw: &str) -> Result<FreadPrecisionSpec, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("precision argument must not be empty".to_string());
    }
    let lower = trimmed.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix('*') {
        parse_star_precision(rest.trim())
    } else if let Some((lhs, rhs)) = lower.split_once("=>") {
        let (repeat, input) = parse_repeated_input_label(lhs.trim())?;
        let output = parse_output_label(rhs.trim())?;
        Ok(FreadPrecisionSpec {
            input,
            output,
            repeat,
        })
    } else {
        let (repeat, input) = parse_repeated_input_label(lower.trim())?;
        let wants_char =
            lower == "char" || (matches!(input, FreadInputType::UInt8) && lower.contains("char"));
        let output = if wants_char {
            FreadOutputKind::Char
        } else {
            FreadOutputKind::Numeric(NumericClass::Double)
        };
        Ok(FreadPrecisionSpec {
            input,
            output,
            repeat,
        })
    }
}

fn parse_star_precision(label: &str) -> Result<FreadPrecisionSpec, String> {
    if label == "char" {
        return Ok(FreadPrecisionSpec {
            input: FreadInputType::UInt8,
            output: FreadOutputKind::Char,
            repeat: 1,
        });
    }
    let input = parse_input_label(label)?;
    Ok(FreadPrecisionSpec {
        input,
        output: FreadOutputKind::Numeric(input.numeric_class()),
        repeat: 1,
    })
}

fn parse_repeated_input_label(label: &str) -> Result<(usize, FreadInputType), String> {
    if let Some((repeat, source)) = label.split_once('*') {
        if repeat.trim().chars().all(|ch| ch.is_ascii_digit()) {
            let repeat = repeat
                .trim()
                .parse::<usize>()
                .ok()
                .filter(|&repeat| repeat > 0)
                .ok_or_else(|| "precision repeat count must be a positive integer".to_string())?;
            return Ok((repeat, parse_input_label(source.trim())?));
        }
    }
    Ok((1, parse_input_label(label)?))
}

fn parse_input_label(label: &str) -> Result<FreadInputType, String> {
    match label {
        "double" | "float64" | "real*8" => Ok(FreadInputType::Float64),
        "single" | "float" | "float32" | "real*4" => Ok(FreadInputType::Float32),
        "int8" | "schar" | "signedchar" | "signed char" | "integer*1" => Ok(FreadInputType::Int8),
        "uint8" | "uchar" | "unsignedchar" | "unsigned char" | "char" | "byte" => {
            Ok(FreadInputType::UInt8)
        }
        "int16" | "short" | "integer*2" => Ok(FreadInputType::Int16),
        "uint16" | "ushort" | "unsignedshort" | "unsigned short" => Ok(FreadInputType::UInt16),
        "int32" | "int" | "integer*4" | "long" => Ok(FreadInputType::Int32),
        "uint32" | "uint" | "unsignedint" | "unsigned int" | "unsignedlong" | "unsigned long" => {
            Ok(FreadInputType::UInt32)
        }
        "int64" | "integer*8" | "longlong" | "long long" => Ok(FreadInputType::Int64),
        "uint64" | "unsignedlonglong" | "unsigned long long" => Ok(FreadInputType::UInt64),
        other => Err(format!("unsupported precision '{other}'")),
    }
}

fn parse_output_label(label: &str) -> Result<FreadOutputKind, String> {
    let dtype = match label {
        "double" | "float64" | "real*8" => NumericClass::Double,
        "single" | "float" | "float32" | "real*4" => NumericClass::Single,
        "int8" | "schar" | "signedchar" | "signed char" | "integer*1" => NumericClass::Int8,
        "int16" | "short" | "integer*2" => NumericClass::Int16,
        "int32" | "int" | "integer*4" => NumericClass::Int32,
        "int64" | "integer*8" | "longlong" | "long long" => NumericClass::Int64,
        "uint8" | "uchar" | "unsignedchar" | "unsigned char" => NumericClass::UInt8,
        "uint16" | "ushort" | "unsignedshort" | "unsigned short" => NumericClass::UInt16,
        "uint32" | "uint" | "unsignedint" | "unsigned int" | "unsignedlong" | "unsigned long" => {
            NumericClass::UInt32
        }
        "uint64" | "unsignedlonglong" | "unsigned long long" => NumericClass::UInt64,
        "char" => return Ok(FreadOutputKind::Char),
        other => return Err(format!("output class '{other}' is not implemented yet")),
    };
    Ok(FreadOutputKind::Numeric(dtype))
}

use runmat_value::Tensor;

use crate::builtins::common::tensor;

pub(crate) fn tensor_char_codes_to_string(value: &Tensor) -> Option<String> {
    if let Some(storage) = value.integer_storage() {
        let mut text = String::with_capacity(storage.len());
        for code in storage.exact_values() {
            let code = u32::try_from(code.try_to_usize()?).ok()?;
            text.push(char::from_u32(code)?);
        }
        return Some(text);
    }

    let codes = tensor::tensor_values_f64(value);
    let mut text = String::with_capacity(codes.len());
    for code in codes {
        if !code.is_finite() {
            return None;
        }
        let rounded = code.round();
        if (code - rounded).abs() > 1e-6 {
            return None;
        }
        let int_code = rounded as i64;
        if !(0..=0x10FFFF).contains(&int_code) {
            return None;
        }
        text.push(char::from_u32(int_code as u32)?);
    }
    Some(text)
}

#[cfg(test)]
mod tests;

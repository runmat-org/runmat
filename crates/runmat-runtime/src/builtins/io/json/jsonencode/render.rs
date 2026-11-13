use super::*;

pub(super) fn render_json(value: &JsonValue, options: &JsonEncodeOptions) -> String {
    let mut writer = JsonWriter::new(options.pretty_print);
    writer.write_value(value);
    writer.finish()
}

struct JsonWriter {
    output: String,
    pretty: bool,
    indent: usize,
}

impl JsonWriter {
    fn new(pretty: bool) -> Self {
        Self {
            output: String::new(),
            pretty,
            indent: 0,
        }
    }

    fn finish(self) -> String {
        self.output
    }

    fn write_value(&mut self, value: &JsonValue) {
        match value {
            JsonValue::Null => self.output.push_str("null"),
            JsonValue::Bool(true) => self.output.push_str("true"),
            JsonValue::Bool(false) => self.output.push_str("false"),
            JsonValue::Number(number) => self.write_number(number),
            JsonValue::String(text) => {
                self.output.push('"');
                self.output.push_str(&escape_json_string(text));
                self.output.push('"');
            }
            JsonValue::Array(items) => self.write_array(items),
            JsonValue::Object(fields) => self.write_object(fields),
        }
    }

    fn write_number(&mut self, number: &JsonNumber) {
        match number {
            JsonNumber::Float(f) => {
                if f.is_nan() || !f.is_finite() {
                    self.output.push_str("null");
                } else {
                    self.output.push_str(&format_number(*f));
                }
            }
            JsonNumber::I64(i) => {
                let _ = write!(self.output, "{i}");
            }
            JsonNumber::U64(u) => {
                let _ = write!(self.output, "{u}");
            }
        }
    }

    fn write_array(&mut self, items: &[JsonValue]) {
        if items.is_empty() {
            self.output.push_str("[]");
            return;
        }
        let inline = if self.pretty {
            items.iter().all(|item| {
                matches!(
                    item,
                    JsonValue::Null
                        | JsonValue::Bool(_)
                        | JsonValue::Number(_)
                        | JsonValue::String(_)
                )
            })
        } else {
            false
        };
        if inline {
            self.output.push('[');
            for (index, item) in items.iter().enumerate() {
                self.write_value(item);
                if index + 1 < items.len() {
                    self.output.push(',');
                }
            }
            self.output.push(']');
            return;
        }
        self.output.push('[');
        if self.pretty {
            self.output.push('\n');
            self.indent += 1;
        }
        for (index, item) in items.iter().enumerate() {
            if self.pretty {
                self.write_indent();
            }
            self.write_value(item);
            if index + 1 < items.len() {
                if self.pretty {
                    self.output.push_str(",\n");
                } else {
                    self.output.push(',');
                }
            }
        }
        if self.pretty {
            self.output.push('\n');
            if self.indent > 0 {
                self.indent -= 1;
            }
            self.write_indent();
        }
        self.output.push(']');
    }

    fn write_object(&mut self, fields: &[(String, JsonValue)]) {
        if fields.is_empty() {
            self.output.push_str("{}");
            return;
        }
        self.output.push('{');
        if self.pretty {
            self.output.push('\n');
            self.indent += 1;
        }
        for (index, (key, value)) in fields.iter().enumerate() {
            if self.pretty {
                self.write_indent();
            }
            self.output.push('"');
            self.output.push_str(&escape_json_string(key));
            self.output.push('"');
            if self.pretty {
                self.output.push_str(": ");
            } else {
                self.output.push(':');
            }
            self.write_value(value);
            if index + 1 < fields.len() {
                if self.pretty {
                    self.output.push_str(",\n");
                } else {
                    self.output.push(',');
                }
            }
        }
        if self.pretty {
            self.output.push('\n');
            if self.indent > 0 {
                self.indent -= 1;
            }
            self.write_indent();
        }
        self.output.push('}');
    }

    fn write_indent(&mut self) {
        if self.pretty {
            for _ in 0..self.indent {
                self.output.push_str("    ");
            }
        }
    }
}

fn escape_json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\u{08}' => escaped.push_str("\\b"),
            '\u{0C}' => escaped.push_str("\\f"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(escaped, "\\u{:04X}", c as u32);
            }
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        // Display integer-like doubles without decimal point
        format!("{:.0}", value)
    } else {
        format!("{}", value)
    }
}

//! High-level plotting utilities that use the unified renderer.

pub fn generate_major_ticks(min: f64, max: f64) -> Vec<f64> {
    if !(min.is_finite() && max.is_finite()) || max <= min {
        return Vec::new();
    }
    let range = (max - min).max(1e-9);
    let step = calculate_tick_interval(range);
    if !(step.is_finite() && step > 0.0) {
        return Vec::new();
    }

    let mut ticks = Vec::new();
    let mut value = (min / step).ceil() * step;
    let epsilon = range * 1e-6 + step * 1e-6;
    while value <= max + epsilon {
        let snapped = if value.abs() < epsilon { 0.0 } else { value };
        ticks.push(snapped);
        value += step;
        if ticks.len() > 64 {
            break;
        }
    }

    // Only the nice values are ticks; MATLAB does not label an axis limit that falls
    // between them.
    ticks
}

/// Calculate nice tick intervals for axis labeling
pub fn calculate_tick_interval(range: f64) -> f64 {
    let magnitude = 10.0_f64.powf(range.log10().floor());
    let normalized = range / magnitude;

    let nice_interval = if normalized <= 1.0 {
        0.2
    } else if normalized <= 2.0 {
        0.5
    } else if normalized <= 5.0 {
        1.0
    } else {
        2.0
    };

    nice_interval * magnitude
}

/// Format a tick label value for display
pub fn format_tick_label(value: f64) -> String {
    fn trim_fixed(mut s: String) -> String {
        if s.contains('.') {
            while s.ends_with('0') {
                s.pop();
            }
            if s.ends_with('.') {
                s.pop();
            }
        }
        if s == "-0" {
            "0".to_string()
        } else {
            s
        }
    }

    // Tick values are multiples of a nice step, so ten significant digits show each one
    // exactly (12.65, 0.0002) while trimming the floating-point tail (0.30000000000000004).
    if !value.is_finite() {
        return format!("{value}");
    }
    if value.abs() < 1.0e-12 {
        return "0".to_string();
    }
    let magnitude = value.abs().log10().floor() as i32;
    let decimals = (9 - magnitude).clamp(0, 15) as usize;
    trim_fixed(format!("{value:.decimals$}"))
}

pub fn canonical_tick_label_format(format: &str) -> String {
    match format.trim().to_ascii_lowercase().as_str() {
        "auto" => "%g".to_string(),
        "usd" => "$%,.2f".to_string(),
        "eur" => "\u{20AC}%,.2f".to_string(),
        "gbp" => "\u{00A3}%,.2f".to_string(),
        "jpy" => "\u{00A5}%,d".to_string(),
        "degrees" => "%g\u{00B0}".to_string(),
        "percentage" => "%g%%".to_string(),
        _ => format.to_string(),
    }
}

pub fn format_tick_label_with_format(value: f64, format: Option<&str>) -> String {
    TickLabelFormatter::new(format).format(value)
}

#[derive(Debug, Clone)]
pub struct TickLabelFormatter {
    spec: Option<NumericTickFormat>,
}

impl TickLabelFormatter {
    pub fn new(format: Option<&str>) -> Self {
        let spec = format
            .map(canonical_tick_label_format)
            .and_then(|format| NumericTickFormat::parse(&format));
        Self { spec }
    }

    pub fn format(&self, value: f64) -> String {
        let Some(spec) = &self.spec else {
            return format_tick_label(value);
        };
        spec.format(value)
    }
}

#[derive(Debug, Clone)]
struct NumericTickFormat {
    prefix: String,
    suffix: String,
    flags: TickFormatFlags,
    width: Option<usize>,
    precision: Option<usize>,
    conversion: char,
}

#[derive(Debug, Clone, Copy, Default)]
struct TickFormatFlags {
    thousands: bool,
    plus: bool,
    zero_pad: bool,
    left: bool,
    alternate: bool,
}

impl NumericTickFormat {
    fn parse(format: &str) -> Option<Self> {
        let chars: Vec<char> = format.chars().collect();
        let mut index = 0usize;
        while index < chars.len() {
            if chars[index] == '%' {
                if chars.get(index + 1) == Some(&'%') {
                    index += 2;
                    continue;
                }
                break;
            }
            index += 1;
        }
        if index >= chars.len() {
            return None;
        }

        let prefix = decode_percent_literals(&chars[..index]);
        index += 1;
        let mut flags = TickFormatFlags::default();
        while index < chars.len() {
            match chars[index] {
                ',' => flags.thousands = true,
                '+' => flags.plus = true,
                '0' => flags.zero_pad = true,
                '-' => flags.left = true,
                '#' => flags.alternate = true,
                _ => break,
            }
            index += 1;
        }

        let width_start = index;
        while index < chars.len() && chars[index].is_ascii_digit() {
            index += 1;
        }
        let width = parse_usize(&chars[width_start..index]);

        let precision = if chars.get(index) == Some(&'.') {
            index += 1;
            let precision_start = index;
            while index < chars.len() && chars[index].is_ascii_digit() {
                index += 1;
            }
            parse_usize(&chars[precision_start..index]).or(Some(0))
        } else {
            None
        };

        let conversion = *chars.get(index)?;
        if !matches!(conversion, 'd' | 'i' | 'f' | 'e' | 'E' | 'g' | 'G') {
            return None;
        }
        index += 1;
        let suffix = decode_percent_literals(&chars[index..]);
        Some(Self {
            prefix,
            suffix,
            flags,
            width,
            precision,
            conversion,
        })
    }

    fn format(&self, value: f64) -> String {
        let mut body = match self.conversion {
            'd' | 'i' => format_integer(value, self.precision, self.flags),
            'f' => format_fixed(value, self.precision.unwrap_or(6), self.flags),
            'e' => format_exponential(value, self.precision.unwrap_or(6), false),
            'E' => format_exponential(value, self.precision.unwrap_or(6), true),
            'g' => format_general(value, self.precision.unwrap_or(6), false, self.flags),
            'G' => format_general(value, self.precision.unwrap_or(6), true, self.flags),
            _ => format_tick_label(value),
        };
        if self.flags.plus && value.is_sign_positive() && !body.starts_with('+') {
            body.insert(0, '+');
        }
        if self.flags.thousands {
            body = add_thousands_separators(&body);
        }
        body = apply_width(body, self.width, self.flags);
        format!("{}{}{}", self.prefix, body, self.suffix)
    }
}

fn parse_usize(chars: &[char]) -> Option<usize> {
    if chars.is_empty() {
        return None;
    }
    chars.iter().collect::<String>().parse().ok()
}

fn decode_percent_literals(chars: &[char]) -> String {
    let mut out = String::new();
    let mut index = 0usize;
    while index < chars.len() {
        if chars[index] == '%' && chars.get(index + 1) == Some(&'%') {
            out.push('%');
            index += 2;
        } else {
            out.push(chars[index]);
            index += 1;
        }
    }
    out
}

fn format_integer(value: f64, precision: Option<usize>, flags: TickFormatFlags) -> String {
    let rounded = if value.is_finite() {
        value.round()
    } else {
        value
    };
    let mut text = format!("{rounded:.0}");
    if let Some(precision) = precision {
        let negative = text.starts_with('-');
        let digits = if negative { &text[1..] } else { &text };
        if digits.len() < precision {
            let mut padded = "0".repeat(precision - digits.len());
            padded.push_str(digits);
            text = if negative {
                format!("-{padded}")
            } else {
                padded
            };
        }
    }
    if flags.alternate && !text.contains('.') {
        text.push('.');
    }
    text
}

fn format_fixed(value: f64, precision: usize, flags: TickFormatFlags) -> String {
    let mut text = format!("{value:.precision$}");
    if precision == 0 && flags.alternate && !text.contains('.') {
        text.push('.');
    }
    text
}

fn format_exponential(value: f64, precision: usize, upper: bool) -> String {
    let text = format!("{value:.precision$e}");
    if upper {
        text.to_ascii_uppercase()
    } else {
        text
    }
}

fn format_general(value: f64, precision: usize, upper: bool, flags: TickFormatFlags) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let abs = value.abs();
    let use_exp = abs < 1e-4 || abs >= 10f64.powi(precision as i32);
    let text = if use_exp {
        let decimals = precision.saturating_sub(1);
        format_exponential(value, decimals, upper)
    } else {
        let integer_digits = abs.log10().floor().max(0.0) as usize + 1;
        let decimals = precision.saturating_sub(integer_digits);
        format_fixed(value, decimals, flags)
    };
    if flags.alternate {
        text
    } else {
        trim_general_zeros(text)
    }
}

fn trim_general_zeros(mut text: String) -> String {
    let exp_index = text.find(['e', 'E']);
    let suffix = exp_index.map(|idx| text.split_off(idx));
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    if let Some(suffix) = suffix {
        text.push_str(&suffix);
    }
    text
}

fn add_thousands_separators(text: &str) -> String {
    let (sign, rest) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => match text.strip_prefix('+') {
            Some(rest) => ("+", rest),
            None => ("", text),
        },
    };
    let (integer, suffix) = rest
        .find(['.', 'e', 'E'])
        .map(|idx| rest.split_at(idx))
        .unwrap_or((rest, ""));
    let chars: Vec<char> = integer.chars().collect();
    let mut out = String::new();
    for (idx, ch) in chars.iter().enumerate() {
        if idx > 0 && (chars.len() - idx).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*ch);
    }
    format!("{sign}{out}{suffix}")
}

fn apply_width(mut text: String, width: Option<usize>, flags: TickFormatFlags) -> String {
    let Some(width) = width else {
        return text;
    };
    let len = text.chars().count();
    if len >= width {
        return text;
    }
    let pad = width - len;
    if flags.left {
        text.push_str(&" ".repeat(pad));
        text
    } else if flags.zero_pad {
        let mut chars = text.chars();
        match chars.next() {
            Some(sign @ ('-' | '+')) => {
                let rest = chars.collect::<String>();
                format!("{sign}{}{rest}", "0".repeat(pad))
            }
            _ => format!("{}{text}", "0".repeat(pad)),
        }
    } else {
        format!("{}{text}", " ".repeat(pad))
    }
}

/// Generate grid lines for plotting
pub fn generate_grid_lines(
    bounds: (f64, f64, f64, f64),
    plot_rect: (f32, f32, f32, f32), // (left, right, bottom, top)
) -> Vec<(f32, f32, f32, f32)> {
    // Vector of (x1, y1, x2, y2) line segments
    let (x_min, x_max, y_min, y_max) = bounds;
    let (left, right, bottom, top) = plot_rect;

    let mut lines = Vec::new();

    // X-axis grid lines
    let x_range = x_max - x_min;
    let x_interval = calculate_tick_interval(x_range);
    let mut x_val = (x_min / x_interval).ceil() * x_interval;

    while x_val <= x_max {
        let x_screen = left + ((x_val - x_min) / x_range) as f32 * (right - left);
        lines.push((x_screen, bottom, x_screen, top));
        x_val += x_interval;
    }

    // Y-axis grid lines
    let y_range = y_max - y_min;
    let y_interval = calculate_tick_interval(y_range);
    let mut y_val = (y_min / y_interval).ceil() * y_interval;

    while y_val <= y_max {
        let y_screen = bottom + ((y_val - y_min) / y_range) as f32 * (top - bottom);
        lines.push((left, y_screen, right, y_screen));
        y_val += y_interval;
    }

    lines
}

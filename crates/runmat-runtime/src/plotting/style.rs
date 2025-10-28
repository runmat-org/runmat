use runmat_builtins::Value;

pub fn is_style_string(spec: &str) -> bool {
    spec.chars().all(|ch| "-:.ox+*sd^v<>phrgbcmykw".contains(ch))
}

#[derive(Clone, Debug)]
pub struct SeriesStyle {
    pub color: glam::Vec4,
    pub line_style: Option<runmat_plot::plots::line::LineStyle>,
    pub line_width: f32,
    pub marker: Option<runmat_plot::plots::scatter::MarkerStyle>,
    pub marker_size: f32,
}

impl Default for SeriesStyle {
    fn default() -> Self {
        Self {
            color: glam::Vec4::new(0.0, 0.4470, 0.7410, 1.0),
            line_style: Some(runmat_plot::plots::line::LineStyle::Solid),
            line_width: 1.0,
            marker: None,
            // Default scatter marker size in pixels (approx MATLAB default visual size)
            marker_size: 12.0,
        }
    }
}

pub fn parse_line_spec(spec: &str) -> SeriesStyle {
    use runmat_plot::plots::line::LineStyle;
    use runmat_plot::plots::scatter::MarkerStyle;
    let mut st = SeriesStyle::default();
    for ch in spec.chars() {
        st.color = match ch {
            'r' => glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            'g' => glam::Vec4::new(0.0, 1.0, 0.0, 1.0),
            'b' => glam::Vec4::new(0.0, 0.0, 1.0, 1.0),
            'c' => glam::Vec4::new(0.0, 1.0, 1.0, 1.0),
            'm' => glam::Vec4::new(1.0, 0.0, 1.0, 1.0),
            'y' => glam::Vec4::new(1.0, 1.0, 0.0, 1.0),
            'k' => glam::Vec4::new(0.0, 0.0, 0.0, 1.0),
            'w' => glam::Vec4::new(1.0, 1.0, 1.0, 1.0),
            _ => st.color,
        };
    }
    if spec.contains("--") { st.line_style = Some(LineStyle::Dashed); }
    else if spec.contains("-.") { st.line_style = Some(LineStyle::DashDot); }
    else if spec.contains(":") { st.line_style = Some(LineStyle::Dotted); }
    else if spec.contains("-") { st.line_style = Some(LineStyle::Solid); }

    for ch in spec.chars() {
        st.marker = match ch {
            '.' | 'o' => Some(MarkerStyle::Circle),
            'x' | '+' => Some(MarkerStyle::Plus),
            '*' => Some(MarkerStyle::Star),
            's' => Some(MarkerStyle::Square),
            'd' => Some(MarkerStyle::Diamond),
            '^' | 'v' | '<' | '>' => Some(MarkerStyle::Triangle),
            'h' => Some(MarkerStyle::Hexagon),
            _ => st.marker,
        };
        if st.marker.is_some() { break; }
    }
    if st.marker.is_some() && !(spec.contains("--") || spec.contains("-.") || spec.contains(":") || spec.contains("-")) {
        st.line_style = None;
    }
    st
}

pub fn parse_color_value(v: &Value) -> Option<glam::Vec4> {
    match v {
        Value::String(s) => {
            let c = s.to_ascii_lowercase();
            match c.as_str() {
                "r" => Some(glam::Vec4::new(1.0,0.0,0.0,1.0)),
                "g" => Some(glam::Vec4::new(0.0,1.0,0.0,1.0)),
                "b" => Some(glam::Vec4::new(0.0,0.0,1.0,1.0)),
                "c" => Some(glam::Vec4::new(0.0,1.0,1.0,1.0)),
                "m" => Some(glam::Vec4::new(1.0,0.0,1.0,1.0)),
                "y" => Some(glam::Vec4::new(1.0,1.0,0.0,1.0)),
                "k" => Some(glam::Vec4::new(0.0,0.0,0.0,1.0)),
                "w" => Some(glam::Vec4::new(1.0,1.0,1.0,1.0)),
                _ => None,
            }
        }
        Value::Tensor(t) => {
            if t.data.len() >= 3 { Some(glam::Vec4::new(t.data[0] as f32, t.data[1] as f32, t.data[2] as f32, 1.0)) } else { None }
        }
        _ => None,
    }
}



use glam::Vec4;
use runmat_builtins::{CellArray, Tensor, Value};
use runmat_plot::plots::{
    ColorMap, LineMarkerAppearance, LineStyle, MarkerStyle as PlotMarkerStyle, ShadingMode,
    SurfacePlot,
};

#[derive(Clone, Copy, Debug)]
pub struct BarStyleDefaults {
    pub face_color: Vec4,
    pub bar_width: f32,
}

impl BarStyleDefaults {
    pub fn new(face_color: Vec4, bar_width: f32) -> Self {
        Self {
            face_color,
            bar_width,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BarStyle {
    pub face_color: Vec4,
    pub face_alpha: f32,
    pub edge_color: Option<Vec4>,
    pub edge_alpha: f32,
    pub line_width: f32,
    pub bar_width: f32,
    pub label: Option<String>,
    pub face_color_flat: bool,
    pub layout: BarLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarLayout {
    Grouped,
    Stacked,
}

impl BarStyle {
    pub fn face_rgba(&self) -> Vec4 {
        let mut color = self.face_color;
        color.w *= self.face_alpha;
        color
    }

    pub fn edge_rgba(&self) -> Option<Vec4> {
        self.edge_color.map(|mut color| {
            color.w *= self.edge_alpha;
            color
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LineStyleParseOptions {
    pub builtin_name: &'static str,
    pub forbid_leading_numeric: bool,
    pub forbid_interleaved_numeric: bool,
}

impl LineStyleParseOptions {
    pub const fn plot() -> Self {
        Self {
            builtin_name: "plot",
            forbid_leading_numeric: true,
            forbid_interleaved_numeric: false,
        }
    }

    pub const fn scatter() -> Self {
        Self {
            builtin_name: "scatter",
            forbid_leading_numeric: true,
            forbid_interleaved_numeric: true,
        }
    }

    pub const fn scatter3() -> Self {
        Self {
            builtin_name: "scatter3",
            forbid_leading_numeric: true,
            forbid_interleaved_numeric: true,
        }
    }

    pub const fn stairs() -> Self {
        Self {
            builtin_name: "stairs",
            forbid_leading_numeric: true,
            forbid_interleaved_numeric: true,
        }
    }

    pub const fn generic(name: &'static str) -> Self {
        Self {
            builtin_name: name,
            forbid_leading_numeric: false,
            forbid_interleaved_numeric: false,
        }
    }
}

pub const DEFAULT_LINE_WIDTH: f32 = 2.0;
pub const DEFAULT_LINE_COLOR: Vec4 = Vec4::new(0.0, 0.4, 0.8, 1.0);
pub const DEFAULT_LINE_MARKER_SIZE: f32 = 6.0;

fn ctx_err(opts: &LineStyleParseOptions, msg: impl Into<String>) -> String {
    format!("{}: {}", opts.builtin_name, msg.into())
}

#[derive(Clone, Debug)]
pub struct LineAppearance {
    pub color: Vec4,
    pub line_width: f32,
    pub line_style: LineStyle,
    pub marker: Option<MarkerAppearance>,
}

impl Default for LineAppearance {
    fn default() -> Self {
        Self {
            color: DEFAULT_LINE_COLOR,
            line_width: DEFAULT_LINE_WIDTH,
            line_style: LineStyle::Solid,
            marker: None,
        }
    }
}

pub fn marker_metadata_from_appearance(
    appearance: &LineAppearance,
) -> Option<LineMarkerAppearance> {
    let marker = appearance.marker.as_ref()?;
    let size = marker.size.unwrap_or(DEFAULT_LINE_MARKER_SIZE);
    let edge_color = marker_color_to_vec4(&marker.edge_color, appearance.color);
    let face_color = match marker.face_color {
        MarkerColor::None => Vec4::new(edge_color.x, edge_color.y, edge_color.z, 0.0),
        MarkerColor::Flat => appearance.color,
        MarkerColor::Auto | MarkerColor::Color(_) => {
            marker_color_to_vec4(&marker.face_color, appearance.color)
        }
    };
    Some(LineMarkerAppearance {
        kind: marker.kind.to_plot_marker(),
        size,
        edge_color,
        face_color,
        filled: !matches!(marker.face_color, MarkerColor::None),
    })
}

pub fn marker_color_to_vec4(color: &MarkerColor, fallback: Vec4) -> Vec4 {
    match color {
        MarkerColor::Auto | MarkerColor::Flat => fallback,
        MarkerColor::None => Vec4::new(fallback.x, fallback.y, fallback.z, 0.0),
        MarkerColor::Color(value) => *value,
    }
}

#[derive(Clone, Debug)]
pub struct MarkerAppearance {
    pub kind: MarkerKind,
    pub size: Option<f32>,
    pub edge_color: MarkerColor,
    pub face_color: MarkerColor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerKind {
    Circle,
    Plus,
    Star,
    Point,
    Cross,
    TriangleUp,
    TriangleDown,
    TriangleLeft,
    TriangleRight,
    Square,
    Diamond,
    Pentagram,
    Hexagram,
}

impl MarkerKind {
    pub fn to_plot_marker(self) -> PlotMarkerStyle {
        match self {
            MarkerKind::Circle => PlotMarkerStyle::Circle,
            MarkerKind::Plus => PlotMarkerStyle::Plus,
            MarkerKind::Star => PlotMarkerStyle::Star,
            MarkerKind::Point => PlotMarkerStyle::Circle,
            MarkerKind::Cross => PlotMarkerStyle::Cross,
            MarkerKind::TriangleUp => PlotMarkerStyle::Triangle,
            MarkerKind::TriangleDown => PlotMarkerStyle::Triangle,
            MarkerKind::TriangleLeft => PlotMarkerStyle::Triangle,
            MarkerKind::TriangleRight => PlotMarkerStyle::Triangle,
            MarkerKind::Square => PlotMarkerStyle::Square,
            MarkerKind::Diamond => PlotMarkerStyle::Diamond,
            MarkerKind::Pentagram => PlotMarkerStyle::Star,
            MarkerKind::Hexagram => PlotMarkerStyle::Star,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MarkerColor {
    Auto,
    None,
    Flat,
    Color(Vec4),
}

#[derive(Clone, Copy, Debug)]
pub struct SurfaceStyleDefaults {
    pub colormap: ColorMap,
    pub shading: ShadingMode,
    pub wireframe: bool,
    pub alpha: f32,
    pub flatten_z: bool,
    pub lighting_enabled: bool,
    pub visible: bool,
}

impl SurfaceStyleDefaults {
    pub fn new(
        colormap: ColorMap,
        shading: ShadingMode,
        wireframe: bool,
        alpha: f32,
        flatten_z: bool,
        lighting_enabled: bool,
    ) -> Self {
        Self {
            colormap,
            shading,
            wireframe,
            alpha,
            flatten_z,
            lighting_enabled,
            visible: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SurfaceStyle {
    pub colormap: ColorMap,
    pub shading: ShadingMode,
    pub wireframe: bool,
    pub alpha: f32,
    pub flatten_z: bool,
    pub lighting_enabled: bool,
    pub label: Option<String>,
    pub visible: Option<bool>,
}

impl SurfaceStyle {
    fn from_defaults(defaults: SurfaceStyleDefaults) -> Self {
        Self {
            colormap: defaults.colormap,
            shading: defaults.shading,
            wireframe: defaults.wireframe,
            alpha: defaults.alpha,
            flatten_z: defaults.flatten_z,
            lighting_enabled: defaults.lighting_enabled,
            label: None,
            visible: Some(defaults.visible),
        }
    }

    pub fn apply_to_plot(&self, plot: &mut SurfacePlot) {
        plot.colormap = self.colormap;
        plot.shading_mode = self.shading;
        plot.wireframe = self.wireframe;
        plot.alpha = self.alpha;
        plot.flatten_z = self.flatten_z;
        plot.lighting_enabled = self.lighting_enabled;
        if let Some(label) = &self.label {
            plot.label = Some(label.clone());
        }
        if let Some(visible) = self.visible {
            plot.visible = visible;
        }
    }
}

#[derive(Default, Clone)]
struct LineStyleOptions {
    line_style: Option<LineStyle>,
    color: Option<Vec4>,
    line_width: Option<f32>,
    requires_cpu_fallback: bool,
    marker_kind: Option<MarkerKind>,
    marker_disabled: bool,
    marker_size: Option<f32>,
    marker_edge_color: Option<MarkerColor>,
    marker_face_color: Option<MarkerColor>,
    line_style_order: Option<Vec<LineStyle>>,
    label: Option<String>,
}

impl LineStyleOptions {
    fn merge(&mut self, other: LineStyleOptions) {
        if other.line_style.is_some() {
            self.line_style = other.line_style;
        }
        if other.color.is_some() {
            self.color = other.color;
        }
        if other.line_width.is_some() {
            self.line_width = other.line_width;
        }
        self.requires_cpu_fallback |= other.requires_cpu_fallback;
        if other.marker_disabled {
            self.marker_disabled = true;
            self.marker_kind = None;
        } else if other.marker_kind.is_some() {
            self.marker_kind = other.marker_kind;
            self.marker_disabled = false;
        }
        if other.marker_size.is_some() {
            self.marker_size = other.marker_size;
        }
        if other.marker_edge_color.is_some() {
            self.marker_edge_color = other.marker_edge_color;
        }
        if other.marker_face_color.is_some() {
            self.marker_face_color = other.marker_face_color;
        }
        if other.line_style_order.is_some() {
            self.line_style_order = other.line_style_order;
        }
        if other.label.is_some() {
            self.label = other.label;
        }
    }

    fn resolve(&self) -> LineAppearance {
        let mut appearance = LineAppearance::default();
        if let Some(color) = self.color {
            appearance.color = color;
        }
        if let Some(width) = self.line_width {
            appearance.line_width = width;
        }
        if let Some(style) = self.line_style {
            appearance.line_style = style;
        }
        if !self.marker_disabled {
            if let Some(kind) = self.marker_kind {
                appearance.marker = Some(MarkerAppearance {
                    kind,
                    size: self.marker_size,
                    edge_color: self.marker_edge_color.clone().unwrap_or(MarkerColor::Auto),
                    face_color: self.marker_face_color.clone().unwrap_or(MarkerColor::Auto),
                });
            }
        }
        appearance
    }
}

#[derive(Clone, Debug)]
pub struct ParsedLineStyle {
    pub appearance: LineAppearance,
    pub requires_cpu_fallback: bool,
    pub line_style_explicit: bool,
    pub line_style_order: Option<Vec<LineStyle>>,
    pub label: Option<String>,
}

pub fn parse_line_style_args(
    rest: &[Value],
    opts: &LineStyleParseOptions,
) -> Result<ParsedLineStyle, String> {
    if rest.is_empty() {
        return Ok(ParsedLineStyle {
            appearance: LineAppearance::default(),
            requires_cpu_fallback: false,
            line_style_explicit: false,
            line_style_order: None,
            label: None,
        });
    }

    if opts.forbid_leading_numeric && begins_with_numeric(rest) {
        return Err(ctx_err(
            opts,
            "additional data series must precede style arguments",
        ));
    }

    let mut options = LineStyleOptions::default();
    let mut idx = 0usize;
    if let Some(token) = rest.get(0).and_then(value_as_string) {
        if is_style_token(&token) {
            options.merge(parse_style_string(opts, &token)?);
            idx = 1;
        }
    }

    let remaining = &rest[idx..];
    if !remaining.is_empty() {
        if opts.forbid_interleaved_numeric && begins_with_numeric(remaining) {
            return Err(ctx_err(
                opts,
                "per-series styles interleaved with data are not supported yet",
            ));
        }
        if remaining.len() % 2 != 0 {
            return Err(ctx_err(opts, "name-value arguments must come in pairs"));
        }
        options.merge(parse_name_value_pairs(opts, remaining)?);
    }

    let appearance = options.resolve();
    Ok(ParsedLineStyle {
        requires_cpu_fallback: options.requires_cpu_fallback,
        appearance,
        line_style_explicit: options.line_style.is_some(),
        line_style_order: options.line_style_order.clone(),
        label: options.label.clone(),
    })
}

pub fn parse_surface_style_args(
    builtin: &'static str,
    rest: &[Value],
    defaults: SurfaceStyleDefaults,
) -> Result<SurfaceStyle, String> {
    let opts = LineStyleParseOptions::generic(builtin);
    let mut style = SurfaceStyle::from_defaults(defaults);
    if rest.is_empty() {
        return Ok(style);
    }
    if rest.len() % 2 != 0 {
        return Err(ctx_err(
            &opts,
            "name-value arguments must come in pairs for surface plots",
        ));
    }
    for pair in rest.chunks_exact(2) {
        let key = value_as_string(&pair[0])
            .ok_or_else(|| ctx_err(&opts, "option names must be char arrays or strings"))?;
        let lower = key.trim().to_ascii_lowercase();
        match lower.as_str() {
            "colormap" => {
                style.colormap = parse_colormap_option(&opts, &pair[1])?;
            }
            "shading" => {
                style.shading = parse_shading_option(&opts, &pair[1])?;
            }
            "facecolor" => {
                apply_face_color_option(&opts, &pair[1], &mut style)?;
            }
            "facealpha" | "alpha" => {
                style.alpha = parse_alpha_value(&opts, &pair[1])?;
            }
            "edgecolor" => {
                apply_edge_color_option(&opts, &pair[1], &mut style)?;
            }
            "flattenz" => {
                style.flatten_z = parse_surface_bool(&opts, "FlattenZ", &pair[1])?;
            }
            "displayname" | "label" => {
                let Some(name) = value_as_string(&pair[1]) else {
                    return Err(ctx_err(&opts, "DisplayName must be a char array or string"));
                };
                style.label = Some(name);
            }
            "lighting" => {
                style.lighting_enabled = parse_lighting_option(&opts, &pair[1])?;
            }
            "visible" => {
                let visible = parse_surface_bool(&opts, "Visible", &pair[1])?;
                style.visible = Some(visible);
            }
            other => {
                return Err(ctx_err(
                    &opts,
                    format!("unsupported surface option `{other}`"),
                ));
            }
        }
    }
    Ok(style)
}

fn begins_with_numeric(rest: &[Value]) -> bool {
    matches!(
        rest.first(),
        Some(
            Value::Tensor(_) | Value::GpuTensor(_) | Value::Num(_) | Value::Int(_) | Value::Bool(_)
        )
    )
}

fn parse_style_string(
    opts: &LineStyleParseOptions,
    token: &str,
) -> Result<LineStyleOptions, String> {
    let mut options = LineStyleOptions::default();
    let mut chars = token.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '-' => {
                if let Some(next) = chars.peek() {
                    match next {
                        '-' => {
                            chars.next();
                            options.line_style = Some(LineStyle::Dashed);
                        }
                        '.' => {
                            chars.next();
                            options.line_style = Some(LineStyle::DashDot);
                        }
                        _ => options.line_style = Some(LineStyle::Solid),
                    }
                } else {
                    options.line_style = Some(LineStyle::Solid);
                }
            }
            ':' => {
                options.line_style = Some(LineStyle::Dotted);
            }
            'o' | '+' | '*' | '.' | 'x' | '^' | 'v' | '<' | '>' | 's' | 'd' | 'p' | 'h' => {
                if let Some(kind) = marker_kind_from_token(ch) {
                    options.marker_kind = Some(kind);
                }
            }
            c if color_from_token(c).is_some() => {
                options.color = color_from_token(c);
            }
            _ => {
                return Err(ctx_err(opts, format!("unrecognised style token `{ch}`")));
            }
        }
    }
    Ok(options)
}

fn parse_name_value_pairs(
    opts: &LineStyleParseOptions,
    args: &[Value],
) -> Result<LineStyleOptions, String> {
    let mut options = LineStyleOptions::default();
    let mut iter = args.chunks_exact(2);
    for pair in iter.by_ref() {
        let key = value_as_string(&pair[0])
            .ok_or_else(|| ctx_err(opts, "option names must be char arrays or strings"))?;
        let lower = key.trim().to_ascii_lowercase();
        match lower.as_str() {
            "linewidth" => {
                let width = value_as_f64(&pair[1])
                    .ok_or_else(|| ctx_err(opts, "LineWidth must be numeric"))?;
                if width <= 0.0 {
                    return Err(ctx_err(opts, "LineWidth must be positive"));
                }
                options.line_width = Some(width as f32);
            }
            "color" => {
                options.color = Some(parse_color_value(opts, &pair[1])?);
            }
            "linestyle" => {
                let style_text = value_as_string(&pair[1])
                    .ok_or_else(|| ctx_err(opts, "LineStyle must be a string"))?;
                options.line_style = Some(parse_line_style_name(opts, &style_text)?);
            }
            "marker" => {
                let marker_text = value_as_string(&pair[1])
                    .ok_or_else(|| ctx_err(opts, "Marker must be a string"))?;
                let parsed = parse_marker_string(opts, &marker_text)?;
                match parsed {
                    Some(kind) => {
                        options.marker_kind = Some(kind);
                        options.marker_disabled = false;
                    }
                    None => {
                        options.marker_kind = None;
                        options.marker_disabled = true;
                    }
                }
            }
            "markersize" => {
                let size = value_as_f64(&pair[1])
                    .ok_or_else(|| ctx_err(opts, "MarkerSize must be numeric"))?;
                if size <= 0.0 {
                    return Err(ctx_err(opts, "MarkerSize must be positive"));
                }
                options.marker_size = Some(size as f32);
            }
            "markeredgecolor" => {
                options.marker_edge_color = Some(parse_marker_color_value(opts, &pair[1])?);
            }
            "markerfacecolor" => {
                options.marker_face_color = Some(parse_marker_color_value(opts, &pair[1])?);
            }
            "linestyleorder" => {
                let order = parse_line_style_order_value(opts, &pair[1])?;
                options.line_style_order = Some(order);
            }
            "displayname" | "label" => {
                let Some(name) = value_as_string(&pair[1]) else {
                    return Err(ctx_err(opts, "DisplayName must be a char array or string"));
                };
                options.label = Some(name);
            }
            other => {
                return Err(ctx_err(opts, format!("unsupported option `{other}`")));
            }
        }
    }
    Ok(options)
}

fn parse_line_style_name(opts: &LineStyleParseOptions, value: &str) -> Result<LineStyle, String> {
    match value.trim() {
        "-" => Ok(LineStyle::Solid),
        "--" => Ok(LineStyle::Dashed),
        ":" => Ok(LineStyle::Dotted),
        "-." => Ok(LineStyle::DashDot),
        other => Err(ctx_err(opts, format!("unsupported LineStyle `{other}`"))),
    }
}

fn parse_marker_string(
    opts: &LineStyleParseOptions,
    value: &str,
) -> Result<Option<MarkerKind>, String> {
    let trimmed = value.trim();
    if trimmed.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    if trimmed.is_empty() {
        return Err(ctx_err(opts, "Marker cannot be empty"));
    }
    if let Some(kind) = trimmed.chars().next().and_then(marker_kind_from_token) {
        return Ok(Some(kind));
    }
    Err(ctx_err(opts, format!("unsupported Marker `{value}`")))
}

fn marker_kind_from_token(token: char) -> Option<MarkerKind> {
    match token {
        'o' | 'O' => Some(MarkerKind::Circle),
        '+' => Some(MarkerKind::Plus),
        '*' => Some(MarkerKind::Star),
        '.' => Some(MarkerKind::Point),
        'x' | 'X' => Some(MarkerKind::Cross),
        '^' => Some(MarkerKind::TriangleUp),
        'v' => Some(MarkerKind::TriangleDown),
        '<' => Some(MarkerKind::TriangleLeft),
        '>' => Some(MarkerKind::TriangleRight),
        's' | 'S' => Some(MarkerKind::Square),
        'd' | 'D' => Some(MarkerKind::Diamond),
        'p' | 'P' => Some(MarkerKind::Pentagram),
        'h' | 'H' => Some(MarkerKind::Hexagram),
        _ => None,
    }
}

fn parse_marker_color_value(
    opts: &LineStyleParseOptions,
    value: &Value,
) -> Result<MarkerColor, String> {
    if let Some(name) = value_as_string(value) {
        let lower = name.trim().to_ascii_lowercase();
        return match lower.as_str() {
            "auto" => Ok(MarkerColor::Auto),
            "flat" => Ok(MarkerColor::Flat),
            "none" => Ok(MarkerColor::None),
            _ => Ok(MarkerColor::Color(parse_color_value(opts, value)?)),
        };
    }
    Ok(MarkerColor::Color(parse_color_value(opts, value)?))
}

fn parse_line_style_order_value(
    opts: &LineStyleParseOptions,
    value: &Value,
) -> Result<Vec<LineStyle>, String> {
    let strings = match value {
        Value::StringArray(arr) => arr.data.clone(),
        Value::Cell(cell) => {
            collect_cell_strings(cell).map_err(|e| ctx_err(opts, format!("LineStyleOrder: {e}")))?
        }
        _ => {
            if let Some(text) = value_as_string(value) {
                vec![text]
            } else {
                return Err(ctx_err(
                    opts,
                    "LineStyleOrder must be a string, string array, or cell array of strings",
                ));
            }
        }
    };

    if strings.is_empty() {
        return Err(ctx_err(opts, "LineStyleOrder cannot be empty"));
    }

    let mut styles = Vec::with_capacity(strings.len());
    for entry in strings {
        styles.push(parse_line_style_name(opts, &entry)?);
    }
    Ok(styles)
}

fn collect_cell_strings(cell: &CellArray) -> Result<Vec<String>, String> {
    let mut strings = Vec::with_capacity(cell.rows * cell.cols);
    for row in 0..cell.rows {
        for col in 0..cell.cols {
            let value = cell.get(row, col)?;
            if let Some(text) = value_as_string(&value) {
                strings.push(text);
            } else {
                return Err(format!(
                    "cell entry at ({row}, {col}) is not a char array or string"
                ));
            }
        }
    }
    Ok(strings)
}

pub(crate) fn parse_color_value(
    opts: &LineStyleParseOptions,
    value: &Value,
) -> Result<Vec4, String> {
    if let Some(name) = value_as_string(value) {
        if let Some(color) = name.trim().chars().next().and_then(color_from_token) {
            return Ok(color);
        }
        return Err(ctx_err(
            opts,
            format!("unsupported color specification `{name}`"),
        ));
    }
    let tensor = Tensor::try_from(value).map_err(|e| ctx_err(opts, e))?;
    if tensor.data.len() != 3 {
        return Err(ctx_err(opts, "color vectors must contain three elements"));
    }
    Ok(Vec4::new(
        tensor.data[0] as f32,
        tensor.data[1] as f32,
        tensor.data[2] as f32,
        1.0,
    ))
}

pub(crate) fn color_from_token(token: char) -> Option<Vec4> {
    match token {
        'r' | 'R' => Some(Vec4::new(1.0, 0.0, 0.0, 1.0)),
        'g' | 'G' => Some(Vec4::new(0.0, 1.0, 0.0, 1.0)),
        'b' | 'B' => Some(Vec4::new(0.0, 0.0, 1.0, 1.0)),
        'c' | 'C' => Some(Vec4::new(0.0, 1.0, 1.0, 1.0)),
        'm' | 'M' => Some(Vec4::new(1.0, 0.0, 1.0, 1.0)),
        'y' | 'Y' => Some(Vec4::new(1.0, 1.0, 0.0, 1.0)),
        'k' | 'K' => Some(Vec4::new(0.0, 0.0, 0.0, 1.0)),
        'w' | 'W' => Some(Vec4::new(1.0, 1.0, 1.0, 1.0)),
        _ => None,
    }
}

fn is_style_token(token: &str) -> bool {
    !token.trim().is_empty() && !looks_like_option_name(token)
}

pub fn looks_like_option_name(token: &str) -> bool {
    matches!(
        token.trim().to_ascii_lowercase().as_str(),
        "linewidth"
            | "color"
            | "linestyle"
            | "marker"
            | "markersize"
            | "markeredgecolor"
            | "markerfacecolor"
            | "linestyleorder"
            | "displayname"
            | "label"
    )
}

pub fn value_as_string(value: &Value) -> Option<String> {
    match value {
        Value::CharArray(chars) => Some(chars.data.iter().collect()),
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

pub(crate) fn value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Num(v) => Some(*v),
        Value::Int(i) => Some(i.to_f64()),
        Value::Tensor(tensor) => tensor.data.first().copied(),
        _ => None,
    }
}

pub(crate) fn value_as_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(b) => Some(*b),
        Value::Num(v) => Some(*v != 0.0),
        Value::Int(i) => Some(!i.is_zero()),
        Value::CharArray(chars) => {
            let text: String = chars.data.iter().collect();
            bool_from_text(&text)
        }
        Value::String(text) => bool_from_text(text),
        _ => None,
    }
}

fn bool_from_text(text: &str) -> Option<bool> {
    match text.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "yes" | "y" => Some(true),
        "off" | "false" | "no" | "n" => Some(false),
        _ => None,
    }
}

fn parse_colormap_option(opts: &LineStyleParseOptions, value: &Value) -> Result<ColorMap, String> {
    if let Some(name) = value_as_string(value) {
        if let Some(map) = parse_colormap_name(&name) {
            return Ok(map);
        }
        return Err(ctx_err(opts, format!("unsupported colormap `{name}`")));
    }
    let color = parse_color_value(opts, value)?;
    Ok(ColorMap::Custom(color, color))
}

fn parse_colormap_name(name: &str) -> Option<ColorMap> {
    match name.trim().to_ascii_lowercase().as_str() {
        "parula" => Some(ColorMap::Parula),
        "jet" => Some(ColorMap::Jet),
        "turbo" => Some(ColorMap::Turbo),
        "viridis" => Some(ColorMap::Viridis),
        "plasma" => Some(ColorMap::Plasma),
        "inferno" => Some(ColorMap::Inferno),
        "magma" => Some(ColorMap::Magma),
        "hot" => Some(ColorMap::Hot),
        "cool" => Some(ColorMap::Cool),
        "spring" => Some(ColorMap::Spring),
        "summer" => Some(ColorMap::Summer),
        "autumn" => Some(ColorMap::Autumn),
        "winter" => Some(ColorMap::Winter),
        "gray" | "grey" => Some(ColorMap::Gray),
        "bone" => Some(ColorMap::Bone),
        "copper" => Some(ColorMap::Copper),
        "pink" => Some(ColorMap::Pink),
        "lines" => Some(ColorMap::Lines),
        _ => None,
    }
}

fn parse_shading_option(
    opts: &LineStyleParseOptions,
    value: &Value,
) -> Result<ShadingMode, String> {
    let Some(text) = value_as_string(value) else {
        return Err(ctx_err(opts, "Shading must be a string"));
    };
    match text.trim().to_ascii_lowercase().as_str() {
        "flat" => Ok(ShadingMode::Flat),
        "interp" | "gouraud" => Ok(ShadingMode::Smooth),
        "faceted" => Ok(ShadingMode::Faceted),
        "none" => Ok(ShadingMode::None),
        other => Err(ctx_err(opts, format!("unsupported Shading `{other}`"))),
    }
}

fn parse_alpha_value(opts: &LineStyleParseOptions, value: &Value) -> Result<f32, String> {
    let alpha =
        value_as_f64(value).ok_or_else(|| ctx_err(opts, "Alpha/FacesAlpha must be numeric"))?;
    Ok(alpha.clamp(0.0, 1.0) as f32)
}

fn apply_face_color_option(
    opts: &LineStyleParseOptions,
    value: &Value,
    style: &mut SurfaceStyle,
) -> Result<(), String> {
    if let Some(text) = value_as_string(value) {
        let lower = text.trim().to_ascii_lowercase();
        return match lower.as_str() {
            "none" => {
                style.alpha = 0.0;
                Ok(())
            }
            "flat" => {
                style.shading = ShadingMode::Flat;
                Ok(())
            }
            "interp" => {
                style.shading = ShadingMode::Smooth;
                Ok(())
            }
            "texturemap" => {
                style.flatten_z = true;
                Ok(())
            }
            _ => {
                let color = parse_color_value(opts, value)?;
                style.colormap = ColorMap::Custom(color, color);
                style.shading = ShadingMode::None;
                style.lighting_enabled = false;
                Ok(())
            }
        };
    }
    let color = parse_color_value(opts, value)?;
    style.colormap = ColorMap::Custom(color, color);
    style.shading = ShadingMode::None;
    style.lighting_enabled = false;
    Ok(())
}

fn apply_edge_color_option(
    opts: &LineStyleParseOptions,
    value: &Value,
    style: &mut SurfaceStyle,
) -> Result<(), String> {
    if let Some(text) = value_as_string(value) {
        let lower = text.trim().to_ascii_lowercase();
        return match lower.as_str() {
            "none" => {
                style.wireframe = false;
                Ok(())
            }
            "auto" | "flat" | "interp" => {
                style.wireframe = true;
                Ok(())
            }
            _ => Err(ctx_err(
                opts,
                "EdgeColor only supports 'auto', 'flat', or 'none' in this build",
            )),
        };
    }
    Err(ctx_err(
        opts,
        "EdgeColor does not support custom RGB values yet",
    ))
}

fn parse_surface_bool(
    opts: &LineStyleParseOptions,
    field: &str,
    value: &Value,
) -> Result<bool, String> {
    value_as_bool(value).ok_or_else(|| ctx_err(opts, format!("{field} must be logical (on/off)")))
}

fn parse_lighting_option(opts: &LineStyleParseOptions, value: &Value) -> Result<bool, String> {
    if let Some(text) = value_as_string(value) {
        let lower = text.trim().to_ascii_lowercase();
        return match lower.as_str() {
            "none" => Ok(false),
            "flat" | "gouraud" | "phong" | "auto" => Ok(true),
            _ => Err(ctx_err(
                opts,
                format!("unsupported Lighting value `{lower}`"),
            )),
        };
    }
    parse_surface_bool(opts, "Lighting", value)
}

pub fn parse_bar_style_args(
    builtin: &'static str,
    rest: &[Value],
    defaults: BarStyleDefaults,
) -> Result<BarStyle, String> {
    let opts = LineStyleParseOptions::generic(builtin);
    let mut style = BarStyle {
        face_color: defaults.face_color,
        face_alpha: 1.0,
        edge_color: None,
        edge_alpha: 1.0,
        line_width: DEFAULT_LINE_WIDTH,
        bar_width: defaults.bar_width.clamp(0.1, 1.0),
        label: None,
        face_color_flat: false,
        layout: BarLayout::Grouped,
    };

    if rest.is_empty() {
        return Ok(style);
    }

    let filtered = strip_bar_layout_tokens(rest, &mut style);
    let rest = filtered.as_slice();

    let mut idx = 0usize;
    if let Some(token) = rest.get(0).and_then(value_as_string) {
        let trimmed = token.trim();
        if !trimmed.is_empty() && !is_bar_option_name(trimmed) {
            style.face_color = parse_bar_color_literal(&opts, trimmed)?;
            idx = 1;
        }
    }

    let remaining = &rest[idx..];
    if remaining.is_empty() {
        return Ok(style);
    }
    if remaining.len() % 2 != 0 {
        return Err(bar_ctx_err(
            builtin,
            "name-value arguments must come in pairs",
        ));
    }

    for pair in remaining.chunks_exact(2) {
        let key_value = pair[0].clone();
        let Some(key) = value_as_string(&pair[0]) else {
            return Err(bar_ctx_err(
                builtin,
                "option names must be char arrays or strings",
            ));
        };
        let lower = key.trim().to_ascii_lowercase();
        match lower.as_str() {
            "facecolor" | "color" => match parse_bar_face_color(&opts, &pair[1])? {
                FaceColorSpec::Auto => {
                    style.face_color_flat = false;
                }
                FaceColorSpec::Flat => {
                    style.face_color_flat = true;
                    style.face_alpha = 1.0;
                }
                FaceColorSpec::None => {
                    style.face_alpha = 0.0;
                    style.face_color_flat = false;
                }
                FaceColorSpec::Color(color) => {
                    style.face_color = color;
                    style.face_alpha = 1.0;
                    style.face_color_flat = false;
                }
            },
            "edgecolor" => {
                style.edge_color = match parse_bar_edge_color(&opts, &pair[1])? {
                    EdgeColorSpec::Auto | EdgeColorSpec::Flat => None,
                    EdgeColorSpec::None => None,
                    EdgeColorSpec::Color(color) => Some(color),
                };
            }
            "linewidth" => {
                let width = value_as_f64(&pair[1])
                    .ok_or_else(|| bar_ctx_err(builtin, "LineWidth must be numeric"))?;
                if width <= 0.0 {
                    return Err(bar_ctx_err(builtin, "LineWidth must be positive"));
                }
                style.line_width = width as f32;
            }
            "barwidth" => {
                let width = value_as_f64(&pair[1])
                    .ok_or_else(|| bar_ctx_err(builtin, "BarWidth must be numeric"))?;
                if width <= 0.0 {
                    return Err(bar_ctx_err(builtin, "BarWidth must be positive"));
                }
                style.bar_width = width as f32;
            }
            "facealpha" => {
                let alpha = value_as_f64(&pair[1])
                    .ok_or_else(|| bar_ctx_err(builtin, "FaceAlpha must be numeric"))?;
                style.face_alpha = alpha.clamp(0.0, 1.0) as f32;
            }
            "edgealpha" => {
                let alpha = value_as_f64(&pair[1])
                    .ok_or_else(|| bar_ctx_err(builtin, "EdgeAlpha must be numeric"))?;
                style.edge_alpha = alpha.clamp(0.0, 1.0) as f32;
            }
            "displayname" | "label" => {
                let Some(name) = value_as_string(&pair[1]) else {
                    return Err(bar_ctx_err(
                        builtin,
                        "DisplayName must be a char array or string",
                    ));
                };
                style.label = Some(name);
            }
            other => {
                return Err(bar_ctx_err(
                    builtin,
                    format!("unsupported option `{other}`"),
                ));
            }
        }
        drop(key_value);
    }

    Ok(style)
}

fn strip_bar_layout_tokens(rest: &[Value], style: &mut BarStyle) -> Vec<Value> {
    let mut filtered = Vec::with_capacity(rest.len());
    let mut expecting_value = false;
    for value in rest {
        if expecting_value {
            filtered.push(value.clone());
            expecting_value = false;
            continue;
        }
        if let Some(text) = value_as_string(value) {
            let lower = text.trim().to_ascii_lowercase();
            match lower.as_str() {
                "stacked" => {
                    style.layout = BarLayout::Stacked;
                    continue;
                }
                "grouped" => {
                    style.layout = BarLayout::Grouped;
                    continue;
                }
                _ => {
                    if is_bar_option_name(&lower) {
                        filtered.push(value.clone());
                        expecting_value = true;
                        continue;
                    }
                }
            }
        }
        filtered.push(value.clone());
    }
    filtered
}

fn is_bar_option_name(token: &str) -> bool {
    matches!(
        token.trim().to_ascii_lowercase().as_str(),
        "facecolor"
            | "color"
            | "edgecolor"
            | "linewidth"
            | "barwidth"
            | "facealpha"
            | "edgealpha"
            | "displayname"
            | "label"
    )
}

impl BarStyle {
    pub fn requires_cpu_path(&self) -> bool {
        self.face_color_flat
    }
}

enum FaceColorSpec {
    Auto,
    Flat,
    None,
    Color(Vec4),
}

enum EdgeColorSpec {
    Auto,
    Flat,
    None,
    Color(Vec4),
}

fn parse_bar_face_color(
    opts: &LineStyleParseOptions,
    value: &Value,
) -> Result<FaceColorSpec, String> {
    if let Some(text) = value_as_string(value) {
        let lower = text.trim().to_ascii_lowercase();
        return match lower.as_str() {
            "auto" => Ok(FaceColorSpec::Auto),
            "none" => Ok(FaceColorSpec::None),
            "flat" => Ok(FaceColorSpec::Flat),
            _ => {
                let color = parse_bar_color_literal(opts, &text)?;
                Ok(FaceColorSpec::Color(color))
            }
        };
    }
    let color = parse_color_value(opts, value)?;
    Ok(FaceColorSpec::Color(color))
}

fn parse_bar_edge_color(
    opts: &LineStyleParseOptions,
    value: &Value,
) -> Result<EdgeColorSpec, String> {
    if let Some(text) = value_as_string(value) {
        let lower = text.trim().to_ascii_lowercase();
        return match lower.as_str() {
            "auto" => Ok(EdgeColorSpec::Auto),
            "none" => Ok(EdgeColorSpec::None),
            "flat" => Ok(EdgeColorSpec::Flat),
            _ => {
                let color = parse_bar_color_literal(opts, &text)?;
                Ok(EdgeColorSpec::Color(color))
            }
        };
    }
    let color = parse_color_value(opts, value)?;
    Ok(EdgeColorSpec::Color(color))
}

fn parse_bar_color_literal(opts: &LineStyleParseOptions, token: &str) -> Result<Vec4, String> {
    if let Some(ch) = token.trim().chars().next() {
        if let Some(color) = color_from_token(ch) {
            return Ok(color);
        }
    }
    Err(ctx_err(
        opts,
        format!("unsupported color specification `{token}`"),
    ))
}

fn bar_ctx_err(builtin: &str, msg: impl Into<String>) -> String {
    format!("{builtin}: {}", msg.into())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn style_string_with_marker_no_longer_forces_cpu_fallback() {
        let rest = vec![Value::String("o--".into())];
        let opts = LineStyleParseOptions::plot();
        let parsed = parse_line_style_args(&rest, &opts).expect("parsed");
        assert!(parsed.appearance.marker.is_some());
        assert!(!parsed.requires_cpu_fallback);
    }

    #[test]
    fn marker_none_disables_marker_without_fallback() {
        let rest = vec![Value::String("Marker".into()), Value::String("none".into())];
        let opts = LineStyleParseOptions::plot();
        let parsed = parse_line_style_args(&rest, &opts).expect("parsed");
        assert!(parsed.appearance.marker.is_none());
        assert!(!parsed.requires_cpu_fallback);
    }

    #[test]
    fn marker_color_flat_literal_is_supported() {
        let opts = LineStyleParseOptions::plot();
        let color = parse_marker_color_value(&opts, &Value::String("flat".into())).unwrap();
        assert_eq!(color, MarkerColor::Flat);
    }

    #[test]
    fn bar_style_parses_face_and_edge_colors() {
        let defaults = BarStyleDefaults::new(Vec4::new(0.2, 0.6, 0.9, 1.0), 0.8);
        let rest = vec![
            Value::String("FaceColor".into()),
            Value::String("r".into()),
            Value::String("EdgeColor".into()),
            Value::String("k".into()),
            Value::String("BarWidth".into()),
            Value::from(0.5),
        ];
        let style = parse_bar_style_args("bar", &rest, defaults).expect("parsed");
        assert!((style.bar_width - 0.5).abs() < f32::EPSILON);
        assert_eq!(style.face_color, Vec4::new(1.0, 0.0, 0.0, 1.0));
        assert_eq!(style.edge_color, Some(Vec4::new(0.0, 0.0, 0.0, 1.0)));
    }

    #[test]
    fn bar_style_accepts_label() {
        let defaults = BarStyleDefaults::new(Vec4::new(0.2, 0.6, 0.9, 1.0), 0.8);
        let rest = vec![
            Value::String("DisplayName".into()),
            Value::String("My Bars".into()),
        ];
        let style = parse_bar_style_args("bar", &rest, defaults).expect("parsed");
        assert_eq!(style.label.as_deref(), Some("My Bars"));
    }

    #[test]
    fn bar_style_accepts_flat_facecolor() {
        let defaults = BarStyleDefaults::new(Vec4::new(0.2, 0.6, 0.9, 1.0), 0.8);
        let rest = vec![
            Value::String("FaceColor".into()),
            Value::String("flat".into()),
        ];
        let style = parse_bar_style_args("bar", &rest, defaults).expect("parsed");
        assert!(style.face_color_flat);
        assert!(style.requires_cpu_path());
    }
}

use super::*;
pub(super) fn guess_request_format(value: &Value) -> RequestFormat {
    match value {
        Value::Struct(_) => RequestFormat::Form,
        Value::Cell(cell) if cell.cols == 2 => RequestFormat::Form,
        Value::CharArray(ca) if ca.rows == 1 => RequestFormat::Text,
        Value::String(runmat_value::StringElement::Text(_)) => RequestFormat::Text,
        Value::StringArray(sa) => {
            if sa.data().len() == 1 {
                RequestFormat::Text
            } else {
                RequestFormat::Json
            }
        }
        Value::Tensor(_) | Value::LogicalArray(_) => RequestFormat::Json,
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => RequestFormat::Json,
        _ => RequestFormat::Json,
    }
}

pub(super) fn infer_request_format(media_type: &str) -> RequestFormat {
    let lower = media_type.trim().to_ascii_lowercase();
    if lower.contains("json") {
        RequestFormat::Json
    } else if lower.starts_with("text/") || lower.contains("xml") {
        RequestFormat::Text
    } else if lower == "application/x-www-form-urlencoded" {
        RequestFormat::Form
    } else {
        RequestFormat::Binary
    }
}

pub(super) fn default_content_type_for(format: RequestFormat) -> Option<String> {
    match format {
        RequestFormat::Form => Some("application/x-www-form-urlencoded".to_string()),
        RequestFormat::Json => Some("application/json".to_string()),
        RequestFormat::Text => Some("text/plain; charset=utf-8".to_string()),
        RequestFormat::Binary => Some("application/octet-stream".to_string()),
        RequestFormat::Auto => None,
    }
}

#[derive(Clone, Debug)]
pub(super) struct PreparedBody {
    pub(super) bytes: Vec<u8>,
    pub(super) content_type: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum ResolvedContentType {
    Text,
    Json,
    Binary,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum RequestFormat {
    Auto,
    Form,
    Json,
    Text,
    Binary,
}

#[derive(Clone, Debug)]
pub(super) struct WebWriteOptions {
    pub(super) content_type: HttpResponseMode,
    pub(super) timeout: Duration,
    pub(super) headers: Vec<(String, String)>,
    pub(super) api_key_header: Option<(String, String)>,
    pub(super) user_agent: Option<String>,
    pub(super) username: Option<String>,
    pub(super) password: Option<String>,
    pub(super) method: HttpMethod,
    pub(super) request_format: RequestFormat,
    pub(super) request_format_explicit: bool,
    pub(super) media_type: Option<String>,
}

impl Default for WebWriteOptions {
    fn default() -> Self {
        Self {
            content_type: HttpResponseMode::Auto,
            timeout: Duration::from_secs_f64(DEFAULT_TIMEOUT_SECONDS),
            headers: Vec::new(),
            api_key_header: None,
            user_agent: None,
            username: None,
            password: None,
            method: HttpMethod::Post,
            request_format: RequestFormat::Auto,
            request_format_explicit: false,
            media_type: None,
        }
    }
}

impl WebWriteOptions {
    pub(super) fn resolve_content_type(&self, header: Option<&str>) -> ResolvedContentType {
        match self.content_type {
            HttpResponseMode::Json => ResolvedContentType::Json,
            HttpResponseMode::Text => ResolvedContentType::Text,
            HttpResponseMode::Binary => ResolvedContentType::Binary,
            HttpResponseMode::Auto => infer_response_content_type(header),
        }
    }
}

pub(super) fn infer_response_content_type(header: Option<&str>) -> ResolvedContentType {
    if let Some(raw) = header {
        let mime = raw
            .split(';')
            .next()
            .map(|part| part.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if mime == "application/json" || mime == "text/json" || mime.ends_with("+json") {
            ResolvedContentType::Json
        } else if mime.starts_with("text/")
            || mime == "application/xml"
            || mime.ends_with("+xml")
            || mime == "application/xhtml+xml"
            || mime == "application/javascript"
            || mime == "application/x-www-form-urlencoded"
        {
            ResolvedContentType::Text
        } else {
            ResolvedContentType::Binary
        }
    } else {
        ResolvedContentType::Text
    }
}

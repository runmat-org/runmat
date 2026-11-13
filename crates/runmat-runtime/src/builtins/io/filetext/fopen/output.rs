use runmat_value::{Tensor, Value};

use crate::builtins::io::filetext::{helpers::char_array_value, registry::FileInfo};
use crate::BuiltinResult;

use super::{fopen_error_with_detail, make_cell_column, FOPEN_ERROR_INTERNAL};

#[derive(Clone)]
pub struct FopenEval {
    kind: FopenEvalKind,
}

#[derive(Clone)]
enum FopenEvalKind {
    Open(OpenOutputs),
    Query(QueryOutputs),
    List(Box<ListOutputs>),
}

#[derive(Clone)]
pub(crate) struct OpenOutputs {
    pub fid: f64,
    pub message: String,
    pub machinefmt: String,
    pub encoding: String,
}

#[derive(Clone)]
pub(crate) struct QueryOutputs {
    pub filename: String,
    pub permission: String,
    pub machinefmt: String,
    pub encoding: String,
}

#[derive(Clone)]
pub(crate) struct ListOutputs {
    pub handles: Tensor,
    pub names: Value,
    pub machinefmts: Value,
    pub encodings: Value,
}

impl FopenEval {
    pub(super) fn open(outputs: OpenOutputs) -> Self {
        Self {
            kind: FopenEvalKind::Open(outputs),
        }
    }

    pub(super) fn query(outputs: QueryOutputs) -> Self {
        Self {
            kind: FopenEvalKind::Query(outputs),
        }
    }

    pub(super) fn list(outputs: ListOutputs) -> Self {
        Self {
            kind: FopenEvalKind::List(Box::new(outputs)),
        }
    }

    pub fn first_output(&self) -> Value {
        match &self.kind {
            FopenEvalKind::Open(outputs) => Value::Num(outputs.fid),
            FopenEvalKind::Query(outputs) => char_array_value(&outputs.filename),
            FopenEvalKind::List(outputs) => Value::Tensor(outputs.handles.clone()),
        }
    }

    pub fn outputs(&self) -> Vec<Value> {
        match &self.kind {
            FopenEvalKind::Open(outputs) => outputs.outputs(),
            FopenEvalKind::Query(outputs) => outputs.outputs(),
            FopenEvalKind::List(outputs) => outputs.outputs(),
        }
    }

    #[cfg(test)]
    pub(crate) fn as_open(&self) -> Option<&OpenOutputs> {
        match &self.kind {
            FopenEvalKind::Open(outputs) => Some(outputs),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn as_query(&self) -> Option<&QueryOutputs> {
        match &self.kind {
            FopenEvalKind::Query(outputs) => Some(outputs),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn as_list(&self) -> Option<&ListOutputs> {
        match &self.kind {
            FopenEvalKind::List(outputs) => Some(outputs),
            _ => None,
        }
    }
}

impl OpenOutputs {
    pub(super) fn success(fid: i32, machinefmt: String, encoding: String) -> Self {
        Self {
            fid: fid as f64,
            message: String::new(),
            machinefmt,
            encoding,
        }
    }

    pub(super) fn failure(message: String) -> Self {
        Self {
            fid: -1.0,
            message,
            machinefmt: String::new(),
            encoding: String::new(),
        }
    }

    fn outputs(&self) -> Vec<Value> {
        vec![
            Value::Num(self.fid),
            char_array_value(&self.message),
            char_array_value(&self.machinefmt),
            char_array_value(&self.encoding),
        ]
    }
}

impl QueryOutputs {
    pub(super) fn from_info(info: FileInfo) -> Self {
        let filename = match info.path {
            Some(path) => path.to_string_lossy().to_string(),
            None => info.name,
        };
        Self {
            filename,
            permission: info.permission,
            machinefmt: info.machinefmt,
            encoding: info.encoding,
        }
    }

    pub(super) fn not_found() -> Self {
        Self {
            filename: String::new(),
            permission: String::new(),
            machinefmt: String::new(),
            encoding: String::new(),
        }
    }

    fn outputs(&self) -> Vec<Value> {
        vec![
            char_array_value(&self.filename),
            char_array_value(&self.permission),
            char_array_value(&self.machinefmt),
            char_array_value(&self.encoding),
        ]
    }
}

impl ListOutputs {
    pub(super) fn from_infos(infos: Vec<FileInfo>) -> BuiltinResult<Self> {
        let mut handles: Vec<f64> = infos.iter().map(|info| info.id as f64).collect();
        let rows = handles.len();
        if rows == 0 {
            handles = Vec::new();
        }
        let shape = if rows == 0 { vec![0, 1] } else { vec![rows, 1] };
        let tensor = Tensor::new(handles, shape)
            .map_err(|e| fopen_error_with_detail(&FOPEN_ERROR_INTERNAL, &e))?;

        let mut name_values = Vec::with_capacity(infos.len());
        let mut machine_values = Vec::with_capacity(infos.len());
        let mut encoding_values = Vec::with_capacity(infos.len());
        for info in &infos {
            let display = match &info.path {
                Some(path) => path.to_string_lossy().to_string(),
                None => info.name.clone(),
            };
            name_values.push(char_array_value(&display));
            machine_values.push(char_array_value(&info.machinefmt));
            encoding_values.push(char_array_value(&info.encoding));
        }

        let names = make_cell_column(name_values)?;
        let machinefmts = make_cell_column(machine_values)?;
        let encodings = make_cell_column(encoding_values)?;

        Ok(Self {
            handles: tensor,
            names,
            machinefmts,
            encodings,
        })
    }

    fn outputs(&self) -> Vec<Value> {
        vec![
            Value::Tensor(self.handles.clone()),
            self.names.clone(),
            self.machinefmts.clone(),
            self.encodings.clone(),
        ]
    }
}

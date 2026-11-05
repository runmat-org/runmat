//! Columnar fastener tables: their binary form, an in-memory source, and loading a table.

use super::*;

// ------------------------------------------------------------------------------ tables

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ColumnData {
    F64 { unit: String, values: Vec<f64> },
    I64 { values: Vec<i64> },
    Text { values: Vec<String> },
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::F64 { values, .. } => values.len(),
            ColumnData::I64 { values } => values.len(),
            ColumnData::Text { values } => values.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub data: ColumnData,
}

/// A canonical columnar table (irregular instance sets, imported arrays).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ColumnarTable {
    pub columns: Vec<Column>,
}

pub(super) const TABLE_MAGIC: &[u8; 8] = b"RMTBL\x00\x00\x01";

impl ColumnarTable {
    pub fn rows(&self) -> usize {
        self.columns.first().map(|c| c.data.len()).unwrap_or(0)
    }

    pub fn column(&self, name: &str) -> Option<&ColumnData> {
        self.columns
            .iter()
            .find(|c| c.name == name)
            .map(|c| &c.data)
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.rows();
        let mut names = BTreeSet::new();
        for c in &self.columns {
            if !names.insert(c.name.as_str()) {
                return Err(format!("column `{}` appears twice", c.name));
            }
            if c.data.len() != n {
                return Err(format!(
                    "column `{}` has {} rows, expected {}",
                    c.name,
                    c.data.len(),
                    n
                ));
            }
        }
        Ok(())
    }

    /// The binary encoding: magic, column count, then per column its name, type tag, unit and
    /// values, all little-endian with length prefixes.
    pub fn encode_binary(&self) -> Vec<u8> {
        fn put_str(out: &mut Vec<u8>, s: &str) {
            out.extend_from_slice(&(s.len() as u32).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        let mut out = Vec::with_capacity(16 + self.rows() * self.columns.len() * 8);
        out.extend_from_slice(TABLE_MAGIC);
        out.extend_from_slice(&(self.columns.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.rows() as u64).to_le_bytes());
        for c in &self.columns {
            put_str(&mut out, &c.name);
            match &c.data {
                ColumnData::F64 { unit, values } => {
                    out.push(0);
                    put_str(&mut out, unit);
                    for v in values {
                        out.extend_from_slice(&v.to_bits().to_le_bytes());
                    }
                }
                ColumnData::I64 { values } => {
                    out.push(1);
                    for v in values {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
                ColumnData::Text { values } => {
                    out.push(2);
                    for v in values {
                        put_str(&mut out, v);
                    }
                }
            }
        }
        out
    }

    pub fn decode_binary(bytes: &[u8]) -> Result<ColumnarTable, String> {
        struct R<'a> {
            b: &'a [u8],
            at: usize,
        }
        impl R<'_> {
            fn take(&mut self, n: usize) -> Result<&[u8], String> {
                if self.at + n > self.b.len() {
                    return Err("truncated table".into());
                }
                let s = &self.b[self.at..self.at + n];
                self.at += n;
                Ok(s)
            }
            fn u32(&mut self) -> Result<u32, String> {
                Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
            }
            fn u64(&mut self) -> Result<u64, String> {
                Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            }
            fn string(&mut self) -> Result<String, String> {
                let n = self.u32()? as usize;
                String::from_utf8(self.take(n)?.to_vec()).map_err(|e| e.to_string())
            }
        }
        let mut r = R { b: bytes, at: 0 };
        if r.take(8)? != TABLE_MAGIC {
            return Err("not a RunMat columnar table".into());
        }
        let ncols = r.u32()? as usize;
        let nrows = r.u64()? as usize;
        let mut columns = Vec::with_capacity(ncols);
        for _ in 0..ncols {
            let name = r.string()?;
            let tag = r.take(1)?[0];
            let data = match tag {
                0 => {
                    let unit = r.string()?;
                    let mut values = Vec::with_capacity(nrows);
                    for _ in 0..nrows {
                        values.push(f64::from_bits(r.u64()?));
                    }
                    ColumnData::F64 { unit, values }
                }
                1 => {
                    let mut values = Vec::with_capacity(nrows);
                    for _ in 0..nrows {
                        values.push(r.u64()? as i64);
                    }
                    ColumnData::I64 { values }
                }
                2 => {
                    let mut values = Vec::with_capacity(nrows);
                    for _ in 0..nrows {
                        values.push(r.string()?);
                    }
                    ColumnData::Text { values }
                }
                t => return Err(format!("unknown column type {}", t)),
            };
            columns.push(Column { name, data });
        }
        if r.at != bytes.len() {
            return Err("trailing bytes after the table".into());
        }
        let t = ColumnarTable { columns };
        t.validate()?;
        Ok(t)
    }

    /// The text encoding: a header line of `name:f64[unit]`, `name:i64` or `name:text`
    /// fields, then one line per row; fields are tab-separated. Numbers are written in their
    /// shortest round-trip form, so the conversion is lossless. Text escapes `\\`, `\t`, `\n`.
    pub fn encode_text(&self) -> String {
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\")
                .replace('\t', "\\t")
                .replace('\n', "\\n")
        }
        let mut out = String::new();
        let header: Vec<String> = self
            .columns
            .iter()
            .map(|c| match &c.data {
                ColumnData::F64 { unit, .. } => format!("{}:f64[{}]", esc(&c.name), esc(unit)),
                ColumnData::I64 { .. } => format!("{}:i64", esc(&c.name)),
                ColumnData::Text { .. } => format!("{}:text", esc(&c.name)),
            })
            .collect();
        out.push_str(&header.join("\t"));
        out.push('\n');
        for r in 0..self.rows() {
            let fields: Vec<String> = self
                .columns
                .iter()
                .map(|c| match &c.data {
                    ColumnData::F64 { values, .. } => format!("{:?}", values[r]),
                    ColumnData::I64 { values } => values[r].to_string(),
                    ColumnData::Text { values } => esc(&values[r]),
                })
                .collect();
            out.push_str(&fields.join("\t"));
            out.push('\n');
        }
        out
    }

    pub fn decode_text(text: &str) -> Result<ColumnarTable, String> {
        fn unesc(s: &str) -> String {
            let mut out = String::new();
            let mut it = s.chars();
            while let Some(c) = it.next() {
                if c == '\\' {
                    match it.next() {
                        Some('t') => out.push('\t'),
                        Some('n') => out.push('\n'),
                        Some(o) => out.push(o),
                        None => out.push('\\'),
                    }
                } else {
                    out.push(c);
                }
            }
            out
        }
        let mut lines = text.lines();
        let header = lines.next().ok_or("empty table text")?;
        let mut columns = Vec::new();
        for field in header.split('\t') {
            let (name, ty) = field
                .rsplit_once(':')
                .ok_or_else(|| format!("header field {:?} has no type", field))?;
            let data = if let Some(unit) = ty.strip_prefix("f64[").and_then(|u| u.strip_suffix(']'))
            {
                ColumnData::F64 {
                    unit: unesc(unit),
                    values: Vec::new(),
                }
            } else if ty == "i64" {
                ColumnData::I64 { values: Vec::new() }
            } else if ty == "text" {
                ColumnData::Text { values: Vec::new() }
            } else {
                return Err(format!("unknown column type {:?}", ty));
            };
            columns.push(Column {
                name: unesc(name),
                data,
            });
        }
        for (ln, line) in lines.enumerate() {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != columns.len() {
                return Err(format!("row {} has {} fields", ln + 1, fields.len()));
            }
            for (c, f) in columns.iter_mut().zip(fields) {
                match &mut c.data {
                    ColumnData::F64 { values, .. } => values.push(
                        f.parse::<f64>()
                            .map_err(|e| format!("row {}: {}", ln + 1, e))?,
                    ),
                    ColumnData::I64 { values } => values.push(
                        f.parse::<i64>()
                            .map_err(|e| format!("row {}: {}", ln + 1, e))?,
                    ),
                    ColumnData::Text { values } => values.push(unesc(f)),
                }
            }
        }
        let t = ColumnarTable { columns };
        t.validate()?;
        Ok(t)
    }

    /// `sha256:<hex>` of the binary encoding.
    pub fn digest(&self) -> String {
        format!(
            "sha256:{}",
            hex::encode(Sha256::digest(self.encode_binary()))
        )
    }
}

/// Fetches tables by digest.
pub trait TableSource {
    /// The table's binary encoding.
    fn fetch(&self, digest: &str) -> Result<Vec<u8>, String>;
}

#[derive(Clone, Debug, Default)]
pub struct MemoryTables {
    pub blobs: BTreeMap<String, Vec<u8>>,
}

impl MemoryTables {
    /// Store a table; returns its digest.
    pub fn put(&mut self, t: &ColumnarTable) -> String {
        let d = t.digest();
        self.blobs.insert(d.clone(), t.encode_binary());
        d
    }
}

impl TableSource for MemoryTables {
    fn fetch(&self, digest: &str) -> Result<Vec<u8>, String> {
        self.blobs
            .get(digest)
            .cloned()
            .ok_or_else(|| format!("no table `{}`", digest))
    }
}

/// Load and integrity-check a table.
pub fn load_table(
    source: &dyn TableSource,
    digest: &str,
) -> Result<Arc<ColumnarTable>, FastenerProblem> {
    let bytes = source
        .fetch(digest)
        .map_err(|message| FastenerProblem::TableIntegrity { message })?;
    let got = format!("sha256:{}", hex::encode(Sha256::digest(&bytes)));
    if got != digest {
        return Err(FastenerProblem::TableIntegrity {
            message: format!("table content hashes to {}, not {}", got, digest),
        });
    }
    ColumnarTable::decode_binary(&bytes)
        .map(Arc::new)
        .map_err(|message| FastenerProblem::TableIntegrity { message })
}

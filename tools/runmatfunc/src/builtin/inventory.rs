use std::collections::HashMap;

use anyhow::Result;
use runmat_builtins::{builtin_docs, builtin_functions, AccelTag, BuiltinDoc, BuiltinFunction};

use crate::builtin::metadata::{BuiltinManifest, BuiltinRecord};

pub fn collect_manifest() -> Result<BuiltinManifest> {
    let docs_index = index_docs();
    let builtins = builtin_functions()
        .into_iter()
        .map(|func| build_record(func, &docs_index))
        .collect();
    Ok(BuiltinManifest { builtins })
}

fn index_docs() -> HashMap<&'static str, &'static BuiltinDoc> {
    builtin_docs()
        .into_iter()
        .map(|doc| (doc.name, doc))
        .collect()
}

fn build_record(
    func: &'static BuiltinFunction,
    docs: &HashMap<&'static str, &'static BuiltinDoc>,
) -> BuiltinRecord {
    let doc = docs.get(func.name).copied();
    BuiltinRecord {
        name: func.name.to_string(),
        category: doc
            .and_then(|d| d.category.map(|s| s.to_string()))
            .or_else(|| Some(func.category.to_string()).filter(|s| !s.is_empty())),
        summary: doc
            .and_then(|d| d.summary.map(|s| s.to_string()))
            .or_else(|| Some(func.description.to_string()).filter(|s| !s.is_empty())),
        keywords: doc
            .and_then(|d| {
                d.keywords
                    .map(|s| s.split(',').map(|k| k.trim().to_string()).collect())
            })
            .unwrap_or_default(),
        accel_tags: func.accel_tags.iter().map(tag_to_str).collect(),
        is_sink: func.is_sink,
        param_types: func.param_types.iter().map(|t| format!("{t:?}")).collect(),
        return_type: format!("{:?}", func.return_type),
    }
}

fn tag_to_str(tag: &AccelTag) -> String {
    match tag {
        AccelTag::Unary => "unary".to_string(),
        AccelTag::Elementwise => "elementwise".to_string(),
        AccelTag::Reduction => "reduction".to_string(),
        AccelTag::MatMul => "matmul".to_string(),
        AccelTag::Transpose => "transpose".to_string(),
        AccelTag::ArrayConstruct => "array_construct".to_string(),
    }
}

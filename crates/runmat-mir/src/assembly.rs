use crate::{MirBody, MirFunctionMetadata};
mod class_sources;
use runmat_types::FunctionId;
use runmat_types::{ClassDeclaration, ClassId, IntegerLiteral, MemberName};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MirAssembly {
    pub bodies: BTreeMap<FunctionId, MirBody>,
    pub functions: BTreeMap<FunctionId, MirFunctionMetadata>,
    pub classes: Vec<ClassDeclaration>,
    /// Literal class defaults projected once from the canonical HIR class body.
    #[serde(default)]
    pub class_property_defaults: Vec<MirClassPropertyDefault>,
    pub entrypoints: Vec<FunctionId>,
}

impl MirAssembly {
    /// Project the HIR containment edge into a retained callable-definition
    /// parent. A script's synthetic entrypoint contains anonymous expressions,
    /// but it is not a function definition published into the session registry.
    pub fn retained_callable_parent(
        &self,
        function: FunctionId,
    ) -> Result<Option<FunctionId>, String> {
        let metadata = self
            .functions
            .get(&function)
            .ok_or_else(|| format!("function {function:?} has no MIR metadata"))?;
        let retained_parent = metadata.parent;
        let mut visited = BTreeSet::new();
        let mut cursor = function;
        loop {
            if !visited.insert(cursor) {
                return Err(format!("function {function:?} has a cyclic lexical parent"));
            }
            let current = self
                .functions
                .get(&cursor)
                .ok_or_else(|| format!("lexical ancestor {cursor:?} has no MIR metadata"))?;
            let Some(parent) = current.parent else {
                if current.source_role == runmat_types::FunctionSourceRole::Nested {
                    return Err(format!("nested function {cursor:?} has no lexical parent"));
                }
                break;
            };
            let parent_metadata = self
                .functions
                .get(&parent)
                .ok_or_else(|| format!("lexical parent {parent:?} has no MIR metadata"))?;
            if current.source != parent_metadata.source {
                return Err(format!(
                    "function {cursor:?} has a cross-source lexical parent {parent:?}"
                ));
            }
            if parent_metadata.kind == runmat_types::FunctionKind::SyntheticEntrypoint {
                if current.kind != runmat_types::FunctionKind::Anonymous
                    || current.source_role != runmat_types::FunctionSourceRole::Anonymous
                    || parent_metadata.source_role != runmat_types::FunctionSourceRole::Synthetic
                    || parent_metadata.parent.is_some()
                {
                    return Err(format!(
                        "function {cursor:?} has an invalid synthetic lexical parent {parent:?}"
                    ));
                }
                return if cursor == function {
                    Ok(None)
                } else {
                    Ok(retained_parent)
                };
            }
            cursor = parent;
        }
        Ok(retained_parent)
    }
}

#[cfg(all(test, feature = "compiler"))]
mod tests {
    use super::*;

    fn lower(source: &str) -> MirAssembly {
        let parsed = runmat_parser::parse(source).expect("parse lexical parent fixture");
        let hir = runmat_hir::lower(&parsed, &runmat_hir::LoweringContext::empty())
            .expect("lower lexical parent fixture");
        crate::lowering::lower_assembly(&hir.assembly).expect("lower lexical parent MIR")
    }

    #[test]
    fn script_containment_is_not_a_retained_callable_parent() {
        let mut mir = lower("offset = 1; f = @(x) x + offset;");
        let (anonymous, metadata) = mir
            .functions
            .iter()
            .find(|(_, metadata)| metadata.kind == runmat_hir::FunctionKind::Anonymous)
            .expect("anonymous function");
        let anonymous = *anonymous;
        let script = metadata.parent.expect("HIR script containment");
        assert_eq!(
            mir.functions[&script].kind,
            runmat_hir::FunctionKind::SyntheticEntrypoint
        );
        assert_eq!(mir.retained_callable_parent(anonymous), Ok(None));

        let mut child = mir.functions[&anonymous].clone();
        child.parent = Some(anonymous);
        let descendant = FunctionId(usize::MAX - 1);
        mir.functions.insert(descendant, child);
        assert_eq!(
            mir.retained_callable_parent(descendant),
            Ok(Some(anonymous))
        );

        mir.functions.get_mut(&anonymous).unwrap().source_role =
            runmat_types::FunctionSourceRole::Nested;
        assert!(mir.retained_callable_parent(anonymous).is_err());
        mir.functions.get_mut(&anonymous).unwrap().source_role =
            runmat_types::FunctionSourceRole::Anonymous;
        mir.functions.get_mut(&script).unwrap().source = runmat_types::ProgramSourceId(999);
        assert!(mir.retained_callable_parent(anonymous).is_err());
        let source = mir.functions[&anonymous].source;
        mir.functions.get_mut(&script).unwrap().source = source;
        mir.functions.get_mut(&anonymous).unwrap().parent = Some(anonymous);
        assert!(mir.retained_callable_parent(anonymous).is_err());
        mir.functions.get_mut(&anonymous).unwrap().parent = Some(descendant);
        assert!(mir.retained_callable_parent(descendant).is_err());
        mir.functions.get_mut(&anonymous).unwrap().parent = Some(FunctionId(usize::MAX));
        assert!(mir.retained_callable_parent(anonymous).is_err());
    }

    #[test]
    fn anonymous_inside_named_function_retains_its_callable_parent() {
        let mut mir = lower(
            "function f = outer()\n offset = 1;\n f = @(x) inner(x) + offset;\n function y = inner(x)\n y = x + offset;\n end\n end",
        );
        let (anonymous, metadata) = mir
            .functions
            .iter()
            .find(|(_, metadata)| metadata.kind == runmat_hir::FunctionKind::Anonymous)
            .expect("anonymous function");
        let anonymous = *anonymous;
        let outer = metadata.parent.expect("named enclosing function");
        assert_eq!(mir.functions[&outer].name.0, "outer");
        assert_eq!(mir.retained_callable_parent(anonymous), Ok(Some(outer)));
        let nested = mir
            .functions
            .iter()
            .find_map(|(id, metadata)| (metadata.name.0 == "inner").then_some(*id))
            .expect("nested function");
        assert!(crate::analysis::lexical_reference_graph(&mir)[&anonymous].contains(&nested));
        mir.functions.get_mut(&outer).unwrap().source = runmat_types::ProgramSourceId(999);
        assert!(mir.retained_callable_parent(anonymous).is_err());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirClassPropertyDefault {
    pub class: ClassId,
    pub property: MemberName,
    pub literal: MirClassDefaultLiteral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MirClassDefaultLiteral {
    Num(#[serde(with = "runmat_types::serde_f64")] f64),
    Int(IntegerLiteral),
    Bool(bool),
    String(String),
}

impl PartialEq for MirClassDefaultLiteral {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Num(left), Self::Num(right)) => left.to_bits() == right.to_bits(),
            (Self::Int(left), Self::Int(right)) => left == right,
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::String(left), Self::String(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for MirClassDefaultLiteral {}

impl PartialEq for MirClassPropertyDefault {
    fn eq(&self, other: &Self) -> bool {
        self.class == other.class
            && self.property == other.property
            && self.literal == other.literal
    }
}

impl Eq for MirClassPropertyDefault {}

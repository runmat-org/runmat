use super::ctx::Ctx;
use crate::{HirClassMember, HirLValue, HirStmt, SemanticError, VarId};
use runmat_parser::{self as parser, Stmt as AstStmt};

impl Ctx {
    pub(crate) fn lower_stmts(&mut self, stmts: &[AstStmt]) -> Result<Vec<HirStmt>, SemanticError> {
        stmts.iter().map(|s| self.lower_stmt(s)).collect()
    }

    pub(crate) fn lower_stmt(&mut self, stmt: &AstStmt) -> Result<HirStmt, SemanticError> {
        let span = stmt.span();
        match stmt {
            AstStmt::ExprStmt(e, semicolon_terminated, _) => Ok(HirStmt::ExprStmt(
                self.lower_expr(e)?,
                *semicolon_terminated,
                span,
            )),
            AstStmt::Assign(name, expr, semicolon_terminated, _) => {
                let id = match self.lookup(name) {
                    Some(id) => id,
                    None => self.define(name.clone()),
                };
                let value = self.lower_expr(expr)?;
                if id.0 < self.var_types.len() {
                    self.var_types[id.0] = value.ty.clone();
                }
                Ok(HirStmt::Assign(id, value, *semicolon_terminated, span))
            }
            AstStmt::MultiAssign(names, expr, semicolon_terminated, _) => {
                let ids: Vec<Option<VarId>> = names
                    .iter()
                    .map(|n| {
                        if n == "~" {
                            None
                        } else {
                            Some(match self.lookup(n) {
                                Some(id) => id,
                                None => self.define(n.to_string()),
                            })
                        }
                    })
                    .collect();
                let value = self.lower_expr(expr)?;
                Ok(HirStmt::MultiAssign(
                    ids,
                    value,
                    *semicolon_terminated,
                    span,
                ))
            }
            AstStmt::If {
                cond,
                then_body,
                elseif_blocks,
                else_body,
                ..
            } => {
                let cond = self.lower_expr(cond)?;
                let then_body = self.lower_stmts(then_body)?;
                let mut elseif_vec = Vec::new();
                for (c, b) in elseif_blocks {
                    elseif_vec.push((self.lower_expr(c)?, self.lower_stmts(b)?));
                }
                let else_body = match else_body {
                    Some(b) => Some(self.lower_stmts(b)?),
                    None => None,
                };
                Ok(HirStmt::If {
                    cond,
                    then_body,
                    elseif_blocks: elseif_vec,
                    else_body,
                    span,
                })
            }
            AstStmt::While { cond, body, .. } => Ok(HirStmt::While {
                cond: self.lower_expr(cond)?,
                body: self.lower_stmts(body)?,
                span,
            }),
            AstStmt::For {
                var, expr, body, ..
            } => {
                let id = match self.lookup(var) {
                    Some(id) => id,
                    None => self.define(var.clone()),
                };
                let expr = self.lower_expr(expr)?;
                let body = self.lower_stmts(body)?;
                Ok(HirStmt::For {
                    var: id,
                    expr,
                    body,
                    span,
                })
            }
            AstStmt::Switch {
                expr,
                cases,
                otherwise,
                ..
            } => {
                let control = self.lower_expr(expr)?;
                let mut cases_hir = Vec::new();
                for (v, b) in cases {
                    let ve = self.lower_expr(v)?;
                    let vb = self.lower_stmts(b)?;
                    cases_hir.push((ve, vb));
                }
                let otherwise_hir = otherwise
                    .as_ref()
                    .map(|b| self.lower_stmts(b))
                    .transpose()?;
                Ok(HirStmt::Switch {
                    expr: control,
                    cases: cases_hir,
                    otherwise: otherwise_hir,
                    span,
                })
            }
            AstStmt::TryCatch {
                try_body,
                catch_var,
                catch_body,
                ..
            } => {
                let try_hir = self.lower_stmts(try_body)?;
                let catch_var_id = catch_var.as_ref().map(|name| match self.lookup(name) {
                    Some(id) => id,
                    None => self.define(name.clone()),
                });
                let catch_hir = self.lower_stmts(catch_body)?;
                Ok(HirStmt::TryCatch {
                    try_body: try_hir,
                    catch_var: catch_var_id,
                    catch_body: catch_hir,
                    span,
                })
            }
            AstStmt::Global(names, _) => {
                let pairs: Vec<(VarId, String)> = names
                    .iter()
                    .map(|n| {
                        let id = match self.lookup(n) {
                            Some(id) => id,
                            None => self.define(n.to_string()),
                        };
                        (id, n.clone())
                    })
                    .collect();
                Ok(HirStmt::Global(pairs, span))
            }
            AstStmt::Persistent(names, _) => {
                let pairs: Vec<(VarId, String)> = names
                    .iter()
                    .map(|n| {
                        let id = match self.lookup(n) {
                            Some(id) => id,
                            None => self.define(n.to_string()),
                        };
                        (id, n.clone())
                    })
                    .collect();
                Ok(HirStmt::Persistent(pairs, span))
            }
            AstStmt::Break(_) => Ok(HirStmt::Break(span)),
            AstStmt::Continue(_) => Ok(HirStmt::Continue(span)),
            AstStmt::Return(_) => Ok(HirStmt::Return(span)),
            AstStmt::Function {
                name,
                params,
                outputs,
                body,
                ..
            } => {
                self.push_scope();
                let param_ids: Vec<VarId> = params.iter().map(|p| self.define(p.clone())).collect();
                let output_ids: Vec<VarId> = outputs
                    .iter()
                    .map(|o| {
                        self.lookup_current_scope(o)
                            .unwrap_or_else(|| self.define(o.clone()))
                    })
                    .collect();
                let body_hir = self.lower_stmts(body)?;
                self.pop_scope();

                let has_varargin = params
                    .last()
                    .map(|s| s.as_str() == "varargin")
                    .unwrap_or(false);
                let has_varargout = outputs
                    .last()
                    .map(|s| s.as_str() == "varargout")
                    .unwrap_or(false);

                let func_stmt = HirStmt::Function {
                    name: name.clone(),
                    params: param_ids,
                    outputs: output_ids,
                    body: body_hir,
                    has_varargin,
                    has_varargout,
                    span,
                };

                self.functions.insert(name.clone(), func_stmt.clone());
                Ok(func_stmt)
            }
            AstStmt::ClassDef {
                name,
                super_class,
                members,
                ..
            } => {
                let members_hir = members
                    .iter()
                    .map(|m| match m {
                        parser::ClassMember::Properties { attributes, names } => {
                            HirClassMember::Properties {
                                attributes: attributes.clone(),
                                names: names.clone(),
                            }
                        }
                        parser::ClassMember::Events { attributes, names } => {
                            HirClassMember::Events {
                                attributes: attributes.clone(),
                                names: names.clone(),
                            }
                        }
                        parser::ClassMember::Enumeration { attributes, names } => {
                            HirClassMember::Enumeration {
                                attributes: attributes.clone(),
                                names: names.clone(),
                            }
                        }
                        parser::ClassMember::Arguments { attributes, names } => {
                            HirClassMember::Arguments {
                                attributes: attributes.clone(),
                                names: names.clone(),
                            }
                        }
                        parser::ClassMember::Methods { attributes, body } => {
                            match self.lower_stmts(body) {
                                Ok(s) => HirClassMember::Methods {
                                    attributes: attributes.clone(),
                                    body: s,
                                },
                                Err(_) => HirClassMember::Methods {
                                    attributes: attributes.clone(),
                                    body: Vec::new(),
                                },
                            }
                        }
                    })
                    .collect();
                Ok(HirStmt::ClassDef {
                    name: name.clone(),
                    super_class: super_class.clone(),
                    members: members_hir,
                    span,
                })
            }
            AstStmt::AssignLValue(lv, rhs, suppressed, _) => {
                let hir_lv = self.lower_lvalue(lv)?;
                let value = self.lower_expr(rhs)?;
                if let HirLValue::Var(var_id) = hir_lv {
                    if var_id.0 < self.var_types.len() {
                        self.var_types[var_id.0] = value.ty.clone();
                    }
                    return Ok(HirStmt::Assign(var_id, value, *suppressed, span));
                }
                Ok(HirStmt::AssignLValue(hir_lv, value, *suppressed, span))
            }
            AstStmt::Import { .. } => {
                if let AstStmt::Import { path, wildcard, .. } = stmt {
                    Ok(HirStmt::Import {
                        path: path.clone(),
                        wildcard: *wildcard,
                        span,
                    })
                } else {
                    unreachable!()
                }
            }
        }
    }
}

//! Parameter expressions: `{rval*2}`, `.param gain = {1 + rf/rg}`.
//!
//! Numbers (with SPICE suffixes), parameter names, `+ - * / ^ **`, unary minus, parentheses,
//! comparisons, the ternary `c ? a : b`, and the functions `sqrt exp log ln log10 abs pow pwr
//! min max sin cos tan atan atan2 sinh cosh tanh floor ceil int sgn if limit`.

use super::value::parse_number;
use std::collections::BTreeMap;

pub type Scope = BTreeMap<String, f64>;

pub fn eval(text: &str, scope: &Scope) -> Result<f64, String> {
    let t = text.trim();
    let t = t
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .or_else(|| t.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(t);
    let toks = lex(t)?;
    let mut p = P { t: &toks, i: 0, scope };
    let v = p.ternary()?;
    if p.i != toks.len() {
        return Err(format!("unexpected `{}` in `{t}`", toks[p.i].text()));
    }
    Ok(v)
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum T {
    Num(f64),
    Id(String),
    Op(&'static str),
}

impl T {
    pub(super) fn text(&self) -> String {
        match self {
            T::Num(v) => v.to_string(),
            T::Id(s) => s.clone(),
            T::Op(o) => o.to_string(),
        }
    }
}

pub(super) fn lex(s: &str) -> Result<Vec<T>, String> {
    const OPS: [&str; 19] = [
        "**", "<=", ">=", "==", "!=", "&&", "||", "+", "-", "*", "/", "^", "(", ")", ",", "<", ">", "?",
        ":",
    ];
    let b = s.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()) {
            let mut j = i;
            while j < b.len() {
                let d = b[j] as char;
                let exp_sign = (d == '+' || d == '-') && j > i && matches!(b[j - 1], b'e' | b'E')
                    && b[i..j - 1].iter().all(|x| x.is_ascii_digit() || *x == b'.');
                if d.is_ascii_alphanumeric() || d == '.' || exp_sign {
                    j += 1;
                } else {
                    break;
                }
            }
            let v = parse_number(&s[i..j]).ok_or_else(|| format!("bad number `{}`", &s[i..j]))?;
            out.push(T::Num(v));
            i = j;
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let mut j = i;
            while j < b.len() && ((b[j] as char).is_ascii_alphanumeric() || b[j] == b'_' || b[j] == b'.') {
                j += 1;
            }
            out.push(T::Id(s[i..j].to_ascii_lowercase()));
            i = j;
            continue;
        }
        match OPS.iter().find(|o| s[i..].starts_with(**o)) {
            Some(o) => {
                out.push(T::Op(o));
                i += o.len();
            }
            None => return Err(format!("unexpected `{c}`")),
        }
    }
    Ok(out)
}

struct P<'a> {
    t: &'a [T],
    i: usize,
    scope: &'a Scope,
}

impl P<'_> {
    fn peek_op(&self, o: &str) -> bool {
        matches!(self.t.get(self.i), Some(T::Op(x)) if *x == o)
    }
    fn eat(&mut self, o: &str) -> bool {
        if self.peek_op(o) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn ternary(&mut self) -> Result<f64, String> {
        let c = self.or()?;
        if self.eat("?") {
            let a = self.ternary()?;
            if !self.eat(":") {
                return Err("expected `:`".into());
            }
            let b = self.ternary()?;
            return Ok(if c != 0.0 { a } else { b });
        }
        Ok(c)
    }
    fn or(&mut self) -> Result<f64, String> {
        let mut a = self.and()?;
        while self.eat("||") {
            let b = self.and()?;
            a = ((a != 0.0) || (b != 0.0)) as i32 as f64;
        }
        Ok(a)
    }
    fn and(&mut self) -> Result<f64, String> {
        let mut a = self.cmp()?;
        while self.eat("&&") {
            let b = self.cmp()?;
            a = ((a != 0.0) && (b != 0.0)) as i32 as f64;
        }
        Ok(a)
    }
    fn cmp(&mut self) -> Result<f64, String> {
        let a = self.sum()?;
        for o in ["<=", ">=", "==", "!=", "<", ">"] {
            if self.eat(o) {
                let b = self.sum()?;
                let r = match o {
                    "<=" => a <= b,
                    ">=" => a >= b,
                    "==" => a == b,
                    "!=" => a != b,
                    "<" => a < b,
                    _ => a > b,
                };
                return Ok(r as i32 as f64);
            }
        }
        Ok(a)
    }
    fn sum(&mut self) -> Result<f64, String> {
        let mut a = self.prod()?;
        loop {
            if self.eat("+") {
                a += self.prod()?;
            } else if self.eat("-") {
                a -= self.prod()?;
            } else {
                return Ok(a);
            }
        }
    }
    fn prod(&mut self) -> Result<f64, String> {
        let mut a = self.unary()?;
        loop {
            if self.eat("*") {
                a *= self.unary()?;
            } else if self.eat("/") {
                a /= self.unary()?;
            } else {
                return Ok(a);
            }
        }
    }
    fn unary(&mut self) -> Result<f64, String> {
        if self.eat("-") {
            return Ok(-self.unary()?);
        }
        if self.eat("+") {
            return self.unary();
        }
        self.power()
    }
    fn power(&mut self) -> Result<f64, String> {
        let a = self.atom()?;
        if self.eat("^") || self.eat("**") {
            let b = self.unary()?;
            return Ok(a.powf(b));
        }
        Ok(a)
    }
    fn atom(&mut self) -> Result<f64, String> {
        match self.t.get(self.i).cloned() {
            Some(T::Num(v)) => {
                self.i += 1;
                Ok(v)
            }
            Some(T::Op("(")) => {
                self.i += 1;
                let v = self.ternary()?;
                if !self.eat(")") {
                    return Err("expected `)`".into());
                }
                Ok(v)
            }
            Some(T::Id(name)) => {
                self.i += 1;
                if self.eat("(") {
                    let mut args = vec![];
                    if !self.eat(")") {
                        loop {
                            args.push(self.ternary()?);
                            if self.eat(")") {
                                break;
                            }
                            if !self.eat(",") {
                                return Err("expected `,` or `)`".into());
                            }
                        }
                    }
                    return call(&name, &args);
                }
                match name.as_str() {
                    "pi" => Ok(std::f64::consts::PI),
                    "e" if !self.scope.contains_key("e") => Ok(std::f64::consts::E),
                    _ => self
                        .scope
                        .get(&name)
                        .copied()
                        .ok_or_else(|| format!("unknown parameter `{name}`")),
                }
            }
            Some(t) => Err(format!("unexpected `{}`", t.text())),
            None => Err("unexpected end of expression".into()),
        }
    }
}

fn call(name: &str, a: &[f64]) -> Result<f64, String> {
    let need = |n: usize| {
        if a.len() == n {
            Ok(())
        } else {
            Err(format!("`{name}` takes {n} argument(s)"))
        }
    };
    let one = |f: fn(f64) -> f64| -> Result<f64, String> {
        need(1)?;
        Ok(f(a[0]))
    };
    match name {
        "sqrt" => one(f64::sqrt),
        "exp" => one(f64::exp),
        "log" | "ln" => one(f64::ln),
        "log10" => one(f64::log10),
        "abs" => one(f64::abs),
        "sin" => one(f64::sin),
        "cos" => one(f64::cos),
        "tan" => one(f64::tan),
        "atan" => one(f64::atan),
        "sinh" => one(f64::sinh),
        "cosh" => one(f64::cosh),
        "tanh" => one(f64::tanh),
        "floor" => one(f64::floor),
        "ceil" => one(f64::ceil),
        "int" => one(f64::trunc),
        "sgn" => one(|x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 }),
        "pow" | "pwr" => {
            need(2)?;
            Ok(if name == "pwr" { a[0].abs().powf(a[1]).copysign(a[0]) } else { a[0].powf(a[1]) })
        }
        "atan2" => {
            need(2)?;
            Ok(a[0].atan2(a[1]))
        }
        "min" => {
            need(2)?;
            Ok(a[0].min(a[1]))
        }
        "max" => {
            need(2)?;
            Ok(a[0].max(a[1]))
        }
        "if" => {
            need(3)?;
            Ok(if a[0] != 0.0 { a[1] } else { a[2] })
        }
        "limit" => {
            need(3)?;
            Ok(a[0].clamp(a[1].min(a[2]), a[1].max(a[2])))
        }
        _ => Err(format!("unknown function `{name}`")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expressions() {
        let mut s = Scope::new();
        s.insert("rf".into(), 10e3);
        s.insert("rg".into(), 1e3);
        assert_eq!(eval("{1 + rf/rg}", &s), Ok(11.0));
        assert_eq!(eval("2**3", &s), Ok(8.0));
        assert_eq!(eval("-2^2", &s), Ok(-4.0));
        assert_eq!(eval("1k*2", &s), Ok(2000.0));
        assert_eq!(eval("1e-3*2", &s), Ok(2e-3));
        assert_eq!(eval("max(rf, rg) > 5k ? 1 : 0", &s), Ok(1.0));
        assert!(eval("nope*2", &s).is_err());
    }
}

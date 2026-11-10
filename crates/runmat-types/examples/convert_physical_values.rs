//! Convert documentation verification values through the runtime's unit authority.
use runmat_types::physical::parse_unit_expression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let source = parse_unit_expression(&arguments.next().ok_or("source unit required")?)
        .map_err(|error| error.to_string())?;
    let target = parse_unit_expression(&arguments.next().ok_or("target unit required")?)
        .map_err(|error| error.to_string())?;
    if source.dimension() != target.dimension() {
        return Err("incompatible physical dimensions".into());
    }
    let values = arguments
        .map(|argument| {
            let value: f64 = argument.parse()?;
            Ok(target.from_canonical(source.to_canonical(value)?)?)
        })
        .collect::<Result<Vec<f64>, Box<dyn std::error::Error>>>()?;
    if values.is_empty() {
        return Err("at least one value required".into());
    }
    println!("{}", serde_json::to_string(&values)?);
    Ok(())
}

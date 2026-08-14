pub fn value(value: &runmat_value::Value) {
    println!("{value}");
}

pub fn error(message: &str) {
    eprintln!("RunMat standalone error: {message}");
}

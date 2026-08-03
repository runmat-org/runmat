use tokio::io::{stdin, stdout, BufReader, BufWriter, Stdin, Stdout};

pub fn endpoint() -> (BufReader<Stdin>, BufWriter<Stdout>) {
    (BufReader::new(stdin()), BufWriter::new(stdout()))
}

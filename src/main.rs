use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use clap::Parser;

pub mod graph;
pub mod input_format;
pub mod schedule;
pub mod tiling;
use crate::input_format::InputFormat;

#[derive(Parser)]
struct Cli {
    input_file: PathBuf,
    output_file: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    let file = File::open(&cli.input_file).expect("failed to open input file");
    let reader = BufReader::new(file);
    let input: InputFormat =
        serde_json::from_reader(reader).expect("failed to parse input file as JSON");

    // Temporary smoke test output.
    println!("{:?}", input);
}

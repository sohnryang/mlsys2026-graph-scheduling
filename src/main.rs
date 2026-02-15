use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
struct Cli {
    input_file: PathBuf,
    output_file: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    println!("input file path: {:?}", cli.input_file);
    println!("output file path: {:?}", cli.output_file);
}

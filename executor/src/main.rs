mod py_compare;

use clap::{App, Arg, ArgMatches, SubCommand};
use rust_ncc::experiment_setups::ExperimentType;
use rust_ncc::{experiment_setups, world, DEFAULT_OUTPUT_DIR};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

pub const EXP_CHOICES: [&str; 2] = ["sep-pair", "py-comp"];

fn main() {
    let parsed_args = App::new("rust-ncc executor")
        .version("0.1")
        .about("Execute rust-ncc simulations from the command line.")
        .arg(
            Arg::with_name("exp-setup")
                .short("es")
                .long("exp-setup")
                .help(&format!(
                    "Choose type of experiment to run. Choices are:\
                 {:?}",
                    EXP_CHOICES
                ))
                .required(true),
        )
        .arg(
            Arg::with_name("sep-pair-params")
                .long("sep-pair-params")
                .short("spp")
                .help("Separated pair parameters: sep_dist_in_cell_diams")
                .required_if("exp-setup", "sep-pair"),
        )
        .arg(
            Arg::with_name("cil")
                .long("cil")
                .short("cil")
                .help(&format!(
                    "Magnitude of CIL signal."
                ))
                .required(true)
        )
        .arg(
            Arg::with_name("coa")
                .long("coa")
                .short("coa")
                .help(&format!(
                    "Magnitude of COA signal."
                ))
        )
        .arg(
            Arg::with_name("cal")
                .long("cal")
                .short("cal")
                .help(&format!(
                    "Magnitude of CAL signal."
                ))
        )
        .arg(
            Arg::with_name("adh")
                .long("adh")
                .short("adh")
                .help(&format!(
                    "Magnitude of adhesion."
                ))
        )
        .arg(
            Arg::with_name("seeds")
                .short("s")
                .long("seeds")
                .help(&format!(
                    "Choose type of experiment to run. Choices are:\
                 {:?}",
                    EXP_CHOICES
                ))
                .required(true)
                .multiple(true)
        )
        .get_matches();

    let exp_type_val = parsed_args.value_of("exp-setup").unwrap();
    let exp_type = match exp_type_val {
        "sep-pair" => ExperimentType::SeparatedPair {
            sep_in_cell_diams: parsed_args
                .values_of("sep-pair-params")
                .unwrap()
                .parse::<usize>()
                .unwrap(),
        },
        "py-compare" => ExperimentType::PythonComparison,
        _ => panic!("Unknown experiment given: {}", exp_type_val),
    };
    let (cil, coa, cal, adh, seeds) =
        handle_general_args(&parsed_args);

    for (ix, &seed) in seeds.iter().enumerate() {
        let exp = experiment_setups::generate(
            ix, cil, cal, coa, adh, seed, exp_type,
        );

        let mut w = world::World::new(
            exp,
            Some(PathBuf::from(DEFAULT_OUTPUT_DIR)),
            10,
            1000,
        );

        let now = Instant::now();
        w.simulate(3.0 * 3600.0, true);

        println!(
            "Simulation complete. {} s.",
            now.elapsed().as_secs()
        );

        generate_animation(&exp.file_name)
    }
}

fn handle_general_args(
    parsed_args: &ArgMatches,
) -> (f64, Option<f64>, Option<f64>, Option<f64>, Vec<Option<u64>>) {
    let cil =
        parsed_args.value_of("cil").unwrap().parse::<f64>().unwrap();

    let coa = parsed_args
        .value_of("coa")
        .map(|v| v.parse::<f64>().unwrap());
    let cal = parsed_args
        .value_of("coa")
        .map(|v| v.parse::<f64>().unwrap());
    let adh = parsed_args
        .value_of("coa")
        .map(|v| v.parse::<f64>().unwrap());

    let seeds = {
        let mut seeds: Vec<Option<u64>> = vec![];
        let values = parsed_args.values_of("seeds").unwrap();
        for val in values {
            seeds.push(match val {
                "None" => None,
                _ => Some(val.parse::<u64>().unwrap()),
            });
        }
        seeds
    };

    (cil, coa, cal, adh, seeds)
}

fn generate_animation(file_name: &str) {
    let file_path = PathBuf::from(format!(
        "{}/{}.cbor",
        DEFAULT_OUTPUT_DIR, file_name
    ));

    let output = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(&["/C", "echo hello"])
            .output()
            .expect("failed to execute process");

        Command::new()
    } else {
        Command::new("sh")
            .arg("-c")
            .arg("echo hello")
            .output()
            .expect("failed to execute process")
    };
}

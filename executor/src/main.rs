mod py_compare;

use clap::{App, Arg, ArgMatches};
use rust_ncc::exp_setups::ExperimentType;
use rust_ncc::{exp_setups, world, DEFAULT_OUTPUT_DIR};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

pub const SEP_PAIR: &str = "sep-pair";
pub const PY_COMP: &str = "py-comp";
pub const EXP_CHOICES: [&str; 2] = [SEP_PAIR, PY_COMP];

fn main() {
    let parsed_args = App::new("rust-ncc executor")
        .version("0.1")
        .about("Execute rust-ncc simulations from the command line.")
        .arg(
            Arg::with_name("etype")
                .long("etype")
                .help(&format!(
                    "Choose type of experiment to run. Choices are:\
                 {:?}",
                    EXP_CHOICES
                )).takes_value(true).required(true),
        )
        .arg(
            Arg::with_name("sep-pair-params")
                .long("spp")
                .help("Separated pair parameters: sep_dist_in_cell_diams")
                .required_if("etype", SEP_PAIR).takes_value(true),
        )
        .arg(
            Arg::with_name("cil")
                .long("cil")
                .help(&"Magnitude of CIL signal.".to_string())
                .required(true).takes_value(true)
        )
        .arg(
            Arg::with_name("coa")
                .long("coa")
                .help(&"Magnitude of COA signal.".to_string()).takes_value(true)
        )
        .arg(
            Arg::with_name("cal")
                .long("cal")
                .help(&"Magnitude of CAL signal.".to_string()).takes_value(true)
        )
        .arg(
            Arg::with_name("adh")
                .long("adh")
                .help(&"Magnitude of adhesion.".to_string()).takes_value(true)
        )
        .arg(
            Arg::with_name("seeds")
                .long("seeds")
                .required(true)
                .multiple(true).takes_value(true)
        )
        .arg(
            Arg::with_name("path")
            .long("p")
            .help(&"Output directory".to_string()).takes_value(true)
        )
        .get_matches();

    let exp_type_val = parsed_args.value_of("etype").unwrap();
    let exp_type = match exp_type_val {
        SEP_PAIR => ExperimentType::SeparatedPair {
            sep_in_cell_diams: parsed_args
                .value_of("sep-pair-params")
                .unwrap()
                .parse::<usize>()
                .unwrap(),
        },
        PY_COMP => ExperimentType::PythonComparison,
        _ => panic!("Unknown experiment given: {}", exp_type_val),
    };
    let exp_args = refine_experiment_args(&parsed_args);

    for &exp_args in exp_args.iter() {
        let ExperimentArgs {
            ix,
            cil,
            coa,
            cal,
            adh,
            seed,
            path,
        } = exp_args;
        let exp = exp_setups::generate(
            ix, cil, coa, cal, adh, seed, exp_type,
        );

        let path = match path {
            Some(p) => p,
            None => DEFAULT_OUTPUT_DIR
        };

        let file_name = exp.file_name.clone();

        let mut w = world::World::new(
            exp,
            Some(PathBuf::from(path)),
            10,
            1000,
        );

        let now = Instant::now();
        w.simulate(3.0 * 3600.0, true);

        println!(
            "Simulation complete. {} s.",
            now.elapsed().as_secs()
        );

        generate_animation(&file_name)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct ExperimentArgs<'a> {
    ix: usize,
    cil: f64,
    coa: Option<f64>,
    cal: Option<f64>,
    adh: Option<f64>,
    seed: Option<u64>,
    path: Option<&'a str>,
}

fn refine_experiment_args<'a>(
    parsed_args: &'a ArgMatches,
) -> Vec<ExperimentArgs<'a>> {
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

    let path = parsed_args.value_of("path");

    seeds
        .iter()
        .enumerate()
        .map(|(ix, &seed)| ExperimentArgs {
            ix,
            cil,
            coa,
            cal,
            adh,
            seed,
            path,
        })
        .collect::<Vec<ExperimentArgs>>()
}

fn generate_animation(file_name: &str) {
    let _file_path = PathBuf::from(format!(
        "{}/{}.cbor",
        DEFAULT_OUTPUT_DIR, file_name
    ));

    let output = if cfg!(target_os = "windows") {
        Command::new("cmd")
            .args(&["/K", "echo hello"])
            .output()
            .expect("failed to execute process")
    } else {
        Command::new("sh")
            .arg("-c")
            .arg("echo hello")
            .output()
            .expect("failed to execute process")
    };
    println!("{:?}", output);
}

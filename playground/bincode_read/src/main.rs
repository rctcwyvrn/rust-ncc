use bincode::deserialize_from;
use rust_ncc::world::{hardio, WorldInfo, WorldSnapshot};
use std::borrow::Borrow;
use std::fs::OpenOptions;
use std::io;

fn main() {
    let mut src = OpenOptions::new()
        .read(true)
        .open("../output/history_n_cells.binc")
        .unwrap();

    let _world_info: WorldInfo = deserialize_from(&mut src).unwrap();
    let mut snapshots: Vec<WorldSnapshot> = vec![];

    loop {
        let rd: bincode::Result<Vec<WorldSnapshot>> =
            deserialize_from(&mut src);
        match rd {
            Ok(mut snaps) => snapshots.append(&mut snaps),
            Err(err) => {
                if let bincode::ErrorKind::Io(std_err) = err.borrow()
                {
                    if let io::ErrorKind::UnexpectedEof =
                        std_err.kind()
                    {
                        break;
                    }
                }
                panic!("{}", err);
            }
        }
    }

    println!("num_snapshots: {}", snapshots.len());
}

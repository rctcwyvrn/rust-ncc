// Copyright © 2020 Brian Merchant.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod hardio;
#[cfg(debug_assertions)]
use crate::cell::confirm_volume_exclusion;
use crate::cell::core_state::CoreState;
use crate::cell::ModelCell;
use crate::experiments::{CellGroup, Experiment};
use crate::interactions::{CellInteractions, InteractionGenerator, RgtpState};
use crate::math::v2d::V2d;
use crate::parameters::{Parameters, WorldParameters};
use crate::world::hardio::{save_data, save_schema};
use crate::NVERTS;
use avro_rs::Writer;
use avro_schema_derive::Schematize;
use once_cell::unsync::Lazy;
use rand::prelude::ThreadRng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{thread_rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::PathBuf;

#[derive(Clone, Deserialize, Serialize, Schematize)]
pub struct Cells {
    pub tstep: u32,
    pub cells: Vec<ModelCell>,
    pub interactions: Vec<CellInteractions>,
}

impl Cells {
    fn simulate(
        &self,
        rng: &mut ThreadRng,
        cell_regs: &mut [Option<RandomEventGenerator>],
        world_parameters: &WorldParameters,
        group_parameters: &[Parameters],
        interaction_generator: &mut InteractionGenerator,
    ) -> Cells {
        let mut cells = vec![self.cells[0]; self.cells.len()];
        let shuffled_cells = {
            let mut crs = self.cells.iter().collect::<Vec<&ModelCell>>();
            crs.shuffle(rng);
            crs
        };
        for c in shuffled_cells {
            println!("------------------");
            let ci = c.ix as usize;
            println!("ci: {}", ci);
            println!(
                "contacts: {:?}",
                interaction_generator.get_physical_contacts(ci)
            );
            let contact_polys = interaction_generator.get_physical_contact_polys(ci);
            let new_cell_state = c.simulate_euler(
                self.tstep,
                &self.interactions[ci],
                contact_polys,
                cell_regs[ci].as_mut(),
                world_parameters,
                &group_parameters[c.group_ix as usize],
            );
            println!("----------------------");
            interaction_generator.update(ci, &new_cell_state.state.vertex_coords);
            cells[ci] = new_cell_state;
        }
        for (ci, c) in cells.iter().enumerate() {
            println!("+++++++++++++++++");
            println!("ci: {}", ci);
            println!(
                "contacts: {:?}",
                interaction_generator.get_physical_contacts(ci)
            );
            let vcs = &c.state.vertex_coords;
            let contact_polys = interaction_generator.get_physical_contact_polys(ci);
            #[cfg(debug_assertions)]
            confirm_volume_exclusion(vcs, contact_polys.as_slice(), &format!("world cell {}", ci));
        }
        println!("+++++++++++++++++");
        let cell_interactions = interaction_generator.generate();
        Cells {
            tstep: self.tstep + 1,
            cells,
            interactions: cell_interactions,
        }
    }
}

pub struct RandomEventGenerator {
    pub(crate) rng: SmallRng,
    distrib: Normal<f32>,
}

impl RandomEventGenerator {
    pub fn sample(&mut self) -> f32 {
        self.distrib.sample(&mut self.rng)
    }
}

pub struct World {
    tstep_length: f32,
    global_params: WorldParameters,
    cell_group_params: Vec<Parameters>,
    pub history: Vec<Cells>,
    cell_regs: Vec<Option<RandomEventGenerator>>,
    cell_states: Cells,
    interaction_generator: InteractionGenerator,
    pub rng: ThreadRng,
}

fn gen_vertex_coords(centroid: &V2d, radius: f32) -> [V2d; NVERTS] {
    let mut r = [V2d::default(); NVERTS];
    (0..NVERTS).for_each(|vix| {
        let vf = (vix as f32) / (NVERTS as f32);
        let theta = 2.0 * PI * vf;
        r[vix] = V2d {
            x: centroid.x + theta.cos() * radius,
            y: centroid.y + theta.sin() * radius,
        };
    });
    r
}

impl World {
    pub fn new(experiment: Experiment) -> World {
        let Experiment {
            basic_quants,
            world_parameters,
            cell_groups,
            ..
        } = experiment;
        let group_parameters = cell_groups
            .iter()
            .map(|cg| cg.parameters.clone())
            .collect::<Vec<Parameters>>();

        let mut cell_group_ixs = vec![];
        let mut cell_centroids = vec![];
        cell_groups.iter().enumerate().for_each(|(gix, cg)| {
            cell_group_ixs.append(&mut vec![gix; cg.num_cells as usize]);
            cell_centroids.append(&mut gen_cell_centroids(cg).unwrap())
        });
        let cell_vcs = cell_group_ixs
            .iter()
            .zip(cell_centroids.iter())
            .map(|(&gix, cc)| gen_vertex_coords(cc, group_parameters[gix].cell_r))
            .collect::<Vec<[V2d; NVERTS]>>();
        let cell_states = cell_group_ixs
            .iter()
            .zip(cell_vcs.iter())
            .map(|(&gix, vcs)| {
                let parameters = &group_parameters[gix];
                CoreState::new(*vcs, parameters.init_rac, parameters.init_rho)
            })
            .collect::<Vec<CoreState>>();
        let cell_rgtps = cell_group_ixs
            .iter()
            .zip(cell_states.iter())
            .map(|(&gix, state)| {
                let parameters = &group_parameters[gix];
                state.calc_rgtp_state(parameters)
            })
            .collect::<Vec<[RgtpState; NVERTS]>>();
        let interaction_generator = InteractionGenerator::new(
            &cell_vcs,
            &cell_rgtps,
            world_parameters.interactions.clone(),
        );
        let cell_interactions = interaction_generator.generate();
        let mut cells = vec![];
        let mut rng = thread_rng();
        let mut cell_regs = vec![];
        for (ix, gix) in cell_group_ixs.into_iter().enumerate() {
            let parameters = &group_parameters[gix];
            let normal =
                Lazy::new(|| Normal::new(parameters.rand_avg_t, parameters.rand_std_t).unwrap());
            let mut creg = if parameters.randomization {
                Some(RandomEventGenerator {
                    rng: SmallRng::from_rng(&mut rng).unwrap(),
                    distrib: *normal,
                })
            } else {
                None
            };
            cells.push(ModelCell::new(
                ix as u32,
                gix as u32,
                cell_states[gix],
                &cell_interactions[ix],
                parameters,
                creg.as_mut(),
            ));
            cell_regs.push(creg);
        }
        let state = Cells {
            tstep: 0,
            cells,
            interactions: cell_interactions.clone(),
        };
        let history = vec![state.clone()];
        World {
            tstep_length: basic_quants.time(),
            global_params: world_parameters,
            cell_group_params: group_parameters,
            cell_regs,
            history,
            cell_states: state,
            interaction_generator,
            rng,
        }
    }

    pub fn simulate(&mut self, final_tpoint: f32) {
        let num_tsteps = (final_tpoint / self.tstep_length).ceil() as u32;
        while self.cell_states.tstep < num_tsteps {
            println!("========================================");
            println!("tstep: {}/{}", self.cell_states.tstep, num_tsteps);
            let new_state = self.cell_states.simulate(
                &mut self.rng,
                self.cell_regs.as_mut_slice(),
                &self.global_params,
                &self.cell_group_params,
                &mut self.interaction_generator,
            );
            self.history.push(new_state.clone());
            self.cell_states = new_state;
            println!("========================================")
        }
    }

    pub fn save_history(&self, output_dir: &PathBuf) {
        let name = "history";
        let schema = Cells::schematize(None);
        save_schema(name, &schema, output_dir);
        let mut writer = Writer::new(&schema, Vec::new()); //Writer::with_codec(&self.state_schema, Vec::new(), avro_rs::Codec::Deflate);
        for ws in self.history.iter() {
            writer.append_ser(ws).unwrap();
        }
        let encoded = writer.into_inner().unwrap();
        save_data(name, &encoded, output_dir);
    }
}

fn gen_cell_centroids(cg: &CellGroup) -> Result<Vec<V2d>, String> {
    let CellGroup {
        num_cells,
        layout,
        parameters,
    } = cg;
    let cell_r = parameters.cell_r;
    if layout.width * layout.height >= *num_cells {
        let mut r = vec![];
        let first_cell_centroid = V2d {
            x: layout.bottom_left.x + cell_r,
            y: layout.bottom_left.y + cell_r,
        };
        let delta_x = V2d {
            x: 2.0 * cell_r,
            y: 0.0,
        };
        let delta_y = V2d {
            x: 0.0,
            y: 2.0 * cell_r,
        };
        for ix in 0..*num_cells {
            let row = ix / layout.width;
            let col = ix - layout.width * row;
            r.push(first_cell_centroid + (col as f32) * delta_x + (row as f32) * delta_y);
        }
        Ok(r)
    } else {
        Err(format!(
            "Cell group layout area ({}x{}) not large enough to fit {} cells.",
            layout.width, layout.height, num_cells
        ))
    }
}
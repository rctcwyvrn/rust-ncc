use crate::interactions::dat_sym2d::SymCcDat;
use crate::interactions::dat_sym4d::SymCcVvDat;
use crate::interactions::generate_contacts;
use crate::math::close_to_zero;
use crate::math::geometry::{BBox, CheckIntersectResult, LineSeg2D, Poly};
use crate::math::v2d::V2D;
use crate::parameters::CoaParams;
use crate::utils::{circ_ix_minus, circ_ix_plus};
use crate::NVERTS;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Deserialize, Serialize)]
pub struct VertexPairInfo {
    dist: f32,
    num_intersects: f32,
}

impl VertexPairInfo {
    pub fn infinity() -> VertexPairInfo {
        VertexPairInfo {
            dist: f32::INFINITY,
            num_intersects: f32::INFINITY,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct CoaGenerator {
    dat: SymCcVvDat<VertexPairInfo>,
    contact_bbs: Vec<BBox>,
    contacts: SymCcDat<bool>,
    pub params: CoaParams,
}

/// Suppose we have a line segment `lseg`, which goes from vertex `vi_a` of cell `A`
/// to vertex `vi_b` of cell `B`. `lseg` models a COA interaction between cell `A`
/// and `B` due to interaction between vertex `vi_a` and `vi_b`.
///
/// Let `C` be a cell that is not `A` or `B` (an "other" polygon). This function
/// checks to see if `lseg` intersects `C`. Note that `lseg` can only intersect `C` in
/// the following ways: 1. `lseg` passes through two of the vertices of `C` (weak
/// intersection). 2. `lseg` intersects an edge of `C` at a point that is not also
/// a vertex of `C` (strong intersection).
///
/// We are restricted to these cases because `C` is by assumption a cell which
/// does not contain the endpoints of `lseg`. Note that case `1` is unlikely, but
/// possible especially in the initial if cells have been initialized in a
/// regular lattice.
pub fn check_other_poly_intersect(lseg: &LineSeg2D, poly: &Poly) -> bool {
    if lseg.intersects_bbox(&poly.bbox) {
        for edge in poly.edges.iter() {
            match lseg.check_intersection(edge) {
                CheckIntersectResult::No | CheckIntersectResult::Unknown => {
                    continue;
                }
                _ => {
                    return true;
                }
            }
        }
    }
    false
}

pub fn is_left_py(p0: &V2D, p1: &V2D, p2: &V2D) -> f32 {
    let (p0x, p0y) = (p0.x, p0.y);
    let (p1x, p1y) = (p1.x, p1.y);
    let (p2x, p2y) = (p2.x, p2.y);

    (p1x - p0x) * (p2y - p0y) - (p2x - p0x) * (p1y - p0y)
}

/// Version of `check_other_poly_intersect`/`check_root_poly_intersect`
/// emulating old Python behaviour.
pub fn check_poly_intersect_py(
    a: &V2D,
    b: &V2D,
    normal_to_lseg: &V2D,
    polygon: &Poly,
    ignore_vertex_index: Option<usize>,
) -> bool {
    let num_vertices = polygon.verts.len();

    let nls_x = normal_to_lseg.x;
    let nls_y = normal_to_lseg.y;

    for vi in 0..polygon.verts.len() {
        let next_index = circ_ix_plus(vi, num_vertices);

        if let Some(igv) = ignore_vertex_index {
            if vi == igv || next_index == igv {
                continue;
            }
        }

        let this_coords = polygon.verts[vi];
        let next_coords = polygon.verts[next_index];

        let is_left_a = is_left_py(&this_coords, &next_coords, a);
        let is_left_b = is_left_py(&this_coords, &next_coords, b);

        if (is_left_a < 0.0 && is_left_b < 0.0)
            || (is_left_a > 0.0 && is_left_b > 0.0)
        {
            continue;
        }

        if close_to_zero(is_left_a) || close_to_zero(is_left_b) {
            return true;
        }

        let alpha = a - &this_coords;
        let (alpha_x, alpha_y) = (alpha.x, alpha.y);

        let beta = next_coords - this_coords;
        let (beta_x, beta_y) = (beta.x, beta.y);
        let denominator = beta_x * nls_x + beta_y * nls_y;

        if close_to_zero(denominator) {
            return false;
        }

        let t = (alpha_x * nls_x + alpha_y * nls_y) / denominator;

        if close_to_zero(t) || (t > 0.0 && t < 1.0) {
            return true;
        } else {
            continue;
        }
    }

    false
}

fn is_given_vector_between_others_py(x: &V2D, alpha: &V2D, beta: &V2D) -> bool {
    let cp1 = alpha.x * x.y - alpha.y * x.x;
    let cp2 = x.x * beta.y - x.y * beta.x;

    if close_to_zero(cp1) || close_to_zero(cp2) {
        true
    } else {
        cp1 > 0.0 && cp2 > 0.0
    }
}

pub fn check_root_poly_intersect_py(
    start_coord: &V2D,
    end_coord: &V2D,
    polygon_coords: &[V2D; NVERTS],
    start_coord_ix: usize,
) -> bool {
    let si_minus1 = circ_ix_minus(start_coord_ix, NVERTS);
    let si_plus1 = circ_ix_plus(start_coord_ix, NVERTS);

    let si_plus1_coord = polygon_coords[si_plus1];
    let si_minus1_coord = polygon_coords[si_minus1];
    let edge_vector_to_plus = &si_plus1_coord - start_coord;

    let edge_vector_from_minus = start_coord - &si_minus1_coord;

    let rough_tangent_vector = edge_vector_to_plus + edge_vector_from_minus;

    let ipv = rough_tangent_vector.normal();

    let v = end_coord - start_coord;

    if is_given_vector_between_others_py(
        &v,
        &ipv,
        &(-1.0 * edge_vector_from_minus),
    ) {
        return true;
    }

    if is_given_vector_between_others_py(&v, &edge_vector_to_plus, &ipv) {
        return true;
    }

    false
}

/// Suppose we have a line segment `lseg`, which goes from vertex `vi_a` of cell `A`
/// to vertex `vi_b` of cell `B`. `lseg` models a COA interaction between cell `A`
/// and `B` due to interaction between vertex `vi_a` and `vi_b`.
///
/// It could be that that `lseg` one of `A` or `B`, the "root" polygons of `lseg`.
/// This function checks if this has occurred, but ignores intersections
/// involving the source/destination vertices of `lseg`.
pub fn check_root_poly_intersect(
    lseg: &LineSeg2D,
    poly_a: &Poly,
    poly_b: &Poly,
    vi_a: usize,
    vi_b: usize,
) -> bool {
    let (ignore_mi, ignore_i) = (circ_ix_minus(vi_a, NVERTS), vi_a);
    for (ei, edge) in poly_a.edges.iter().enumerate() {
        match lseg.check_intersection(edge) {
            CheckIntersectResult::No | CheckIntersectResult::Unknown => {
                continue;
            }
            CheckIntersectResult::Strong
            | CheckIntersectResult::Self1OnOther0
            | CheckIntersectResult::Self1OnOther1 => {
                return true;
            }
            CheckIntersectResult::Self0OnOther0 => {
                if ei == ignore_i {
                    continue;
                } else {
                    return true;
                }
            }
            CheckIntersectResult::Self0OnOther1 => {
                if ei == ignore_mi {
                    continue;
                } else {
                    return true;
                }
            }
        }
    }

    let (ignore_mi, ignore_i) = (circ_ix_minus(vi_b, NVERTS), vi_b);
    for (ei, edge) in poly_b.edges.iter().enumerate() {
        match lseg.check_intersection(edge) {
            CheckIntersectResult::No | CheckIntersectResult::Unknown => {
                continue;
            }
            CheckIntersectResult::Strong
            | CheckIntersectResult::Self0OnOther0
            | CheckIntersectResult::Self0OnOther1 => {
                return true;
            }
            CheckIntersectResult::Self1OnOther0 => {
                if ei == ignore_i {
                    continue;
                } else {
                    return true;
                }
            }
            CheckIntersectResult::Self1OnOther1 => {
                if ei == ignore_mi {
                    continue;
                } else {
                    return true;
                }
            }
        }
    }
    false
}

/// Calculate clearance and distance.
pub fn calc_pair_info(
    ci: usize,
    skip_vi: usize,
    oci: usize,
    skip_ovi: usize,
    lseg: LineSeg2D,
    cell_polys: &[Poly],
) -> VertexPairInfo {
    if check_root_poly_intersect(
        &lseg,
        &cell_polys[ci],
        &cell_polys[oci],
        skip_vi,
        skip_ovi,
    ) {
        return VertexPairInfo {
            dist: lseg.len,
            num_intersects: f32::INFINITY,
        };
    }
    let clearance = cell_polys
        .iter()
        .enumerate()
        .map(|(pi, poly)| {
            if pi != ci && pi != oci && check_other_poly_intersect(&lseg, poly)
            {
                1.0
            } else {
                0.0
            }
        })
        .sum::<f32>();
    VertexPairInfo {
        dist: lseg.len,
        num_intersects: clearance,
    }
}

/// Calculate clearance and distance, trying to emulate old Python behaviour
/// for line segmet vs. poly intersection.
pub fn calc_pair_info_py(
    ci: usize,
    skip_vi: usize,
    oci: usize,
    skip_ovi: usize,
    lseg: LineSeg2D,
    cell_polys: &[Poly],
) -> VertexPairInfo {
    let poly_a = cell_polys[ci];
    let poly_b = cell_polys[oci];
    let coords_a = lseg.p0;
    let coords_b = lseg.p1;
    let normal_to_lseg = lseg.vector.normal();

    if ci == 0 && oci == 1 && skip_vi == 1 && skip_ovi == 6 {
        let x = 1 + 2;
    }

    let check_a = check_root_poly_intersect_py(
        &coords_a,
        &coords_b,
        &poly_a.verts,
        skip_vi,
    );
    let check_b = check_root_poly_intersect_py(
        &coords_b,
        &coords_a,
        &poly_b.verts,
        skip_ovi,
    );
    let mut num_intersects = 0.0;
    if check_a || check_b {
        VertexPairInfo {
            dist: lseg.len,
            num_intersects: 1e5,
        }
    } else {
        for (pi, poly) in cell_polys.iter().enumerate() {
            if pi == ci || pi == oci {
                continue;
            } else if lseg.intersects_bbox(&poly.bbox)
                && check_poly_intersect_py(
                    &coords_a,
                    &coords_b,
                    &normal_to_lseg,
                    &poly,
                    None,
                )
            {
                num_intersects += 1.0;
            }
        }
        VertexPairInfo {
            dist: lseg.len,
            num_intersects,
        }
    }
}

impl CoaGenerator {
    /// Calculates a matrix storing whether two vertices have clear line of sight if in contact range.
    pub fn new(cell_polys: &[Poly], params: CoaParams) -> CoaGenerator {
        let num_cells = cell_polys.len();
        let contact_bbs = cell_polys
            .iter()
            .map(|cp| cp.bbox.expand_by(params.range))
            .collect::<Vec<BBox>>();
        let contacts = generate_contacts(&contact_bbs);
        let mut dat = SymCcVvDat::empty(num_cells, VertexPairInfo::infinity());

        for (ci, poly) in cell_polys.iter().enumerate() {
            for (ocj, opoly) in cell_polys[(ci + 1)..].iter().enumerate() {
                let oci = (ci + 1) + ocj;
                for (vi, v) in poly.verts.iter().enumerate() {
                    for (ovi, ov) in opoly.verts.iter().enumerate() {
                        let lseg = LineSeg2D::new(v, ov);
                        dat.set(
                            ci,
                            vi,
                            oci,
                            ovi,
                            calc_pair_info_py(
                                ci, vi, oci, ovi, lseg, cell_polys,
                            ),
                        )
                    }
                }
            }
        }
        CoaGenerator {
            dat,
            contact_bbs,
            contacts,
            params,
        }
    }

    pub fn update(&mut self, ci: usize, cell_polys: &[Poly]) {
        let this_poly = cell_polys[ci];
        let bb = this_poly.bbox.expand_by(self.params.range);
        self.contact_bbs[ci] = bb;
        // Update contacts.
        for (oci, obb) in self.contact_bbs.iter().enumerate() {
            if oci != ci {
                self.contacts.set(ci, oci, obb.intersects(&bb))
            }
        }
        for (oci, other_poly) in cell_polys.iter().enumerate() {
            if oci == ci || !self.contacts.get(ci, oci) {
                continue;
            }
            for (vi, v) in this_poly.verts.iter().enumerate() {
                for (ovi, ov) in other_poly.verts.iter().enumerate() {
                    let lseg = LineSeg2D::new(v, ov);
                    self.dat.set(
                        ci,
                        vi,
                        oci,
                        ovi,
                        calc_pair_info_py(ci, vi, oci, ovi, lseg, cell_polys),
                    );
                }
            }
        }
    }

    pub fn generate(&self) -> Vec<[f32; NVERTS]> {
        let num_cells = self.contacts.num_cells;
        let mut all_x_coas = vec![[0.0f32; NVERTS]; num_cells];
        let CoaParams {
            los_penalty,
            mag,
            distrib_exp,
            ..
        } = self.params;
        for (ci, x_coas) in all_x_coas.iter_mut().enumerate() {
            for (vi, x_coa) in x_coas.iter_mut().enumerate() {
                for oci in 0..num_cells {
                    if oci != ci {
                        for ovi in 0..NVERTS {
                            let VertexPairInfo {
                                dist,
                                num_intersects,
                            } = self.dat.get(ci, vi, oci, ovi);
                            // println!("====================");
                            // println!(
                            //     "(ci: {}, vi: {}, ovi: {}, oci: {}):",
                            //     ci, vi, oci, ovi
                            // );
                            let los_factor =
                                1.0 / (num_intersects + 1.0).powf(los_penalty);
                            // println!("dist: {}", dist);
                            // println!("num_intersects: {}", num_intersects);
                            // println!("los_factor: {}", los_factor);
                            let coa_signal =
                                (distrib_exp * dist).exp() * los_factor;
                            let additional_signal = mag * coa_signal;
                            let old_coa = *x_coa;
                            *x_coa += additional_signal;

                            // println!("coa_signal: {}", coa_signal);
                            // println!("coa_mag: {}", mag);
                            //
                            // println!(
                            //     "new = {} + {}",
                            //     old_coa, additional_signal
                            // );
                        }
                    }
                }
            }
        }
        let mut max_coa = 0.0;
        for x_coas in all_x_coas.iter() {
            for &x_coa in x_coas.iter() {
                if x_coa > max_coa {
                    max_coa = x_coa;
                }
            }
        }
        //println!("{}", max_coa);
        all_x_coas
    }
}

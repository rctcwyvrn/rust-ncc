use crate::interactions::dat_4d::CvCvDat;
use crate::interactions::dat_sym2d::SymCcDat;
use crate::interactions::{generate_contacts, RelativeRgtpActivity};
use crate::math::geometry::{BBox, LineSeg2D, Poly};
use crate::math::v2d::V2D;
use crate::math::{
    capped_linear_fn, close_to_zero, in_unit_interval, InUnitInterval,
};
use crate::parameters::{CloseBounds, PhysicalContactParams};
use crate::utils::circ_ix_plus;
use crate::NVERTS;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy)]
pub struct Dist(f32);
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct LineSegParam(f32);

#[derive(Clone, Copy, Serialize, Deserialize, Debug, PartialEq)]
pub enum ClosePoint {
    Vertex {
        lseg: LineSeg2D,
        smooth_factor: f32,
    },
    OnEdge {
        edge_point_param: f32,
        lseg: LineSeg2D,
        smooth_factor: f32,
    },
    None,
}

impl Default for ClosePoint {
    fn default() -> Self {
        ClosePoint::None
    }
}

impl ClosePoint {
    /// Returns `ClosePoint` on line segment `k = (b - a)*t + a, 0 <= t < 1`,
    /// which is the point closest to the `focus` point.
    pub fn calc(range: CloseBounds, focus: V2D, a: V2D, b: V2D) -> ClosePoint {
        // Is `p` close to `a`? Then it interacts directly with `a`.
        let focus_to_a = LineSeg2D::new(&focus, &a);
        if focus_to_a.len < range.zero_at {
            let smooth_factor =
                capped_linear_fn(focus_to_a.len, range.zero_at, range.one_at);
            return ClosePoint::Vertex {
                lseg: focus_to_a,
                smooth_factor,
            };
        }

        let focus_to_b = LineSeg2D::new(&focus, &b);
        if focus_to_b.len < range.zero_at {
            ClosePoint::None
        } else {
            let a_to_b = b - a;
            let a_to_p = focus_to_a.vector.scale(-1.0);
            let t = a_to_b.dot(&a_to_p) / a_to_b.mag_squared();
            // Is `t` in the interval `[0, 1)`? If yes, then the close
            // point lies on the edge.
            match in_unit_interval(t) {
                InUnitInterval::Zero | InUnitInterval::In => {
                    let c = t * (a_to_b) + a;
                    let focus_to_c = LineSeg2D::new(&focus, &c);
                    if focus_to_c.len < range.zero_at {
                        ClosePoint::OnEdge {
                            edge_point_param: t,
                            lseg: focus_to_c,
                            smooth_factor: capped_linear_fn(
                                focus_to_c.len,
                                range.zero_at,
                                range.one_at,
                            ),
                        }
                    } else {
                        ClosePoint::None
                    }
                }
                _ => ClosePoint::None,
            }
        }
    }
}

impl fmt::Display for ClosePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClosePoint::Vertex {
                lseg,
                smooth_factor,
            } => {
                write!(
                    f,
                    "Vertex(vector_to: {}, smooth_factor: {})",
                    lseg.len, smooth_factor
                )
            }
            ClosePoint::OnEdge {
                edge_point_param,
                lseg: vector_to,
                smooth_factor,
            } => {
                write!(f, "OnEdge(edge_point_param: {}, vector_to: {}, smooth_factor: {})", edge_point_param, vector_to.mag(), smooth_factor)
            }
            ClosePoint::None => write!(f, "None"),
        }
    }
}

/// Generates CIL/CAL/adhesion related interaction information. These
/// are the interactions that require cells to engage in
/// physical contact.
#[derive(Clone, Deserialize, Serialize, PartialEq, Default, Debug)]
pub struct PhysicalContactGenerator {
    dat: CvCvDat<ClosePoint>,
    pub contact_bbs: Vec<BBox>,
    pub contacts: SymCcDat<bool>,
    pub params: PhysicalContactParams,
}

pub struct PhysContactFactors {
    pub adh: Vec<[V2D; NVERTS]>,
    pub cil: Vec<[f32; NVERTS]>,
    pub cal: Vec<[f32; NVERTS]>,
}

impl PhysicalContactGenerator {
    /// Calculate distances between vertices of cells in contact.
    pub fn new(
        cell_polys: &[Poly],
        params: PhysicalContactParams,
    ) -> PhysicalContactGenerator {
        let num_cells = cell_polys.len();
        let mut dat = CvCvDat::empty(num_cells, ClosePoint::None);
        let contact_bbs = cell_polys
            .iter()
            .map(|cp| cp.bbox.expand_by(params.range.zero_at))
            .collect::<Vec<BBox>>();
        let contacts = generate_contacts(&contact_bbs);
        for (ai, poly) in cell_polys.iter().enumerate() {
            for (bi, other) in cell_polys.iter().enumerate() {
                if ai != bi && contacts.get(ai, bi) {
                    for (avi, p) in poly.verts.iter().enumerate() {
                        for (bvi, a) in other.verts.iter().enumerate() {
                            let b = &other.verts[circ_ix_plus(bvi, NVERTS)];
                            dat.set(
                                ai,
                                avi,
                                bi,
                                bvi,
                                ClosePoint::calc(params.range, *p, *a, *b),
                            )
                        }
                    }
                }
            }
        }
        PhysicalContactGenerator {
            dat,
            contact_bbs,
            contacts,
            params,
        }
    }

    /// Get edges containing points on cell `oci` which are close to vertex `vi` on cell `ci`.
    pub fn get_close_edges_on_cell(
        &self,
        ci: usize,
        vi: usize,
        oci: usize,
        rel_rgtps_per_cell: &[[RelativeRgtpActivity; NVERTS]],
    ) -> Vec<CloseEdge> {
        let v_rgtp = rel_rgtps_per_cell[ci][vi];
        (0..NVERTS)
            .filter_map(|ovi| match self.dat.get(ci, vi, oci, ovi) {
                ClosePoint::None => None,
                ClosePoint::OnEdge {
                    lseg,
                    smooth_factor,
                    edge_point_param,
                } => {
                    //TODO: confirm that we don't want:
                    // let edge_rgtp = (1.0 - edge_point_param) * cell_rgtps[oci][ovi]
                    //     + edge_point_param * cell_rgtps[oci]
                    //         [circ_ix_plus(ovi, NVERTS)];
                    let edge_rgtp =
                        RelativeRgtpActivity::mix_rel_rgtp_act_across_edge(
                            rel_rgtps_per_cell[oci][ovi],
                            rel_rgtps_per_cell[oci][circ_ix_plus(ovi, NVERTS)],
                            edge_point_param,
                        );
                    Some(CloseEdge {
                        cell_ix: oci,
                        vert_ix: ovi,
                        crl: CrlEffect::calc_crl_on_focus(v_rgtp, edge_rgtp),
                        lseg,
                        edge_point_param,
                        smooth_factor,
                    })
                }
                ClosePoint::Vertex {
                    lseg,
                    smooth_factor,
                } => Some(CloseEdge {
                    cell_ix: oci,
                    vert_ix: ovi,
                    crl: CrlEffect::calc_crl_on_focus(
                        v_rgtp,
                        rel_rgtps_per_cell[oci][ovi],
                    ),
                    lseg,
                    edge_point_param: 0.0,
                    smooth_factor,
                }),
            })
            .collect::<Vec<CloseEdge>>()
    }

    /// Get edges which contain points close to vertex `vi` on cell `ci`.
    pub fn get_close_edges_to(
        &self,
        ci: usize,
        vi: usize,
        cell_rgtps: &[[RelativeRgtpActivity; NVERTS]],
    ) -> Vec<CloseEdge> {
        let mut r = vec![];
        for oci in 0..self.dat.num_cells {
            r.append(&mut self.get_close_edges_on_cell(ci, vi, oci, cell_rgtps))
        }
        r
    }

    pub fn update(&mut self, ci: usize, cell_polys: &[Poly]) {
        let poly = cell_polys[ci];
        let bb = cell_polys[ci].bbox.expand_by(self.params.range.zero_at);
        self.contact_bbs[ci] = bb;
        for (oci, obb) in self.contact_bbs.iter().enumerate() {
            if oci != ci {
                self.contacts.set(ci, oci, obb.intersects(&bb));
            }
        }
        for (oci, other) in cell_polys.iter().enumerate() {
            if ci != oci && self.contacts.get(ci, oci) {
                for (pi, p) in poly.verts.iter().enumerate() {
                    for (ai, a) in other.verts.iter().enumerate() {
                        let bi = circ_ix_plus(ai, NVERTS);
                        let b = &other.verts[bi];
                        self.dat.set(
                            ci,
                            pi,
                            oci,
                            ai,
                            ClosePoint::calc(self.params.range, *p, *a, *b),
                        );
                    }
                }
            }
        }
    }

    pub fn generate(
        &self,
        rel_rgtps_per_cell: &[[RelativeRgtpActivity; NVERTS]],
    ) -> PhysContactFactors {
        let num_cells = self.contacts.num_cells;
        let mut adh_per_cell = vec![[V2D::default(); NVERTS]; num_cells];
        let mut cal_per_cell = vec![[0.0f32; NVERTS]; num_cells];
        let mut cil_per_cell = vec![[0.0f32; NVERTS]; num_cells];
        for ci in 0..num_cells {
            let x_cals = &mut cal_per_cell[ci];
            let x_cils = &mut cil_per_cell[ci];
            for vi in 0..NVERTS {
                for CloseEdge {
                    cell_ix: oci,
                    vert_ix: ovi,
                    crl,
                    lseg,
                    edge_point_param,
                    smooth_factor,
                } in self
                    .get_close_edges_to(ci, vi, rel_rgtps_per_cell)
                    .into_iter()
                {
                    match (self.params.cal_mag, crl) {
                        (Some(cal_mag), CrlEffect::Cal) => {
                            x_cals[vi] = smooth_factor * cal_mag;
                        }
                        (Some(_), CrlEffect::Cil) | (None, _) => {
                            x_cils[vi] = smooth_factor * self.params.cil_mag;
                        }
                    }

                    if let Some(adh_mag) = self.params.adh_mag {
                        let x = -1.0
                            * (lseg.len / self.params.range.one_at)
                            * smooth_factor;
                        let adh_force = -1.0 * adh_mag * x * lseg.vector;
                        //* ((1.0 / self.params.range) * delta);
                        // We are close to the vertex.
                        if close_to_zero(edge_point_param) {
                            adh_per_cell[oci][ovi] =
                                adh_per_cell[oci][ovi] - adh_force;
                            adh_per_cell[ci][vi] =
                                adh_per_cell[ci][vi] + adh_force;
                        } else {
                            adh_per_cell[oci][ovi] = adh_per_cell[oci][ovi]
                                - (1.0 - edge_point_param) * adh_force;
                            let owi = circ_ix_plus(ovi, NVERTS);
                            adh_per_cell[oci][owi] = adh_per_cell[oci][owi]
                                - edge_point_param * adh_force;
                            adh_per_cell[ci][vi] =
                                adh_per_cell[ci][vi] + adh_force;
                        }
                    };
                }
            }
        }
        PhysContactFactors {
            adh: adh_per_cell,
            cil: cil_per_cell,
            cal: cal_per_cell,
        }
    }
}

pub enum CrlEffect {
    Cil,
    Cal,
}

impl CrlEffect {
    //TODO: should CIL/CAL be modelled with a "relative strength"?
    pub fn calc_crl_on_focus(
        focus_vertex: RelativeRgtpActivity,
        other: RelativeRgtpActivity,
    ) -> CrlEffect {
        use RelativeRgtpActivity::{RacDominant, RhoDominant};
        match (focus_vertex, other) {
            (RacDominant(_), RhoDominant(_)) => CrlEffect::Cal,
            (RhoDominant(_), RacDominant(_))
            | (RhoDominant(_), RhoDominant(_))
            | (RacDominant(_), RacDominant(_)) => CrlEffect::Cil,
        }
    }
}

/// Information relevant to calculating CIL and adhesion due to an edge in proximity to the focus
/// vertex.
pub struct CloseEdge {
    /// Cell containing edge close to focus vertex.
    pub cell_ix: usize,
    /// Close edge runs from `vert_ix` to `vert_ix + 1`.
    pub vert_ix: usize,
    /// Contact regulation of motion.
    pub crl: CrlEffect,
    /// Let the position of the focus vertex be denoted `p`, and
    /// the point on the close edge closest to the focus vertex
    /// be denoted `c`. `lseg` is the line segment from `p` to `c`.
    pub lseg: LineSeg2D,
    /// Let the position of `vert_ix` be `p0`, and the position of `vert_ix + 1` be `p1`. Let `p`
    /// be the point on the close edge closest to the focus vertex. Then, `t` is such that
    /// `(p1 - p0)*t + p0 = p`.
    pub edge_point_param: f32,
    pub smooth_factor: f32,
}

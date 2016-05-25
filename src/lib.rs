///! The Paeth decomposition of a N-dimensional rotation is an equivalent 
///! composition of N one-dimensional shear operations.  Paeth decompositions
///! come from the paper "A Fast Algorithm for General Raster Rotation" by 
///! Alan W. Paeth in 1986 (DOI 10.1016/B978-0-08-050753-8.50046-2)
///!
///! This crate implements Paeth decompositions for the 2d and 3d rotation
///! objects in the `nalgebra` package.

extern crate nalgebra as na;
use na::{BaseFloat, Rotation2, Matrix2};

/// Two-dimensional Paeth rotation
///
/// Decomposes a given rotation into an Y-shear followed by and X-shear:
/// `R = X*Y`.  The parameters of the shears are stored in a compact
/// form in the `PaethRotation2` struct:
///
/// ```ignore
/// X = [ gamma eta
///         0    1   ]
///
/// Y = [ 0      1
///       alpha beta ]
/// ```
///
/// The individual shear matrices can be computed with `shear_x()` and
/// `shear_y().
pub struct PaethRotation2<N>
where N: BaseFloat + Clone {
    pub alpha: N,
    pub beta: N,
    pub gamma: N,
    pub eta: N,
}

impl<N> PaethRotation2<N>
where N: BaseFloat + Clone {
    pub fn new(rot2: &Rotation2<N>) -> Self {
        let m = rot2.submatrix();
        PaethRotation2{
            alpha: m.m21,
            beta: m.m22,
            gamma: m.m11 - m.m12*m.m21/m.m22,
            eta: m.m12 / m.m22
        }
    }

    pub fn shear_y(self: &Self) -> Matrix2<N> {
        Matrix2{
            m11: N::one(),
            m12: N::zero(),
            m21: self.alpha,
            m22: self.beta,
        }
    }

    pub fn shear_x(self: &Self) -> Matrix2<N> {
        Matrix2{
            m11: self.gamma,
            m12: self.eta,
            m21: N::zero(),
            m22: N::one(),
        }
    }
}

/// Three-dimensional Paeth rotation
///
/// Decomposes a given rotation into a compositon of three one-dimensional
/// shears: `R = X*Y*Z`.  The parameters for the shears are stored in a 
/// compact form in the `PathRotation3` struct:
///
/// ```ignore
///
/// X = [  xx   xy   xz
///         0    1    0 
///         0    0    1]
///
/// Y = [   1    0    0
///        yx   yy   yz
///         0    0    1 ]
///
/// Z = [   1    0    0
///         0    1    0 
///        zx   zy   zz ]
///
/// ```
///
/// The individual shear matrices can be computed with `shear_x()`, `shear_y()`
/// and `shear_z()`.
pub struct PaethRotation3<N>
where N: BaseFloat + Clone {
    pub xx: N,
    pub xy: N,
    pub xz: N,
    pub yx: N,
    pub yy: N,
    pub yz: N,
    pub zx: N,
    pub zy: N,
    pub zz: N,
}

impl<N> PaethRotation3<N> 
where N: BaseFloat + Clone {
    pub fn new(rot3: &Rotation3<N>) -> Self {
        let m = rot3.submatrix();
    }

    pub fn shear_x(self: &Self) -> Matrix3<N> {
    }

    pub fn shear_y(self: &Self) -> Matrix3<N> {
    }

    pub fn shear_z(self: &Self) -> Matrix3<N> {
    }
}

#[test]
fn test_rot2() {
    use na::Vector1;

    let r = Rotation2::new(Vector1::new(32f32));
    let rm = r.submatrix();
    let p = PaethRotation2::new(&r);
    let px = p.shear_x();
    let py = p.shear_y();
    let rr = px * py;

    assert_eq!(rr.m11, rm.m11);
    assert_eq!(rr.m12, rm.m12);
    assert_eq!(rr.m21, rm.m21);
    assert_eq!(rr.m22, rm.m22);
}


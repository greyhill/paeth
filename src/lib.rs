///! The Paeth decomposition of a N-dimensional rotation is an equivalent 
///! composition of N one-dimensional shear operations.  Paeth decompositions
///! come from the paper "A Fast Algorithm for General Raster Rotation" by 
///! Alan W. Paeth in 1986 (DOI 10.1016/B978-0-08-050753-8.50046-2)
///!
///! This crate implements Paeth decompositions for the 2d and 3d rotation
///! objects in the `nalgebra` package.

#[macro_use]
extern crate nalgebra as na;
use na::{BaseFloat, Rotation2, Matrix2, Rotation3, Matrix3, inverse, ApproxEq};

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::*;

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
        Self::new_from_matrix(rot2.submatrix())
    }

    pub fn new_from_matrix(m: &Matrix2<N>) -> Self {
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

fn shear_x_from_entries<N: BaseFloat + Clone>(xx: N, xy: N, xz: N) -> Matrix3<N> {
    Matrix3{
        m11: xx,
        m12: xy,
        m13: xz,

        m21: N::zero(),
        m22: N::one(),
        m23: N::zero(),

        m31: N::zero(),
        m32: N::zero(),
        m33: N::one(),
    }
}

fn shear_y_from_entries<N: BaseFloat + Clone>(yx: N, yy: N, yz: N) -> Matrix3<N> {
    Matrix3{
        m11: N::one(),
        m12: N::zero(),
        m13: N::zero(),

        m21: yx,
        m22: yy,
        m23: yz,

        m31: N::zero(),
        m32: N::zero(),
        m33: N::one(),
    }
}

fn shear_z_from_entries<N: BaseFloat + Clone>(zx: N, zy: N, zz: N) -> Matrix3<N> {
    Matrix3{
        m11: N::one(),
        m12: N::zero(),
        m13: N::zero(),

        m21: N::zero(),
        m22: N::one(),
        m23: N::zero(),

        m31: zx,
        m32: zy,
        m33: zz,
    }
}

impl<N> PaethRotation3<N> 
where N: BaseFloat + Clone + ApproxEq<N> {
    pub fn new(rot3: &Rotation3<N>) -> Self {
        Self::new_from_matrix(rot3.submatrix())
    }

    pub fn new_from_matrix(m: &Matrix3<N>) -> Self {
        // one could write explicit expressions for the factorization,
        // but for the sake of simplicity here, we successively
        // "peel off" dimensions of the rotation 

        let zx = m.m31;
        let zy = m.m32;
        let zz = m.m33;

        let shear_z = shear_z_from_entries(zx.clone(), zy.clone(), zz.clone());
        let m_less_z = *m * inverse(&shear_z).expect("shear_z was not invertible?");
        
        let yx = m_less_z.m21;
        let yy = m_less_z.m22;
        let yz = m_less_z.m23;

        let shear_y = shear_y_from_entries(yx.clone(), yy.clone(), yz.clone());
        let m_less_yz = m_less_z * inverse(&shear_y).expect("shear_y not invertible?");

        let xx = m_less_yz.m11;
        let xy = m_less_yz.m12;
        let xz = m_less_yz.m13;

        PaethRotation3{
            xx: xx,
            xy: xy,
            xz: xz,

            yx: yx,
            yy: yy,
            yz: yz,

            zx: zx,
            zy: zy,
            zz: zz,
        }
    }

    pub fn shear_x(self: &Self) -> Matrix3<N> {
        shear_x_from_entries(self.xx.clone(), self.xy.clone(), self.xz.clone())
    }

    pub fn shear_y(self: &Self) -> Matrix3<N> {
        shear_y_from_entries(self.yx.clone(), self.yy.clone(), self.yz.clone())
    }

    pub fn shear_z(self: &Self) -> Matrix3<N> {
        shear_z_from_entries(self.zx.clone(), self.zy.clone(), self.zz.clone())
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

    assert_approx_eq!(rr.m11, rm.m11);
    assert_approx_eq!(rr.m12, rm.m12);
    assert_approx_eq!(rr.m21, rm.m21);
    assert_approx_eq!(rr.m22, rm.m22);
}

#[test]
fn test_rot3() {
    use na::Vector3;

    let r = Rotation3::new(Vector3::new(2f32, 5f32, -3f32));
    let rm = r.submatrix();
    let p = PaethRotation3::new(&r);
    let px = p.shear_x();
    let py = p.shear_y();
    let pz = p.shear_z();
    let rr = px * py * pz;

    assert_approx_eq!(rr.m11, rm.m11);
    assert_approx_eq!(rr.m12, rm.m12);
    assert_approx_eq!(rr.m13, rm.m13);

    assert_approx_eq!(rr.m21, rm.m21);
    assert_approx_eq!(rr.m22, rm.m22);
    assert_approx_eq!(rr.m23, rm.m23);

    assert_approx_eq!(rr.m31, rm.m31);
    assert_approx_eq!(rr.m32, rm.m32);
    assert_approx_eq!(rr.m33, rm.m33);
}


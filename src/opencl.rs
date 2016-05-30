extern crate proust;
extern crate nalgebra as na;
extern crate num;
use self::proust::*;
use std::marker::*;
use na::{BaseFloat, Absolute};
use ::PaethRotation2;
use std::mem::size_of;
use self::num::FromPrimitive;

fn fmin<N: BaseFloat + Clone>(x: N, y: N) -> N {
    if x < y {
        x
    } else {
        y
    }
}

pub trait ClRotate {
    fn rotate2_source() -> &'static str;
    fn rotate3_source() -> &'static str;
}

impl ClRotate for f32 {
    fn rotate2_source() -> &'static str {
        include_str!("../opencl/rotate2_f32.opencl")
    }

    fn rotate3_source() -> &'static str {
        include_str!("../opencl/rotate3_f32.opencl")
    }
}

pub struct ClRotator2<N>
where N: BaseFloat + Clone + ClRotate + Absolute<N> {
    nx: usize,
    ny: usize,
    wx: N,
    wy: N,
    queue: CommandQueue,
    shear_x: Kernel,
    shear_y: Kernel,
    tmp: Mem,
    d: PhantomData<N>,
}

impl<N> ClRotator2<N>
where N: BaseFloat + Clone + ClRotate + Absolute<N> + FromPrimitive {
    pub fn new(queue: CommandQueue,
               nx: usize, ny: usize,) -> Result<Self, Error> {
        let context = try!(queue.context());
        let device = try!(queue.device());
        let source = N::rotate2_source();
        let unbuilt = try!(Program::new_from_source(context.clone(), &[source]));
        let built = try!(unbuilt.build(&[device]));
        let shear_x = try!(built.create_kernel("shear_x"));
        let shear_y = try!(built.create_kernel("shear_y"));

        let tmp = try!(queue.create_buffer(size_of::<N>() * nx * ny));

        Ok(ClRotator2{
            nx: nx,
            ny: ny,
            wx: N::from_usize(nx - 1usize).unwrap() / N::from_f32(2f32).unwrap(),
            wy: N::from_usize(ny - 1usize).unwrap() / N::from_f32(2f32).unwrap(),
            queue: queue,
            shear_x: shear_x,
            shear_y: shear_y,
            tmp: tmp,
            d: PhantomData,
        })
    }

    fn forw_x(self: &mut Self,
              src: &Mem,
              rot: &PaethRotation2<N>,
              wait_for: &[Event]) -> Result<Event, Error> {
        let cx = N::one() / rot.xx;
        let cy = - rot.xy / rot.xx;
        let h = fmin(N::one(), N::one() / Absolute::<N>::abs(&rot.xy));

        let half = N::one() / (N::one() + N::one());
        let mut taus = vec![ half/rot.xx + half*rot.xy/rot.xx,
                             half/rot.xx - half*rot.xy/rot.xx,
                            -half/rot.xx + half*rot.xy/rot.xx,
                            -half/rot.xx - half*rot.xy/rot.xx ];
        taus.sort_by(|l, r| l.partial_cmp(r).unwrap());

        try!(self.shear_x.bind_scalar(0, &N::to_f32(&cx).unwrap()));
        try!(self.shear_x.bind_scalar(1, &N::to_f32(&cy).unwrap()));
        try!(self.shear_x.bind_scalar(2, &N::to_f32(&h).unwrap()));
        try!(self.shear_x.bind_scalar(3, &N::to_f32(&taus[0]).unwrap()));
        try!(self.shear_x.bind_scalar(4, &N::to_f32(&taus[1]).unwrap()));
        try!(self.shear_x.bind_scalar(5, &N::to_f32(&taus[2]).unwrap()));
        try!(self.shear_x.bind_scalar(6, &N::to_f32(&taus[3]).unwrap()));
        try!(self.shear_x.bind_scalar(7, &(self.nx as i32)));
        try!(self.shear_x.bind_scalar(8, &(self.ny as i32)));
        try!(self.shear_x.bind_scalar(9, &N::to_f32(&self.wx).unwrap()));
        try!(self.shear_x.bind_scalar(10, &N::to_f32(&self.wy).unwrap()));
        try!(self.shear_x.bind(11, src));
        try!(self.shear_x.bind_mut(12, &mut self.tmp));

        let local_size = (32, 8, 1);
        let global_size = (self.nx, self.ny, 1);

        self.queue.run_with_events(&mut self.shear_x, local_size, global_size,
                                   wait_for)
    }

    fn forw_y(self: &mut Self,
              dst: &mut Mem,
              rot: &PaethRotation2<N>,
              wait_for: &[Event]) -> Result<Event, Error> {
        let cy = N::one() / rot.yy;
        let cx = - rot.yx / rot.yy;
        let h = fmin(N::one(), N::one() / Absolute::<N>::abs(&rot.yx));

        let half = N::one() / (N::one() + N::one());
        let mut taus = vec![ half/rot.yy + half*rot.yx/rot.yy,
                             half/rot.yy - half*rot.yx/rot.yy,
                            -half/rot.yy + half*rot.yx/rot.yy,
                            -half/rot.yy - half*rot.yx/rot.yy ];
        taus.sort_by(|l, r| l.partial_cmp(r).unwrap());

        try!(self.shear_y.bind_scalar(0, &N::to_f32(&cy).unwrap()));
        try!(self.shear_y.bind_scalar(1, &N::to_f32(&cx).unwrap()));
        try!(self.shear_y.bind_scalar(2, &N::to_f32(&h).unwrap()));
        try!(self.shear_y.bind_scalar(3, &N::to_f32(&taus[0]).unwrap()));
        try!(self.shear_y.bind_scalar(4, &N::to_f32(&taus[1]).unwrap()));
        try!(self.shear_y.bind_scalar(5, &N::to_f32(&taus[2]).unwrap()));
        try!(self.shear_y.bind_scalar(6, &N::to_f32(&taus[3]).unwrap()));
        try!(self.shear_y.bind_scalar(7, &(self.nx as i32)));
        try!(self.shear_y.bind_scalar(8, &(self.ny as i32)));
        try!(self.shear_y.bind_scalar(9, &N::to_f32(&self.wx).unwrap()));
        try!(self.shear_y.bind_scalar(10, &N::to_f32(&self.wy).unwrap()));
        try!(self.shear_y.bind(11, &self.tmp));
        try!(self.shear_y.bind_mut(12, dst));

        let local_size = (32, 8, 1);
        let global_size = (self.ny, self.nx, 1);

        self.queue.run_with_events(&mut self.shear_y, local_size, global_size,
                                   wait_for)
    }

    pub fn forw(self: &mut Self,
                src: &Mem,
                dst: &mut Mem,
                rot: &PaethRotation2<N>,
                wait_for: &[Event]) -> Result<Event, Error> {
        let evt_x = try!(self.forw_x(src, rot, wait_for));
        self.forw_y(dst, rot, &[evt_x])
    }
}

#[test]
fn test_rot2_cl() {
    use na::{Vector1, Rotation2, ApproxEq};

    // setup rotation
    let r = Rotation2::new(Vector1::new(32f32));
    let p = PaethRotation2::new(&r);

    // setup opencl
    let platforms = Platform::platforms().expect("no platforms?");
    let devices: Vec<Device> = platforms[0].devices().expect("no devices?").filter(
        |d| match d.device_type() {
            Ok(DeviceType::GPU) => true,
            _ => false,
        }).collect();
    let context = Context::new(&devices).expect("can't get context?");
    let queue = CommandQueue::new(context, devices[0].clone()).expect("no queue?");

    // get rotator
    let rotator = ClRotator2::<f32>::new(queue, 512, 512).expect("error setting up rotator");
}


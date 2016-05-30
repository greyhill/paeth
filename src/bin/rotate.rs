extern crate paeth;
extern crate image;
extern crate proust;
extern crate nalgebra as na;

use self::image::*;
use self::proust::*;
use self::paeth::*;
use std::env;
use na::{Rotation2, Vector1, BaseFloat};

fn main() -> () {
    let args: Vec<String> = env::args().collect();
    let degrees: f32 = args[1].parse().expect("Couldn't understand degrees");
    let in_path = &args[2];
    let out_path = &args[3];

    let r = Rotation2::new(Vector1::new(degrees * f32::pi() / 180f32));
    let p = PaethRotation2::new(&r);

    let image_in = open(in_path).expect("Error opening input image").to_luma();
    let nx = image_in.width() as usize;
    let ny = image_in.height() as usize;
    println!("Image is {}x{}", nx, ny);
    let pixels_in: Vec<f32> = image_in.into_raw().into_iter().map(|p| p as f32).collect();

    // setup opencl
    let platforms = Platform::platforms().expect("no cl platforms");
    let devices: Vec<Device> = platforms[0].devices().expect("no devices?").into_iter().filter(
        |d| match d.device_type() {
            Ok(DeviceType::GPU) => true,
            _ => false,
        }).collect();
    let context = Context::new(&devices).expect("cannot create context");
    let queue = CommandQueue::new(context, devices[0].clone()).expect("cannot create queue");

    let mut rotator = ClRotator2::<f32>::new(queue.clone(), nx, ny).expect("error setting up rotator");

    // rotate in-place
    let buf_in = queue.create_buffer_from_slice(&pixels_in).expect("error loading image");
    let mut buf_out = buf_in.clone();
    rotator.forw(&buf_in, &mut buf_out, &p, &[]).expect("error rotating").wait().expect("error waiting for rotation");

    let mut out_f32: Vec<f32> = vec![0f32; nx*ny];
    queue.read_buffer(&buf_out, &mut out_f32).expect("error reading");
    let out_u8 = out_f32.into_iter().map(|p| p as u8).collect();
    let out = GrayImage::from_raw(nx as u32, ny as u32, out_u8).expect("error creating output image");

    out.save(out_path).expect("error saving output image");
}


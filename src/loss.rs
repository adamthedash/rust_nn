use crate::nd_tensor::NDTensor;

pub struct SumSquaredError {}

impl SumSquaredError {
    pub fn forward(gt: &NDTensor<f32>, pred: &NDTensor<f32>) -> f32 {
        (gt - pred).pow(2.).sum()
    }

    pub fn gradient(gt: &NDTensor<f32>, pred: &NDTensor<f32>) -> NDTensor<f32> {
        let diff = gt - pred;

        let mut c = NDTensor::from_vector(&vec![-2. / gt.numel() as f32; gt.numel()]);
        c.reshape(&gt.shape);

        &c * &diff
    }
}
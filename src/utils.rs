/// Copy a vector
#[macro_export]
macro_rules! copy_vec {
    ($var:expr) => {$var.iter().copied().collect::<Vec<_>>()};
}
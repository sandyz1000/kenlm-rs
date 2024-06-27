use std::fmt;
use std::error::Error;


#[derive(Debug)]
struct BadDiscountException;

impl fmt::Display for BadDiscountException {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BadDiscountException")
    }
}

impl Error for BadDiscountException {}

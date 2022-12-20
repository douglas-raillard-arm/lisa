mod cparser;
mod grammar;
mod header;
mod parser;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_test() {
        let data = include_bytes!("../../../../doc/traces/trace.dat");
        // println!("bytes:\n{}", &data[0..100].to_hex(8));
        let res = header::header(data);
        match res {
            Ok((_, _x)) => {
                // println!("{x:?}")
            }
            Err(err) => {
                println!("{err:?}");
                panic!("failed to parse header")
            }
        }
    }
}

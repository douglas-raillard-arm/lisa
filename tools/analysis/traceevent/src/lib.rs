mod cparser;
mod grammar;
mod header;
mod io;
mod parser;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_test() {
        let data = include_bytes!("../../../../doc/traces/trace.dat");
        // println!("bytes:\n{}", &data[0..100].to_hex(8));

        // let reader = crate::io::BorrowingCursor::new(&data[..]);

        // let reader = std::io::Cursor::new(&data[..]);
        // let reader = crate::io::BorrowingBufReader::new(reader, Some(4096));

        // let mut file =
        //     std::fs::File::open("/home/dourai01/Work/lisa/lisa/doc/traces/trace.dat").unwrap();
        // use std::rc::Rc;
        // use std::cell::RefCell;
        // let reader = crate::io::CursorReader{inner: Rc::new(RefCell::new(file)), offset: 0};
        // let reader = crate::io::BorrowingBufReader::new(reader, Some(4096));

        // let mut file =
        //     std::fs::File::open("/home/dourai01/Work/lisa/lisa/doc/traces/trace.dat").unwrap();
        // let reader = unsafe { crate::io::MmapFile::new(file) }.unwrap();

        let mut file =
            std::fs::File::open("/home/dourai01/Work/lisa/lisa/doc/traces/trace.dat").unwrap();
        let reader = crate::io::FallbackBorrowingReader::new(
            || unsafe { crate::io::MmapFile::new(file) },
            || Ok(crate::io::BorrowingCursor::new(&data[..])),
        ).unwrap();

        let res = header::header(reader);
        match res {
            Ok(x) => {
                println!("{x:?}")
            }
            Err(err) => {
                println!("{err:?}");
                panic!("failed to parse header")
            }
        }
    }
}

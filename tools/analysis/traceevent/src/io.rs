use core::{borrow::Borrow, cell::RefCell, convert::AsRef, ops::Deref};

use std::{
    borrow::Cow,
    io,
    io::{BufRead, BufReader, Error, ErrorKind, Read, Seek, SeekFrom},
    mem::size_of,
    os::fd::AsRawFd,
    rc::Rc,
};

use nom::{Finish as _, IResult, Parser};

use crate::{header::Endianness, parser::NomError};

type FileOffset = u64;
type FileSize = FileOffset;

type MemOffset = usize;
type MemSize = MemOffset;

// We have to know which of MemOffset and FileOffset is the biggest. This module assumes
// MemOffset <= FileOffset, so it converts MemOffset to FileOffset when a common denominator is
// required.
#[inline]
fn mem2file(x: MemOffset) -> FileOffset {
    x.try_into()
        .expect("Could not convert MemOffset to FileOffset")
}

pub trait BorrowingRead {
    // The 'a lifetime is the lifetime of Self, so that the returned Bytes can
    // simply be a &[u8] pointing to an internal buffer of Self.
    type Bytes<'a>: Deref<Target = [u8]>
    where
        Self: 'a;
    type OffsetReader: BorrowingRead;

    fn read<'a>(&mut self, count: MemSize) -> io::Result<Self::Bytes<'_>>;
    fn read_null_terminated(&mut self) -> io::Result<Self::Bytes<'_>>;
    fn split_at_offsets(
        self,
        offsets: &[(FileOffset, FileSize)],
    ) -> io::Result<Vec<Self::OffsetReader>>;

    #[inline]
    fn parse<P, O, E>(&mut self, count: MemSize, mut p: P) -> io::Result<Result<O, E>>
    where
        // Sadly we can't take an for<'b>impl Parser<Input<'b>, _, _> as it
        // seems impossible to build real-world parsers complying with for<'b>.
        // In practice, the error type needs bounds involving 'b, and there is
        // no way to shove these bounds inside the scope of for<'b> (for now at
        // least). As a result, 'b needs to be a generic param on the parser
        // function and our caller gets to choose the lifetime, not us.
        P: for<'b> Fn(&'b [u8]) -> IResult<&'b [u8], O, NomError<E, ()>>,
    {
        let buf = self.read(count)?;
        Ok(match p.parse(&buf).finish() {
            Err(err) => Err(err.data),
            Ok((_, x)) => Ok(x),
        })
    }

    #[inline]
    fn read_int<T>(&mut self, endianness: Endianness) -> io::Result<T>
    where
        T: DecodeBinary,
    {
        Ok(DecodeBinary::decode(
            &self.read(size_of::<T>())?,
            endianness,
        )?)
    }

    #[inline]
    fn read_tag<'b, T, E>(&mut self, tag: T, or: E) -> io::Result<Result<(), E>>
    where
        T: IntoIterator<Item = &'b u8>,
        T::IntoIter: ExactSizeIterator,
    {
        let tag = tag.into_iter();
        let buff = self.read(tag.len())?;
        let eq = buff.iter().eq(tag);
        Ok(if eq { Ok(()) } else { Err(or) })
    }
}

/// Newtype wrapper for &[u8] that allows zero-copy operations from
/// [`BorrowingRead`]. It is similar to what [`Cursor`] provides to [`Read`].
pub struct BorrowingCursor<'a>(&'a [u8]);

impl<'a> BorrowingCursor<'a> {
    #[inline]
    pub fn new(buf: &'a [u8]) -> Self {
        BorrowingCursor(buf)
    }

    #[inline]
    fn advance(&mut self, count: MemOffset) {
        self.0 = &self.0[count..];
    }
}

impl<'a> BorrowingRead for BorrowingCursor<'a> {
    type Bytes<'b> = &'b [u8] where Self: 'b;
    type OffsetReader = BorrowingCursor<'a>;

    #[inline]
    fn read(&mut self, count: MemSize) -> io::Result<Self::Bytes<'_>> {
        let buf = &self.0[..count];
        self.advance(count);

        if buf.len() == count {
            Ok(buf)
        } else {
            #[cold]
            Err(ErrorKind::UnexpectedEof.into())
        }
    }

    #[inline]
    fn read_null_terminated(&mut self) -> io::Result<Self::Bytes<'_>> {
        match self.0.iter().position(|x| *x == 0) {
            Some(end) => {
                self.advance(end + 1);
                Ok(&self.0[..end])
            }
            #[cold]
            None => {
                self.advance(self.0.len());
                Err(ErrorKind::UnexpectedEof.into())
            }
        }
    }

    fn split_at_offsets(
        self,
        offsets: &[(FileOffset, FileSize)],
    ) -> io::Result<Vec<Self::OffsetReader>> {
        offsets
            .iter()
            .map(|(offset, len)| {
                #[inline]
                fn convert(x: FileOffset) -> io::Result<MemOffset> {
                    x.try_into().map_err(
                        #[cold]
                        |_| ErrorKind::UnexpectedEof.into(),
                    )
                }

                let offset = convert(*offset)?;
                let len = convert(*len)?;

                Ok(BorrowingCursor::new(&self.0[offset..offset + len]))
            })
            .collect()
    }
}

/////////////
// Memory map
/////////////

/// Bounded view in a mmap mapping. This is equivalent to a &[u8] except that it
/// also holds a ref-counted ref to the underlying map for automatic unmapping.
/// At times, it can also simply contain a Vec<u8> if the requested data could
/// not be serviced with mmap for some reason.
pub enum MmapView {
    Mmap {
        mmap: Rc<memmap2::Mmap>,
        start: MemOffset,
        end: MemOffset,
    },
    Buffer(Vec<u8>),
}

impl MmapView {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        match self {
            MmapView::Mmap { mmap, start, end } => &mmap[*start..*end],
            #[cold]
            MmapView::Buffer(vec) => &vec,
        }
    }
}

impl Deref for MmapView {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_bytes()
    }
}

impl AsRef<[u8]> for MmapView {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl Borrow<[u8]> for MmapView {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self
    }
}

pub struct Mmap {
    file_offset: FileOffset,
    read_offset: MemOffset,
    len: MemSize,
    mmap: Rc<memmap2::Mmap>,
}

impl Mmap {
    unsafe fn new<T>(file: &T, offset: FileOffset, mut len: MemSize) -> io::Result<Self>
    where
        T: AsRawFd,
    {
        //SAFETY: mmap is inherently unsafe as the memory content could change
        // without notice if the backing file is modified. We have to rely on
        // the user/OS being nice to us and not do that, or we might crash,
        // there is no way around it unfortunately.
        let mmap = loop {
            let mmap = unsafe {
                memmap2::MmapOptions::new()
                    .offset(offset)
                    .len(len)
                    .populate()
                    .map(&*file)
            };
            match mmap {
                Ok(mmap) => break Ok(mmap),
                Err(err) => {
                    len /= 2;
                    if len == 0 {
                        break Err(err);
                    }
                }
            }
        }?;

        mmap.advise(memmap2::Advice::WillNeed);
        mmap.advise(memmap2::Advice::Sequential);

        Ok(Mmap {
            len,
            file_offset: offset,
            read_offset: 0,
            mmap: Rc::new(mmap),
        })
    }

    #[inline]
    fn file_read_offset(&self) -> FileOffset {
        let read_offset = mem2file(self.read_offset);
        self.file_offset + read_offset
    }

    #[inline]
    fn read(&mut self, count: MemSize) -> Option<MmapView> {
        let read_offset = self.read_offset;

        if self.len - read_offset >= count {
            let view = MmapView::Mmap {
                mmap: Rc::clone(&self.mmap),
                start: read_offset,
                end: read_offset + count,
            };
            self.read_offset += count;
            Some(view)
        } else {
            #[cold]
            None
        }
    }
}

impl Deref for Mmap {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        // The memory map already has a length setup, so we only need to take
        // the offset into that.
        &self.mmap[self.read_offset..]
    }
}

pub struct MmapFile<T> {
    // Use a Rc<RefCell<_>> so that we can clone the reference when creating
    // MmapFile::OffsetReader
    file: Rc<RefCell<T>>,
    len: FileSize,

    mmap: Mmap,
}

impl<T> MmapFile<T> {
    pub unsafe fn new(mut file: T) -> io::Result<MmapFile<T>>
    where
        T: AsRawFd + Seek,
    {
        let offset = 0;
        let len = file_len(&mut file)?;
        let file = Rc::new(RefCell::new(file));
        Self::from_cell(file, offset, len)
    }

    unsafe fn from_cell(
        file: Rc<RefCell<T>>,
        offset: FileOffset,
        len: FileSize,
    ) -> io::Result<MmapFile<T>>
    where
        T: AsRawFd,
    {
        let mmap_len = len.try_into().unwrap_or(MemSize::MAX);
        let mmap = Mmap::new(&*RefCell::borrow(&file), offset, mmap_len)?;

        Ok(MmapFile { file, len, mmap })
    }

    #[inline]
    unsafe fn remap(&mut self, offset: FileOffset) -> io::Result<Mmap>
    where
        T: AsRawFd,
    {
        // Try to map all the remainder of the file range of interest.
        let len = self.len - offset;
        // Saturate at the max size possible for a mmap
        let len: MemSize = len.try_into().unwrap_or(MemSize::MAX);
        Ok(Mmap::new(&*RefCell::borrow(&self.file), offset, len)?)
    }
}

impl<T> BorrowingRead for MmapFile<T>
where
    T: AsRawFd + Read + Seek,
{
    type Bytes<'a> = MmapView where Self: 'a;
    type OffsetReader = Self;

    #[inline]
    fn read(&mut self, count: MemSize) -> io::Result<Self::Bytes<'_>> {
        match self.mmap.read(count) {
            Some(view) => Ok(view),
            // Not enough bytes left in the mmap, so we need to remap to catch
            // up with the read offset.
            #[cold]
            None => {
                let mut mmap = unsafe { self.remap(self.mmap.file_read_offset())? };
                match mmap.read(count) {
                    Some(view) => {
                        // Useful mmap, we keep it for next time
                        self.mmap = mmap;
                        Ok(view)
                    }
                    // Remapping was not enough, we need to fallback on read()
                    // syscall. We discard the mapping we just created, as it's
                    // useless since it cannot service even the first read.
                    #[cold]
                    None => {
                        let file_offset = self.mmap.file_read_offset();

                        // Remap for future reads after the syscall we are abount to do.
                        self.mmap = unsafe { self.remap(file_offset + mem2file(count))? };

                        let mut file = RefCell::borrow_mut(&self.file);
                        file.seek(SeekFrom::Start(file_offset))?;
                        let vec = read(&mut *file, count)?;
                        Ok(MmapView::Buffer(vec))
                    }
                }
            }
        }
    }

    #[inline]
    fn read_null_terminated(&mut self) -> io::Result<Self::Bytes<'_>> {
        let find = |buf: &[u8]| buf.iter().position(|x| *x == 0);

        for _ in 0..2 {
            match find(&self.mmap) {
                Some(end) => {
                    let mut view = self.mmap.read(end + 1).unwrap();
                    // Remove the null terminator from the view
                    match &mut view {
                        MmapView::Mmap { end, .. } => {
                            *end -= 1;
                        }
                        MmapView::Buffer(vec) => {
                            vec.pop();
                        }
                    };
                    return Ok(view);
                }
                // Update the mapping to catch up with the current read offset and try again.
                #[cold]
                None => {
                    if mem2file(self.mmap.len) == self.len {
                        return Err(ErrorKind::UnexpectedEof.into());
                    } else {
                        self.mmap = unsafe { self.remap(self.mmap.file_read_offset())? };
                    }
                }
            }
        }

        // We failed to find the pattern in the area covered by mmap, so try
        // again with read() syscall.

        let vec: Vec<u8>;
        {
            let mut file = &mut *RefCell::borrow_mut(&self.file);
            let file_offset = self.mmap.file_read_offset();
            file.seek(SeekFrom::Start(file_offset))?;

            let buf_size = 4096;
            let mut vec_vec = Vec::new();

            loop {
                let mut read_vec = Vec::with_capacity(buf_size);
                file.take(mem2file(buf_size)).read_to_end(&mut read_vec)?;

                match find(&read_vec) {
                    Some(end) => {
                        read_vec.truncate(end);
                        vec_vec.push(read_vec);
                        vec = vec_vec.into_iter().flatten().collect();
                        break;
                    }
                    None => {
                        vec_vec.push(read_vec);
                    }
                }
            }
        }

        // Remap for future reads after the syscall we are about to do.
        let mmap_offset = self.mmap.file_read_offset() + mem2file(vec.len()) + 1;
        self.mmap = unsafe { self.remap(mmap_offset)? };
        Ok(MmapView::Buffer(vec))
    }

    fn split_at_offsets(
        self,
        offsets: &[(FileOffset, FileSize)],
    ) -> io::Result<Vec<Self::OffsetReader>> {
        let file = self.file;
        offsets
            .iter()
            .map(move |(offset, len)| unsafe { Self::from_cell(file.clone(), *offset, *len) })
            .collect()
    }
}

// libtraceevent deals with it by keeping a loaded page for each CPU buffer. The page either comes from:
// * a mmap (not necessarily the whole file, it can deal with a small bit)
// * a simple seek + read + seek sequence to load a page in memory.
//
// This means it cannot consume a non-seek fd (tested with cat trace.dat | trace-cmd report /dev/stdin).
//
// This allows efficient access that avoids seeking all the time (only a few
// seeks to load a whole page), and there is one read buffer for each CPU buffer
// (as opposed to a single BufReader that would never preload the right piece of
// info since it would be shared for multiple offsets).
//

pub struct BorrowingBufReader<T> {
    inner: BufReader<T>,
    consume: MemSize,
}

impl<T> BorrowingBufReader<T>
where
    T: Read,
{
    pub fn new(reader: T, buf_size: Option<MemOffset>) -> Self {
        let buf_size = buf_size.unwrap_or(4096);
        BorrowingBufReader {
            inner: BufReader::with_capacity(buf_size, reader),
            consume: 0,
        }
    }
    fn consume(&mut self) {
        self.inner.consume(self.consume);
        self.consume = 0;
    }
}

impl<T> BorrowingRead for BorrowingBufReader<T>
where
    T: Read + Seek,
{
    type Bytes<'a> = Cow<'a, [u8]> where Self: 'a;
    type OffsetReader = BorrowingBufReader<CursorReader<T>>;

    #[inline]
    fn read<'a>(&'a mut self, count: MemSize) -> io::Result<Self::Bytes<'a>> {
        self.consume();

        let buf = self.inner.fill_buf()?;
        let len = buf.len();

        if len == 0 && count > 0 {
            #[cold]
            Err(ErrorKind::UnexpectedEof.into())
        } else if count < len {
            self.consume = count;
            Ok(Cow::Borrowed(&self.inner.buffer()[..count]))
        } else {
            #[cold]
            // Pre-filled buffer not large enough for that read, fallback on
            // read() syscall
            Ok(Cow::Owned(read(&mut self.inner, count)?))
        }
    }

    #[inline]
    fn read_null_terminated<'a>(&'a mut self) -> io::Result<Self::Bytes<'a>> {
        self.consume();

        let buf = self.inner.fill_buf()?;
        let end = buf.iter().position(|x| *x == 0);

        match end {
            Some(end) => {
                if end == 0 {
                    Ok(Cow::Owned(Vec::new()))
                } else {
                    self.consume = end + 1;
                    // For some reason, the borrow checker is not happy for us
                    // to use buf directly, so we fetch it again with
                    // self.inner.buffer()
                    Ok(Cow::Borrowed(&self.inner.buffer()[..end]))
                }
            }
            #[cold]
            None => {
                // If we could not find the data in the pre-loaded buffer, just read
                // as much as needed
                let mut vec = Vec::new();
                self.inner.read_until(0, &mut vec)?;
                if vec.last() == Some(&0) {
                    // Remove the null terminator
                    vec.pop();
                    Ok(Cow::Owned(vec))
                } else {
                    #[cold]
                    // read_until() read until the end of the file and did not find the
                    // null terminator.
                    Err(ErrorKind::UnexpectedEof.into())
                }
            }
        }
    }

    fn split_at_offsets(
        mut self,
        offsets: &[(FileOffset, FileSize)],
    ) -> io::Result<Vec<Self::OffsetReader>> {
        self.consume();

        let capacity = Some(self.inner.capacity());
        let reader = self.inner.into_inner();
        let reader = Rc::new(RefCell::new(reader));

        Ok(offsets
            .iter()
            .map(|(offset, _len)| {
                BorrowingBufReader::new(
                    CursorReader {
                        inner: reader.clone(),
                        offset: *offset,
                    },
                    capacity,
                )
            })
            .collect())
    }
}

pub struct CursorReader<T> {
    inner: Rc<RefCell<T>>,
    offset: FileOffset,
}

impl<T> Read for CursorReader<T>
where
    T: Read + Seek,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<MemSize> {
        let mut reader = RefCell::borrow_mut(&self.inner);
        reader.seek(SeekFrom::Start(self.offset))?;
        let count = reader.read(buf)?;
        self.offset += mem2file(count);
        Ok(count)
    }
}

impl<T> Seek for CursorReader<T>
where
    T: Seek,
{
    fn seek(&mut self, pos: SeekFrom) -> io::Result<FileOffset> {
        let mut reader = RefCell::borrow_mut(&self.inner);
        self.offset = reader.seek(pos)?;
        Ok(self.offset)
    }
}

// pub enum AttemptMmapFile<'a, R> {
//     Mmap(MmapFile<'a, R>),

// }

// pub trait BorrowingRead {
//     // The 'a lifetime is the lifetime of Self, so that the returned Bytes can
//     // simply be a &[u8] pointing to an internal buffer of Self.
//     type Bytes<'a>: Deref<Target = [u8]>
//     where
//         Self: 'a;
//     type OffsetReader: BorrowingRead;

//     fn read<'a>(&mut self, count: MemSize) -> io::Result<Self::Bytes<'_>>;
//     fn read_null_terminated(&mut self) -> io::Result<Self::Bytes<'_>>;
//     fn split_at_offsets(self, offsets: &[(FileOffset, FileSize)]) -> io::Result<Vec<Self::OffsetReader>>;

fn read<T>(reader: &mut T, count: MemSize) -> io::Result<Vec<u8>>
where
    T: Read,
{
    let mut vec = Vec::with_capacity(count);
    let take_nr = count.try_into().map_err(
        #[cold]
        |_| Error::from(ErrorKind::UnexpectedEof),
    )?;

    let nr_read = reader.take(take_nr).read_to_end(&mut vec)?;

    if nr_read == count {
        Ok(vec)
    } else {
        #[cold]
        Err(ErrorKind::UnexpectedEof.into())
    }
}

#[inline]
fn file_len<T>(stream: &mut T) -> io::Result<FileSize>
where
    T: Seek,
{
    let old_pos = stream.stream_position()?;
    let len = stream.seek(SeekFrom::End(0))?;
    stream.seek(SeekFrom::Start(old_pos))?;
    Ok(len)
}

pub trait DecodeBinary: Sized {
    fn decode(buf: &[u8], endianness: Endianness) -> io::Result<Self>;
}

macro_rules! impl_DecodeBinary {
    ( $($ty:ty),* ) => {
        $(
            impl DecodeBinary for $ty {
                #[inline]
                fn decode(buf: &[u8], endianness: Endianness) -> io::Result<Self> {
                    match buf.try_into() {
                        Ok(buf) => Ok(match endianness {
                            Endianness::Little => Self::from_le_bytes(buf),
                            Endianness::Big => Self::from_be_bytes(buf),
                        }),
                        #[cold]
                        Err(_) => Err(ErrorKind::UnexpectedEof.into())
                    }
                }
            }
        )*
    }
}

impl_DecodeBinary!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);

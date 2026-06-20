import struct
import time
import numpy as np
from multiprocessing import shared_memory

__all__ = ["SharedMemoryBuffer"]


import struct
import time
import numpy as np
from multiprocessing import shared_memory

class SharedMemoryBuffer:
    MAGIC = 0x5349474E
    VERSION = 1
    MAX_NAME_LEN = 32

    def __init__(self, name=None, create=False, num_signals=None, buffer_size=None, signal_names=None):
        self.name = name
        self.shm = None
        self.data = None
        self.attached = False

        if create:
            # ----- Writer side -----
            if None in (num_signals, buffer_size, signal_names):
                raise ValueError("Must provide num_signals, buffer_size, signal_names when creating")
            if len(signal_names) != num_signals:
                raise ValueError("Number of signal names must match num_signals")

            self.num_signals = num_signals
            self.buffer_size = buffer_size
            self.signal_names = list(signal_names)

            # Header layout: magic(4), version(4), num_signals(4), buffer_size(4), write_idx(4), names[]
            header_size = 5 * 4 + self.MAX_NAME_LEN * num_signals
            header_size = (header_size + 7) & ~7   # align to 8 bytes
            self.header_size = header_size

            data_size = buffer_size * num_signals * 8   # float64
            total_size = header_size + data_size

            # Create shared memory (anonymous if name is None)
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=total_size)
            # If name was None, the OS assigns a unique name; store it
            self.name = self.shm.name

            # Write header
            header = struct.pack('<IIIII', self.MAGIC, self.VERSION, num_signals, buffer_size, 0)
            names_bytes = b''
            for s in signal_names:
                encoded = s.encode('utf-8')[:self.MAX_NAME_LEN-1]
                names_bytes += encoded.ljust(self.MAX_NAME_LEN, b'\0')
            header += names_bytes
            header = header.ljust(header_size, b'\0')
            self.shm.buf[:header_size] = header

            # Data view
            self.data = np.frombuffer(self.shm.buf, dtype=np.float64, offset=header_size)
            self.data = self.data.reshape((buffer_size, num_signals))
            self.data[:] = 0.0
            self.write_index = 0
            self.attached = True

        else:
            # ----- Reader side (name must be provided) -----
            if name is None:
                raise ValueError("Reader must provide a shared memory name")
            self.shm = shared_memory.SharedMemory(name=name)

            # Read header
            header_prefix = self.shm.buf[:20]   # first five uint32
            magic, version, n_sig, b_size, w_idx = struct.unpack('<IIIII', header_prefix)
            if magic != self.MAGIC:
                raise ValueError("Invalid magic – wrong shared memory?")
            if version != self.VERSION:
                raise ValueError(f"Unsupported header version {version}")

            self.num_signals = n_sig
            self.buffer_size = b_size
            self.write_index = w_idx

            # Compute header size (must match writer)
            header_size = 5 * 4 + self.MAX_NAME_LEN * n_sig
            header_size = (header_size + 7) & ~7
            self.header_size = header_size

            # Read names
            names_start = 20
            names_bytes = self.shm.buf[names_start:names_start + self.MAX_NAME_LEN * n_sig]
            self.signal_names = []
            for i in range(n_sig):
                start = i * self.MAX_NAME_LEN
                end = start + self.MAX_NAME_LEN
                # Convert memoryview slice to bytes before splitting
                raw = bytes(names_bytes[start:end])
                name = raw.split(b'\0', 1)[0].decode('utf-8')
                self.signal_names.append(name)

            # Data view
            self.data = np.frombuffer(self.shm.buf, dtype=np.float64, offset=header_size)
            self.data = self.data.reshape((b_size, n_sig))
            self.attached = True

    def write_samples(self, samples):
        """Write one time step of data (list/array of values for all signals)."""
        if not self.attached:
            raise RuntimeError("Buffer not attached")
        if len(samples) != self.num_signals:
            raise ValueError(f"Expected {self.num_signals} values, got {len(samples)}")
        idx = self.write_index % self.buffer_size
        self.data[idx] = samples
        self.write_index += 1
        # Update header write index
        self.shm.buf[16:20] = struct.pack('<I', self.write_index)

    def read_latest(self, n_points=None):
        """
        Return a copy of the most recent data in chronological order.
        If n_points is None: returns the whole buffer, oldest first, newest last.
        If n_points is given: returns the most recent n_points samples.
        """
        if not self.attached:
            raise RuntimeError("Buffer not attached")

        # Read current write index from header
        current_write = struct.unpack('<I', self.shm.buf[16:20])[0]
        pos = current_write % self.buffer_size

        if n_points is None:
            # Return whole buffer in chronological order: oldest first
            if pos == 0:
                return self.data.copy()
            else:
                # data[pos:] is the older part (written before wrap)
                # data[:pos] is the newer part (written after wrap)
                return np.vstack((self.data[pos:], self.data[:pos]))
        else:
            # Return the most recent n_points in chronological order
            n = min(n_points, self.buffer_size)
            if pos >= n:
                return self.data[pos-n:pos].copy()
            else:
                # Need to wrap: take from end and beginning
                part1 = self.data[:pos]               # newest (beginning)
                part2 = self.data[-(n-pos):]          # older (end)
                # Stack so that oldest is first: part2 (older) then part1 (newer)
                return np.vstack((part2, part1))

    def get_write_index(self):
        """Total number of samples written so far (read live from the header)."""
        return struct.unpack('<I', self.shm.buf[16:20])[0]

    def reset(self):
        """Restart recording at sample 0 (e.g. at the beginning of the cut, so
        the monitor shows the cut rather than the long approach/IK phase)."""
        self.write_index = 0
        self.shm.buf[16:20] = struct.pack('<I', 0)

    def get_signal_names(self):
        return self.signal_names

    def close(self):
        """Detach from shared memory (both reader and writer)."""
        if self.shm:
            # Release numpy reference first
            self.data = None
            self.shm.close()
        self.attached = False

    def unlink(self):
        """Remove the shared memory segment (writer only)."""
        if self.shm:
            self.data = None
            self.shm.close()
            self.shm.unlink()
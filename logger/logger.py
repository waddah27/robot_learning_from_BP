import os
import sys
import time
import threading
from typing import Optional

__all__ = ["Logger"]


"""
Static Logger Utility  No instantiation needed.
Logs to console and file. File created automatically.
Automatic rotation when size exceeds max_bytes.
Manual clear_log() truncates the main log and optionally deletes backups.
Thread safe.
"""

import os
import sys
import time
import threading
from typing import Optional

class Logger:
    """Static logger  use class methods directly, e.g., Logger.info('...')"""

    # Class variables (shared state)
    _log_file: Optional[str] = None
    _level: int = 20                     # INFO default
    _max_bytes: Optional[int] = None
    _backup_count: int = 3
    _format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    _name: str = "StaticLogger"

    _file_handle = None                  # File object for writing
    _lock = threading.Lock()
    _configured: bool = False

    # Log level constants
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    _level_names = {
        DEBUG: "DEBUG",
        INFO: "INFO",
        WARNING: "WARNING",
        ERROR: "ERROR",
        CRITICAL: "CRITICAL",
    }

    @classmethod
    def configure(
        cls,
        log_file: str = "app.log",
        level: int = INFO,
        max_bytes: Optional[int] = None,
        backup_count: int = 3,
        format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        name: str = "StaticLogger",
    ) -> None:
        """
        Set up the logger. Call once at program start (optional – if not called,
        defaults will be used on first log).
        """
        with cls._lock:
            cls._log_file = log_file
            cls._level = level
            cls._max_bytes = max_bytes
            cls._backup_count = max(1, backup_count) if max_bytes else 0
            cls._format = format_string
            cls._name = name
            cls._configured = True

            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Open file in append mode (creates if missing)
            cls._open_file()

    @classmethod
    def _open_file(cls) -> None:
        """Open (or reopen) the log file in append mode."""
        if not cls._log_file:
            raise RuntimeError("Logger not configured. Call configure() first.")
        if cls._file_handle:
            cls._file_handle.close()
        cls._file_handle = open(cls._log_file, "a", encoding="utf-8")

    @classmethod
    def _rotate(cls) -> None:
        """Rotate log file if size exceeds max_bytes."""
        if not cls._max_bytes or not cls._log_file:
            return
        try:
            if os.path.getsize(cls._log_file) <= cls._max_bytes:
                return
        except OSError:
            return  # file may not exist yet

        # Close current file
        if cls._file_handle:
            cls._file_handle.close()
            cls._file_handle = None

        # Shift existing backups: .1 -> .2, .2 -> .3, etc.
        for i in range(cls._backup_count - 1, 0, -1):
            src = f"{cls._log_file}.{i}"
            dst = f"{cls._log_file}.{i+1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)

        # Rename current log to .1
        backup = f"{cls._log_file}.1"
        if os.path.exists(backup):
            os.remove(backup)
        os.rename(cls._log_file, backup)

        # Reopen fresh log file
        cls._open_file()

    @classmethod
    def _write(cls, level: int, message: str) -> None:
        """Internal: format and write a log record."""
        if not cls._configured:
            # Auto‑configure with defaults on first use
            cls.configure()

        if level < cls._level:
            return

        # Format the log line
        record = {
            "asctime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "name": cls._name,
            "levelname": cls._level_names.get(level, f"LEVEL{level}"),
            "message": message,
        }
        log_line = cls._format % record

        with cls._lock:
            # Write to console
            print(log_line, file=sys.stdout)

            # Write to file
            if cls._file_handle:
                cls._file_handle.write(log_line + "\n")
                cls._file_handle.flush()
                if cls._max_bytes:
                    cls._rotate()

    # Public logging methods
    @classmethod
    def debug(cls, message: str) -> None:
        cls._write(cls.DEBUG, message)

    @classmethod
    def info(cls, message: str) -> None:
        cls._write(cls.INFO, message)

    @classmethod
    def warning(cls, message: str) -> None:
        cls._write(cls.WARNING, message)

    @classmethod
    def error(cls, message: str) -> None:
        cls._write(cls.ERROR, message)

    @classmethod
    def critical(cls, message: str) -> None:
        cls._write(cls.CRITICAL, message)

    @classmethod
    def clear_log(cls, keep_backups: bool = False) -> None:
        """
        Manually clear the main log file.
        :param keep_backups: If True, backup files are left untouched;
                             if False, they are also deleted.
        """
        if not cls._configured:
            return

        with cls._lock:
            # Close and truncate main log
            if cls._file_handle:
                cls._file_handle.close()
                cls._file_handle = None

            # Truncate main log file
            with open(cls._log_file, "w", encoding="utf-8") as f:
                f.truncate(0)

            # Reopen for future writes
            cls._open_file()

            # Optionally delete backups
            if not keep_backups and cls._max_bytes:
                for i in range(1, cls._backup_count + 1):
                    backup = f"{cls._log_file}.{i}"
                    try:
                        os.remove(backup)
                    except OSError:
                        pass

    @classmethod
    def close(cls) -> None:
        """Explicitly close the log file (optional, call at program exit)."""
        with cls._lock:
            if cls._file_handle:
                cls._file_handle.close()
                cls._file_handle = None

Logger.configure(
        log_file="test.log",
        level=Logger.DEBUG,
        max_bytes=1024*1024*50,        # rotate after 50 MB
        backup_count=2
    )
# --- Usage examples (just uncomment to test) ---
if __name__ == "__main__":
    # Optional configuration (if omitted, defaults are used on first log)
    Logger.configure(
        log_file="test.log",
        level=Logger.DEBUG,
        max_bytes=1024,        # rotate after 1 KB (for testing)
        backup_count=2
    )

    Logger.debug("Application started")
    Logger.debug("Debug message")
    Logger.debug("Warning message")
    Logger.debug("Error message")
    Logger.debug("Critical message")

    # Manually clear the log (and delete backups)
    # Logger.clear_log(keep_backups=False)

    # Close file handle (good practice, but not strictly required)
    Logger.close()
